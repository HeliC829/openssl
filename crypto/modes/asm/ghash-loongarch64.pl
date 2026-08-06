#! /usr/bin/env perl
# Copyright 2026 The OpenSSL Project Authors. All Rights Reserved.
#
# Licensed under the Apache License 2.0 (the "License").  You may not use
# this file except in compliance with the License.  You can obtain a copy
# in the file LICENSE in the source distribution or at
# https://www.openssl.org/source/license.html

use strict;

######################################################################
# GHASH for LoongArch64 SCALAR/LSX/LASX (128/256-bit SIMD)
#
# Carryless multiply via vshuf.b 4-bit table lookup with Karatsuba
# decomposition. No hardware clmul/PMULL required.
#
# LASX mode: 2-block parallel GHASH against H and H^2.  Both multiplies
# (A[0]*H^2 and A[1]*H) happen simultaneously in the two 128-bit lanes
# of 256-bit LASX registers.
#
# Algorithm overview:
#   1. Convert data to polynomial representation via brev8 (bit-reverse
#      within each byte). After brev8 on LE LoongArch:
#        elem0 = x^0..x^63, elem1 = x^64..x^127
#      This is the natural layout for our vshuf.b-based clmul64.
#
#   2. For 128x128 GF(2^128) multiply, use Karatsuba decomposition:
#        P_lo  = clmul64(Xi.lo, H.lo)
#        P_hi  = clmul64(Xi.hi, H.hi)
#        P_mid = clmul64(Xi.lo^Xi.hi, H.lo^H.hi)
#      Then combine to get 256-bit product and reduce mod
#        f(x) = x^128 + x^7 + x^2 + x + 1.
#
#   3. clmul64(a, b) = polynomial multiply of two 64-bit values:
#      - Build T[0..15] = {i * b : i=0..15} (4-bit × 64-bit products)
#      - Transpose T into column vectors (byte-level)
#      - Process each nibble of 'a' via vshuf.b table lookup
#      - Accumulate with byte shifts for positional weighting
#
# Performance: the multiply itself is ~222 SIMD instructions per block
# (vs ~320 serial instructions for the 4-bit table code), with good ILP.
# On top of that every call pays for building and transposing the T
# tables: ~245 instructions for the LSX entry point, and roughly twice
# that for the LASX one (two table builds, one multiply to re-derive
# H^2, and the stack round-trip that packs the two 128-bit lanes).
#
# The T tables do not fit in the 256-byte Htable, so they cannot be
# hoisted into gcm_init and the setup is repaid only over a long enough
# buffer.  Hence only gcm_ghash is vectorised here: a single-block
# multiply is cheaper with the 4-bit table, so gcm_gmult stays scalar
# and each entry point hands short buffers down to the next cheaper
# one (LASX -> LSX -> scalar).  All of them read the one Htable built
# by gcm_init_4bit, so the hand-off is a plain tail call.
#
# Throughput from "openssl speed ghash" on a Loongson-3A5000, in 1000s
# of bytes per second:
#
#   len (bytes)          16       64      256     1024    16384
#   C 4-bit        127906.4 166778.9 179062.4 183619.2 185040.9
#   scalar         135981.3 179133.0 194081.3 198743.0 200518.3
#   LSX            136632.2 230439.1 309498.2 338928.0 349268.7
#   LASX           135293.8 229294.4 431701.1 583032.0 651198.5
#
# The thresholds come from calling each implementation directly, with
# the hand-off disabled, in ns per call:
#
#   len (bytes)      16     32     64     96
#   scalar         81.0  161.1  319.5  482.4
#   LSX           101.4  152.7  238.5  330.3
#   LASX            n/a  211.0  259.7  308.4
#
# that is, the scalar code wins below 32 bytes and the LSX code below
# 96.  A single-block multiply costs 80.4 ns scalar against 97.0 ns for
# the equivalent LSX code, which is why gcm_gmult is not vectorised.
######################################################################

# Scalar register aliases
my ($zero,$ra,$tp,$sp)=("\$zero","\$ra","\$tp","\$sp");
my ($a0,$a1,$a2,$a3,$a4,$a5,$a6,$a7)=map("\$a$_",(0..7));
my ($t0,$t1,$t2,$t3,$t4,$t5,$t6,$t7,$t8)=map("\$t$_",(0..8));

# The saved floating-point registers in the LP64D ABI.  In LoongArch
# with vector extension, the low 64 bits of a vector register alias with
# the corresponding FPR.  So we must save and restore the corresponding
# FPR if we'll write into a vector register.  The ABI only requires
# saving and restoring the FPR (i.e. 64 bits of the corresponding vector
# register), not the entire vector register.
#
# Every routine below writes $vr24-$vr31 while building the T_hi table.
my ($fs0,$fs1,$fs2,$fs3,$fs4,$fs5,$fs6,$fs7)=map("\$f$_",(24..31));

# Vector register aliases
my @vr = map { "\$vr$_" } (0..31);
my ($vr0,$vr1,$vr2,$vr3,$vr4,$vr5,$vr6,$vr7,
    $vr8,$vr9,$vr10,$vr11,$vr12,$vr13,$vr14,$vr15,
    $vr16,$vr17,$vr18,$vr19,$vr20,$vr21,$vr22,$vr23,
    $vr24,$vr25,$vr26,$vr27,$vr28,$vr29,$vr30,$vr31) = @vr;

# LASX (256-bit) register aliases
my @xr = map { "\$xr$_" } (0..31);

my $output = $#ARGV >= 0 && $ARGV[$#ARGV] =~ m|\.\w+$| ? pop : undef;
open STDOUT, ">$output" if $output;

my $code = "";

######################################################################
# Working register assignments (all in zero-column slots after transpose)
######################################################################
my $BREV  = $vr31;   # brev8 nibble-reverse lookup table
my $XI    = $vr29;   # Xi accumulator (128-bit GHASH state)
my $INP   = $vr27;   # input block / general temp
my $NIBLO = $vr25;   # low nibbles of current operand
my $NIBHI = $vr23;   # high nibbles of current operand
my $REVEN = $vr21;   # even accumulator for clmul64 lookup
my $RODD  = $vr19;   # odd accumulator for clmul64 lookup
my $PLO   = $vr15;   # Karatsuba P_lo result
my $PHI   = $vr13;   # Karatsuba P_hi result
my $PMID  = $vr11;   # Karatsuba P_mid result
my $T1    = $vr9;    # temp 1
my $T2    = $vr7;    # temp 2
my $T3    = $vr5;    # temp 3
my $T4    = $vr3;    # temp 4

######################################################################
# Column VR mapping after 16x16 transpose
# VR[i] contains column bitrev4(i). Columns 0-8 hold the T table data
# (T values are at most 67 bits = 9 bytes), columns 9-15 are all zero.
######################################################################
# T_lo (VR0-VR15): column j -> VR[bitrev4(j)]
my @Tcol_lo = ($vr0, $vr8, $vr4, $vr12, $vr2, $vr10, $vr6, $vr14, $vr1);
# T_hi (VR16-VR31): column j -> VR[16+bitrev4(j)]
my @Tcol_hi = ($vr16, $vr24, $vr20, $vr28, $vr18, $vr26, $vr22, $vr30, $vr17);

# LASX packed column arrays (lo lane = H columns, hi lane = H^2 columns)
my @XTcol_lo = ($xr[0], $xr[8], $xr[4], $xr[12], $xr[2], $xr[10], $xr[6], $xr[14], $xr[1]);
my @XTcol_hi = ($xr[16], $xr[24], $xr[20], $xr[28], $xr[18], $xr[26], $xr[22], $xr[30], $xr[17]);

# LASX working registers (same odd-numbered slots, 256-bit)
my $XXI    = $xr[29];
my $XINP   = $xr[27];
my $XNIBLO = $xr[25];
my $XNIBHI = $xr[23];
my $XREVEN = $xr[21];
my $XRODD  = $xr[19];
my $XPLO   = $xr[15];
my $XPHI   = $xr[13];
my $XPMID  = $xr[11];
my $XT1    = $xr[9];
my $XT2    = $xr[7];
my $XT3    = $xr[5];
my $XT4    = $xr[3];

######################################################################
# Helper: emit brev8 (bit-reverse within each byte)
# Uses a 16-byte nibble-reverse table in $brev.
# brev8 is its own inverse.
######################################################################
sub emit_brev8 {
    my ($dst, $src, $brev, $tmp) = @_;
    return <<___;
    vandi.b    $tmp, $src, 0x0f
    vsrli.b    $dst, $src, 4
    vshuf.b    $tmp, $brev, $brev, $tmp
    vshuf.b    $dst, $brev, $brev, $dst
    vslli.b    $tmp, $tmp, 4
    vor.v      $dst, $dst, $tmp
___
}

######################################################################
# Helper: build T[0..15] in 16 consecutive VRs
# T[i] = i * b (4-bit x 64-bit polynomial multiply)
# b is in $b_reg (elem0 = value, elem1 = 0)
# $base = first VR number (0 or 16), $tmp = temp VR number outside range
######################################################################
sub emit_build_table {
    my ($base, $b_reg, $tmp) = @_;
    my @v = map { $vr[$base + $_] } (0..15);
    my $vtmp = $vr[$tmp];
    return <<___;
    vxor.v     $v[0], $v[0], $v[0]
    vor.v      $v[1], $b_reg, $b_reg
    # T[2] = b << 1 (128-bit polynomial shift)
    vsrli.d    $vtmp, $b_reg, 63
    vslli.d    $v[2], $b_reg, 1
    vbsll.v    $vtmp, $vtmp, 8
    vor.v      $v[2], $v[2], $vtmp
    # T[4] = b << 2
    vsrli.d    $vtmp, $b_reg, 62
    vslli.d    $v[4], $b_reg, 2
    vbsll.v    $vtmp, $vtmp, 8
    vor.v      $v[4], $v[4], $vtmp
    # T[8] = b << 3
    vsrli.d    $vtmp, $b_reg, 61
    vslli.d    $v[8], $b_reg, 3
    vbsll.v    $vtmp, $vtmp, 8
    vor.v      $v[8], $v[8], $vtmp
    # XOR combinations
    vxor.v     $v[3], $v[1], $v[2]
    vxor.v     $v[5], $v[1], $v[4]
    vxor.v     $v[6], $v[2], $v[4]
    vxor.v     $v[7], $v[1], $v[6]
    vxor.v     $v[9], $v[1], $v[8]
    vxor.v     $v[10], $v[2], $v[8]
    vxor.v     $v[11], $v[1], $v[10]
    vxor.v     $v[12], $v[4], $v[8]
    vxor.v     $v[13], $v[1], $v[12]
    vxor.v     $v[14], $v[2], $v[12]
    vxor.v     $v[15], $v[1], $v[14]
___
}

######################################################################
# Helper: 16x16 byte transpose (in-place)
# Converts T[0..15] from row-major to column-major layout.
# After transpose, VR[base+i] contains column bitrev4(i).
# $base = first VR number, $tmp = temp VR outside range.
######################################################################
sub emit_transpose16 {
    my ($base, $tmp) = @_;
    my @v = map { $vr[$base + $_] } (0..15);
    my $vtmp = $vr[$tmp];
    my $out = "";

    # Level 1: byte interleave, stride 1
    for my $i (0..7) {
        my ($a, $b) = ($v[2*$i], $v[2*$i+1]);
        $out .= <<___;
    vilvl.b    $vtmp, $b, $a
    vilvh.b    $b, $b, $a
    vor.v      $a, $vtmp, $vtmp
___
    }
    # Level 2: halfword interleave, stride 2
    for my $i (0..3) {
        for my $j (0..1) {
            my ($a, $b) = ($v[4*$i+$j], $v[4*$i+$j+2]);
            $out .= <<___;
    vilvl.h    $vtmp, $b, $a
    vilvh.h    $b, $b, $a
    vor.v      $a, $vtmp, $vtmp
___
        }
    }
    # Level 3: word interleave, stride 4
    for my $i (0..1) {
        for my $j (0..3) {
            my ($a, $b) = ($v[8*$i+$j], $v[8*$i+$j+4]);
            $out .= <<___;
    vilvl.w    $vtmp, $b, $a
    vilvh.w    $b, $b, $a
    vor.v      $a, $vtmp, $vtmp
___
        }
    }
    # Level 4: doubleword interleave, stride 8
    for my $j (0..7) {
        my ($a, $b) = ($v[$j], $v[$j+8]);
        $out .= <<___;
    vilvl.d    $vtmp, $b, $a
    vilvh.d    $b, $b, $a
    vor.v      $a, $vtmp, $vtmp
___
    }
    return $out;
}

######################################################################
# Helper: clmul64 lookup for T_lo or T_hi columns
# Computes polynomial multiply of operand (64-bit in elem0) with the
# value whose table columns are in @cols[0..8].
# Result (128-bit) goes to $result.
######################################################################
sub emit_clmul64 {
    my ($result, $operand, @cols) = @_;
    my $out = <<___;
    # Extract nibbles of operand
    vandi.b    $NIBLO, $operand, 0x0f
    vsrli.b    $NIBHI, $operand, 4
    # Column 0: initialize accumulators (no byte shift needed)
    vshuf.b    $REVEN, $cols[0], $cols[0], $NIBLO
    vshuf.b    $RODD, $cols[0], $cols[0], $NIBHI
___
    for my $j (1..8) {
        $out .= <<___;
    vshuf.b    $T1, $cols[$j], $cols[$j], $NIBLO
    vshuf.b    $T2, $cols[$j], $cols[$j], $NIBHI
    vbsll.v    $T1, $T1, $j
    vbsll.v    $T2, $T2, $j
    vxor.v     $REVEN, $REVEN, $T1
    vxor.v     $RODD, $RODD, $T2
___
    }
    # Combine: R_odd contributes with a 4-bit left shift
    $out .= <<___;
    vslli.d    $T1, $RODD, 4
    vsrli.d    $T2, $RODD, 60
    vbsll.v    $T2, $T2, 8
    vor.v      $T1, $T1, $T2
    vxor.v     $result, $REVEN, $T1
___
    return $out;
}

######################################################################
# LASX: clmul64 on both 128-bit lanes simultaneously
######################################################################
sub emit_clmul64_lasx {
    my ($result, $operand, @cols) = @_;
    my $out = <<___;
    xvandi.b   $XNIBLO, $operand, 0x0f
    xvsrli.b   $XNIBHI, $operand, 4
    xvshuf.b   $XREVEN, $cols[0], $cols[0], $XNIBLO
    xvshuf.b   $XRODD, $cols[0], $cols[0], $XNIBHI
___
    for my $j (1..8) {
        $out .= <<___;
    xvshuf.b   $XT1, $cols[$j], $cols[$j], $XNIBLO
    xvshuf.b   $XT2, $cols[$j], $cols[$j], $XNIBHI
    xvbsll.v   $XT1, $XT1, $j
    xvbsll.v   $XT2, $XT2, $j
    xvxor.v    $XREVEN, $XREVEN, $XT1
    xvxor.v    $XRODD, $XRODD, $XT2
___
    }
    $out .= <<___;
    xvslli.d   $XT1, $XRODD, 4
    xvsrli.d   $XT2, $XRODD, 60
    xvbsll.v   $XT2, $XT2, 8
    xvor.v     $XT1, $XT1, $XT2
    xvxor.v    $result, $XREVEN, $XT1
___
    return $out;
}

######################################################################
# Helper: clmul64 lookup for mid term (columns computed on-the-fly)
# H.mid columns = H.lo columns XOR H.hi columns
######################################################################
sub emit_clmul64_mid {
    my ($result, $operand, $cols_lo_ref, $cols_hi_ref) = @_;
    my @lo = @$cols_lo_ref;
    my @hi = @$cols_hi_ref;
    # Use INP (VR27) as extra temp for mid-column XOR since the operand
    # has already been consumed (nibbles extracted above)
    my $TMID = $INP;
    my $out = <<___;
    # Extract nibbles of operand
    vandi.b    $NIBLO, $operand, 0x0f
    vsrli.b    $NIBHI, $operand, 4
    # Column 0: XOR lo/hi to get mid column, init accumulators
    vxor.v     $TMID, $lo[0], $hi[0]
    vshuf.b    $REVEN, $TMID, $TMID, $NIBLO
    vshuf.b    $RODD, $TMID, $TMID, $NIBHI
___
    for my $j (1..8) {
        $out .= <<___;
    vxor.v     $TMID, $lo[$j], $hi[$j]
    vshuf.b    $T1, $TMID, $TMID, $NIBLO
    vshuf.b    $T2, $TMID, $TMID, $NIBHI
    vbsll.v    $T1, $T1, $j
    vbsll.v    $T2, $T2, $j
    vxor.v     $REVEN, $REVEN, $T1
    vxor.v     $RODD, $RODD, $T2
___
    }
    $out .= <<___;
    vslli.d    $T1, $RODD, 4
    vsrli.d    $T2, $RODD, 60
    vbsll.v    $T2, $T2, 8
    vor.v      $T1, $T1, $T2
    vxor.v     $result, $REVEN, $T1
___
    return $out;
}

######################################################################
# LASX: clmul64_mid on both lanes (mid columns computed on the fly)
######################################################################
sub emit_clmul64_mid_lasx {
    my ($result, $operand, $cols_lo_ref, $cols_hi_ref) = @_;
    my @lo = @$cols_lo_ref;
    my @hi = @$cols_hi_ref;
    my $XTMID = $XINP;
    my $out = <<___;
    xvandi.b   $XNIBLO, $operand, 0x0f
    xvsrli.b   $XNIBHI, $operand, 4
    xvxor.v    $XTMID, $lo[0], $hi[0]
    xvshuf.b   $XREVEN, $XTMID, $XTMID, $XNIBLO
    xvshuf.b   $XRODD, $XTMID, $XTMID, $XNIBHI
___
    for my $j (1..8) {
        $out .= <<___;
    xvxor.v    $XTMID, $lo[$j], $hi[$j]
    xvshuf.b   $XT1, $XTMID, $XTMID, $XNIBLO
    xvshuf.b   $XT2, $XTMID, $XTMID, $XNIBHI
    xvbsll.v   $XT1, $XT1, $j
    xvbsll.v   $XT2, $XT2, $j
    xvxor.v    $XREVEN, $XREVEN, $XT1
    xvxor.v    $XRODD, $XRODD, $XT2
___
    }
    $out .= <<___;
    xvslli.d   $XT1, $XRODD, 4
    xvsrli.d   $XT2, $XRODD, 60
    xvbsll.v   $XT2, $XT2, 8
    xvor.v     $XT1, $XT1, $XT2
    xvxor.v    $result, $XREVEN, $XT1
___
    return $out;
}

######################################################################
# Helper: Karatsuba combine + GF(2^128) reduction
# Input: PLO, PHI, PMID (128-bit each)
# Output: XI = reduced 128-bit result
######################################################################
sub emit_karatsuba_reduce {
    return <<___;
    # Karatsuba combine: K = P_mid ^ P_lo ^ P_hi
    vxor.v     $T1, $PMID, $PLO
    vxor.v     $T1, $T1, $PHI
    # Fold K into R_lo and R_hi
    vbsll.v    $T2, $T1, 8
    vbsrl.v    $T3, $T1, 8
    vxor.v     $PLO, $PLO, $T2
    vxor.v     $PHI, $PHI, $T3

    # GF(2^128) reduction: fold R_hi into R_lo
    # f(x) = x^128 + x^7 + x^2 + x + 1
    # x^(128+i) = x^(i+7) + x^(i+2) + x^(i+1) + x^i

    # Phase 1: main per-element 64-bit shifts
    vslli.d    $T1, $PHI, 1
    vslli.d    $T2, $PHI, 2
    vslli.d    $T3, $PHI, 7
    vxor.v     $T1, $T1, $T2
    vxor.v     $T1, $T1, $T3
    vxor.v     $T1, $T1, $PHI
    vxor.v     $PLO, $PLO, $T1

    # Phase 2: overflow bits that crossed 64-bit element boundaries
    vsrli.d    $T1, $PHI, 63
    vsrli.d    $T2, $PHI, 62
    vsrli.d    $T3, $PHI, 57
    vxor.v     $T1, $T1, $T2
    vxor.v     $T1, $T1, $T3

    # D2 overflow (elem0) -> R_lo.elem1
    vbsll.v    $T2, $T1, 8
    vxor.v     $PLO, $PLO, $T2

    # D3 overflow (elem1) -> reduce to elem0
    vbsrl.v    $T2, $T1, 8
    vslli.d    $T3, $T2, 1
    vslli.d    $T4, $T2, 2
    vxor.v     $T3, $T3, $T4
    vslli.d    $T4, $T2, 7
    vxor.v     $T3, $T3, $T4
    vxor.v     $T3, $T3, $T2
    vxor.v     $XI, $PLO, $T3
___
}

######################################################################
# LASX: Karatsuba combine + GF(2^128) reduction (both lanes)
######################################################################
sub emit_karatsuba_reduce_lasx {
    return <<___;
    xvxor.v    $XT1, $XPMID, $XPLO
    xvxor.v    $XT1, $XT1, $XPHI
    xvbsll.v   $XT2, $XT1, 8
    xvbsrl.v   $XT3, $XT1, 8
    xvxor.v    $XPLO, $XPLO, $XT2
    xvxor.v    $XPHI, $XPHI, $XT3
    xvslli.d   $XT1, $XPHI, 1
    xvslli.d   $XT2, $XPHI, 2
    xvslli.d   $XT3, $XPHI, 7
    xvxor.v    $XT1, $XT1, $XT2
    xvxor.v    $XT1, $XT1, $XT3
    xvxor.v    $XT1, $XT1, $XPHI
    xvxor.v    $XPLO, $XPLO, $XT1
    xvsrli.d   $XT1, $XPHI, 63
    xvsrli.d   $XT2, $XPHI, 62
    xvsrli.d   $XT3, $XPHI, 57
    xvxor.v    $XT1, $XT1, $XT2
    xvxor.v    $XT1, $XT1, $XT3
    xvbsll.v   $XT2, $XT1, 8
    xvxor.v    $XPLO, $XPLO, $XT2
    xvbsrl.v   $XT2, $XT1, 8
    xvslli.d   $XT3, $XT2, 1
    xvslli.d   $XT4, $XT2, 2
    xvxor.v    $XT3, $XT3, $XT4
    xvslli.d   $XT4, $XT2, 7
    xvxor.v    $XT3, $XT3, $XT4
    xvxor.v    $XT3, $XT3, $XT2
    xvxor.v    $XXI, $XPLO, $XT3
___
}

######################################################################
# Helper: emit full table setup for both lo and hi halves
# Input: H (brev8'd) in $INP (VR27)
# After: T_lo columns in VR0-set, T_hi columns in VR16-set
#        BREV table reloaded in VR31
# Clobbers: all 32 VRs except VR5 (holds H.hi temporarily)
######################################################################
sub emit_table_setup {
    my $out = "";

    # Extract H.lo (elem0 with elem1 zeroed) into VR19
    # VR19 is in VR16-31 range, safe during T_lo build/transpose
    $out .= <<___;
    vbsll.v    $RODD, $INP, 8
    vbsrl.v    $RODD, $RODD, 8
___

    # Build T_lo in VR0-VR15 from H.lo (in VR19), temp VR16
    $out .= emit_build_table(0, $RODD, 16);

    # Transpose T_lo (VR0-VR15, temp VR16)
    $out .= emit_transpose16(0, 16);

    # Extract H.hi into VR5 (in VR0-15, safe during T_hi build/transpose)
    # VR27 still has the original brev8'd H (untouched by T_lo ops)
    $out .= <<___;
    vbsrl.v    $T3, $INP, 8
___

    # Build T_hi in VR16-VR31 from H.hi (in VR5), temp VR3
    $out .= emit_build_table(16, $T3, 3);

    # Transpose T_hi (VR16-VR31, temp VR3)
    $out .= emit_transpose16(16, 3);

    # Reload brev8 table into VR31 (was overwritten as part of VR16-31)
    $out .= <<___;
    la.local   $t0, .Lbrev8
    vld        $BREV, $t0, 0
___

    return $out;
}

######################################################################
# Helper: emit the Karatsuba multiply body
# Input: XI contains the value to multiply by H (tables already set up)
# Output: XI = XI * H (reduced)
# Preserves table columns.
######################################################################
######################################################################
# Helper: load H from the standard 4-bit table and convert it to the
# polynomial representation used by the code below.
#
# gcm_init_4bit() stores H itself at Htable[8] (byte offset 128), with
# the same .hi/.lo layout as the H[2] array it was handed.  Reading it
# back lets the SIMD ghash share the one 256-byte table with the scalar
# 4-bit gmult/ghash, instead of needing a private table format and a
# private gcm_init.
#
# The GCM framework applies BSWAP8 to H on little-endian targets, which
# reverses bytes within each 64-bit word, so undo that before brev8.
#
# Result: $INP = brev8(H).  Clobbers $T1.
######################################################################
sub emit_load_h {
    my $out = <<___;
    vld        $INP, $a1, 128
    # Undo BSWAP8: reverse bytes within each 64-bit element.
    vshuf4i.b  $INP, $INP, 0x1B
    vshuf4i.h  $INP, $INP, 0x4E
___
    $out .= emit_brev8($INP, $INP, $BREV, $T1);

    return $out;
}

sub emit_multiply {
    my $out = "";

    # Extract Xi.lo and Xi.hi for Karatsuba
    # Xi.lo = {Xi.elem0, 0}, Xi.hi = {Xi.elem1, 0}
    # Use T3 (VR5) and T4 (VR3) — both are free zero-column slots
    $out .= <<___;
    vbsll.v    $T3, $XI, 8
    vbsrl.v    $T3, $T3, 8
    vbsrl.v    $T4, $XI, 8
___

    # P_lo = clmul64(Xi.lo, H.lo)
    $out .= emit_clmul64($PLO, $T3, @Tcol_lo);

    # P_hi = clmul64(Xi.hi, H.hi)
    $out .= emit_clmul64($PHI, $T4, @Tcol_hi);

    # Xi.mid = Xi.lo ^ Xi.hi
    # T3 and T4 survive clmul64 (they're VR5 and VR3, not used by clmul64)
    $out .= <<___;
    vxor.v     $T3, $T3, $T4
___

    # P_mid = clmul64_mid(Xi.mid, H.mid)
    $out .= emit_clmul64_mid($PMID, $T3, \@Tcol_lo, \@Tcol_hi);

    # Karatsuba combine + reduction -> result in XI
    $out .= emit_karatsuba_reduce();

    return $out;
}

######################################################################
# LASX 2-block parallel multiply
# Input: XXI = {(Xi^A[0]) | A[1]} packed in hi|lo lanes
# Output: XXI with results in both lanes (caller XORs them)
# Packed table columns: lo lane = H, hi lane = H^2
######################################################################
sub emit_multiply_lasx {
    my $out = "";

    # Extract lo and hi 64-bit elements from each 128-bit lane
    $out .= <<___;
    xvbsll.v   $XT3, $XXI, 8
    xvbsrl.v   $XT3, $XT3, 8
    xvbsrl.v   $XT4, $XXI, 8
___

    # P_lo = clmul64(lo_elements, T_lo_columns)
    $out .= emit_clmul64_lasx($XPLO, $XT3, @XTcol_lo);

    # P_hi = clmul64(hi_elements, T_hi_columns)
    $out .= emit_clmul64_lasx($XPHI, $XT4, @XTcol_hi);

    # mid = lo ^ hi
    $out .= <<___;
    xvxor.v    $XT3, $XT3, $XT4
___

    # P_mid = clmul64_mid(mid, T_lo, T_hi)
    $out .= emit_clmul64_mid_lasx($XPMID, $XT3, \@XTcol_lo, \@XTcol_hi);

    # Karatsuba combine + reduce
    $out .= emit_karatsuba_reduce_lasx();

    return $out;
}

######################################################################
# Constants
######################################################################
$code .= <<___;
.section .rodata
.align 4
.Lbrev8:
    # Nibble bit-reverse lookup table: maps nibble i -> bitrev4(i)
    # Used for brev8: reverse bits within each byte
    .byte 0x00, 0x08, 0x04, 0x0c, 0x02, 0x0a, 0x06, 0x0e
    .byte 0x01, 0x09, 0x05, 0x0d, 0x03, 0x0b, 0x07, 0x0f

.text
___

######################################################################
# void gcm_ghash_loongarch64_lsx(u64 Xi[2], const u128 Htable[16],
#                             const u8 *inp, size_t len)
#
# Xi = ((Xi ^ inp[0]) * H ^ inp[1]) * H ... (multi-block GHASH)
# a0 = Xi, a1 = Htable, a2 = inp, a3 = len (bytes, multiple of 16)
#
# Building the T tables costs more than a single 4-bit table pass, so
# one-block calls are handed to the scalar code instead.
######################################################################

# Minimum length, in bytes, for the LSX path to beat the scalar one.
# Measured on a Loongson-3A5000.
my $LSX_MIN_LEN = 32;

$code .= <<___;

.globl gcm_ghash_loongarch64_lsx
.type gcm_ghash_loongarch64_lsx, \@function
.align 4
gcm_ghash_loongarch64_lsx:
    # Fall back to the scalar 4-bit code for less than 2 blocks.  Both
    # read the same Htable, so this is a plain tail call, made before
    # the frame is set up.  It also covers len == 0.
    ori        $t0, $zero, $LSX_MIN_LEN
    bltu       $a3, $t0, gcm_ghash_loongarch64

    addi.d     $sp, $sp, -64
    fst.d      $fs0, $sp, 0
    fst.d      $fs1, $sp, 8
    fst.d      $fs2, $sp, 16
    fst.d      $fs3, $sp, 24
    fst.d      $fs4, $sp, 32
    fst.d      $fs5, $sp, 40
    fst.d      $fs6, $sp, 48
    fst.d      $fs7, $sp, 56
    # Load brev8 table
    la.local   $t0, .Lbrev8
    vld        $BREV, $t0, 0
___
$code .= emit_load_h();

# Build T_lo and T_hi tables, transpose
$code .= emit_table_setup();

$code .= <<___;
    # Load Xi and apply brev8
    vld        $XI, $a0, 0
___
$code .= emit_brev8($XI, $XI, $BREV, $T1);

$code .= <<___;
.Lghash_loop:
    # Load input block, brev8, XOR with Xi
    vld        $INP, $a2, 0
___
$code .= emit_brev8($INP, $INP, $BREV, $T1);
$code .= <<___;
    vxor.v     $XI, $XI, $INP
___

# Multiply Xi * H
$code .= emit_multiply();

$code .= <<___;
    addi.d     $a2, $a2, 16
    addi.d     $a3, $a3, -16
    bnez       $a3, .Lghash_loop

    # Store result (brev8 back to memory format)
___
$code .= emit_brev8($XI, $XI, $BREV, $T1);
$code .= <<___;
    vst        $XI, $a0, 0
    fld.d      $fs0, $sp, 0
    fld.d      $fs1, $sp, 8
    fld.d      $fs2, $sp, 16
    fld.d      $fs3, $sp, 24
    fld.d      $fs4, $sp, 32
    fld.d      $fs5, $sp, 40
    fld.d      $fs6, $sp, 48
    fld.d      $fs7, $sp, 56
    addi.d     $sp, $sp, 64
    jr         $ra
.size gcm_ghash_loongarch64_lsx, .-gcm_ghash_loongarch64_lsx
___

######################################################################
# void gcm_ghash_loongarch64_lasx(u64 Xi[2], const u128 Htable[16],
#                                  const u8 *inp, size_t len)
#
# LASX 2-block parallel GHASH. Processes 2 blocks per iteration:
#   Xi_new = (Xi ^ A[0]) * H^2 + A[1] * H
#
# H^2 is derived here rather than at gcm_init time: the Htable is the
# standard 4-bit table and has no spare room for a private H^2 copy, so
# each call pays one extra multiply to rebuild it.  That only pays for
# itself over a long enough buffer, hence the fallback threshold.
######################################################################

# Minimum length, in bytes, for the LASX path to beat the LSX one.
# Measured on a Loongson-3A5000; see the note above about setup cost.
# The 2-block loop is entered unconditionally, so this must never be
# set below 32 regardless of what the timings say.
my $LASX_MIN_LEN = 96;

$code .= <<___;

.globl gcm_ghash_loongarch64_lasx
.type gcm_ghash_loongarch64_lasx, \@function
.align 4
gcm_ghash_loongarch64_lasx:
    # Below the crossover, hand off to the LSX path (which in turn hands
    # short buffers to the scalar code).  This is a tail call made before
    # the frame is set up, so the callee saves its own FPRs.
    ori        $t0, $zero, $LASX_MIN_LEN
    bltu       $a3, $t0, gcm_ghash_loongarch64_lsx

    # Stack: 18 columns x 32 bytes = 576 bytes of column scratch, then
    # 64 bytes for the callee-saved \$f24-\$f31, then 16 bytes holding
    # brev8(H) across the first table build.
    # Layout per column: bytes [0..15] = H col, [16..31] = H^2 col
    addi.d     $sp, $sp, -656
    fst.d      $fs0, $sp, 576
    fst.d      $fs1, $sp, 584
    fst.d      $fs2, $sp, 592
    fst.d      $fs3, $sp, 600
    fst.d      $fs4, $sp, 608
    fst.d      $fs5, $sp, 616
    fst.d      $fs6, $sp, 624
    fst.d      $fs7, $sp, 632
    # Load brev8 table
    la.local   $t0, .Lbrev8
    vld        $BREV, $t0, 0

    # === Phase 1: Build tables for H ===
___
$code .= emit_load_h();
$code .= <<___;
    # Stash brev8(H): building the tables overwrites every vector register
    # that is not a column, and H is needed again below to form H^2.
    vst        $INP, $sp, 640
___
$code .= emit_table_setup();

# Phase 2: Save H columns to lo halves of 32-byte stack slots
for my $j (0..8) {
    my $off = $j * 32;
    $code .= "    vst        $Tcol_lo[$j], \$sp, $off\n";
}
for my $j (0..8) {
    my $off = 288 + $j * 32;
    $code .= "    vst        $Tcol_hi[$j], \$sp, $off\n";
}

# Phase 3: H^2 = H * H via the tables just built, then build its tables
$code .= <<___;
    # === Phase 3: H^2 = H * H, then build tables for H^2 ===
    vld        $XI, $sp, 640
___
$code .= emit_multiply();
$code .= <<___;
    vor.v      $INP, $XI, $XI
___
$code .= emit_table_setup();

# Phase 3.5: Save H^2 columns to hi halves of 32-byte stack slots
for my $j (0..8) {
    my $off = $j * 32 + 16;
    $code .= "    vst        $Tcol_lo[$j], \$sp, $off\n";
}
for my $j (0..8) {
    my $off = 288 + $j * 32 + 16;
    $code .= "    vst        $Tcol_hi[$j], \$sp, $off\n";
}

# Phase 4: Load packed {H^2 | H} columns via 256-bit xvld
# The stack is only guaranteed 16-byte aligned, so half of these 32-byte
# loads straddle a 32-byte boundary.  That is deliberate: LoongArch vector
# loads do not require natural alignment, and realigning the frame would
# cost more than the 18 loads it saves, which run once per call.
$code .= "    # === Phase 4: Load packed columns via xvld ===\n";
for my $j (0..8) {
    my $off = $j * 32;
    $code .= "    xvld       $XTcol_lo[$j], \$sp, $off\n";
}
for my $j (0..8) {
    my $off = 288 + $j * 32;
    $code .= "    xvld       $XTcol_hi[$j], \$sp, $off\n";
}

$code .= <<___;
    # The column scratch is dead from here on, but the frame stays put:
    # it still holds the saved FPRs.
    # Load Xi and brev8
    vld        $XI, $a0, 0
___
$code .= emit_brev8($XI, $XI, $BREV, $T1);

# Main 2-block loop
$code .= <<___;

    # Loop bound, kept in a register across the loop.  Nothing in the
    # loop body touches a scratch GPR.
    ori        $t0, $zero, 32

.Lghash_lasx_loop:
    # Load A[0], brev8, XOR with Xi
    vld        $T1, $a2, 0
___
$code .= emit_brev8($T1, $T1, $BREV, $T2);
$code .= <<___;
    vxor.v     $T1, $T1, $XI

    # Load A[1], brev8
    vld        $INP, $a2, 16
___
$code .= emit_brev8($INP, $INP, $BREV, $T2);

# Pack XXI: hi = Xi^brev8(A[0]) for H^2, lo = brev8(A[1]) for H
# xvpermi.q imm=0x02: result.lo = xd.lo, result.hi = xj.lo
$code .= <<___;
    vor.v      $XI, $INP, $INP
    xvpermi.q  $XXI, $XT1, 0x02
___

# LASX 2-block multiply
$code .= emit_multiply_lasx();

# Combine lanes: Xi = XXI.hi ^ XXI.lo
# xvpermi.q imm=0x01: result.lo = xj.hi
$code .= <<___;
    xvpermi.q  $XINP, $XXI, 0x01
    vxor.v     $XI, $XI, $INP

    addi.d     $a2, $a2, 32
    addi.d     $a3, $a3, -32
    bgeu       $a3, $t0, .Lghash_lasx_loop

    # Tail: handle remaining single block (if any)
    beqz       $a3, .Lghash_lasx_done

    vld        $INP, $a2, 0
___
$code .= emit_brev8($INP, $INP, $BREV, $T1);
$code .= <<___;
    vxor.v     $XI, $XI, $INP
___

# LSX multiply for tail (lo lanes of packed XVRs = H's tables)
$code .= emit_multiply();

$code .= <<___;
.Lghash_lasx_done:
___
$code .= emit_brev8($XI, $XI, $BREV, $T1);
$code .= <<___;
    vst        $XI, $a0, 0
    fld.d      $fs0, $sp, 576
    fld.d      $fs1, $sp, 584
    fld.d      $fs2, $sp, 592
    fld.d      $fs3, $sp, 600
    fld.d      $fs4, $sp, 608
    fld.d      $fs5, $sp, 616
    fld.d      $fs6, $sp, 624
    fld.d      $fs7, $sp, 632
    addi.d     $sp, $sp, 656
    jr         $ra
.size gcm_ghash_loongarch64_lasx, .-gcm_ghash_loongarch64_lasx
___

######################################################################
# Scalar 4-bit table GHASH — optimized fallback when LSX unavailable
#
# Uses the standard gcm_init_4bit Htable (16 × u128, 256 bytes).
# Each u128 entry: .hi at offset 0, .lo at offset 8 (LE native).
#
# Key LoongArch64-specific optimizations:
#  - revb.d for byte swap (1 instruction)
#  - alsl.d for table index computation (nibble*16 + base in 1 instruction)
#  - Htable loads hoisted ahead of the nibble step that consumes them,
#    giving the load ~8 instructions of slack to cover L1 latency
#  - Xi and inp loaded as u64 pairs, not byte-by-byte
#  - Z accumulator and both Xi halves stay in registers for the whole
#    block; only the 16-byte result is written back per block
######################################################################

# Scalar register assignments for 4-bit GHASH:
#  $a0 = Xi pointer
#  $a1 = Htable pointer
#  $a2 = inp pointer (ghash only)
#  $a3 = len (ghash only)
#
# Working registers:
my $Zhi  = $t0;   # Z accumulator high 64 bits
my $Zlo  = $t1;   # Z accumulator low 64 bits
my $rem  = $t2;   # reduction index (4 bits)
my $nlo  = $t3;   # current low nibble
my $nhi  = $t4;   # current high nibble
my $Hhi  = $t5;   # prefetched Htable[n].hi
my $Hlo  = $t6;   # prefetched Htable[n].lo
my $addr = $t7;   # table lookup address
my $tmp  = $t8;   # general temp
my $Xl   = $a4;   # Xi bytes[15..8] (lower address half, after revb.d)
my $Xh   = $a5;   # Xi bytes[7..0] (upper address half, after revb.d)
my $rp   = $a6;   # rem_4bit table pointer
my $cnt  = $a7;   # byte counter

$code .= <<___;

.section .rodata
.align 3
.Lrem_4bit:
.dword 0x0000000000000000
.dword 0x1C20000000000000
.dword 0x3840000000000000
.dword 0x2460000000000000
.dword 0x7080000000000000
.dword 0x6CA0000000000000
.dword 0x48C0000000000000
.dword 0x54E0000000000000
.dword 0xE100000000000000
.dword 0xFD20000000000000
.dword 0xD940000000000000
.dword 0xC560000000000000
.dword 0x9180000000000000
.dword 0x8DA0000000000000
.dword 0xA9C0000000000000
.dword 0xB5E0000000000000

.text
___

# Emit one nibble of GHASH: shift Z right by 4, apply reduction, XOR Htable entry.
# Htable entry ($Hhi/$Hlo) must already be loaded before calling this.
sub emit_nibble {
    return <<___;
    andi       $rem, $Zlo, 0xf
    srli.d     $Zlo, $Zlo, 4
    slli.d     $tmp, $Zhi, 60
    or         $Zlo, $Zlo, $tmp
    srli.d     $Zhi, $Zhi, 4
    alsl.d     $addr, $rem, $rp, 3
    ld.d       $tmp, $addr, 0
    xor        $Zhi, $Zhi, $tmp
    xor        $Zhi, $Zhi, $Hhi
    xor        $Zlo, $Zlo, $Hlo
___
}

$code .= <<___;
######################################################################
# void gcm_gmult_loongarch64(u64 Xi[2], const u128 Htable[16])
######################################################################
.globl gcm_gmult_loongarch64
.type  gcm_gmult_loongarch64, \@function
.align 4
gcm_gmult_loongarch64:
    # Load Xi as two u64, byte-swap so byte[15] is in Xl LSByte
    ld.d       $Xl, $a0, 8
    ld.d       $Xh, $a0, 0
    revb.d     $Xl, $Xl
    revb.d     $Xh, $Xh

    la.local   $rp, .Lrem_4bit

    # Extract first byte nibbles
    andi       $nlo, $Xl, 0xf
    srli.d     $nhi, $Xl, 4
    andi       $nhi, $nhi, 0xf

    # Prefetch Htable[nlo]
    alsl.d     $addr, $nlo, $a1, 4
    ld.d       $Hhi, $addr, 0
    ld.d       $Hlo, $addr, 8

    # Process low nibble of byte 15 (Z is zero, so just load from Htable)
    move       $Zhi, $Hhi
    move       $Zlo, $Hlo

    # Load Htable[nhi] for high nibble
    alsl.d     $addr, $nhi, $a1, 4
    ld.d       $Hhi, $addr, 0
    ld.d       $Hlo, $addr, 8
___

# Process high nibble of byte 15
$code .= emit_nibble();

$code .= <<___;
    # Advance to byte 14
    srli.d     $Xl, $Xl, 8

    # Process bytes 14..8 (7 bytes from Xl)
    ori        $cnt, $zero, 7
.Lgmult_lo_loop:
    andi       $nlo, $Xl, 0xf
    srli.d     $nhi, $Xl, 4
    andi       $nhi, $nhi, 0xf

    # Prefetch Htable[nlo]
    alsl.d     $addr, $nlo, $a1, 4
    ld.d       $Hhi, $addr, 0
    ld.d       $Hlo, $addr, 8
___
$code .= emit_nibble();
$code .= <<___;

    # Load Htable[nhi]
    alsl.d     $addr, $nhi, $a1, 4
    ld.d       $Hhi, $addr, 0
    ld.d       $Hlo, $addr, 8
___
$code .= emit_nibble();
$code .= <<___;

    srli.d     $Xl, $Xl, 8
    addi.d     $cnt, $cnt, -1
    bnez       $cnt, .Lgmult_lo_loop

    # Process bytes 7..0 (8 bytes from Xh)
    ori        $cnt, $zero, 8
.Lgmult_hi_loop:
    andi       $nlo, $Xh, 0xf
    srli.d     $nhi, $Xh, 4
    andi       $nhi, $nhi, 0xf

    alsl.d     $addr, $nlo, $a1, 4
    ld.d       $Hhi, $addr, 0
    ld.d       $Hlo, $addr, 8
___
$code .= emit_nibble();
$code .= <<___;

    alsl.d     $addr, $nhi, $a1, 4
    ld.d       $Hhi, $addr, 0
    ld.d       $Hlo, $addr, 8
___
$code .= emit_nibble();
$code .= <<___;

    srli.d     $Xh, $Xh, 8
    addi.d     $cnt, $cnt, -1
    bnez       $cnt, .Lgmult_hi_loop

    # Store result: byte-swap Z and write to Xi
    revb.d     $Zhi, $Zhi
    revb.d     $Zlo, $Zlo
    st.d       $Zhi, $a0, 0
    st.d       $Zlo, $a0, 8

    jr         $ra
.size gcm_gmult_loongarch64, .-gcm_gmult_loongarch64

######################################################################
# void gcm_ghash_loongarch64(u64 Xi[2], const u128 Htable[16],
#                                  const u8 *inp, size_t len)
######################################################################
.globl gcm_ghash_loongarch64
.type  gcm_ghash_loongarch64, \@function
.align 4
gcm_ghash_loongarch64:
    # The loop below is bottom-tested, and the SIMD entry points forward
    # short buffers here, so a zero length has to be rejected up front.
    beqz       $a3, .Lghash_4bit_ret

    la.local   $rp, .Lrem_4bit

.Lghash_4bit_outer:
    # Load Xi and input, XOR together, byte-swap.  inp is caller data and
    # carries no alignment guarantee; LoongArch64 handles unaligned ld.d
    # in hardware, so it is loaded directly rather than byte at a time.
    ld.d       $Xl, $a0, 8
    ld.d       $Xh, $a0, 0
    ld.d       $tmp, $a2, 8
    xor        $Xl, $Xl, $tmp
    ld.d       $tmp, $a2, 0
    xor        $Xh, $Xh, $tmp
    revb.d     $Xl, $Xl
    revb.d     $Xh, $Xh

    # First byte (byte 15): extract nibbles
    andi       $nlo, $Xl, 0xf
    srli.d     $nhi, $Xl, 4
    andi       $nhi, $nhi, 0xf

    # Z = Htable[nlo] (first nibble, Z was zero)
    alsl.d     $addr, $nlo, $a1, 4
    ld.d       $Zhi, $addr, 0
    ld.d       $Zlo, $addr, 8

    # Load Htable[nhi]
    alsl.d     $addr, $nhi, $a1, 4
    ld.d       $Hhi, $addr, 0
    ld.d       $Hlo, $addr, 8
___

$code .= emit_nibble();

$code .= <<___;
    srli.d     $Xl, $Xl, 8

    # Bytes 14..8 (7 bytes from Xl)
    ori        $cnt, $zero, 7
.Lghash_4bit_lo_loop:
    andi       $nlo, $Xl, 0xf
    srli.d     $nhi, $Xl, 4
    andi       $nhi, $nhi, 0xf

    alsl.d     $addr, $nlo, $a1, 4
    ld.d       $Hhi, $addr, 0
    ld.d       $Hlo, $addr, 8
___
$code .= emit_nibble();
$code .= <<___;

    alsl.d     $addr, $nhi, $a1, 4
    ld.d       $Hhi, $addr, 0
    ld.d       $Hlo, $addr, 8
___
$code .= emit_nibble();
$code .= <<___;

    srli.d     $Xl, $Xl, 8
    addi.d     $cnt, $cnt, -1
    bnez       $cnt, .Lghash_4bit_lo_loop

    # Bytes 7..0 (8 bytes from Xh)
    ori        $cnt, $zero, 8
.Lghash_4bit_hi_loop:
    andi       $nlo, $Xh, 0xf
    srli.d     $nhi, $Xh, 4
    andi       $nhi, $nhi, 0xf

    alsl.d     $addr, $nlo, $a1, 4
    ld.d       $Hhi, $addr, 0
    ld.d       $Hlo, $addr, 8
___
$code .= emit_nibble();
$code .= <<___;

    alsl.d     $addr, $nhi, $a1, 4
    ld.d       $Hhi, $addr, 0
    ld.d       $Hlo, $addr, 8
___
$code .= emit_nibble();
$code .= <<___;

    srli.d     $Xh, $Xh, 8
    addi.d     $cnt, $cnt, -1
    bnez       $cnt, .Lghash_4bit_hi_loop

    # Store result
    revb.d     $Zhi, $Zhi
    revb.d     $Zlo, $Zlo
    st.d       $Zhi, $a0, 0
    st.d       $Zlo, $a0, 8

    # Next block
    addi.d     $a2, $a2, 16
    addi.d     $a3, $a3, -16
    bnez       $a3, .Lghash_4bit_outer

.Lghash_4bit_ret:
    jr         $ra
.size gcm_ghash_loongarch64, .-gcm_ghash_loongarch64
___

print $code;
close STDOUT or die "error closing STDOUT: $!";
