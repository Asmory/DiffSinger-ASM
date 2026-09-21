# M41 — Hot-shape Conv profiler

M40.1 reached a real end-to-end RTF of about 1.10 with selective K11 AVX-VNNI,
while ordinary FP32 Conv remained the largest vocoder cost.  M41 deliberately
changes no model math and no default kernel selection.  It adds a profile-only
shape breakdown so the next ASM kernel is chosen from measured end-to-end cost.

Set `DSASM_PROFILE_SHAPES=1` together with the existing CLI `--profile` flag to
print every Conv/ConvTranspose node in descending wall-time order.  Each line
includes the graph op index, Cin/Cout, K, time length, dilation/stride, packing
width and FP32/VNNI path.

`scripts/run_m41_profile.sh` reuses one packed 48-frame real NSF-HiFiGAN bundle
and runs the current K11 winner plus controlled A/B cases for fixed-K kernels,
2-D late-stage scheduling, and 4/6/8 worker counts.

No M41 code path is active unless profiling is explicitly enabled.
