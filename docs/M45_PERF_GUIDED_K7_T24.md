# M45 perf-guided K7 oc4 x t24

M45.1 perf on the i5-13420H showed fixed-K Conv1d dominates P-core cycles:
K7 full 29.43%, K11 full 21.89%, K3 full 14.48%, plus range variants.
`perf annotate` placed substantial samples on the repeated memory-source
`vbroadcastss` instructions inside the Cin loop.

The M45 K7 kernel changes only the register tile. Existing packed8 weights and
NCT tensors are unchanged. It computes four output channels across 24 time
samples (three YMM time vectors), keeping 12 accumulators live. One scalar
weight broadcast is therefore reused by three t8 vectors. Compared with three
old oc8xt8 tiles over the same 8x24 output region, this roughly changes the hot
inner work from 21 input vector loads + 168 scalar broadcasts to 42 input vector
loads + 56 scalar broadcasts, while keeping the same 168 FMAs and the same
per-output accumulation order.

Enable independently of the old K-specialization switch with:

    DSASM_K7_T24=1

Only full K7 Conv with Tout divisible by 24 uses the new path. M37 2-D range
jobs stay on the existing range kernels. This deliberately targets the hottest
128x128/T3072 and 64x64/T6144 K7 layers first.
