# M43 parallel Add + VNNI scope sprint

M42 crossed realtime on the target machine at 514.621 ms end-to-end (RTF 0.923),
with the stride-8/K16 ConvTranspose specialization reducing its two hottest
operators substantially.  The next engineering goal is RTF <= 0.8.

M43 takes two low-risk steps before another large convolution-kernel rewrite:

1. Large same-shape residual `Add` operations are dispatched through the
   persistent worker pool instead of leaving seven P-core logical CPUs idle.
   `DSASM_PARALLEL_ADD=0` restores the old single-thread ASM path for A/B.
2. The packer already supports `--vnni-scope all-k711`, but production stayed
   on `stage128`.  M43 automatically races the quality-passing stage128/all-K
   candidates at 8 workers, then tests only the winner at 10 and 12 workers.
   This answers whether the four E cores and broader AVX-VNNI coverage are now
   useful after the M42 ConvTranspose surgery.

The script selects the lowest median vocoder time among candidates that pass
cosine >= 0.999 and SNR >= 25 dB, then runs the real CPU/C/ASM E2E pipeline
with that exact bundle/mode/worker count.
