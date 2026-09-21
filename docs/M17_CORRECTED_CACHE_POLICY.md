# Milestone 17 — Corrected promoted cache policy

M16 promoted `N-owner + Kblock=512` from an earlier target run. The following
8-worker target-machine validation disproved the owner part:

- interleaved whole-forward A/B: `dynamic kb512` was best while
  `owner+kb512` was effectively neutral/slower;
- `perf` showed `dynamic kb512` cutting wall-clock dramatically in the same
  run, while `owner+kb512` remained much slower;
- owner mode does reduce cache references, but that does not translate into
  useful target latency.

M17 therefore corrects the default policy:

- `N-owner = OFF` by default;
- `K-blocked ATan Linear = ON` for the auto-selected 8-worker P-core pool;
- `Kblock = 512`;
- the existing job guard remains `M>=32, N>=512, K>=512`, so small shapes keep
  the legacy full-K path;
- `parallel depthwise`, `spin dispatch`, and `indexed Linear` remain off.

The public setters and `DSASM_*` environment variables remain available for
same-process A/B and portability experiments.
