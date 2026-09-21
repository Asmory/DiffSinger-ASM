# M48 — perf-guided K3 register-tile sweep

M45/M46 established that the full-channel owner path benefits from reusing a
scalar packed weight across several t8 vectors. M47 showed the same strategy is
not automatically profitable in the 2-D range path: fewer instructions can
still lose cycles when load pressure lowers IPC.

M48 therefore leaves range kernels disabled and isolates only full K=3 Conv.
Two bit-exact kernels are added on top of the existing packed8 layout:

- `oc4 x t16`: 8 live accumulators, moderate register pressure.
- `oc4 x t24`: 12 live accumulators, fewer weight broadcasts.

`DSASM_K3_TMODE=0|16|24` selects the implementation independently of
`DSASM_KSPEC`, so the current production combination remains:

- `DSASM_KSPEC=0`
- `DSASM_K7_T24=1`
- `DSASM_K11_T24=1`
- `DSASM_RANGE_T24=0`
- `DSASM_VOCODER_T_TILE=504`

The runner mirrors baseline/t16/t24 order and selects the winner by P-core
cycles when perf counters are available. Wall time remains secondary because
sustained AVX2 load on the i5-13420H showed large thermal/power-frequency drift.
