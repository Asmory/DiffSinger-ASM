# M47 — perf-guided t24 range kernels for 2-D vocoder Conv

M45/M46 proved that `oc4 x t24` reduces scalar-weight broadcasts and is bit-exact
for full K7/K11 Conv1d.  Perf still showed substantial cycles in the range kernels
used by the 2-D scheduler (`rci_7` and `rci_11`).

M47 adds bit-exact `oc4 x t24` range kernels for K7 and K11, plus a runtime-selectable
vocoder time tile.  `DSASM_VOCODER_T_TILE=504` keeps every 2-D tile divisible by both
24 and 8 for the real 12288/24576 late-stage lengths; `DSASM_RANGE_T24=1` enables the
new range kernels independently of `DSASM_KSPEC`.

The intended winning policy remains:

- `DSASM_KSPEC=0`
- `DSASM_K7_T24=1`
- `DSASM_K11_T24=1`
- test `DSASM_VOCODER_T_TILE=512/504`
- test `DSASM_RANGE_T24=0/1`

The runner separates tile geometry from kernel selection and chooses by P-core cycles
when perf counters are available.
