# M46 perf-guided K11 oc4 x t24

M45 established that the perf-guided oc4 x t24 register tile is bit-exact and
reduces P-core work for K7. M46 applies the same transformation to full K11
Conv1d while leaving K3 generic and leaving M37 2-D range jobs untouched.

Enable independently with `DSASM_K11_T24=1`. The intended production candidate
is `DSASM_KSPEC=0 DSASM_K7_T24=1 DSASM_K11_T24=1`, so only the two perf-proven
new kernels override the generic path.
