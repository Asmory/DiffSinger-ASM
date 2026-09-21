# M50 release profile (historical baseline)

M50 freezes the perf-guided M45-M49 vocoder choices as production defaults:

- `DSASM_KSPEC=0`
- `DSASM_K3_TMODE=24`
- `DSASM_K7_T24=1`
- `DSASM_K11_T24=1`
- `DSASM_RANGE_T24=0`
- `DSASM_VOCODER_T_TILE=504`
- `DSASM_RESIDUAL_T24=1`
- `DSASM_VNNI=k11`, symmetric quantization, `DSASM_VNNI_CIN=128`
- M42 stride8/K16 ConvTranspose remains default-on
- parallel Add remains default-off

Every setting is still overrideable by its environment variable.  The release
runner compares implicit defaults against the explicit M49 profile, checks
waveform parity, records P-core cycles when perf is available, and runs the
native acoustic -> native vocoder E2E gate.

The later persistent real-model E2E promotion supersedes these runtime
defaults with `config/best-inference.env`; see `docs/PERFORMANCE.md`.
