# M58 — all-stage residual fusion surgery

M58 is compared directly with the verified M55 long-audio baseline. It keeps
the M55 worker/topology policy and removes graph-level memory passes.

Changes:

- New `--residual-scope all3711` packer mode: fuse every aligned K3/K7/K11 Conv->Add residual edge, including C64/C32/C16 late stages.
- Keep the proven full/channel-owner K3/K7/K11 t24 residual-store kernels for Cout >= worker coverage.
- Add exact K7 and K11 oc4 x t24 **range + residual-store** assembly kernels for 2-D late-stage jobs.
- New `DSASM_RANGE_RESIDUAL_T24` escape hatch (default on). This is independent from `DSASM_RANGE_T24`, so the new residual surgery can be isolated in A/B.

The aim is to eliminate residual Add passes and their extra read/write traffic, not to change model math.
