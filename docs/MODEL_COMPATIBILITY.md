# Model compatibility matrix

This document distinguishes an acoustic-main-graph result from a complete
voice-package result. A model is not `complete` until its acoustic, duration,
pitch, variance, and vocoder graphs all convert, load, execute natively, and
pass strict reference parity.

Python, PyTorch, ONNX, and ONNX Runtime are offline conversion and validation
tools only. Online neural arithmetic remains in x86-64 assembly; C owns model
loading, validation, dispatch, scheduling, buffers, cancellation, and callbacks.

## Acoustic main graph

Status as measured on 2026-09-22:

| Model | Convert/load | Strict E2E parity | Maximum absolute error (condition / aux / mel) | Evidence |
|---|---:|---:|---:|---|
| DongFangZhiZi Nectar CE 26.08.31 | pass | pass | `9.24e-7 / 3.24e-5 / 4.72e-5` | `build/model_matrix_scan_20260922/DongFangZhiZi_Nectar_current/parity_adaln_regression/m26_report.json` |
| NeuroSynth-1 | pass | pass | `1.43e-6 / 2.77e-5 / 1.10e-4` | `build/model_matrix_scan_20260922/NeuroSynth/parity_rope_silu_regression/m27_report.json` |
| DongFangZhiZi Nectar V2.5 | pass | pass | `2.86e-6 / 3.24e-5 / 1.71e-3` | `build/model_matrix_scan_20260922/DongFangZhiZi_Nectar_V2_5/parity_silu/m27_report.json` |
| Aotian Sky 003 | pass | pass | `3.61e-6 / 2.43e-5 / 3.05e-5` | `build/model_matrix_scan_20260922/Aotian_Sky_003/parity_adaptive_v2/m26_report.json` |
| DongFangZhiZi Era CE 26.01.22 | pass | pass | `3.34e-6 / 2.62e-5 / 7.25e-5` | `build/model_matrix_scan_20260922/DongFangZhiZi_Era/parity_adaptive_v2/m26_report.json` |
| Liliko 1.4.0 | pass | pass | `1.31e-6 / 2.38e-5 / 2.57e-5` | `build/model_matrix_scan_20260922/Liliko/parity_adaptive_v3/m26_report.json` |
| Yousa 1.56 | pass | pass | `2.86e-6 / 2.57e-5 / 3.00e-5` | `build/model_matrix_scan_20260922/Yousa/parity_adaptive_v2/m26_report.json` |
| Aikie ML v201 | blocked | not run | energy conditioner is not yet represented by DSFS25 | importer rejection |
| Hakukyo zh/ja | blocked | not run | per-residual-layer conditioner projections need a DSLYNX extension | importer rejection |

The adaptive LayerNorm models use instrumented graph semantics, not naming
assumptions: layers 0 and 2 project phoneme-level speaker conditioning to
`[shift, scale]`, then apply `normalized * scale + shift`; layers 1 and 3 use
static LayerNorm. External frame-level speaker embeddings are averaged over
the duration-defined phoneme spans. Frozen speaker embeddings are broadcast.

## Complete package coverage

No package is complete yet. The repository contains 44 ONNX graphs across the
acoustic, duration, linguistic, pitch, variance, and vocoder roles. The current
native packer and strict parity harness cover the acoustic main graph. Remaining
work must inventory graph families by measured topology, add explicit formats
and ASM execution paths, then record per-role parity here.

| Package | Acoustic | Duration | Pitch | Variance | Vocoder | Complete |
|---|---:|---:|---:|---:|---:|---:|
| Aikie ML v201 | blocked | pending | pending | pending | pending | no |
| Aotian Sky 003 | pass | pending | pending | pending | pending | no |
| DongFangZhiZi Era | pass | pending | pending | pending | pending | no |
| DongFangZhiZi Nectar CE | pass | pending | pending | pending | pending | no |
| DongFangZhiZi Nectar V2.5 | pass | pending | pending | pending | pending | no |
| Hakukyo zh/ja | blocked | pending | pending | pending | absent | no |
| Liliko 1.4.0 | pass | pending | pending | pending | pending | no |
| NeuroSynth-1 | pass | pending | pending | pending | absent | no |
| Yousa 1.56 | pass | pending | pending | pending | pending | no |

## Validation

Use the project environment so offline PyTorch is available:

```sh
make PYTHON=.venv/bin/python -B build/libdsasm_m25.so build/dsasm-acoustic
.venv/bin/python tools/validate_deploy_features_m25.py --lib build/libdsasm_m25.so
```

Real-model acceptance uses `tools/validate_real_onnx_m27.py --strict` with the
source ONNX, packed directory, and a model-owned speaker embedding. A pass must
include native load and execution; conversion alone is not acceptance.
