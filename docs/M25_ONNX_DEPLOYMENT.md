# M25 — ONNX deployment import

M25 targets distributed DiffSinger voicebanks whose authoritative acoustic artifact is `acoustic.onnx`, not a training checkpoint.

## New runtime surface

`DSFS25` keeps the DSFS21 base payload and appends deployment-only conditioner data. Supported branches are:

- token-level language embedding;
- frame-level breathiness, voicing and tension linear embeddings;
- gender -> exported clipping/scaling -> key-shift embedding;
- velocity -> exported clipping -> speed embedding;
- frame-wise external speaker embedding;
- the 1001-entry stretch embedding lookup used when ONNX export constant-folds the stretch MLP.

The old DSFS21 loader and default-profile inference entry point remain valid.

## Import

```bash
python tools/pack_acoustic_onnx_m25.py acoustic.onnx --model-dir VOICEBANK --out packed/model
build/dsasm-acoustic inspect packed/model
```

The importer recursively walks ONNX subgraphs because the Rectified-Flow velocity network may live inside `If` bodies. Anonymous MatMul initializers are recovered through the node topology. The packed model contains:

- `fs2_acoustic.dsfs` (DSFS25)
- `aux_convnext.dsa` (DSAUX20)
- `lynxnet2.dsn` (DSLYNX7)
- `model.conf` / `model.json`
- `spec_min.f32` / `spec_max.f32`
- copied `phonemes.json`, `languages.json`, deployment YAML/text metadata and root `.emb` files when available.

## CLI inputs

For DSFS25 models the CLI exposes:

```text
--language-id N | --languages FILE
--speaker-emb FILE
--breathiness FILE
--voicing FILE
--tension FILE
--gender FILE
--velocity FILE
--depth X
```

`--speaker-emb` accepts either one C-float vector (broadcast over frames) or T*C frame-wise floats, in raw float32 or text form. Language IDs are token-level. Variance/gender/velocity inputs are frame-level.

If the model declares language or speaker branches those inputs are mandatory. Other feature curves default to neutral values: breathiness/voicing/tension/gender=0 and velocity=1.

`--depth` is normalized to [0,1] and maps to the observed deployment graph relation `t_start=max(1-depth,0)`. `--t-start` remains available for direct native Rectified-Flow control.

## Validation

`make m25-deploy-check` verifies the extended conditioner against an independent PyTorch/NumPy reference, including the folded stretch-table path, then creates and mmap-loads a synthetic DSFS25 bundle and runs deterministic native CLI inference.

`make m25-onnx-real MODEL_DIR=...` is the first real voicebank acceptance step: import `acoustic.onnx`, copy metadata and inspect the resulting native model. A successful inspect confirms that ONNX topology/weights were converted and all three native bundles pass cross-dimension validation.
