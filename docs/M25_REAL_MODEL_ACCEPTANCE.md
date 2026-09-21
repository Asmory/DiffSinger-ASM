# M25 — Real-model acceptance

M25 turns the M24 packer/CLI into a one-command correctness gate for a real
supported DiffSinger acoustic checkpoint.

`tools/validate_real_model_m25.py` does not import DiffSinger. It reads the
checkpoint tensors directly and reconstructs the supported current-default
PyTorch inference semantics for:

- FS2 token/duration Transformer + stretch GRU + pitch embedding,
- shallow ConvNeXt aux decoder,
- shallow Rectified Flow Euler with LYNXNet2,
- final spec denormalization.

The exact same explicit FP32 noise file is fed to PyTorch reference and the
native CLI. Final raw mel is compared with max-abs, max-rel, RMSE, cosine and
checksums. Native CLI timing/RTF is printed unchanged.

## Real checkpoint

```sh
make m25-real CKPT=/path/model.ckpt CONFIG=/path/acoustic.yaml
```

Optional: `FRAMES=64 TOKENS=32 STEPS=20 THREADS=0`.

The M24 profile guard still rejects unsupported optional branches (language,
speaker, variance embeds, key-shift, speed) instead of silently producing an
incomplete model.
