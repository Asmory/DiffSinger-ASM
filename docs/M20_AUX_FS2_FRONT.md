# Milestone 20 — Shallow Aux ConvNeXt + FastSpeech2 Front Primitives

M20 moves the native boundary one stage earlier than M19.

## Upstream boundary reproduced

Current `DiffSingerAcoustic.forward()` computes:

1. `condition = FastSpeech2Acoustic(...)`
2. shallow `aux_mel = AuxDecoderAdaptor(condition, infer=True)`
3. `aux_mel *= (mel2ph > 0)`
4. Rectified Flow inference from `condition` + `aux_mel`
5. final mel mask

M20 now owns steps 2–5.  The remaining non-native boundary is the learned
FastSpeech2 encoder/GRU itself.

Current upstream shallow aux configuration:

- ConvNeXt1D
- input = 384
- channels = 512
- layers = 6
- kernel = 7
- output = 128 mel bins
- each block: depthwise Conv1d7 -> LayerNorm(eps=1e-6) -> Linear C->4C -> exact GELU -> Linear 4C->C -> gamma -> residual

## Native pieces

### `dsasm_aux_decoder.h`

- `ds_aux_convnext_forward_norm_f32_avx2`
- `ds_aux_convnext_infer_f32_avx2`

The two k=7 dense convolutions use tap-major im2col once per forward and the
existing packed16 4x16 GEMM.  The six block depthwise convolutions use the new
handwritten `depthwise_conv1d_k7_tc_f32_avx2.S` kernel.

### `dsasm_acoustic.h`

`ds_acoustic_post_fs2_f32_avx2` composes:

```
condition[T,384]
  -> native ConvNeXt aux decoder
  -> aux mel[T,128]
  -> M19 mask/norm
  -> M18 20-step RF Euler
  -> denorm/mask
  -> final mel[T,128]
```

### `dsasm_fs2_front.h`

Exact B=1 non-neural FastSpeech2 primitives are native:

- `mel2ph -> duration`
- stretch regulator coordinates
- encoder-to-frame gather by `mel2ph`
- `log1p(f0/700)` pitch input
- `log1p(duration)` duration input used by current variance-scaling config

These are the scaffolding for M21.  The four Transformer encoder layers,
RoPE attention, stretch sinusoidal embedding + GRU, and learned projection
weights are intentionally **not** claimed as native in M20.

## Real checkpoint packing

`tools/pack_aux_convnext_checkpoint.py` locates current aux decoder parameters
by checkpoint suffix and writes `DSAUX20` packed data:

- k=7 dense Conv1d weights flattened in tap-major im2col order then packed16
- k=7 depthwise weights transposed to `[7,C]`
- pointwise Linear weights packed16
- LayerNorm and gamma vectors stored directly

Example:

```bash
python tools/pack_aux_convnext_checkpoint.py model.ckpt \
  --prefix aux_decoder \
  --out packed/aux
```

## Validation gates

- FS2 stretch: bit-exact
- mel2ph gather: bit-exact
- pitch-log: ~3e-8 max abs in development
- current ConvNeXt official synthetic shape: ~5e-6 normalized-domain max abs
- combined post-FS2 path: ~2e-5 raw-mel max abs on official synthetic shape
- padded final mel: exact zero
- M19 regression and DSLYNX7 bundle regression must remain green

## Next milestone

M21 should implement the learned FastSpeech2 encoder in layers, starting with
one encoder layer (`LN -> 2-head RoPE attention -> residual -> LN -> Conv1d3
FFN/GELU -> Linear -> residual`) before adding the 4-layer stack and stretch GRU.
