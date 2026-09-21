# M21 — Native FastSpeech2 acoustic conditioner

## Scope

M21 implements the inference path selected by the current default acoustic
configuration (B=1): hidden size 384, 4 encoder layers, 2 attention heads,
RoPE theta 10000 with non-interleaved rotation, FFN Conv1d kernel 3,
`use_stretch_embed=true`, variance scaling enabled, and the optional language,
speaker, variance, key-shift and speed branches disabled.

## Transformer layer

Each encoder layer is reproduced as:

1. LayerNorm(eps=1e-5)
2. bias-free QKV projection `384 -> 1152`
3. two-head reshape (`head_dim=192`)
4. non-interleaved RoPE on Q and K
5. scaled QK^T (`1/sqrt(192)`), key-padding mask, softmax
6. weighted sum of V
7. bias-free output projection `384 -> 384`
8. residual + padding-row zeroing
9. LayerNorm(eps=1e-5)
10. Conv1d(k=3) `384 -> 1536`, multiplied by `3^-0.5`
11. exact GELU
12. Linear `1536 -> 384`
13. residual + padding-row zeroing

A final LayerNorm follows the four layers.

The dense projections and the im2col representation of the k=3 FFN convolution
reuse the existing packed16 m4n16 ASM kernels. QK dot products use the M21
`dot_f32_avx2_fma.S` kernel.

## Frame-level condition

After token-level encoding M21 performs:

- `mel2ph` gather
- StretchRegulator-compatible fractional position generation
- `round(1000*stretch)` + SinusoidalPosEmb(384)
- `384 -> 1536 -> GELU -> 384` stretch MLP
- single-layer 384-hidden GRU, PyTorch gate order `r,z,n`
- residual addition of GRU output
- `log1p(f0/700)` pitch embedding

The GRU input projection for every frame is performed as one packed GEMM. Only
the recurrent hidden projection remains sequential per frame, avoiding a
threadpool dispatch/barrier for every GRU time step.

## Public entries

- `ds_fs2_encoder_forward_f32_avx2()` — token-level Transformer
- `ds_fs2_acoustic_condition_f32_avx2()` — tokens/mel2ph/f0 -> condition
- `ds_full_acoustic_infer_f32_avx2()` — condition path + M20 aux + M19 RF -> mel

All temporaries come from caller-provided workspace; the inference path has no
per-call heap allocation.

## DSFS21 pack format

`tools/pack_fs2_acoustic_checkpoint.py` packs current checkpoint tensors into a
64-byte-aligned `fs2_acoustic.dsfs` plus JSON section manifest. It contains:

- token and duration embeddings
- all 4 Transformer layer norms, QKV/out projections, FFN weights/biases
- final encoder LayerNorm
- stretch MLP
- GRU input/recurrent weights and biases
- pitch embedding

QKV and attention output biases are stored as explicit zero arrays because the
upstream attention projections are bias-free while the shared packed GEMM ABI
expects a bias pointer.

## Development-machine validation

The development host exposes five workers rather than the target i5-13420H's
8-thread P-core pool. Representative parity obtained before packaging:

- C64/L1 encoder: max_abs ~4.77e-7
- C128/L3 encoder: max_abs ~9.54e-7
- C384/L4, Ttxt=32 encoder: max_abs ~2.50e-6
- C384/L4, Ttxt=64 encoder: max_abs ~3.13e-6
- full condition C384, Ttxt=32 -> Tmel=64: max_abs ~2.62e-6
- small full tokens-to-mel chain: max_abs ~2.86e-6, padded mel exactly zero
- official synthetic tokens-to-mel chain: max_abs ~5.72e-5

The official synthetic full chain measured roughly 0.75 RTF on the five-worker
development host. Target-machine results should be taken from `make m21-bench`.
