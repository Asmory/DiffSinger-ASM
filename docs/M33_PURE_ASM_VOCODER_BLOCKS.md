# M33 — pure-ASM NSF-HiFiGAN structural blocks

M32 proved that the direct FP32 Conv1d kernel is viable on the target
NSF-HiFiGAN shapes.  M33 moves one level above isolated convolutions while
preserving the project's hard runtime rule:

> Production inference is CPU-only and executes only DiffSinger-ASM C runtime
> plus hand-written x86-64 assembly kernels.  ORT/ONNX is offline-only for
> graph import, captured inputs, and golden outputs.

## Residual unit

M33 implements the native HiFi-GAN unit

```
LeakyReLU -> Conv1d -> LeakyReLU -> Conv1d -> residual Add
```

using:

- `leaky_relu_f32_avx2.S`
- M32 `conv1d_nct_f32_avx2_oc4_t8.S`
- `add_f32_avx2.S`
- the existing persistent topology-aware thread pool for both Conv jobs.

No temporary im2col/GEMM matrix is created.  Only two `[C,T]` temporary tensors
plus one reusable padded Conv workspace are required.

## ConvTranspose1d

A zero-insertion lowering was deliberately rejected because a stride-8
upsampler would waste roughly 7/8 of the inner-loop FMAs on inserted zeros.
Instead M33 adds a direct sparse transposed-convolution assembly kernel.

For one output channel it walks real input samples only.  Each input scalar is
broadcast once and updates only its actual `K` output positions.  Interior tap
ranges are accumulated eight samples at a time with AVX2/FMA; boundary taps use
scalar FMA.  The persistent pool parallelizes across output channels.

Supported M33 deployment subset matches the NSF-HiFiGAN upsamplers:

- group = 1
- dilation = 1
- symmetric padding
- output_padding = 0
- arbitrary positive stride

Weights are packed offline from ONNX `[Cin,Cout,K]` to output-major
`[Cout,Cin,K]`; no runtime transpose is performed.

## Acceptance

`make m33-check` runs synthetic parity for both a dilated residual unit and a
stride-4 transposed convolution.  It also verifies the timed executables do not
link ORT, oneDNN, OpenVINO, MKL, or BLAS.

On a real voicebank:

```
make m33-real-blocks MODEL_DIR=/path/to/voicebank M33_ROUNDS=7
```

captures and benchmarks all three residual units in `resblocks.5` and all five
`generator/ups.{0..4}/ConvTranspose` layers.  ORT is used only by the packing
scripts to produce golden bundles; the timed executables are pure C/ASM.

M34 should use these real measurements to wire `conv_pre`, all upsamplers,
resblocks, source injection, and `conv_post` into a complete native vocoder
executor, then measure native-vocoder end-to-end RTF against the M31 budget.
