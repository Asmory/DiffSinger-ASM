# M37 — Pure-ASM vocoder Conv 2-D scheduling

M37 targets the remaining ordinary Conv1d bottleneck in the full native
NSF-HiFiGAN executor. M36 reduced the AOT activation arena to ~11.5 MiB and
fused most LeakyReLU->Conv pairs, but the target-machine profile still showed
ordinary Conv dominating vocoder latency.

The key utilization problem is late HiFi-GAN stages. With packed-8 Conv and an
8-worker P-core pool, C=32 provides only four output-channel blocks and C=16
only two. M36's channel-owner scheduler therefore left half or three quarters
of the worker pool idle even though those layers have very long time axes.

M37 adds `ds_conv1d_nct_f32_avx2_oc8_t8_range`, an AVX2/FMA entry that computes
one packed-8 output-channel block over an arbitrary contiguous time interval.
The persistent pool dynamically schedules `(oc8 block, 512-sample time tile)`
tasks only when the ordinary channel decomposition is under-subscribed.
Large-channel layers keep the M36 owner path to preserve weight-cache locality.

The AOT packer also uses packed-8 for every Conv whose output channel count is
at least 8 and divisible by 8, rather than only for Cout>=64. The bundle
manifest reports `conv2d_candidates_8w` and a Conv shape histogram.

The final runtime remains CPU-only and links no ONNX Runtime, oneDNN, OpenVINO,
MKL or BLAS. ORT remains an offline pack/golden tool only.

Useful targets:

```bash
make m37-check
make m37-real-vocoder MODEL_DIR=/path/to/voicebank
make m37-real-e2e MODEL_DIR=/path/to/voicebank SPEAKER_EMB=/path/to/singer.emb
```

`m37-real-vocoder` runs an 8-worker `DSASM_2D=0` static-OC baseline followed by
4/6/8-worker M37 auto scheduling on the same packed model and golden tensors.
