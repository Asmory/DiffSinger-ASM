# M32 — pure CPU/ASM NSF-HiFiGAN Conv1d kernel lab

M31 established the target-machine performance budget on the i5-13420H:

- 4-step native acoustic: about 309 ms for 0.557 s audio.
- ORT NSF-HiFiGAN: about 418 ms (best measured at 4 CPU threads).
- To cross RTF < 1.0 with the 4-step acoustic, the vocoder must fall below about 248 ms.

M32 deliberately **does not** use oneDNN, OpenVINO, BLAS, or ORT as an
execution backend.  The production direction is a thin C graph/runtime plus
x86-64 assembly kernels.  ONNX Runtime is permitted only as an offline golden
reference: `pack_vocoder_hot_conv_m32.py` exposes one internal Conv input and
output, then writes them into a `.dsv32` benchmark bundle.  The timed benchmark
process itself never links against or invokes ORT.

## Kernel

`ds_conv1d_nct_f32_avx2_oc4_t8` implements group=1/stride=1 Conv1d directly on
NCT tensors.  It uses a 4-output-channel x 8-time tile:

- one contiguous 8-float input load is reused by four output accumulators;
- each packed weight tuple contains four output-channel weights;
- AVX2/FMA performs the hot multiply-accumulate loop;
- dilation is native (HiFi-GAN residual blocks use dilated convolutions);
- 1..7 trailing time samples use scalar FMA in the same assembly routine;
- no im2col/GEMM materialization is used.

The C wrapper only zero-pads input and submits output-channel blocks to the
existing persistent, topology-aware P-core thread pool.

## Bundle

`DSVOC32` contains one real Conv layer:

- shape/stride/padding/dilation header;
- bias;
- output-channel-4 packed weights;
- a captured real NCT input;
- the corresponding ORT FP32 output as golden reference.

## Tests

```
make m32-check
```

runs both ordinary and dilated synthetic Conv1d parity tests.

On the real DongFangZhiZi vocoder:

```
make m32-real-kernel MODEL_DIR=/path/to/voicebank M32_ROUNDS=5
```

extracts and benchmarks three M31 hotspots, including
`resblocks.5/convs1.0`, `resblocks.5/convs2.0`, and
`resblocks.4/convs1.0`.  It sweeps 1/2/4/8 persistent CPU workers and reports
GFLOP/s plus FP32 error.

M32 is a kernel milestone, not yet a full native vocoder.  M33 should use the
measured real-layer results to decide whether this direct FP32 kernel is already
sufficient or whether the hot path must move to a specialized AVX-VNNI INT8
kernel before wiring the complete 93-Conv + 5-ConvTranspose graph executor.
