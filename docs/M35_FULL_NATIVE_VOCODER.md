# M35 — Fixed-shape full pure-native NSF-HiFiGAN executor

M35 is the first milestone that executes the entire deployed `nsf_hifigan.onnx`
without ONNX Runtime, oneDNN, OpenVINO, MKL, BLAS, or another inference engine
in the timed runtime path.

The offline compiler `tools/pack_vocoder_graph_m35.py` specializes the ONNX
model to a frame count (48 for the current real-time acceptance clip), captures
static tensor shapes, packs Conv1d weights into the M32 `oc4×time8` layout, and
transposes ConvTranspose1d weights into the M33/M34 output-major layout. The
result is an mmap-able `DSVOC35` bundle.

The runtime supports exactly the operator family present in the real deployed
vocoder graph: Conv, ConvTranspose, LeakyRelu, Add, Sub, Mul, Div, Mod, Slice,
Pad, CumSum, Reshape, Squeeze, Transpose, Sin, and Tanh. Heavy operators use our
x86-64 AVX2/FMA assembly kernels. Small NSF source/control math remains a thin C
implementation and is not delegated to a third-party runtime.

M35 deliberately uses a simple static tensor arena with no lifetime reuse. This
keeps the first complete graph executor easy to validate. M36 may compact the
arena after full real-model parity and timing are established.

`m35-real-e2e` runs native 4-step acoustic inference, feeds that mel directly to
the native vocoder, and reports the pure-native end-to-end RTF. ORT is invoked
only before the timed vocoder run to generate a golden waveform for parity; its
time is explicitly excluded from the runtime budget.


## M35.1 real-graph compatibility fix

The real NSF-HiFiGAN graph contains an `Unsqueeze` view on the source/f0 path. M35.1 adds `Unsqueeze` to DSVOC35 as a fixed-shape dense view/copy op and adds an unsupported-op preflight so future graph gaps fail with the exact operator list before compilation starts. Runtime remains pure C/ASM.
