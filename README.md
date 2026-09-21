# DiffSinger-ASM

**Real-time DiffSinger synthesis on a laptop CPU. No GPU required.**

[![Platform](https://img.shields.io/badge/platform-Linux%20x86--64-1793d1)](https://github.com/Asmory/DiffSinger-ASM)
[![Runtime](https://img.shields.io/badge/runtime-C%20%2B%20x86--64%20ASM-555555)](https://github.com/Asmory/DiffSinger-ASM)
[![ISA](https://img.shields.io/badge/ISA-AVX2%20%7C%20FMA%20%7C%20AVX--VNNI-e34f26)](https://github.com/Asmory/DiffSinger-ASM)
[![CI](https://github.com/Asmory/DiffSinger-ASM/actions/workflows/ci.yml/badge.svg)](https://github.com/Asmory/DiffSinger-ASM/actions/workflows/ci.yml)
[![Upstream](https://img.shields.io/badge/upstream-OpenVPI%2FDiffSinger-2ea44f)](https://github.com/openvpi/DiffSinger)
[![Paper](https://img.shields.io/badge/arXiv-2105.02446-b31b1b)](https://arxiv.org/abs/2105.02446)

DiffSinger-ASM runs the complete acoustic-to-waveform pipeline in C and
handwritten x86-64 assembly. On an Intel Core i5-13420H, it synthesizes a
4.458-second phrase in a median **3.992 seconds**: an end-to-end real-time factor
of **0.895**, entirely on the CPU.

Exported DiffSinger acoustic and NSF-HiFiGAN ONNX models are compiled offline
into memory-mapped native bundles. ONNX Runtime helps pack and validate a
voicebank, but the production inference path needs no ONNX Runtime, PyTorch,
oneDNN, OpenVINO, MKL, BLAS, or GPU.

The product-facing `libdsasm.so` keeps one native engine per singer in process,
reuses its model mappings, worker pool, and inference buffers, and publishes
completed PCM blocks directly to the playback mixer. OpenUtau can therefore
start playback from the first vocoder bucket instead of waiting for a complete
phrase WAV.

> [!IMPORTANT]
> This remains an experimental, CPU-specific runtime, not a standalone singing
> editor. OpenUtau integration is available in
> [`AntheaLaffy/OpenUtau` at `f5efea82`](https://github.com/AntheaLaffy/OpenUtau/commit/f5efea82),
> but it is not part of an upstream OpenUtau release. Bring your own compatible
> exported voicebank and prepare its native bundles before playback.

## CPU Real-Time Performance

The promoted `e2e-cin64-asym-pass-20260921` profile passed ten persistent,
end-to-end requests. Every request finished faster than the audio it generated.

| Measured result | Current release |
| --- | ---: |
| Generated audio | 4458.231 ms |
| Median synthesis latency | **3991.507 ms** |
| Median real-time factor | **0.895** |
| p90 / worst latency | 4120.739 / 4136.881 ms |
| Requests below real time | **10 / 10** |
| Waveform quality | cosine 0.999016, SNR 27.06 dB |

Measured on an Intel Core i5-13420H with eight P-core/SMT workers, a 384-frame
fixture, four Rectified-Flow steps, persistent warm requests, and performance
power mode. See [the full performance record](docs/PERFORMANCE.md) and
[raw samples](benchmarks/artifacts/2026-09-21-e2e-cin64-asym/samples.tsv).
Results on other CPUs and voicebanks will vary.

## Why Native Assembly?

- **CPU real time is the release target.** FastSpeech2, the auxiliary decoder,
  Rectified Flow, and NSF-HiFiGAN are timed together, not as isolated kernels.
- **The hot path stays small.** Runtime dependencies are only libc, libm, and
  pthread, with no framework startup, graph planner, or provider dispatch.
- **Kernels match deployed shapes.** Handwritten AVX2/FMA code handles FP32
  operators; asymmetric AVX-VNNI accelerates quality-gated vocoder stages.
- **Models are prepared ahead of time.** Fixed shapes, packed weights, fused
  residual operations, and memory-mapped bundles move work out of inference.
- **Persistent workers keep the CPU busy.** Scheduling is tuned for the
  performance cores and avoids rebuilding execution state for every request.
- **Speed never bypasses quality.** Releases must pass waveform parity,
  stability, p90, and worst-case gates before they are promoted.

## Built for OpenVPI DiffSinger

This project is an independent native inference runtime for deployment models
exported by the [OpenVPI-maintained DiffSinger](https://github.com/openvpi/DiffSinger)
project. It does not train voicebanks and does not replace the upstream
variance model or score frontend. The integrated OpenUtau renderer constructs
those inputs and selects this runtime as an alternative to ONNX Runtime.

<p align="center">
  <img src="https://raw.githubusercontent.com/openvpi/DiffSinger/8333dd615eef8a04dab6c5e0401215f16b6461b1/docs/resources/arch-overview.jpg" width="520" alt="OpenVPI DiffSinger synthesis architecture">
</p>

<p align="center"><sub>DiffSinger architecture overview from OpenVPI/DiffSinger, licensed under Apache-2.0.</sub></p>

Use the upstream ecosystem to prepare a model and musical inputs, then use
DiffSinger-ASM for the acoustic and vocoder execution stages:

- [DiffSinger user guidance](https://diffsinger.com) and
  [Getting Started](https://github.com/openvpi/DiffSinger/blob/main/docs/GettingStarted.md)
- [MakeDiffSinger](https://github.com/openvpi/MakeDiffSinger) for dataset and
  voicebank preparation
- [OpenUtau](https://github.com/stakira/OpenUtau) for production-oriented score
  editing and synthesis workflows
- [DiffSinger paper](https://arxiv.org/abs/2105.02446) and
  [OpenVPI implementation](https://github.com/openvpi/DiffSinger)

## Requirements

### Native runtime

- Linux on x86-64
- GCC or Clang, GNU Make, and binutils
- AVX2 and FMA; the promoted configuration additionally requires AVX-VNNI
- pthread, libc, and libm

Check the relevant CPU features before enabling the release profile:

```bash
grep -m1 -oE 'avx2|fma|avx_vnni' /proc/cpuinfo | sort -u
```

### Offline model packing

- Python 3.10+
- NumPy
- ONNX
- ONNX Runtime (CPU)

PyTorch is only needed by checkpoint importers and validation tools. It is not
needed to run an already packed model.

## Quick Start

Build the product shared library and its ABI checks:

```bash
git clone https://github.com/Asmory/DiffSinger-ASM.git
cd DiffSinger-ASM
make -j"$(nproc)" engine-check
```

This produces `build/libdsasm.so`. The opaque engine keeps models, one shared
worker pool, and inference buffers resident, then publishes vocoder output as
contiguous PCM callbacks. See [the engine ABI and frame-bucket contract](docs/ENGINE_ABI.md).

The standalone acoustic and vocoder executables remain available for model
inspection, parity checks, and debugging:

```bash
make -j"$(nproc)" build/dsasm-acoustic build/dsasm-vocoder-m40
```

Create an environment for offline packing:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install numpy onnx onnxruntime
```

Pack the acoustic model into the singer's default ASM directory:

```bash
python tools/pack_acoustic_onnx_m25.py /path/to/voicebank/acoustic.onnx \
  --model-dir /path/to/voicebank \
  --out /path/to/singer/dsasm/acoustic

build/dsasm-acoustic inspect /path/to/singer/dsasm/acoustic
```

Compile the streaming vocoder buckets. The filenames are part of the engine
contract; each graph remains fixed-shape internally:

```bash
mkdir -p /path/to/singer/dsasm/vocoder
for frames in 64 128 256 384; do
  python tools/pack_vocoder_graph_m35.py \
    /path/to/voicebank/dsvocoder/nsf_hifigan.onnx \
    --frames "$frames" \
    --vnni-scope all-k711 \
    --residual-scope all3711 \
    --out "/path/to/singer/dsasm/vocoder/$frames.dsv35" \
    --work "build/packer-$frames"
done
```

### OpenUtau

Build or install the OpenUtau integration referenced above, then point it at
this DiffSinger-ASM checkout. The promoted inference profile is built into the
runtime defaults, so OpenUtau does not need tuning environment variables:

```bash
cd /absolute/path/to/DiffSinger-ASM
export DIFFSINGER_ASM_HOME=/absolute/path/to/DiffSinger-ASM
# Launch OpenUtau from this environment.
```

By default, each singer uses these prepared directories:

```text
<singer>/dsasm/acoustic
<singer>/dsasm/vocoder
```

They can instead be configured in the singer's `dsconfig.yaml`:

```yaml
asm_acoustic: /absolute/path/to/packed/acoustic
asm_vocoder: /absolute/path/to/packed/vocoder
```

In OpenUtau, select **Preferences -> Rendering -> DiffSinger backend**, then
choose **Auto**, **ASM**, or **ONNX Runtime**. `Auto` uses ASM only when the OS,
CPU features, ABI version, packed model, sample rate, hop size, mel bins, and
model features are compatible; otherwise it falls back to ONNX Runtime.
Selecting `ASM` explicitly reports the incompatibility instead of silently
falling back.

ABI v1 rejects configurations it cannot reproduce, including energy
conditioning and pitch-controllable vocoders. ASM and ONNX renders use separate
WAV cache keys. Partial PCM is published only to the active playback session;
it enters the complete render cache only after the unique final chunk arrives.

The native engine validates contiguous callback offsets and streams a short
first bucket before switching to larger throughput-oriented buckets. OpenUtau
keeps ungenerated regions pending, so the audio callback reads immutable
published chunks without treating missing audio as a completed phrase.

For standalone CLI debugging, text vector files accept whitespace- or
comma-separated values; F0 and optional variance curves contain one value per
acoustic frame:

```bash
build/dsasm-acoustic infer /path/to/singer/dsasm/acoustic \
  --tokens tokens.txt \
  --durations durations.txt \
  --f0 f0.txt \
  --language-id 4 \
  --speaker-emb /path/to/singer.emb \
  --depth 0.6 \
  --steps 4 \
  --out mel.f32
```

Then synthesize a waveform with a bundle whose fixed frame count exactly
matches the CLI input:

```bash
build/dsasm-vocoder-m40 infer /path/to/singer/dsasm/vocoder/384.dsv35 \
  --mel mel.f32 \
  --f0 f0.f32 \
  --out wave.f32 \
  --wav output.wav \
  --workers 8 \
  --rounds 3
```

The acoustic CLI does not perform text/phoneme parsing or score preparation.
`sum(durations)` must equal the F0 frame count, and the vocoder bundle must have
been compiled for that same frame count. A deployed model may also require
language and speaker inputs; `inspect` reports the enabled features.
`dsasm-acoustic` accepts F0 as text, while `dsasm-vocoder-m40` expects the same
curve as raw native-endian float32 (`f0.f32`). Mel and waveform `.f32` files are
also headerless float32 arrays.

## Best Inference Profile

The runtime defaults enable the exact graph, scheduling, residual-fusion, and
asymmetric VNNI choices that passed the current end-to-end gate.
[`config/best-inference.env`](config/best-inference.env) records those values
explicitly for reproducible benchmarking and lets deployments override or
disable individual optimizations.

```bash
. config/best-inference.env
env | grep '^DSASM_' | sort
```

Do not use the promoted AVX-VNNI settings on a CPU without AVX-VNNI. Local
source builds default to `-march=native`; CI release archives use an explicit
AVX2/FMA baseline, while optional AVX-VNNI vocoder paths remain runtime-gated.

## Verification

Build the shared library, validate ABI layout and exported dependencies, and
run the real-model streaming test when local packed artifacts are available:

```bash
make -j"$(nproc)" engine-check
make engine-real-stream-check \
  ENGINE_REAL_ACOUSTIC=/path/to/packed/acoustic \
  ENGINE_REAL_VOCODER=/path/to/packed/vocoder/384.dsv35
```

Run the core native regression checks:

```bash
make -j"$(nproc)" m40-1-check m38-check m58-check
```

Run the promoted residual-kernel benchmark and its parity gate:

```bash
make m58-bench
```

The full real-voicebank E2E comparison needs locally prepared model and fixture
artifacts and is documented in [docs/PERFORMANCE.md](docs/PERFORMANCE.md).
Voicebanks and singer embeddings are intentionally not distributed here.

The native ABI v2 keeps the 192-byte `dsasm_request` layout and adds explicit
fixed vocoder-bucket selection. A request using zero selects the smallest
loaded bucket; a nonzero unavailable bucket is rejected instead of silently
expanding the render and delaying its next PCM callback. The external OpenUtau
binding must require ABI version 2 and map the former trailing reserved field
to `vocoder_bucket_frames`.

## Architecture

```text
Exported voicebank (offline)
  acoustic.onnx ----> DSFS25 + DSAUX20 + DSLYNX7
  nsf_hifigan.onnx -> 64 / 128 / 256 / 384-frame DSVOC35 buckets

Native inference (online)
  OpenUtau score + curves -> stable C ABI -> persistent singer engine
                          -> FastSpeech2 -> Aux decoder -> Rectified Flow
                          -> mel + F0 -> bucketed NSF-HiFiGAN
                          -> PCM chunk callbacks -> MixPlanner -> playback
```

The packed files are validated and memory-mapped. Large operators use
handwritten assembly; scheduling, graph execution, and control math remain in
small C runtimes. [`include/dsasm_engine.h`](include/dsasm_engine.h) is the
stable product boundary; lower-level headers remain implementation-oriented.

## Documentation

- [Performance policy and current evidence](docs/PERFORMANCE.md)
- [Dual-architecture optimization strategy](docs/DUAL_ARCHITECTURE_OPTIMIZATION.md)
- [Real-time streaming performance policy](docs/STREAMING_PERFORMANCE.md)
- [Block batch performance policy](docs/BATCH_PERFORMANCE.md)
- [CPU runtime implementation policy](docs/RUNTIME_IMPLEMENTATION_POLICY.md)
- [Stable engine ABI and streaming contract](docs/ENGINE_ABI.md)
- [CI and release process](docs/RELEASING.md)
- [Deployment ONNX import](docs/M25_ONNX_DEPLOYMENT.md)
- [Real-model acceptance](docs/M25_REAL_MODEL_ACCEPTANCE.md)
- [Pure-native vocoder executor](docs/M35_FULL_NATIVE_VOCODER.md)
- [Current residual-fusion milestone](docs/M58_ALL_RESIDUAL_FUSION.md)
- [Packed model formats](docs/DSLYNX7_FORMAT.md)

The milestone documents preserve the engineering history. New users should
start with this README and treat older milestone commands as experiments rather
than the recommended release path.

## Project Status

DiffSinger-ASM is an active performance-engineering project from Asmory. The
supported surface is intentionally narrow: Linux, x86-64, and exported
DiffSinger graphs compatible with the included importers. Model compatibility,
quality, and speed should be validated on your own workload before deployment.

No voicebank, singer identity, or third-party model weights are included in the
repository or release archives.

Use voice models only with the voice owner's consent. This project follows the
[upstream DiffSinger responsible-use notice](https://github.com/openvpi/DiffSinger#disclaimer):
do not use it to generate a person's voice without permission.
