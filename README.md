# DiffSinger-ASM

**Streaming and high-throughput DiffSinger synthesis on a laptop CPU. No GPU required.**

[![Platform](https://img.shields.io/badge/platform-Linux%20x86--64-1793d1)](https://github.com/Asmory/DiffSinger-ASM)
[![Runtime](https://img.shields.io/badge/runtime-C%20%2B%20x86--64%20ASM-555555)](https://github.com/Asmory/DiffSinger-ASM)
[![ISA](https://img.shields.io/badge/ISA-AVX2%20%7C%20FMA%20%7C%20AVX--VNNI-e34f26)](https://github.com/Asmory/DiffSinger-ASM)
[![Upstream](https://img.shields.io/badge/upstream-OpenVPI%2FDiffSinger-2ea44f)](https://github.com/openvpi/DiffSinger)
[![Paper](https://img.shields.io/badge/arXiv-2105.02446-b31b1b)](https://arxiv.org/abs/2105.02446)

DiffSinger-ASM runs the complete acoustic-to-waveform pipeline in C and
handwritten x86-64 assembly. It keeps separate execution and performance
contracts for low-latency streaming and large-block batch rendering because a
single long-audio RTF cannot describe both workloads honestly.

On an Intel Core i5-13420H, the current 32-frame streaming profile keeps every
measured region below RTF 1 with zero deadline misses. The independent
384-frame batch profile renders 31.208 seconds of audio in a median 22.031
seconds, an aggregate RTF of **0.706**, entirely on the CPU.

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
> editor. The v0.2 provider package contains the native ABI and offline model
> tool required by the matching OpenUtau integration, but it is not part of an
> upstream OpenUtau release. Bring your own compatible exported voicebank.

## Current CPU Performance

Streaming and batch results are separate champions. For streaming, worst-region
RTF below 1 is a real-time feasibility boundary: every block must finish before
its playable audio is exhausted. It is not the score to minimize after the
service gate passes. The optimization objective then moves to CPU-RTF and
concurrent-track capacity while preserving zero deadline misses. Batch rendering
continues to optimize complete E2E wall-time RTF. Results are never promoted by
comparing one architecture with the other.

| Contract | Current champion |
| --- | ---: |
| Streaming workload | 32 frames, 25 measured regions after 10 warm-ups |
| Streaming service gate | **0.996841** largest worst-region RTF across three runs; required `< 1` |
| Streaming CPU-RTF | **2.391249** maximum across three runs |
| Streaming deadlines | **0 misses in every run** |
| Streaming waveform quality | cosine 0.999292, SNR 28.49 dB |
| Batch workload | 7 x 384-frame regions after 2 warm-ups |
| Batch generated audio | 31207.619 ms |
| Batch median E2E latency | **22031.032 ms** |
| Batch median aggregate RTF | **0.705950** |
| Batch p90 / worst latency | 24730.666 / 25563.829 ms |
| Batch waveform quality | cosine 0.999033, SNR 27.13 dB |

Both profiles were measured on an Intel Core i5-13420H at PL1 45 W with the
performance platform and EPP policies. Streaming uses four workers to reduce
per-track CPU load; batch uses eight P-core/SMT workers for throughput. The
batch champion improved its paired control by **8.02%** while improving p90 and
worst latency.

See the [streaming policy and evidence](docs/STREAMING_PERFORMANCE.md), the
[batch policy and evidence](docs/BATCH_PERFORMANCE.md), and the tracked
[batch raw samples](benchmarks/artifacts/2026-09-22-batch384-k11-convt-oc2-45w/samples.tsv).
Historical M55/M58 long-audio results remain in
[the performance record](docs/PERFORMANCE.md), but are not current streaming or
fixed-block champions. Results on other CPUs and voicebanks will vary.

## Why Two Rendering Architectures?

Interactive playback and offline rendering put pressure on different parts of
the system. One block size cannot minimize response time and maximize sustained
throughput at the same time, so DiffSinger-ASM supplies two independently tuned
engines:

| Mode | Used for | Execution shape | Performance focus |
| --- | --- | --- | --- |
| Real-time streaming | Playback while editing | 32-frame vocoder buckets, 8-frame overlap, 4 workers, progressive PCM callbacks | Service constraint: worst-region RTF below 1 with zero deadline misses. Optimization after that: CPU-RTF and concurrent-track capacity |
| Block batch | Pre-render, mixdown, and export | 384-frame buckets, no overlap, 8 workers, complete blocks | End-to-end throughput, aggregate RTF, p90 and worst latency, and run-to-run stability |

Small streaming blocks bound the time before the mixer receives audio, but
they repeat scheduling, synchronization, and overlap work more often. Large
batch blocks amortize that overhead and keep more arithmetic in flight, but
waiting for a large block would make interactive playback feel unresponsive.
Separate worker counts and kernels let each workload optimize the cost that its
user actually notices.

The streaming metrics form a staged optimization gradient. Before the service
gate is reached, reducing the largest worst-region RTF is necessary. After it
is reached, a lower RTF does not by itself make a better streaming engine: a
candidate advances by reducing CPU-RTF and increasing usable track capacity
while every region remains below 1 and every deadline still passes. Batch RTF
has a different meaning because batch users are waiting for the complete job.

The modes are execution choices, not quality levels. Both pass the same model
correctness and waveform quality gates, and both declare output compatibility
revision 1. Once either mode completes a phrase, OpenUtau may reuse that
canonical PCM for later playback, pre-rendering, mixdown, or export.

## Why Native Assembly?

- **Each product mode has its own target.** Streaming protects deadlines and
  CPU capacity; batch rendering maximizes complete E2E throughput.
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

## Install and Use

DiffSinger-ASM is used through the matching OpenUtau build. Normal users do not
need a compiler, Python, ONNX Runtime, or command-line setup.

1. Download and extract the Linux x64 build from
   [AntheaLaffy's OpenUtau releases](https://github.com/AntheaLaffy/OpenUtau/releases).
2. Download and extract the latest
   [DiffSinger-ASM provider](https://github.com/Asmory/DiffSinger-ASM/releases/latest).
   In OpenUtau, open **Preferences > Rendering**, select the provider's
   `lib/libdsasm.so` under **DiffSinger ASM native library**, and confirm that
   its status is ready.
3. On a DiffSinger track, select **DIFFSINGER-ASM** as the renderer.
4. Open **Tools > Singers**, select the voicebank, and choose
   **Convert current model**. OpenUtau retains the ONNX source and publishes the
   converted model only after validation succeeds.

The singer is ready when its status changes to **ASM model installed**.
Complete pre-renders and completed real-time renders share the canonical PCM
cache, so playback reuses audio OpenUtau has already rendered.

The provider currently requires Linux x86-64 with AVX2 and FMA. Conversion adds
about 550 MB per singer and can take several minutes. Unsupported model graphs
are reported without modifying the source voicebank.

## Develop and Use the CLI

The release package already includes its offline dependencies. A source
checkout needs a C toolchain, GNU Make, binutils, CPython 3.12, and `uv`:

```bash
git clone https://github.com/Asmory/DiffSinger-ASM.git
cd DiffSinger-ASM
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r tools/model-tool-requirements.in
```

Install PyTorch in this environment when working on checkpoint importers or
golden/parity validation. Install Linux `perf` when profiling. See the
[developer guide](docs/DEVELOPER_GUIDE.md) for dependency boundaries, repository
layout, agent and tool routing, experiment scheduling, validation gates,
benchmark rules, packaging, and OpenUtau integration. It translates the core
rules in `AGENTS.md` into a workflow intended for both human contributors and
coding agents.

### Build from source

```bash
make -j"$(nproc)" engine-check
make -j"$(nproc)" package VERSION=v0.2.2
```

This produces `build/libdsasm.so` and a self-contained archive under
`release/`. The package embeds Python, NumPy, ONNX, and ONNX Runtime for offline
conversion; native inference does not load them.

Check CPU features when testing a local build:

```bash
grep -m1 -oE 'avx2|fma|avx_vnni' /proc/cpuinfo | sort -u
```

### Model protocol CLI

Inspect, plan, convert into caller-owned staging, and validate without loading
the native runtime:

```bash
package=/absolute/path/to/diffsinger-asm-v0.2.2-linux-x86_64
singer=/absolute/path/to/singer

"$package/bin/dsasm-model-tool" inspect \
  --protocol 1 --singer-root "$singer" --json
"$package/bin/dsasm-model-tool" plan \
  --protocol 1 --singer-root "$singer" --json
"$package/bin/dsasm-model-tool" convert \
  --protocol 1 --singer-root "$singer" \
  --expected-source-fingerprint SHA256_FROM_PLAN \
  --staging /path/on/singer/filesystem/staging \
  --work /path/to/reusable/work --jsonl
"$package/bin/dsasm-model-tool" validate \
  --protocol 1 --bundle /path/on/singer/filesystem/staging --json
```

Conversion always produces the 32- and 384-frame product buckets and writes
`bundle.json` only after offline validation succeeds. Staging must be empty and
must not overlap the reusable work directory. SIGINT cancels and reaps the
active packer process group before the tool returns. The tool never publishes
`current.json`; OpenUtau owns the atomic generation commit.

The direct packer commands below are retained for development and standalone
debugging. Product integrations should use `dsasm-model-tool` so compatibility,
fingerprints, estimates, reason codes, and validation stay on one versioned
boundary.

The standalone acoustic and vocoder executables remain available for model
inspection, parity checks, and debugging:

```bash
make -j"$(nproc)" build/dsasm-acoustic build/dsasm-vocoder-m40
```

Pack the acoustic model into the singer's default ASM directory:

```bash
python tools/pack_acoustic_onnx_m25.py /path/to/voicebank/acoustic.onnx \
  --model-dir /path/to/voicebank \
  --out /path/to/singer/dsasm/acoustic

build/dsasm-acoustic inspect /path/to/singer/dsasm/acoustic
```

Compile the two product vocoder buckets. The filenames are part of the engine
contract; each graph remains fixed-shape internally:

```bash
mkdir -p /path/to/singer/dsasm/vocoder
for frames in 32 384; do
  python tools/pack_vocoder_graph_m35.py \
    /path/to/voicebank/dsvocoder/nsf_hifigan.onnx \
    --frames "$frames" \
    --vnni-scope all-k711 \
    --residual-scope all3711 \
    --out "/path/to/singer/dsasm/vocoder/$frames.dsv35" \
    --work "build/packer-$frames"
done
```

### OpenUtau package discovery

Selecting an extracted provider's `lib/libdsasm.so` lets OpenUtau discover
`bin/dsasm-model-tool` from the same package root. Automated deployments may
set `OPENUTAU_DSASM_LIBRARY` to the library. Source-checkout development may
use `DIFFSINGER_ASM_HOME` and the explicit `OPENUTAU_DSASM_MODEL_TOOL`
override.

Published singers use an immutable generation selected by one pointer:

```text
<singer>/dsasm/current.json
<singer>/dsasm/generations/<generation>/bundle.json
<singer>/dsasm/generations/<generation>/acoustic/...
<singer>/dsasm/generations/<generation>/vocoder/32.dsv35
<singer>/dsasm/generations/<generation>/vocoder/384.dsv35
```

Projects use the fixed renderer IDs `DIFFSINGER` for ONNX and
`DIFFSINGER-ASM` for this provider. Selecting ASM reports incompatibility or
runtime unavailability instead of silently changing backend. Switching to ASM
may inspect compatibility silently, but conversion remains an explicit user
action.

ABI v3 rejects configurations it cannot reproduce, including energy
conditioning and pitch-controllable vocoders. ASM and ONNX renders use separate
WAV cache keys. Both ASM modes declare output compatibility revision 1, so a
complete pre-render can satisfy later playback and a completed real-time render
can satisfy later batch use. Partial PCM is published only to the active
playback session; it enters the complete render cache only after the unique
final chunk arrives.

The native engine exposes separate four-worker/32-frame real-time and
eight-worker/384-frame batch modes. OpenUtau chooses the mode from whether the
render is consumed progressively during playback or as a complete block.

### Direct inference CLI

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

ABI v3 mode configuration records the worker, region, bucket, and overlap
values that passed each architecture's current end-to-end gate. Runtime
defaults select the corresponding shape-gated graph and kernel paths.
[`config/best-inference.env`](config/best-inference.env) preserves the common
environment controls used for benchmark reproduction.

```bash
. config/best-inference.env
env | grep '^DSASM_' | sort
```

Do not use the promoted AVX-VNNI settings on a CPU without AVX-VNNI. Local
source builds default to `-march=native`; optional AVX-VNNI vocoder paths
remain runtime-gated.

## Verification

Build the shared library, validate ABI layout and exported dependencies, and
run the real-model streaming test when local packed artifacts are available:

```bash
make -j"$(nproc)" engine-check
make model-tool-check
make engine-real-stream-check \
  ENGINE_REAL_ACOUSTIC=/path/to/packed/acoustic \
  ENGINE_REAL_VOCODER=/path/to/packed/vocoder
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

The native ABI v3 exposes explicit real-time streaming and block batch modes.
Mode configuration is queryable, and mode engines reject mismatched request
modes, buckets, or overlap values. See the
[OpenUtau binding contract](docs/ENGINE_ABI.md#openutau-contract).

## Architecture

```text
Exported voicebank (offline)
  acoustic.onnx ----> DSFS25 + DSAUX20 + DSLYNX7
  nsf_hifigan.onnx -> 32 / 384-frame DSVOC35 buckets

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

- [Developer guide](docs/DEVELOPER_GUIDE.md)
- [Performance policy and current evidence](docs/PERFORMANCE.md)
- [Dual-architecture optimization strategy](docs/DUAL_ARCHITECTURE_OPTIMIZATION.md)
- [Real-time streaming performance policy](docs/STREAMING_PERFORMANCE.md)
- [Block batch performance policy](docs/BATCH_PERFORMANCE.md)
- [CPU runtime implementation policy](docs/RUNTIME_IMPLEMENTATION_POLICY.md)
- [Stable engine ABI and streaming contract](docs/ENGINE_ABI.md)
- [Offline model protocol](docs/OPENUTAU_OFFLINE_MODEL_PROTOCOL.md)
- [OpenUtau supply contract draft](docs/OPENUTAU_SUPPLY_CONTRACT_DRAFT.md)
- [v0.2.2 release notes](docs/RELEASE_NOTES_V0.2.2.md)
- [v0.2.1 release notes](docs/RELEASE_NOTES_V0.2.1.md)
- [Manual validation and release process](docs/RELEASING.md)
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
Validation and releases are run locally; the repository intentionally has no
hosted CI or continuous deployment workflow.

No voicebank, singer identity, or third-party model weights are included in the
repository or release archives.

Use voice models only with the voice owner's consent. This project follows the
[upstream DiffSinger responsible-use notice](https://github.com/openvpi/DiffSinger#disclaimer):
do not use it to generate a person's voice without permission.
