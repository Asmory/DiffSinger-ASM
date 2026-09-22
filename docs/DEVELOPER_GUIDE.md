# DiffSinger-ASM Developer Guide

This guide covers source development, validation, profiling, packaging, and
OpenUtau integration. Users installing a released provider should follow the
four-step flow in the project README instead.

## Supported Development Host

The product target is Linux x86-64 with AVX2 and FMA. AVX-VNNI kernels are
selected only when the CPU supports them. A development host needs:

- GCC or Clang, GNU Make, binutils, Git, pthread, libc, and libm;
- CPython 3.12 and `venv` for offline tools;
- `uv` for reproducible provider packaging;
- enough storage for source ONNX models, conversion work, and prepared bundles;
- Linux `perf` for performance investigation.

The installed provider and the source-development environment have different
dependency boundaries:

| Workflow | Dependencies |
| --- | --- |
| Native inference | `libc`, `libm`, and `pthread` |
| Product model conversion | CPython 3.12, NumPy 2.2.6, ONNX 1.22.0, ONNX Runtime 1.27.0, PyYAML 6.0.3, and locked transitive dependencies |
| Checkpoint import and reference validation | Product conversion dependencies plus PyTorch |
| Performance analysis | The selected benchmark dependencies and Linux `perf` |
| Provider packaging | `uv`, Git, a managed CPython 3.12 runtime, and network access for the first download |

Released providers embed the complete product conversion environment. PyTorch
is deliberately development-only because OpenUtau converts exported ONNX
voicebanks and does not import training checkpoints.

## Initial Setup

```bash
git clone https://github.com/Asmory/DiffSinger-ASM.git
cd DiffSinger-ASM

python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r tools/model-tool-requirements.in
```

Install PyTorch in the same environment when working on checkpoint importers,
golden generation, or PyTorch parity tools. Choose the CPU or GPU wheel for the
development host from the official PyTorch installation instructions; it is
not used by `libdsasm.so` or `dsasm-model-tool`.

Check the target CPU before running product profiles:

```bash
grep -m1 -oE 'avx2|fma|avx_vnni' /proc/cpuinfo | sort -u
```

Package downloads are retained under `build/package-cache/`. When a proxy is
needed, set the standard `HTTP_PROXY` and `HTTPS_PROXY` variables; do not move
the cache to a temporary directory.

## Repository Map

| Path | Purpose |
| --- | --- |
| `src/kernels/x86_64/` | Handwritten CPU neural-network kernels |
| `src/runtime/` | Model loading, validation, dispatch, scheduling, buffers, cancellation, and callbacks |
| `src/cli/` | Native acoustic and vocoder diagnostic programs |
| `include/` | Public ABI and internal native interfaces |
| `tools/dsasm_model_tool.py` | Versioned offline model protocol |
| `tools/pack_*` | Offline ONNX and checkpoint conversion tools |
| `tools/validate_*` | Golden and parity validation |
| `tools/bench_*` | Performance instruments |
| `tests/` | ABI, kernel, format, protocol, and regression tests |
| `benchmarks/artifacts/` | Reproducible accepted performance evidence |
| `docs/` | Architecture policies, ABI, protocol, formats, and milestone history |

## Project Operating Model

[`AGENTS.md`](../AGENTS.md) is the authoritative execution policy for coding
agents, but its rules also describe how human contributors should work. The
core idea is that missing tools or dependencies do not justify a weaker
implementation or a skipped validation step. Provision the compiler, profiler,
debugger, library, model, data set, or service the task actually needs. If it
cannot be obtained, record the concrete blocker and the evidence that remains
unknown.

Route work to tools according to the question being answered:

| Question | Appropriate tools and evidence |
| --- | --- |
| Is output incorrect or unstable? | Minimal reproducer, structured logs, debugger, sanitizers, syscall tracing, and a correctness regression test |
| Where is time or CPU being spent? | End-to-end timers, stage and operator instrumentation, `perf stat`, sampling profiles, and hardware counters |
| Did an optimization improve the product? | Separate paired uninstrumented runs against the matching champion |
| Is a format or ABI change safe? | Layout tests, malformed-input tests, real bundle validation, and consumer integration tests |
| Is a result reproducible? | Exact commands, source revision, model hashes, backend/device fingerprint, raw samples, summary, archive, and SHA-256 |

An instrument answers where to work; it does not decide whether a candidate is
faster. Instrumentation overhead and blind spots make profiled runs unsuitable
for promotion. Correctness evidence also comes before timing: measurements from
a failed parity or finite-output run are discarded.

For agent-assisted work, load the task-specific guidance before editing:
performance optimization uses the repository's
[performance-gradient-optimization skill](../skills/performance-gradient-optimization/SKILL.md)
and profiling workflow; correctness failures use debugging tools; test changes
use the testing workflow; commits and releases use the Git and reproducibility
workflows. These routes keep tool choice explicit instead of relying on an
agent's intuition.

### Tool and Experiment Scheduling

Schedule work according to its data dependencies and its effect on the target
machine:

1. **Read and discover in parallel.** Independent source searches, policy
   reads, log inspection, and artifact inventory can run together. This shortens
   investigation without changing repository or machine state.
2. **Serialize dependent changes.** Provisioning, generated artifacts, source
   edits, formatting, and commits consume one another's results. Run them in
   order, and keep concurrent agents away from the same files and worktree.
3. **Establish correctness before performance.** Build and run the applicable
   correctness or parity gate before spending target-machine time on a
   candidate. A failed output cannot produce admissible performance evidence.
4. **Give performance runs exclusive access.** Do not overlap formal baselines,
   paired candidate runs, or profiler captures with builds, conversions, other
   benchmarks, or unrelated CPU-heavy work. Record affinity, worker topology,
   power policy, thermal state, and background-load assumptions.
5. **Separate localization from promotion.** Instrumented runs may execute only
   long enough to identify an attributable cost. Disable the instrumentation,
   stabilize the host, and use paired uninstrumented samples for the promotion
   decision.
6. **Publish only after the gates pass.** Package validation, checksum creation,
   consumer smoke tests, commit, tag, and release form a dependency chain. Each
   step consumes the exact artifact accepted by the preceding step.

Parallelism is useful for independent evidence gathering and isolated test
shards. It is harmful when jobs compete for the CPU being measured, mutate the
same staging directory, share an engine that serializes requests, or make an
artifact's provenance ambiguous. Stop when the task's correctness gate and
declared product objective are answered; extra measurements without a concrete
remaining risk do not strengthen a result.

## Implementation Boundaries

Before changing runtime or benchmark code, read:

- [Dual-architecture optimization](DUAL_ARCHITECTURE_OPTIMIZATION.md)
- [Streaming performance policy](STREAMING_PERFORMANCE.md)
- [Batch performance policy](BATCH_PERFORMANCE.md)
- [Runtime implementation policy](RUNTIME_IMPLEMENTATION_POLICY.md)

Handwritten CPU neural arithmetic belongs in x86-64 assembly. oneDNN and GGML
are approved CPU backends, and GPU frameworks may be introduced as explicit
backends. C and C++ own integration and scheduling. Python is restricted to
offline conversion, correctness validation, golden generation, benchmarks, and
analysis. Unsupported graph branches must fail explicitly.

## Two Product Architectures

The modes share quality gates and canonical PCM compatibility, but their
performance objectives are independent:

| Mode | Product workload | Service constraint | Optimization objective |
| --- | --- | --- | --- |
| Real-time streaming | 32-frame progressive playback regions | Every measured region has worst RTF below 1 and zero deadline misses | Before feasibility: largest worst-region RTF. After feasibility: largest CPU-RTF and concurrent-track capacity |
| Block batch | 384-frame pre-render, mixdown, and export regions | Correct, finite complete output | Complete end-to-end RTF with p90, worst latency, stability, and quality guards |

Do not compare candidates across modes, worker counts, backends, devices,
precisions, or graph partitions. A profiling run locates work; a separate
paired uninstrumented run decides promotion.

## Measured Gradient Optimization

The project's “gradient” is empirical. It does not differentiate a symbolic
cost function. It repeatedly measures the product objective, localizes the
largest attributable cost, changes one controlled factor, and accepts the step
only when the complete end-to-end objective improves.

1. **Fingerprint the stratum.** Record architecture, backend, device, precision,
   graph partition, model, request shape, worker topology, CPU affinity, power
   policy, and build identity.
2. **Start from the champion.** Compare only with the historical best accepted
   result for that exact stratum. A result from another mode or backend is not
   a baseline.
3. **Run correctness first.** Require the model-specific golden/parity gate and
   finite output before admitting timing evidence.
4. **Measure end to end.** The architecture's product metric decides whether a
   candidate already passes. Child benchmarks cannot veto an end-to-end win.
5. **Localize a failure.** If end to end does not pass, use measured stage,
   operator, shape, and hardware-counter evidence to select the next target.
6. **Weight child work from the parent profile.** A faster isolated kernel
   matters only in proportion to its measured share of the complete workload.
   If the weighted model predicts a win that does not appear end to end,
   instrument the unexplained interval instead of inventing a cause.
7. **Change one attributable factor.** Preserve a control path so the candidate
   can be compared under the same environment.
8. **Promote with paired uninstrumented runs.** Use thermally stable samples and
   the architecture's current 5% staircase. Median streaming RTF never replaces
   the required worst-case metric.
9. **Archive accepted evidence.** Store commands, environment, fingerprints,
   raw data, summaries, source revision, tag, archive, and checksums. Diagnostic
   failures may remain for analysis but do not move the champion.

This loop prevents attractive microbenchmarks, unlike machines, or hidden
profiler overhead from steering product decisions. It also explains why the
real-time objective changes after feasibility: worst-region RTF drives the
gradient until every deadline passes; CPU-RTF drives it afterward while the
deadline constraint remains fixed.

## Build and Correctness Gates

Build the shared library and validate ABI v3:

```bash
make -j"$(nproc)" engine-check
```

Run the core native and current synthetic graph gates:

```bash
make -j"$(nproc)" test
make -j"$(nproc)" m35-check m38-check m58-check
```

Run the offline protocol suite:

```bash
make model-tool-check
```

Before a release, all four commands must pass. Performance changes additionally
need the mode-specific real-model golden and promotion gate described by the
streaming or batch policy.

## Offline Model Protocol

The product boundary is the packaged `bin/dsasm-model-tool`, with versioned
`inspect`, `plan`, `convert`, and `validate` operations. Build a provider before
testing the exact shipped environment:

```bash
make -j"$(nproc)" package VERSION=dev
package="$PWD/release/diffsinger-asm-dev-linux-x86_64"
mkdir -p "$package"
tar -xzf release/diffsinger-asm-dev-linux-x86_64.tar.gz \
  -C "$package" --strip-components=1
```

Then run the protocol against a voicebank:

```bash
singer=/absolute/path/to/singer
staging="$PWD/build/dev-staging"
work="$PWD/build/conversion-work/dev"

"$package/bin/dsasm-model-tool" inspect \
  --protocol 1 --singer-root "$singer" --json
"$package/bin/dsasm-model-tool" plan \
  --protocol 1 --singer-root "$singer" --json
"$package/bin/dsasm-model-tool" convert \
  --protocol 1 --singer-root "$singer" \
  --expected-source-fingerprint SOURCE_SHA256_FROM_PLAN \
  --staging "$staging" --work "$work" --jsonl
"$package/bin/dsasm-model-tool" validate \
  --protocol 1 --bundle "$staging" --json
```

Use an empty staging directory. Keep reusable work under `build/`; conversion
work and staging must not overlap. The tool writes `bundle.json` only after the
entire bundle passes offline validation. OpenUtau owns generation publication
and `current.json`.

## Native Diagnostic CLIs

Build the standalone programs:

```bash
make -j"$(nproc)" build/dsasm-acoustic build/dsasm-vocoder-m40
```

These programs inspect packed formats and run controlled inference. They do not
replace the OpenUtau score frontend. Acoustic token, duration, F0, language,
speaker, and variance inputs must already satisfy the model contract, and the
vocoder graph must match the exact fixed frame count.

## Performance Work

Never select a hotspot from source inspection alone. For the exact target
workload:

1. Run the current end-to-end baseline.
2. Use stage timers, operator timers, and hardware counters to locate cost.
3. Change one attributable kernel or scheduling policy.
4. Run correctness and waveform parity first.
5. Compare paired, thermally stable, uninstrumented runs with the current
   champion for the same workload stratum.
6. Archive commands, fingerprints, raw samples, summaries, hashes, and source
   identity only after the end-to-end gate passes.

Streaming uses worst-region RTF only until every region stays below RTF 1 with
zero deadline misses. That threshold means the producer keeps pace with audio;
it is not an offline speed ranking. Once feasible, the measured gradient moves
to the largest CPU-RTF and concurrent-track capacity while preserving the
service constraint. A candidate with a lower RTF but higher CPU cost is not an
automatic improvement. Batch users wait for completion, so complete end-to-end
RTF remains the batch throughput objective. Consult the policy documents for
the current 5% promotion rules and required evidence.

## OpenUtau Integration

OpenUtau loads `lib/libdsasm.so` from an extracted provider and discovers
`bin/dsasm-model-tool` from the same package root. For automated development:

```bash
export OPENUTAU_DSASM_LIBRARY=/absolute/provider/lib/libdsasm.so
export OPENUTAU_DSASM_MODEL_TOOL=/absolute/provider/bin/dsasm-model-tool
```

`DIFFSINGER` selects ONNX Runtime and `DIFFSINGER-ASM` explicitly selects this
provider. OpenUtau owns UI, scheduling, caches, dual-engine lifetime, immutable
model generations, and atomic publication. The provider owns ABI behavior,
model compatibility, packed formats, conversion, and offline validation.

## Packaging and Release

Build a portable AVX2/FMA provider rather than publishing a local
`-march=native` binary:

```bash
make -B -j"$(nproc)" package VERSION=vX.Y.Z DIST=release \
  CFLAGS='-O3 -Wall -Wextra -std=c11 -mavx2 -mfma' \
  CXXFLAGS='-O3 -Wall -Wextra -std=c++17 -mavx2 -mfma'
```

Validate the checksum, extract outside the source checkout, run the packaged
model tool, query both mode configurations, and create both mode engines from a
validated real bundle. The provider build identity must not contain `-dirty`.
The complete release procedure is in [RELEASING.md](RELEASING.md).
