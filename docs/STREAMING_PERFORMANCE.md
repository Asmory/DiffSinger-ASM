# Real-time streaming performance policy

Long-phrase RTF is retained as historical throughput evidence, but it is not a
streaming acceptance metric. Streaming work uses many small render regions on
one resident engine so model loading is excluded while frequency and thermal
variation remain represented.

## Primary workload

- 32 acoustic frames per region at a 512-sample hop and 44.1 kHz
- 32-frame fixed vocoder bucket
- one persistent engine and thread pool
- at least 10 warm-up regions followed by at least 25 measured regions
- real DongFangZhiZi Nectar acoustic and vocoder weights
- deterministic noise seed and finite-output check

Each 32-frame region publishes 16384 samples, or 371.52 ms of playable audio.
Longer requests with the default 8-frame overlap publish 24 new frames per
steady callback, or 278.64 ms. Both quantities must be derived from actual
callback sample counts rather than assumed by the benchmark.

Build and run the instrument with:

```sh
make build/bench_engine_stream
./build/bench_engine_stream PACKED_ACOUSTIC VOCODER_BUCKET_DIR SPEAKER_EMB \
  --frames 32 --regions 25 --warmup 3 --workers 8 --bucket 32 --steps 4 \
  --golden STREAM32_GOLDEN_PCM_F32
```

Formal baseline and promotion runs must supply an offline golden waveform for
the exact request. `finite=PASS` without `quality_pass=PASS` is diagnostic only.

The instrument records aggregate RTF, first-region latency, region and callback
median/p90/worst/CV, every callback interval and playable duration, deadline
miss count, maximum lateness, and an output checksum. A deadline miss occurs
when the next callback arrives after the previously published PCM duration is
exhausted. Initial latency is compared with the first callback's own duration.

Stage/operator timers and hardware counters are required before choosing a
kernel or scheduler to change. In particular, long-audio throughput, source
loop structure, nominal FLOP count, and one fast callback are not evidence of a
streaming hotspot. Profiling runs identify the target; separate uninstrumented
paired runs decide whether the target produced a real improvement.

## Promotion staircase

Before real-time service is achieved, the primary metric is worst-case
steady-state RTF. Run baseline and candidate at least three times after thermal
warm-up and use the largest worst RTF from each set. A candidate advances one
step only when that value is at least 5% lower, output quality passes, and its
deadline-miss count does not increase. This permits several controlled steps
toward the service target without accepting a noisy or merely faster median.

The service target requires every measured region to have RTF below 1 and every
callback to arrive before the previous PCM block is exhausted. Median RTF does
not participate in promotion. Warm-up must be long enough to include sustained
package-power and frequency behavior, rather than measuring only the initial
boost window.

After the service target passes, optimization follows a CPU-load staircase. Run each
baseline and candidate at least three times after thermal warm-up and use the
largest CPU-RTF from each set. A candidate advances one step only when its
worst CPU-RTF is at least 5% lower, every run keeps worst RTF below 1, and every
run has zero deadline misses. Median values never participate in this gate.
The 5% step is deliberately larger than ordinary package-power noise.

Multi-track capacity is a second, discrete staircase. Start with two concurrent
resident engines and increase the track count by one only when every track
independently meets worst RTF below 1 with zero deadline misses. Aggregate RTF
cannot hide one starved track. Correctness and finite-output checks are
mandatory; timing from a failed correctness run is discarded.

Subsystem measurements are diagnostic. When E2E passes, subsystem results do
not block promotion. When E2E fails, measured parent-profile weights determine
which subsystem candidates matter. If the weighted modeled improvement passes
but E2E does not, profile the unmodeled interval and add a new child baseline
only after the measured evidence identifies it.

Accepted milestones must include the exact command, environment, raw samples,
summary, source commit, tag, archive, and SHA-256. The first valid 32-frame run
establishes a new streaming baseline and is not compared with M55/M58.

## Current 32-frame champion

The accepted pre-service champion for the 45 W package-power context uses
twelve additional quality-gated VNNI operators
(`79,77,80,76,74,73,48,49,51,52,54,55`) and six workers. The last six are the
profiled `Cin256/K11` group. Three uninstrumented interleaved pairs measured:

```text
reproduced hot-six: 1.352409  1.007939  0.992979  max=1.352409
hot-twelve:        0.977545  1.001437  0.984307  max=1.001437
```

The paired reduction is 25.95%. More conservatively, the candidate improves
16.77% over the archived hot-six champion maximum of 1.203178. The maximum
deadline-miss count is 1 versus 18 for the paired control and 9 for the archived
champion. Exact-request ONNX-vocoder parity passes at cosine 0.999465810 and SNR
29.71 dB. The candidate remains in the pre-service phase because one run is
slightly above RTF 1; CPU-RTF is diagnostic until every run crosses the target.

This context is distinct from the historical 28 W measurements. Its fingerprint
includes PL1=45 W, a 55.967744 s PL1 window, the performance platform profile,
intel_pstate EPP=performance, six workers, 10 warm-up regions, and 25 measured
regions. Raw evidence and hashes are stored under
`benchmarks/artifacts/2026-09-22-realtime32-vnni-hot12-45w/`. The predecessor
artifact remains preserved under its own tag and directory.

The exact-request waveform golden must match the acoustic implementation used
by the benchmark. When a validated acoustic implementation changes, preserve
the old golden as historical evidence and generate a new golden offline through
the original ONNX vocoder; never compare a newly linked acoustic runtime with a
waveform golden produced from stale mel.
