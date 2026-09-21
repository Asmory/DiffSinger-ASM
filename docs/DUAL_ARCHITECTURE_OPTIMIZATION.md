# Dual-architecture optimization strategy

## Scope

The runtime has two product architectures with independent objective functions,
baselines, kernel dispatch, promotion gates, tags, and archives. A result from
one architecture must never replace or validate the other architecture's
baseline.

The ABI exposes an exact `vocoder_bucket_frames` request field. C uses the
validated fixed shape to dispatch a real-time or batch assembly kernel family.
The engine never silently grows a request to another bucket.

## Instrumentation before optimization

No optimization candidate may be selected from source inspection, intuition,
or a single timing sample. Source inspection explains a measured result; it is
not evidence that a path is hot. Each iteration starts with an instrument on
the exact workload and architecture whose baseline is being challenged.

The minimum evidence is:

- monotonic wall-clock timing at the request, stage, operator, and hot-shape
  levels needed to localize the cost;
- process CPU time and occupied-core estimates for real-time work;
- repeated thermally steady samples, with CPU affinity, worker count, model,
  input, bucket, environment, and build identity held constant;
- hardware counters such as cycles, instructions, cache misses, branch misses,
  and frequency residency when stage timing cannot explain an E2E result;
- correctness/parity measurements produced before performance evidence is
  admitted.

Profiling overhead must be measured or removed from promotion runs. A profiler
run locates work; an uninstrumented paired run decides promotion. Sampling and
instrumentation blind spots are recorded as unknowns, never filled with guessed
weights or guessed causes.

Optimization uses the following decision loop:

1. Run the architecture's E2E gate against its current historical best.
2. If E2E passes, promote it; child metrics are diagnostic and cannot veto it.
3. If E2E fails, use measured parent-profile shares to weight child results and
   change the failing high-impact child path.
4. If the weighted child model passes but E2E still fails, expand the profiler
   into the unmodelled interval and measure the missing work.
5. Add that work as a child baseline only after it is repeatably measurable and
   the corresponding change helps the E2E gate. A causal explanation is useful
   but is not required to preserve an observed, fingerprinted context baseline.

This is a measurement-driven gradient process, not a claim that the runtime is
fully symbolically modelled. Child weights are recomputed when architecture,
block size, model family, worker topology, or thermal/frequency regime changes.

## Real-time streaming

Real-time rendering submits many small regions to one resident engine. The
primary strata are 32 and 64 acoustic frames. Tests include at least 10 warm-up
regions followed by at least 25 measured regions so initial boost clocks do not
hide sustained package-power behavior.

The service target is absolute:

- every measured region has RTF below 1;
- every callback arrives before previously published PCM is exhausted;
- output passes the model-specific golden/parity gate and contains only finite
  samples.

Median RTF does not participate in promotion. Before the service target passes,
optimization follows a worst-RTF staircase: compare at least three thermally
steady runs with the current streaming champion and promote only when the
candidate's largest worst RTF is at least 5% lower, output quality passes, and
deadline misses do not increase. The champion may therefore advance through
several measured 5% steps before it crosses RTF 1.

After every run has worst RTF below 1 with zero deadline misses, the primary
metric switches to CPU load. CPU-RTF is process CPU time divided by generated
audio duration and approximates the number of continuously occupied cores per
track. Take the largest CPU-RTF from at least three baseline and candidate runs,
and require at least a 5% reduction while every run continues to pass the
service target.

Multi-track capacity is a second discrete staircase. Run independent resident
engines concurrently, starting with two tracks. Increase by exactly one track
only when every track independently has worst RTF below 1 and zero deadline
misses. Aggregate throughput cannot hide a starved track.

Real-time profiling records wall time, process CPU time, average occupied cores,
acoustic/vocoder/callback/other stage time, callback interval, playable duration,
deadline misses, maximum lateness, worst region RTF, and output parity. Candidate
selection follows measured hot shapes. Current measurements identify the
64-frame vocoder Conv group, especially `C=128, K=7, T=4096`, as a major small-T
cost; quantization is admitted per channel/op only after golden parity passes.

## Block batch rendering

Batch rendering splits a long phrase into large fixed regions, initially 384
acoustic frames, and submits them through one resident engine. Callback cadence
does not gate this architecture. The primary metric is complete E2E wall-time
RTF; CPU time, p90, worst latency, and variability remain regression guards.

Each candidate is measured in paired stable runs against the best accepted batch
baseline. Promotion requires at least a 5% E2E improvement, no p90/worst/CV
regression, and successful output parity. Failed correctness runs contribute no
timing evidence. Historical M55/M58 numbers remain long-audio evidence but do
not automatically become the new fixed-block batch baseline.

## Assembly specialization

Online neural-network arithmetic is implemented only in assembly. C is limited
to model loading, validated shape dispatch, thread scheduling, buffer ownership,
cancellation, and PCM publication. Python is offline-only for model inspection,
packing validation, golden generation, parity, and benchmark analysis.

Small-T and large-T kernels may differ substantially:

- real-time assembly minimizes synchronization, worker wakeups, packing cost,
  short-loop tails, and sustained CPU load;
- batch assembly maximizes arithmetic throughput with wider tiles, deeper
  parallel work, and amortized packing.

There is no requirement to share an assembly implementation when measurements
show that separate fixed-shape kernels are better. C dispatch must keep the two
families explicit, and every new assembly path requires direct parity coverage.

## Model compatibility

Performance baselines use one fixed reference voicebank so model changes do not
pollute comparisons. Compatibility is a separate matrix covering every supplied
DiffSinger model. Conversion must detect graph capabilities structurally and
must reject unsupported branches explicitly.

Current model families include static LayerNorm with stretch, static LayerNorm
without stretch, energy-conditioned models, and adaptive-LayerNorm models.
Energy and adaptive LayerNorm require assembly kernels before runtime support is
declared. C or Python numerical fallbacks are forbidden. Shared audio metadata
(sample rate, hop, mel bins) is validated but is not evidence that network
topology is compatible.

## Iteration and archive protocol

1. Start only from the best accepted baseline for the same architecture and
   workload stratum.
2. Profile the failing objective with stage, operator, shape, and hardware
   counters. Do not infer hotspots from source structure alone, and do not
   optimize an unmeasured path.
3. Change one attributable kernel or scheduling policy; preserve a control path
   for paired comparison.
4. Run correctness first, then stable performance measurements. Discard timing
   whenever correctness fails.
5. Apply the architecture-specific 5% staircase and stability rules.
6. On acceptance, record commands, environment, CPU topology, model hashes, raw
   samples, profile, summary, source commit, tag, archive, and SHA-256.
7. On rejection, retain diagnostic evidence only; do not move the baseline.

If modeled child baselines improve but E2E does not, profile the unmodeled
interval. Add a new child baseline only after an instrument identifies its
measurable contribution.
