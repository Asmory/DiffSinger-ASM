---
name: performance-gradient-optimization
description: Optimize measurable system performance through instrument-guided experiments, stable champion baselines, explicit promotion gates, calibrated subsystem and composition models, and reproducible archives. Use for latency, throughput, CPU, memory, I/O, energy, capacity, cost, or performance-model refinement; do not use as a substitute for correctness debugging.
---

# Performance Gradient Optimization

Use "gradient" as an experimental method: measure the current system, perturb
one controllable direction, observe the result, and keep only stable progress.
Do not assume the objective is differentiable or that a symbolic performance
model is complete.

## Define the optimization contract

Before changing the implementation, record:

- the workload and its strata, such as request class, input shape, concurrency,
  data set, machine class, and operating regime;
- one primary objective, its direction, and units;
- hard constraints such as correctness, quality, tail latency, memory limits,
  deadlines, compatibility, or cost ceilings;
- a minimum meaningful improvement step large enough to exceed measurement
  noise and engineering cost;
- the warm-up, sampling, pairing, and stability protocol.

Use lexicographic decisions by default: hard constraints first, then the primary
objective, then secondary regression guards. Use a Pareto frontier only when
the product genuinely has multiple coequal objectives. Never hide a failed hard
constraint inside a weighted average.

Maintain a separate champion baseline for every incompatible workload stratum
and objective. Compare a candidate only with the current champion for that same
contract, not with an arbitrary historical version. When hardware, software,
input distribution, or operating conditions change materially, establish a
fingerprinted context baseline instead of mixing populations.

## Instrument before choosing a direction

Do not infer hotspots from source size, nominal operation counts, intuition, or
a single fast sample. Measure the exact target workload and descend only as far
as needed to explain the dominant cost:

1. primary outcome and end-to-end wall time;
2. resource totals such as CPU time, memory, I/O, energy, and occupied capacity;
3. stage, component, operator, query, or hot-shape timing;
4. sampling profiles, traces, system-call evidence, and hardware counters when
   higher-level measurements leave an unexplained residual.

Correctness and other hard constraints are checked before performance evidence
is admitted. Instrumented runs locate work; separate minimally instrumented
paired or interleaved runs decide promotion. Record profiler overhead and blind
spots. Missing measurements remain unknown rather than receiving guessed
weights or invented causal explanations.

## Run one optimization step

1. Reproduce the champion under the current contract.
2. Use measured evidence to select the largest controllable opportunity.
3. State the expected mechanism and the metrics that could falsify it.
4. Change one implementation or policy direction and retain a control path.
5. Run correctness and constraint checks.
6. Collect stable paired or interleaved baseline/candidate samples.
7. Promote, reject, or refine the measurement model from the observed result.

Choose sample count and summary statistics for the objective. Throughput may be
well represented by a central estimate; service latency may require a maximum,
deadline-miss count, or high percentile. Include warm-up and sustained behavior
when caches, allocators, JITs, thermal limits, frequency scaling, or background
work can change the distribution. The fastest sample is diagnostic, not a
promotion result.

Increase sample count or improve experimental control when the confidence
interval or run-to-run variation is too large to distinguish the required step.
Do not lower the gate merely to accept a noisy candidate.

## Use subsystem models to diagnose misses

The primary outcome is authoritative. If it passes, subsystem metrics are
informational and cannot veto promotion unless they are declared hard guards.

If the primary outcome fails, use measured subsystem results to choose the next
direction. For an additive parent cost, a useful predicted ratio is:

`sum(weight_i * candidate_i / baseline_i)`

Derive each weight from the same parent profile and normalize it over the
modeled interval. Report that result as a modeled-interval prediction, not an
end-to-end prediction. For an end-to-end projection, use mutually exclusive
parent shares and include work assumed unchanged:

`unchanged_share + sum(parent_share_i * candidate_i / baseline_i)`

Report the accounted share `sum(parent_share_i)` and the unmodeled residual.
Nested or overlapping timers must not be added as if they were exclusive.
Record confidence and controllability adjustments. A count of winning
subsystems is not a valid substitute for weights.

Do not force additive weighting onto critical paths, parallel stages, queueing,
memory-pressure effects, or other nonlinear systems. Model their actual
composition, or treat the subsystem data only as ranking evidence.

Choose a composition model that matches the measured mechanism:

- For sequential exclusive stages, use additive shares and retain an explicit
  residual term.
- For parallel work, reconstruct the execution DAG and use its measured
  critical path or work-span model. Speeding work outside the critical path has
  no predicted end-to-end benefit unless it changes contention or scheduling.
- For streaming and capacity-limited systems, separate service time from queue
  wait, backpressure, and synchronization. Track arrival rate, service capacity,
  utilization, queue depth, and the latency distribution. Little's Law
  `L = lambda * W` is a steady-state consistency check; it does not make queue
  delay linear as utilization approaches saturation.
- For compute and memory limits, use Roofline or a measured CPU execution-cache-
  memory model to identify the active resource ceiling. Use that ceiling to
  rank directions, not as an end-to-end prediction unless composition and
  overhead are also modeled.

Use measured parent share `p` and local speedup `s` with Amdahl's bound
`1 / ((1 - p) + p / s)` as a plausibility check for an isolated optimization.
Treat a result outside measurement uncertainty as evidence that shares moved,
the change affected another subsystem, or the model omitted work; do not bend
the bound to explain the observation.

If the subsystem model predicts success but the primary outcome still fails,
instrument the unmodeled residual. Promote a newly observed factor to a tracked
sub-baseline when it is repeatably measurable and acting on it improves the
primary gate. Full causal attribution is useful but not mandatory; the context
and observed distribution must be preserved.

Calibrate every predictive model against held-out candidate observations. Track
signed residual `observed - predicted` and relative error
`abs(observed - predicted) / observed`, with uncertainty propagated from the
underlying samples. Declare an error tolerance from measurement noise and the
minimum meaningful improvement before evaluating the candidate. When error
exceeds that tolerance repeatedly, the model may rank diagnostic opportunities
but must not predict gate passage until it is refined and revalidated. Do not
fit and validate a revised model on the same observation.

When known subsystems no longer offer meaningful progress, expand measurement
coverage before selecting another target. This is model refinement, not a
license to guess.

## Promote and preserve

Promote only when the candidate:

- passes correctness and every hard constraint;
- improves the primary objective by at least the declared step;
- satisfies the declared stability and regression guards;
- is compared under the same contract as its champion.

On promotion, update the champion immediately and preserve the exact commands,
raw samples, summaries, profiles, environment, workload and artifact hashes,
source revision, and tool versions. Commit the intended change and evidence;
when the project uses releases or milestone archives, tag it, archive it, record
the archive checksum, and verify recovery instructions.

Rejected candidates may retain diagnostic evidence, but they do not move,
overwrite, or inherit the champion label.

## Respect project overlays

Project instructions define concrete objectives, thresholds, workloads,
implementation boundaries, tools, and archive formats. Read them before acting
and treat them as stricter overlays on this method. Do not turn one project's
threshold or architecture into a universal rule.
