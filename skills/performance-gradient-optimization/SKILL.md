---
name: performance-gradient-optimization
description: Run instrument-driven iterative performance optimization with independent E2E baselines, weighted subsystem diagnostics, stable promotion gates, and reproducible milestone archives. Use when profiling bottlenecks, comparing performance candidates, evolving streaming or batch runtimes, or deciding whether a measured optimization should become the new baseline.
---

# Performance Gradient Optimization

Treat optimization as measured descent against the best accepted baseline, not
as a sequence of plausible source edits. Preserve correctness and compare only
like-for-like workloads.

## Establish the contract

Before changing code, record:

- architecture and workload stratum;
- current best accepted baseline commit/tag;
- fixed model, input, shape or block size, worker topology, affinity, build
  flags, environment, and machine fingerprint;
- correctness gate, primary E2E metric, stability guards, and minimum promotion
  step.

Keep real-time streaming and block batch baselines independent. A result from
one architecture cannot promote the other.

When working in DiffSinger-ASM, read
`docs/DUAL_ARCHITECTURE_OPTIMIZATION.md`,
`docs/STREAMING_PERFORMANCE.md`,
`docs/BATCH_PERFORMANCE.md`, and
`docs/RUNTIME_IMPLEMENTATION_POLICY.md` before editing runtime code. Those
project documents override generic defaults in this skill.

## Measure before hypothesizing

Do not choose a target from source structure, nominal FLOPs, intuition, or one
fast sample. Instrument the exact E2E workload first.

Collect only the depth of evidence needed to localize the unexplained cost:

1. monotonic E2E wall time;
2. stage, operator, and hot-shape timers;
3. process CPU time and occupied-core estimates where CPU burden matters;
4. sampling profiles and hardware counters such as cycles, instructions, cache
   misses, branch misses, and frequency residency when timers leave a residual.

Run correctness/parity before admitting timing evidence. Instrumented runs
locate work; separate paired or interleaved uninstrumented runs decide
promotion. Record profiler overhead and blind spots. Never invent missing
weights or causal explanations.

## Apply the architecture gate

For block batch work, compare complete E2E aggregate wall RTF with the best
accepted batch baseline. Use thermally stable paired or interleaved samples.
Require at least a 5% improvement plus the project's tail, variability, and
correctness guards.

For real-time streaming, first require every thermally steady measured region
to have RTF below 1 and zero callback deadline misses. Median RTF is not an
acceptance metric. Once that service gate passes, descend a CPU-load staircase:
use the largest CPU-RTF from at least three hot runs per side and require at
least a 5% reduction while every run keeps the service gate. Test multi-track
capacity as a separate integer staircase; every track must pass independently.

Do not average incompatible machines, models, shapes, worker layouts, or
thermal regimes. Create a fingerprinted context baseline when the observed
population changes; a symbolic causal proof is not required.

## Use child metrics only to diagnose failed E2E

If E2E passes, child metrics are informational and cannot veto promotion.

If E2E fails:

1. derive each child weight from its measured share of the parent profile;
2. adjust confidence or controllability only with recorded evidence;
3. prioritize a failing, high-impact child path;
4. require the weighted child model to pass before rerunning the expensive E2E
   gate.

A useful predicted ratio is `sum(weight_i * candidate_i / baseline_i)`, with
weights from one parent profile and normalized for the modeled interval. Do not
replace this with a count of winning child cases.

If the weighted child model passes but E2E still fails, instrument the
unmodeled interval. Add a newly observed factor as a child baseline only when it
is repeatably measurable and changing it helps the E2E gate pass. Preserve it
as an empirical context baseline even when complete causal attribution is not
available.

## Iterate narrowly

Keep a control path or switch for paired comparison. Prefer one attributable
kernel or scheduling-policy change per candidate. When correctness fails,
discard its timings and repair that path. When a child fails, target that child;
when it cannot improve further, expand measurement instead of guessing another
hotspot.

For DiffSinger-ASM online CPU inference, neural-network arithmetic belongs in
assembly. C is limited to loading, validation, dispatch, scheduling, buffers,
cancellation, and callbacks. Python is offline-only for conversion, golden
generation, parity, and benchmark analysis. Unsupported graph branches fail
explicitly; do not add numerical C or Python fallbacks.

## Promote and archive

Promote only a stable E2E pass against the historical best for the same
architecture and stratum. Then, in the same iteration:

- update the baseline record and performance document;
- preserve exact commands, raw samples, profiler evidence, environment, CPU
  topology, model/input hashes, correctness results, and summaries;
- commit only intended files, create an annotated tag, build the archive, and
  record its SHA-256;
- verify the archive can identify the source commit and reproduce the command.

Rejected candidates may retain diagnostic evidence, but they do not move,
tag, or overwrite the accepted baseline.
