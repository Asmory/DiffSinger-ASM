# Block batch performance policy

Batch rendering is a separate architecture and baseline from real-time
streaming. A long phrase is divided into large fixed regions and submitted to
one resident engine. The default measurement uses 384 acoustic frames and the
384-frame vocoder bucket.

The primary metric is aggregate wall-time RTF across the complete phrase.
Process CPU time, block p90/worst latency, and variability remain diagnostic,
but callback deadlines do not gate batch promotion. A candidate must improve
median aggregate RTF by at least 5% in paired stable runs without regressing
p90, worst case, output quality, or numerical validity.

Candidate selection is profiler-driven. First measure complete E2E stage time,
then descend through operator and shape timing; use hardware counters when those
timers leave an unexplained residual. Nominal FLOPs, source complexity, and a
standalone kernel result cannot identify or promote a batch candidate. Profiled
runs locate the work, while separate paired uninstrumented runs apply the 5%
E2E gate against the best accepted batch baseline.

Use the same instrument with a batch-sized region:

```sh
make build/bench_engine_stream
./build/bench_engine_stream PACKED_ACOUSTIC VOCODER_BUCKET_DIR SPEAKER_EMB \
  --frames 384 --regions 7 --warmup 2 --workers 8 --bucket 384 --steps 4
```

The explicit ABI v2 bucket field is the scheduling boundary: real-time requests
select a measured small bucket, while batch requests select 384. The engine
does not change modes or bucket sizes implicitly. Streaming and batch releases
have independent best-known baselines, tags, evidence, and archives.

When E2E fails, subsystem results may guide the next edit only after their
weights are measured in the parent E2E profile. If the weighted model predicts
a pass but E2E still fails, the next task is to instrument the unmodelled time,
not to invent a weight or choose another kernel by intuition.
