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
  --frames 384 --regions 7 --warmup 2 --workers 8 --bucket 384 --steps 4 \
  --golden BATCH384_GOLDEN_PCM_F32
```

Formal baseline and promotion runs must supply the golden waveform generated
offline from the exact acoustic request used by the benchmark. A finite-only
run is diagnostic and cannot establish or replace the batch champion.

The explicit ABI v2 bucket field is the scheduling boundary: real-time requests
select a measured small bucket, while batch requests select 384. The engine
does not change modes or bucket sizes implicitly. Streaming and batch releases
have independent best-known baselines, tags, evidence, and archives.

When E2E fails, subsystem results may guide the next edit only after their
weights are measured in the parent E2E profile. If the weighted model predicts
a pass but E2E still fails, the next task is to instrument the unmodelled time,
not to invent a weight or choose another kernel by intuition.

## Current 384-frame champion

The first accepted fixed-block champion combines quality-gated VNNI for
Cin64/K11 operators 92 through 96 with an FP32 stride-2/K4 ConvTranspose kernel
that computes two adjacent output channels per call. The latter shares input
loads while preserving each channel's accumulation order and is bit-exact with
the previous ASM path.

Five interleaved runs per side measured:

```text
control:   median=23797.849 ms  RTF=0.762565  p90=25595.058  worst=25817.030  CV=0.056
candidate: median=22031.032 ms  RTF=0.705950  p90=24730.666  worst=25563.829  CV=0.070
gain: 8.02%
```

Every run passed finite and waveform quality gates. Candidate quality is cosine
`0.999033427` and SNR `27.13 dB`. The context is the i5-13420H at PL1 45 W,
performance platform/EPP policy, eight P-core/SMT workers, two warm-up blocks,
and seven measured blocks. Raw evidence is under
`benchmarks/artifacts/2026-09-22-batch384-k11-convt-oc2-45w/` and the release tag
is `batch384-k11-convt-oc2-45w-20260922`.

The dispatch defaults are restricted to the exact 384-frame graph shapes.
`DSASM_VNNI_EXTRA_OPS=off` and `DSASM_CONVT_S2K4_OC2=0` reproduce the control.
Future batch candidates compare only with this champion under the same context.
