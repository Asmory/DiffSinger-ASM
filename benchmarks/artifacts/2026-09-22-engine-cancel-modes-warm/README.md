# ABI v3 mode cancellation latency

Date: 2026-09-22
Host: `kisaragi`, Intel Core i5-13420H
Gate: p99 cancel-to-return at most 371.52 ms
Result: FAIL

These are paired uninstrumented measurements of cooperative cancellation on
the real product-mode models. Each render was cancelled 10 ms after its render
thread entered `dsasm_engine_render()`. Every measured request returned
`DSASM_E_CANCELLED` and emitted zero callbacks.

| Mode | Workers | Frames | Samples | p50 | p90 | p99 | Worst | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Realtime streaming | 4 | 32 | 25 | 301.436 ms | 500.988 ms | 740.619 ms | 782.688 ms | FAIL |
| Block batch | 8 | 384 | 25 | 1494.680 ms | 1578.341 ms | 1845.819 ms | 1882.523 ms | FAIL |

The realtime result was not stable across repeated runs: an immediately prior
25-sample run measured p99 189.979 ms and worst 190.096 ms. A supplied bound
must hold on repeated runs, so the archived failing run controls the decision.
The batch mode also failed an immediately prior run with p99 1629.070 ms and
worst 1651.759 ms.

Commands:

```sh
./build/bench_engine_cancel \
  build/stream_acoustic \
  build/stream_buckets/current \
  build/stream_acoustic/dongfangzhizi-nectar-xiao.emb \
  --mode realtime --warmup 3 --samples 25 \
  --cancel-delay-ms 10 --gate-ms 371.52 --steps 4

./build/bench_engine_cancel \
  build/stream_acoustic \
  build/stream_buckets/current \
  build/stream_acoustic/dongfangzhizi-nectar-xiao.emb \
  --mode batch --warmup 3 --samples 25 \
  --cancel-delay-ms 10 --gate-ms 371.52 --steps 4
```

`realtime.txt` and `batch.txt` contain the archived per-sample results.
`lscpu.txt`, `uname.txt`, `source-head.txt`, and `sha256.txt` fingerprint the
host, source base, measurement binary, runtime library, and model artifacts.

This measurement rejects the proposed SLA for the current implementation. It
does not establish a replacement bound because the sample count is too small
and the realtime distribution was unstable. Active batch preemption remains
disabled until finer cancellation or another admission policy passes the same
gate on repeated target-machine runs.
