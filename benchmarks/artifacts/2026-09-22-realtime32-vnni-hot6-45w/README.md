# Real-time 32-frame VNNI hot-six promotion

This artifact promotes the six-op selective VNNI path for the 32-frame
real-time stratum. It does not replace the batch champion or any 28 W context
baseline.

## Contract

- CPU: Intel Core i5-13420H, 4 P-cores with SMT plus 4 E-cores
- runtime workers: 6, topology-aware pinned order
- workload: 32 frames, 32-frame bucket, 4 acoustic steps
- warm-up: 10 regions; measured: 25 regions
- primary objective: largest worst-region RTF across three runs
- gate: at least 5% reduction; output quality pass; deadline misses do not rise
- power: PL1 45,000,000 uW, window 55,967,744 us; PL2 80,000,000 uW
- policy: platform `performance`, intel_pstate EPP `performance`
- kernel: Linux 7.1.11-arch1-1; GCC 16.2.1; binutils 2.47
- offline golden: ONNX Runtime 1.30.0, ONNX 1.23.0, NumPy 2.5.3

The control explicitly sets `DSASM_VNNI_EXTRA_OPS=off`. The candidate uses the
promoted 32-frame default, equivalent to:

```text
DSASM_VNNI_EXTRA_OPS=79,77,80,76,74,73
```

The common command after the environment prefix is:

```sh
./build/bench_engine_stream \
  build/stream_acoustic build/stream_buckets/current \
  build/stream_acoustic/dongfangzhizi-nectar-xiao.emb \
  --frames 32 --regions 25 --warmup 10 --workers 6 --bucket 32 --steps 4 \
  --golden benchmarks/artifacts/2026-09-22-realtime32-vnni-hot6-45w/golden_wave.f32
```

Runs were interleaved control/candidate. `tools/check_realtime_gate.py` reports:

```text
REALTIME: PASS phase=pre-service worst_RTF=1.453782->1.203178 reduction=17.24% worst_misses=25->9
```

All six runs pass finite and exact-request quality gates. Candidate quality is
cosine 0.999688334 and SNR 32.05 dB. The candidate remains pre-service because
its largest worst RTF is above 1.

Instrument-aligned aggregate samples were:

```text
control:   worst_RTF=1.646647 Bzy_MHz=1005 PkgWatt=8.08 CoreTmp=51 CoreThr=0
candidate: worst_RTF=1.457160 Bzy_MHz=1024 PkgWatt=7.94 CoreTmp=52 CoreThr=0
```

`turbostat` was used only to characterize the context. Promotion uses the six
uninstrumented logs under `raw/`.

## Artifact hashes

```text
golden_wave.f32  5a8a5bd68fcbb1df6cf91ad40b633fc2a862589fb15bfe41eb9350ee7334c5b1
fs2_acoustic     265604af77e99146af749bd7551841823a9fa5404231dd053682fc63b6712cf8
aux_convnext     62876269608502e7eb8a5c104b587033d1196b586b79df515110d6e9189ffbb0
lynxnet2         a69cb1f744e94c704a6864b1cf8bb2cd7b830dc9a61696c79c9fa0e788d5a315
32.dsv35         7797f80bbcea994a45df4bdf02d2248227a4fbd7ec184bd482e7420ffe34a0b8
vocoder ONNX     5a6d7a25edeec107423d4898151c88509e21bdc19982f15891a14bad6f8f9f9e
```

The source parent is `5bf867c921cbb75c628c0e711b71d17b60667382`. The
promotion is identified by Git tag `realtime32-vnni-hot6-45w-20260922`.
