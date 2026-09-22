# Batch384 K11 VNNI plus paired ConvTranspose promotion

This artifact establishes the first fixed-block 384-frame batch champion. It
does not replace the real-time streaming champion or reinterpret the historical
M55/M58 long-audio measurements.

## Contract

- CPU: Intel Core i5-13420H, 4 P-cores with SMT plus 4 E-cores
- runtime workers: 8, topology-aware P-core/SMT order
- workload: 384 frames, 7 measured regions, 2 warm-up regions, 4 acoustic steps
- primary objective: median aggregate E2E wall time across five paired runs
- gate: at least 5% improvement; p90, worst, and CV guards; finite and quality pass
- power: PL1 45,000,000 uW, window 55,967,744 us; PL2 80,000,000 uW
- policy: platform `performance`, intel_pstate EPP `performance`
- kernel: Linux 7.1.11-arch1-1; GCC 16.2.1; binutils 2.47
- source parent: `1eceb4f41de1704e802f5f2ca0febfe5e07483e4`
- release tag: `batch384-k11-convt-oc2-45w-20260922`

The control sets `DSASM_VNNI_EXTRA_OPS=off` and
`DSASM_CONVT_S2K4_OC2=0`. The candidate enables VNNI only for quality-gated
Cin64/K11 operators 92 through 96 and pairs adjacent output channels in the
stride-2/K4 ConvTranspose ASM kernel. The promoted defaults apply only to the
exact 384-frame graph shapes; both controls remain available for A/B testing.

The common benchmark command was:

```sh
./build/bench_engine_stream \
  build/stream_acoustic build/stream_buckets/current \
  build/stream_acoustic/dongfangzhizi-nectar-xiao.emb \
  --frames 384 --regions 7 --warmup 2 --workers 8 --bucket 384 --steps 4 \
  --golden benchmarks/artifacts/2026-09-22-batch384-k11-convt-oc2-45w/golden_wave.f32
```

Runs used the order `B C C B B C C B B C`. The formal gate reports:

```text
batch384: PASS gain=8.02% baseline(median=23797.849,p90=25595.058,worst=25817.030,cv=0.056) candidate(median=22031.032,p90=24730.666,worst=25563.829,cv=0.070)
```

Median aggregate RTF changed from `0.762565` to `0.705950`. All ten runs
passed finite and exact-request waveform quality. Candidate quality was cosine
`0.999033427` and SNR `27.13 dB`. The paired ConvTranspose path is bit-exact
with the previous FP32 ASM kernel; only the selected VNNI operators change the
waveform.

Stage profiling measured about 35% acoustic and 65% vocoder time. Shape
profiling then identified Cin64 K11 convolutions at roughly 50-54 ms per op and
ConvTranspose at 269 ms total. `perf record` independently sampled the s2/s8
ConvTranspose kernels on P-core cycles. Profiled runs locate work only; the ten
uninstrumented runs under `raw/` decide promotion.

## Artifact hashes

```text
golden_wave.f32  a040c6ab5f06f0b4113981a9ece5fb97158f1b2142f60967831030c8585fc793
fs2_acoustic     265604af77e99146af749bd7551841823a9fa5404231dd053682fc63b6712cf8
aux_convnext     62876269608502e7eb8a5c104b587033d1196b586b79df515110d6e9189ffbb0
lynxnet2         a69cb1f744e94c704a6864b1cf8bb2cd7b830dc9a61696c79c9fa0e788d5a315
384.dsv35        2749a27964486ee7ba2264ab9d8e9abcd03d13507825e2663e1a6f4cc0b890e5
vocoder ONNX     5a6d7a25edeec107423d4898151c88509e21bdc19982f15891a14bad6f8f9f9e
```
