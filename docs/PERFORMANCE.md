# Performance policy and M55/M58 fusion record

## Release rule

Performance changes are measured, not inferred from source shape. A candidate
is promoted only when all of these conditions hold on the target workload:

- output parity passes before timing is considered;
- at least five interleaved or paired samples are collected after warmup;
- candidate median latency improves by at least 3% over the current baseline;
- candidate p90 and worst latency do not regress;
- candidate coefficient of variation (CV) is at most 0.10, or no more than
  0.01 above the paired baseline CV when package power/frequency drifts both;
- a kernel result is labelled as a kernel result. It does not promote the E2E
  release until the 384-frame persistent E2E gate independently passes.

The fastest single sample is diagnostic only. It is never a release gate.

The machine should be otherwise idle. Pin the benchmark to the four P-cores
and their SMT siblings (`0-7` on the measured i5-13420H). Record frequency,
thermal and power state when comparing results across sessions.

## Source inputs

The repository was reconstructed from the M55 full source archive and then the
M55-to-M58 combined patch was applied.

```text
M55 source: 931b368d1292e466f7e34fbef472356098954e9bca2037acfbe23dc622a7aad7
M55-M58 patch bundle: 8fd053d67fdffe2dfaea192b293d1112d354d403a0b3a50cbecf2c1e07e63998
M55 baseline commit: 010d0bc
```

The fusion keeps the M55 long-audio execution policy (`P8SMT`, eight workers,
parallel Add and leaky-copy, tile 2016, range/residual t24, VNNI K11 at Cin128)
and adds M58 `all3711` graph fusion plus K7/K11 t24 range-residual kernels.
`DSASM_RANGE_RESIDUAL_T24` is independent from `DSASM_RANGE_T24`; the imported
M58 K11 dispatch incorrectly coupled them and was fixed during integration.

## Instrumented kernel result

Command:

```bash
M58_ROUNDS=7 bash scripts/run_m58_micro.sh
```

The script pins one P-core, uses AB/BA ordering inside every process, checks
bit-exact output, applies `tools/check_perf_gate.py`, and records `perf stat`
P-core cycles, instructions and cache misses. Results from 2026-09-21:

| Shape | Baseline median | Candidate median | Gain | p90 | Worst | Result |
| --- | ---: | ---: | ---: | --- | --- | --- |
| K7 C128 T24576 | 6.940 ms | 6.470 ms | 7.26% | improved | improved | PASS |
| K11 C128 T24576 | 11.385 ms | 10.867 ms | 4.77% | improved | improved | PASS |
| K7 C32 T98304 | 6.963 ms | 5.875 ms | 18.52% | improved | improved | PASS |
| K11 C32 T98304 | 11.050 ms | 9.279 ms | 19.09% | improved | improved | PASS |

All four shapes had `max_abs=0` and `bitdiff=0`. The instrumented whole-test
run recorded 1,432,397,012 P-core cycles, 5,329,507,943 instructions and
9,166,766 cache misses. These counters cover baseline, candidate and parity
work together; they prove the measurement ran on P-core PMU, but are not used
to attribute the speedup between the two kernels.

Raw generated data lives under `build/m58/` and is intentionally not tracked.

## E2E gate still required

The supplied archives contain neither the real voicebank nor the prepared
384-frame M53 fixture. Therefore this workspace cannot honestly claim a new
E2E release result yet. M55 remains the verified E2E baseline (`median RTF
0.779`), while the fused tree is a kernel-qualified candidate.

After placing the voicebank at `$MODEL` and preparing the M53 fixture, run the
direct historical-best comparison:

```bash
python scripts/run_m58_vs_m55_e2e.py
```

It compares only the M53/M55 `k7ge128` bundle against a newly packed M58
`all3711` bundle, using repeated ABBA order, identical M55 runtime settings and
the same P8SMT affinity. It writes the samples and applies the same gate:

```bash
python tools/check_perf_gate.py e2e-samples.tsv --minimum 3
```

Only an E2E PASS should move the release baseline from M55 to the fused tree.

## Reproduction checks

```bash
make m40-1-check m38-check m58-check
make m58-bench
```

Production binaries remain pure C/x86-64 assembly plus libc, libm and pthread;
ONNX Runtime remains an offline pack/golden dependency.
