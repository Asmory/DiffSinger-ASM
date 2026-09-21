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

Baselines are tracked independently in `benchmarks/baselines.json`:

- the E2E baseline is the only release objective and requires a stable 3%
  improvement by itself;
- when E2E passes, subsystem baselines are informational and cannot veto it;
- only when E2E fails do measured subsystem weights select the next change;
- a subsystem change must make the weighted subsystem total pass before it is
  sent back to the E2E gate, but it never promotes the release on its own;
- a weighted subsystem decision uses measured parent-profile time share,
  statistical confidence and controllability. Missing weights block aggregation
  instead of being guessed;
- parity is a hard veto. E2E regression is also a hard veto regardless of
  subsystem score;
- an observed distribution shift may create a context baseline without a
  causal claim. Its machine/configuration fingerprint and sample distribution
  must be recorded so unlike populations are not silently mixed.

Weighted TSV rows add a fourth field. The weight must come from a measured
parent profile, with confidence/controllability adjustments recorded alongside
the raw profile:

```text
case<TAB>baseline|candidate<TAB>milliseconds<TAB>effective_weight
```

Apply it with `tools/check_perf_gate.py samples.tsv --weighted`. The tool uses
weighted latency ratios, not a count of winning cases.

The machine should be otherwise idle. Pin the benchmark to the four P-cores
and their SMT siblings (`0-7` on the measured i5-13420H). Record frequency,
thermal and power state when comparing results across sessions.

For microbenchmarks, `i5-13420h-performance-pcore0` is the current context
baseline. The runner discards two full-process warmups and writes
`build/k3_range/environment.txt`. Context baselines are empirical strata, not
claims that a particular recorded variable caused the distribution shift.

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

## E2E reproduction

With the voicebank at `$MODEL` and the M53 384-frame fixture prepared, run the
direct historical-best comparison:

```bash
python scripts/run_m58_vs_m55_e2e.py
```

It compares only the M53/M55 `k7ge128` bundle against a newly packed fused
candidate, using persistent warm requests, ABBA process order and the same
P8SMT affinity. M55 retains its historical symmetric K11 VNNI settings; the
candidate enables the independently quality-gated asymmetric K7/K11 path at
Cin128. The runner writes samples and applies the same gate automatically.

```bash
python tools/check_perf_gate.py e2e-samples.tsv --minimum 3
```

Only an E2E PASS moves the release baseline.

## 2026-09-21 persistent E2E promotion

The supplied DongFangZhiZi archive was loaded end to end. Its SHA-256 is
`c171db642e6b26d6802165136f7cf91f90b42458527a7b911ff9ff0bec6c8e18`. The acoustic importer
packed FS2, AUX and RF with feature flags `0x1ff`; native acoustic output was
bit-exact with the prepared fixture. The 384-frame vocoder waveform passed at
cosine `0.999244295` and SNR `28.20 dB` for the promoted candidate.

The first cold-process ABBA attempt failed (`-2.72%`) and exposed a measurement
problem: every sample paid process/load effects and mixed them with one acoustic
and vocoder request. The gate was changed, not the threshold. The final runner
loads each graph once, performs two full E2E warmups, collects five persistent
requests, and uses process order M55/candidate/candidate/M55. An initial
symmetric-K11 candidate appeared 3.22% faster once, but a clean rerun measured
`-0.00%` with p90 regression. It was not promoted.

Subsystem profiling then measured about 5.2% vocoder improvement from fusion,
which should have been enough after weighting by the observed E2E time share,
but acoustic/frequency variation hid that gain in E2E. Shape profiling found
Cin128/K7 convolution as a remaining measured hotspot. Symmetric K7 VNNI was
fast but failed quality (cosine `0.998402204`, SNR `24.96 dB`); asymmetric K7
VNNI passed quality and made the complete E2E gate pass. It is therefore a new
subsystem baseline, based on the E2E result rather than a causal claim.

Final ten-sample-per-side result:

| Metric | M55 baseline | Candidate | Result |
| --- | ---: | ---: | --- |
| Median total | 4516.780 ms | 4169.898 ms | 8.32% faster |
| Median RTF | 1.013 | 0.935 | improved |
| p90 total | 4980.026 ms | 4334.976 ms | improved |
| Worst total | 5036.183 ms | 4650.379 ms | improved |
| CV | 0.050 | 0.045 | improved |

This passes the E2E promotion rule. Subsystem results are informational for this
iteration. If a later E2E candidate fails, measured subsystem weights are used
to guide changes. If their weighted prediction passes while E2E still fails,
profiling targets the unmodelled residual; a factor becomes a new subsystem
baseline only after changing it helps the E2E gate pass.

Tracked raw samples and profiler evidence are under
`benchmarks/artifacts/2026-09-21-e2e-k117-asym/`.

## Reproduction checks

```bash
make m40-1-check m38-check m58-check
make m58-bench
```

Production binaries remain pure C/x86-64 assembly plus libc, libm and pthread;
ONNX Runtime remains an offline pack/golden dependency.
