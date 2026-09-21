# Real-time 32-frame VNNI hot-twelve promotion

This is the second 5% pre-service step for the 45 W, six-worker, 32-frame
streaming context. It preserves the hot-six champion and adds the profiled
`Cin256/K11` group (`48,49,51,52,54,55`).

The workload, machine, warm-up, package-power context, model files, exact ONNX
golden, and quality thresholds are unchanged from the predecessor artifact
`2026-09-22-realtime32-vnni-hot6-45w`.

## Instrument-selected direction

The promoted hot-six E2E stage profile measured acoustic at 43.0%, vocoder at
56.9%, and other work near zero. The standalone vocoder profile identified the
six `Cin256/K11` FP32 convolutions as the largest controllable unquantized group.
Three interleaved `perf stat` pairs over 15 vocoder rounds measured:

```text
control instructions:   33,186,855,168  33,284,529,547  33,298,923,516
candidate instructions: 29,601,876,885  29,683,329,374  29,671,180,940
control cycles:         14,288,327,662  14,330,624,172  14,301,424,147
candidate cycles:       12,447,481,838  12,988,010,664  12,797,465,024
```

This reduced P-core instructions and cycles by about 10.7% and 10.6%,
respectively, before the candidate entered E2E testing.

## E2E gate

The command is the predecessor command with the candidate override:

```text
DSASM_VNNI_EXTRA_OPS=79,77,80,76,74,73,48,49,51,52,54,55
```

`tools/check_realtime_gate.py` reports for the interleaved reproduction:

```text
REALTIME: PASS phase=pre-service worst_RTF=1.352409->1.001437 reduction=25.95% worst_misses=18->1
```

The archived predecessor maximum was 1.203178, so the candidate also clears the
historical-champion comparison by 16.77%. Candidate quality passes at cosine
0.999465810 and SNR 29.71 dB. One run reached worst RTF 1.001437 with one miss,
so this milestone remains pre-service rather than switching to the CPU-load
phase.

The exact golden hash remains:

```text
5a8a5bd68fcbb1df6cf91ad40b633fc2a862589fb15bfe41eb9350ee7334c5b1
```

The promotion is identified by Git tag
`realtime32-vnni-hot12-45w-20260922`; raw E2E logs, hardware-counter samples,
and the selecting vocoder profile are stored under `raw/`.
