# Real-time 32-frame service-target promotion

This third pre-service step is the first 32-frame CPU milestone where every run
has worst RTF below 1 and zero deadline misses. It promotes a 34-op selective
VNNI set and switches subsequent optimization to the CPU-load objective.

## Candidate selection

The hot-twelve champion's measured vocoder profile selected `Cin256/K7` and
`Cin64/K7`. Combined hardware-counter probes reduced P-core instructions by
about 15% and cycles by about 16% while passing exact-request quality.

The next `Cin32/K7/K11` group reduced work further but its complete form failed
the cosine gate at 0.998751068. Per-op golden probes identified ops 139 and 155
as the largest error contributors. Keeping those two in FP32 and quantizing the
other 10 produced cosine 0.999291538 and SNR 28.49 dB.

The promoted override, retained for reproduction, is:

```text
DSASM_VNNI_EXTRA_OPS=79,77,80,76,74,73,48,49,51,52,54,55,38,39,41,42,44,45,106,107,109,110,112,113,140,142,143,145,146,149,150,152,153,156
```

## E2E result

The standard 10-warm-up/25-measured-region command was run for three interleaved
control/candidate pairs. Candidate results were:

```text
worst RTF:       0.947715  0.916095  0.921636
CPU-RTF:         2.957324  2.837905  2.874861
deadline misses: 0         0         0
```

The candidate maximum worst RTF is 5.37% below the archived hot-twelve champion
maximum of 1.001437, satisfying the 5% staircase independently of the slower
paired-control outlier. All finite and quality gates pass.

The primary objective now becomes maximum CPU-RTF. The new champion value is
2.957324; future candidates must reduce it by at least 5%, while all baseline
and candidate runs preserve worst RTF below 1 and zero misses.

The exact ONNX-vocoder golden SHA-256 remains:

```text
5a8a5bd68fcbb1df6cf91ad40b633fc2a862589fb15bfe41eb9350ee7334c5b1
```

Raw E2E logs and the counter/quality evidence that selected the final subset are
under `raw/`. The promotion is identified by Git tag
`realtime32-vnni-hot34-service-45w-20260922`.
