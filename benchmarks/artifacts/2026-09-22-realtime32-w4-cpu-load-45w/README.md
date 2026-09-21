# Real-time 32-frame four-worker CPU-load promotion

The 34-op six-worker milestone established stable service, so this iteration
changes the primary objective from worst RTF to process CPU-RTF. Kernel math,
model artifacts, exact golden, package-power context, and request shape remain
unchanged; only the persistent engine worker count changes from six to four.

An E2E screen measured:

```text
workers 6: CPU-RTF=3.152764 worst_RTF=1.218310 misses=8
workers 5: CPU-RTF=2.707927 worst_RTF=0.994188 misses=0
workers 4: CPU-RTF=2.292162 worst_RTF=0.943903 misses=0
```

Four workers retained more service margin per occupied core, so it entered the
three-run gate. The formal comparison uses the archived service champion runs,
because a paired six-worker control suffered a repeatable platform-frequency
outlier while the historical champion is the declared optimization baseline.
The paired controls are retained under `raw/` as context evidence.

`tools/check_realtime_gate.py` reports:

```text
REALTIME: PASS phase=cpu-load worst_cpu_RTF=2.957324->2.391249 reduction=19.14% service=PASS
```

All three four-worker runs preserve worst RTF below 1, zero deadline misses,
finite output, and exact-request quality. Their maximum worst RTF is 0.996841.
The output checksum and quality are unchanged because worker scheduling does not
change neural arithmetic.

The exact golden SHA-256 remains:

```text
5a8a5bd68fcbb1df6cf91ad40b633fc2a862589fb15bfe41eb9350ee7334c5b1
```

The promotion is identified by Git tag
`realtime32-w4-cpu-load-45w-20260922`.
