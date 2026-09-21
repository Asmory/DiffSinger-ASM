# M16 — promoted cache policy

M15 target results on the i5-13420H showed that K-blocking is a real whole-forward
win and that its best measured combination was N-owner scheduling plus Kblock=512.
M16 promotes that combination for the tuned 8-worker P-core pool.

## Default policy

For an 8-worker pool:

- N-owner scheduling defaults on, but its existing job guards require a wide job
  with enough N tiles to feed all workers. C=256 therefore still falls back to
  the M10/M9 dynamic scheduler.
- ATan K-blocking defaults on with Kblock=512, but M16 only uses it when
  M>=32, N>=512 and K>=512.
- spin dispatch, parallel depthwise and indexed Linear remain off.

All public setters and DSASM_* environment variables remain available as
explicit overrides.

## Why

M15 target A/B measured owner+Kblock=512 at 34.315 ms versus 36.885 ms for the
legacy dynamic full-K path (1.075x). The separate dynamic Kblock=512 perf run
also reduced median latency from 34.239 ms to 32.571 ms while lowering P-core
cache references. M16 adds a dedicated four-way same-process A/B and perf
comparison so the promoted combination can be confirmed directly.
