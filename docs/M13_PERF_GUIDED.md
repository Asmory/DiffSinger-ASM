# M13 perf-guided experiments

M12 target-machine `perf stat` on the i5-13420H showed the official cached-condition forward is no longer dominated by scheduler migration or branch pathology:

- P-core aggregate IPC was about 2.20 (`67.52B instructions / 30.65B cycles`).
- Branch misses were about 1.2% of branches.
- `context-switches=0` and `cpu-migrations=0` under the pinned pool.
- Generic cache misses were high enough to justify testing weight/locality changes, while noting that generic hybrid-Intel cache events are only a coarse signal.

M13 therefore keeps M12's safe defaults (condvar dispatch, serial depthwise, P-core-preferred pool) and adds two target-machine experiments instead of guessing:

1. **Reduced-address-uop packed16 Linear**
   - New `m4n16_idxstrided` and residual variants.
   - Four rows share one K byte offset instead of incrementing four row pointers every K iteration.
   - FMA order and packed-weight ABI are unchanged; regression tests require bit-exact output.
   - The runtime contains a selector for `N_tile <= 64 && K >= 128`, but the indexed kernel is **disabled by default** in M13 because development-host results were not stable enough to promote it. `DSASM_INDEXED_LINEAR=1` or the public setter enables the experiment; wider tiles always retain the M12 kernel.

2. **Weight-ownership / hybrid-worker matrix**
   - Maximum M tile is extended from 32 to 64.
   - `make m13-bench` interleaves `16x64`, `32x64`, and `64x64`, with/without the indexed kernel.
   - It runs once with `requested=8` (P-core logical pool on the target) and once with `requested=12` (P+E on the target). The existing atomic 2-D queue automatically gives faster workers more tasks.
   - No E-core path is enabled by default until the target benchmark demonstrates a win.

Rejected during M13 development:

- K-loop x4 unrolling: bit-exact but slower locally.
- 5x16 row-reuse kernel: bit-exact but inconsistent/slower on larger cases.

The point of M13 is to make the target i5-13420H decide the next default with same-process, rotation-interleaved measurements rather than extrapolating from the development VM. Until that target result is known, production defaults remain the proven M12/M10 path: condvar dispatch, serial depthwise, adaptive tiles, and the original packed16 4x16 strided kernel.
