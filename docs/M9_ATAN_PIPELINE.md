# M9 tile-local ATanGLU pipeline

M9 keeps the M8 P-core-aware persistent pool and 2-D Linear scheduler, but
changes the two ATanGLU feed-forward projections inside each LYNXNet2 block.

## M8 path

For an ATanGLU layer with final width N, M8 did:

1. 2-D Linear `[M,K] x [2N,K] -> [M,2N]`
2. global worker barrier
3. single ATanGLU pass `[M,2N] -> [M,N]`

The full `[M,2N]` activation was written to and read from memory for every GLU.

## M9 path

M9 schedules paired `(M_tile,N_tile)` jobs. Every worker owns a 64-byte aligned
private scratch tile. A job does:

1. Linear left tile -> worker scratch
2. Linear gate tile -> the adjacent half of the same scratch rows
3. AVX2 ATanGLU immediately -> final `[M,N]` output tile
4. pull the next task from the atomic work queue

There is no global Linear-to-ATan barrier and no full `[M,2N]` intermediate.
Only a compact tile-local `[M_tile,2*N_tile]` buffer exists per worker.

The new assembly entry point is:

`ds_atan_glu_f32_avx2_ystrided`

It consumes compact `[left|gate]` scratch rows but writes directly to a slice of
a wider output tensor using an independent output row stride.

## Dynamic scheduling

M8 already used an atomic queue for 2-D Linear jobs. M9 extends that queue to
ATanGLU projection jobs. This matters on SMT P-cores because sibling logical
threads do not have equal effective throughput at every moment; workers that
finish early simply claim the next tile instead of waiting for a statically
assigned range.

## Tile controls

Defaults remain:

- M tile: 16
- N tile: 128

For target-specific tuning:

```bash
make m9-autotune
```

The validator sweeps:

- M = 8, 16, 32
- N = 64, 128, 256

The same settings can be forced without rebuilding:

```bash
DSASM_M_TILE=16 DSASM_N_TILE=128 make m9-bench
```

The public runtime API also provides `ds_threadpool_set_tiles()` so an embedding
application can tune within one process.

## A/B controls

M9 exposes `ds_threadpool_set_atan_pipeline()` and the environment switch:

```bash
DSASM_ATAN_PIPELINE=0
```

The benchmark intentionally measures three modes in the same process:

- M7-style M-only
- M8 2-D Linear
- M9 2-D + tile-local ATan pipeline

A candidate is only useful when it beats M8 on the target machine.
