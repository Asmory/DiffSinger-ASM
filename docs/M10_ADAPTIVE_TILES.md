# M10 shape-aware 2-D tile policy

M9 proved that the tile-local ATanGLU pipeline is useful, but the target
Core i5-13420H also proved that one global default tile is not appropriate for
all LYNXNet2 widths.

Target measurements with the automatically pinned eight P-core logical workers:

- validation shape C/H=256: the sweep favored 8x256;
- official acoustic shape C/H=1024: the sweep favored 32x64, with a final
  repeated-step measurement of 33.211 ms in that run;
- the old M9 default 16x128 was therefore not the correct production default.

## M10 policy

M10 enables adaptive tiles by default. The selector uses both the number of
preferred workers and the output width of each 2-D job.

For pools with at least eight workers:

```text
N >= 768   -> 32x64
N <= 256   -> 8x256
otherwise  -> 16x128
```

For smaller pools the current conservative fallback is 16x64. This avoids
blindly transplanting the 8-worker 13420H result to CPUs with different worker
counts.

The geometry is applied consistently to every eligible 2-D Linear,
Linear+Residual, and tile-local ATanGLU job. This is important: the M9 sweep
measured the whole forward pass, so reproducing the winning configuration means
changing all of those jobs, not only the ATan stage.

## Overrides

`ds_threadpool_set_tiles(pool, m, n)` switches the pool to a fixed tile.
`ds_threadpool_set_auto_tiles(pool, 1)` returns to adaptive mode.

Environment variables at pool creation are also supported:

```text
DSASM_AUTO_TILES=0
DSASM_M_TILE=32
DSASM_N_TILE=64
```

The benchmark tool intentionally switches modes internally to perform
same-process A/B measurements, while the normal runtime starts in adaptive mode.

## Why this is still simple

This is deliberately not a generic online GEMM autotuner. DiffSinger-ASM is a
specialized runtime for a small set of known model shapes. The policy records
measured target behavior with a safe fallback, keeps startup deterministic, and
retains explicit tuning controls for future CPUs.
