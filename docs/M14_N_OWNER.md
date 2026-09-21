# M14 N-tile ownership

## Why

M13 target perf showed that reducing packed16 Linear address-generation work cut
P-core instruction count substantially but did not reduce cycles. The indexed
path therefore exposed the real bottleneck: data/execution stalls rather than
front-end integer instruction count.

The existing 32x64 dynamic schedule represents the official `M=64,N=1024` job as
two M tasks per N tile. Those two tasks may be consumed by different P cores,
causing the same packed `N=64` weight region to enter multiple private caches.

## Scheduler

N-owner represents the job as only N tasks. A worker atomically claims one N tile
and processes M in chunks of at most 64 rows while retaining ownership of the
same weights. For official DiffSinger LYNXNet2:

- normal Linear N64/K1024 weight slice: 256 KiB,
- ATan left+gate N64/K1024 pair: 512 KiB,
- 16 N tiles for C=1024, enough for the 8 preferred logical P workers.

Owner mode is gated by job size and by `N_tiles >= worker_count`; it therefore
falls back automatically when channel width is too small to provide parallelism.

## Safety

The mathematical kernels and packed formats are unchanged. Only task ownership
changes. M14 benchmark candidates must match bit-for-bit before measurements are
accepted.

## Status

Experimental, default off. Promote only if the i5-13420H target demonstrates a
repeatable whole-forward and perf-counter win.
