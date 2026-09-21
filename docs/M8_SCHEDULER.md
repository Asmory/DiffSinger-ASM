# M8 P-core-aware 2-D scheduler

M8 changes only execution scheduling and strided Linear entry points. The
`DSLYNX7` weight format is unchanged and remains fully compatible.

## Why M-only splitting was insufficient

M7 split a Linear only across rows (`M`, the frame/time dimension). The current
DiffSinger LYNXNet2 acoustic backbone commonly runs with `T=64`, while its
Linear outputs are 1024 or 2048 channels. With eight workers, M-only scheduling
leaves each worker just eight rows and offers no load balancing in the wide
output dimension.

M8 represents a wide Linear as an M x N tile grid. The default tile is 16 rows
by 128 output channels. Examples at T=64:

- N=1024 -> 4 x 8 = 32 tasks
- N=2048 -> 4 x 16 = 64 tasks

Workers pull tasks from one atomic queue. N-major ordering makes adjacent tasks
reuse the same packed weight tile across different row ranges while it is hot
in the shared cache.

## Strided assembly kernels

Two new AVX2/FMA assembly entry points separate tile width from full row stride:

- `ds_linear_f32_avx2_m4n16_strided`
- `ds_linear_residual_f32_avx2_m4n16_strided`

The packed16 format remains `[N/16][K][16]`. A scheduler can start at output
column `n0` by advancing the packed weight pointer by
`(n0/16) * K * 16` floats, while the assembly kernel stores each row using the
full output stride.

## Hybrid Intel affinity

Linux does not always expose `/sys/.../topology/core_type`. M8 therefore starts
from the caller's existing affinity mask and groups logical CPUs by physical
package/core ID.

If both SMT-capable core groups and single-thread core groups exist, the
SMT-capable groups are preferred for the heavy AVX2 pool. This matches the
13th-gen Core i5-13420H topology used for this project: four SMT P-cores and
four single-thread E-cores.

Worker selection is physical-core-first, then SMT siblings. For that CPU an
auto eight-worker pool resolves to the logical equivalent of:

`0,2,4,6,1,3,5,7`

The worker itself is pinned with `pthread_setaffinity_np`. Existing `taskset`
restrictions are respected because discovery only considers CPUs in the
process's allowed mask.

Environment switches:

- `DSASM_AFFINITY=0` disables worker pinning.
- `DSASM_2D=0` disables 2-D scheduling and restores M7-style M-only splitting.

The public `ds_threadpool_set_2d()` function is also available for same-process
A/B benchmarks.

## Current measured scheduler result

On the development host, official-shape `T=64, I=128, Q=384, C=1024, H=1024,
L=6, ATanGLU` showed, in the same process and same persistent pool:

- M7-style M-only: ~25.32 ms
- M8 2-D, 16x128 tasks: ~19.62 ms
- scheduler-only improvement: ~1.29x
- serial: ~52.08 ms, making the M8 pool ~2.65x faster on that host

Absolute numbers are host dependent. The important benchmark on the target
Core i5-13420H is `make m8-bench`, which prints selected CPUs and reports M-only
and M8 2-D medians in one run.
