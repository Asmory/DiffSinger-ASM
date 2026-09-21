#ifndef DSASM_THREADPOOL_H
#define DSASM_THREADPOOL_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct DSAsmThreadPool DSAsmThreadPool;

/* M8: Linux-aware persistent worker pool.
   - On hybrid CPUs, logical CPUs belonging to SMT-capable cores are preferred
     for heavy AVX2/FMA work (this selects P-core siblings on i5-13420H).
   - Workers are pinned with pthread_setaffinity_np when possible.
   - threads==0 selects all preferred logical CPUs automatically. */
DSAsmThreadPool *ds_threadpool_create(size_t threads);
/* Selects a disjoint slice of the topology-aware CPU order. This is intended
   for concurrent stage pools; offset is measured in logical CPUs. */
DSAsmThreadPool *ds_threadpool_create_range(size_t threads, size_t cpu_offset);
DSAsmThreadPool *ds_threadpool_create_auto(void);
void ds_threadpool_destroy(DSAsmThreadPool *pool);
size_t ds_threadpool_threads(const DSAsmThreadPool *pool);
int ds_threadpool_cpu_at(const DSAsmThreadPool *pool, size_t worker_index);
int ds_threadpool_affinity_enabled(const DSAsmThreadPool *pool);

/* M43: parallel contiguous float Add. Intended for large same-shape vocoder
   residual merges; falls back to the normal AVX2 kernel when pool is NULL. */
void ds_threadpool_add_f32(DSAsmThreadPool *pool, const float *a, const float *b, float *y, size_t n);

/* M55: parallel channel-sliced leaky copy into padded NCT workspace. */
void ds_threadpool_leaky_copy_nct_f32(DSAsmThreadPool *pool, const float *x, float *dst,
                                      size_t C, size_t Tin, size_t Tp, size_t pad, float alpha);

/* Enable/disable M8 MxN 2-D tiling for wide Linear jobs. Enabled by default.
   Kept public so benchmarks can A/B M7-style M-only scheduling in one binary. */
void ds_threadpool_set_2d(DSAsmThreadPool *pool, int enabled);
int ds_threadpool_get_2d(const DSAsmThreadPool *pool);

/* M9 tile-local ATan pipeline. When enabled, ATan LYNXNet2 GLU layers are
   scheduled as 2-D jobs and never materialize the full [T,2C] tensor. */
void ds_threadpool_set_atan_pipeline(DSAsmThreadPool *pool, int enabled);
int ds_threadpool_get_atan_pipeline(const DSAsmThreadPool *pool);

/* Runtime tile control for same-process autotuning. n_tile must be a multiple
   of 16. The implementation currently supports m<=64 and n<=256.
   Calling ds_threadpool_set_tiles() switches to a fixed override. */
int ds_threadpool_set_tiles(DSAsmThreadPool *pool, size_t m_tile, size_t n_tile);
size_t ds_threadpool_m_tile(const DSAsmThreadPool *pool);
size_t ds_threadpool_n_tile(const DSAsmThreadPool *pool);

/* M10 shape-aware tile selection. Enabled by default. Explicit tile overrides
   disable it until re-enabled. The policy is intentionally tiny and based on
   measured DiffSinger LYNXNet2 shapes rather than a generic GEMM heuristic. */
void ds_threadpool_set_auto_tiles(DSAsmThreadPool *pool, int enabled);
int ds_threadpool_get_auto_tiles(const DSAsmThreadPool *pool);

/* M11: parallelize native k31 depthwise convolution across channel blocks.
   Disabled by default for sufficiently large LYNXNet2 channel counts. */
void ds_threadpool_set_parallel_depthwise(DSAsmThreadPool *pool, int enabled);
int ds_threadpool_get_parallel_depthwise(const DSAsmThreadPool *pool);

/* M13: use the reduced-address-uop packed16 Linear kernel on narrow N tiles.
   Disabled by default only for the N<=64 / K>=128 path where the standalone
   kernel wins; public control keeps same-process A/B possible. */
void ds_threadpool_set_indexed_linear(DSAsmThreadPool *pool, int enabled);
int ds_threadpool_get_indexed_linear(const DSAsmThreadPool *pool);

/* M14-M17 N-owner scheduling: one worker owns a narrow output-channel tile
   and walks all M rows, maximizing packed-weight reuse in private L2.
   Target M16 measurements showed that ownership reduces cache references but
   does not improve wall-clock reliably, so M17 keeps it experimental/off. */
void ds_threadpool_set_n_owner(DSAsmThreadPool *pool, int enabled);
int ds_threadpool_get_n_owner(const DSAsmThreadPool *pool);

/* M15-M18 K-blocked ATan Linear experiment. The packed16 K dimension is
   consumed in cache-sized chunks while preserving the exact FMA order. M18
   keeps it off by default after target-machine runs showed state-dependent
   wins/losses; use the setter/environment override for controlled A/B. */
void ds_threadpool_set_kblocked_atan(DSAsmThreadPool *pool, int enabled);
int ds_threadpool_get_kblocked_atan(const DSAsmThreadPool *pool);
int ds_threadpool_set_k_block(DSAsmThreadPool *pool, size_t k_block);
size_t ds_threadpool_k_block(const DSAsmThreadPool *pool);

#ifdef __cplusplus
}
#endif
#endif
