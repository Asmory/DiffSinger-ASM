#ifndef DSASM_LYNXNET2_H
#define DSASM_LYNXNET2_H

#include <stddef.h>
#include <stdint.h>
#include "dsasm_threadpool.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
    DSASM_GLU_ATAN = 1,
    DSASM_GLU_SOFTSIGN = 2,
    DSASM_GLU_SILU = 3,
};

typedef struct {
    const float *ln_gamma;                 /* [C] */
    const float *ln_beta;                  /* [C] */
    const float *dw_weight_tap_major;      /* [31,C] */
    const float *dw_bias;                  /* [C] */
    const float *glu1_weight;              /* ATAN/SILU: packed16 [2H,C], SOFTSIGN: packed M4N8 */
    const float *glu1_bias;                /* [2H], left then gate */
    const float *glu2_weight;              /* ATAN/SILU: packed16 [2H,H], SOFTSIGN: packed M4N8 */
    const float *glu2_bias;                /* [2H], left then gate */
    const float *out_weight_m4n16;         /* packed16 [C,H] */
    const float *out_bias;                 /* [C] */
} DSAsmLynxNet2Block;

typedef struct {
    uint32_t input_dim;
    uint32_t condition_dim;
    uint32_t channels;
    uint32_t hidden_dim;
    uint32_t num_layers;
    uint32_t kernel_size;
    uint32_t glu_type;

    const float *input_weight_m4n16;       /* [C,input_dim] */
    const float *input_bias;               /* [C] */
    const float *condition_weight_m4n16;   /* [C,condition_dim] */
    const float *condition_bias;           /* [C] */
    const float *time1_weight_m4n16;       /* [4C,C] */
    const float *time1_bias;               /* [4C] */
    const float *time2_weight_m4n16;       /* [C,4C] */
    const float *time2_bias;               /* [C] */

    const DSAsmLynxNet2Block *blocks;      /* [num_layers] */

    const float *post_norm_gamma;          /* [C] */
    const float *post_norm_beta;           /* [C] */
    const float *output_weight_m4n16;      /* [input_dim,C] */
    const float *output_bias;              /* [input_dim] */
} DSAsmLynxNet2Weights;

/* Scratch storage required by the forward functions, in float elements. */
size_t ds_lynxnet2_workspace_floats(const DSAsmLynxNet2Weights *w, size_t frames);

/* Project conditioner once and reuse it for all diffusion/reflow solver steps. */
int ds_lynxnet2_prepare_condition_f32_avx2(
    const DSAsmLynxNet2Weights *w,
    const float *condition_tc,
    float *condition_projected_tc,
    size_t frames);

/* Full forward, including conditioner projection. */
int ds_lynxnet2_forward_f32_avx2(
    const DSAsmLynxNet2Weights *w,
    const float *spec_tc,
    const float *condition_tc,
    float timestep,
    float *output_tc,
    float *workspace,
    size_t frames);

/* Faster repeated-sampling entry point using a preprojected conditioner [T,C]. */
int ds_lynxnet2_forward_cached_condition_f32_avx2(
    const DSAsmLynxNet2Weights *w,
    const float *spec_tc,
    const float *condition_projected_tc,
    float timestep,
    float *output_tc,
    float *workspace,
    size_t frames);

/* Persistent-thread-pool variants. Only GEMM-dominated stages are dispatched;
   small LayerNorm/depthwise/activation stages stay serial to avoid wakeup cost. */
int ds_lynxnet2_prepare_condition_parallel_f32_avx2(
    const DSAsmLynxNet2Weights *w, const float *condition_tc,
    float *condition_projected_tc, size_t frames, DSAsmThreadPool *pool);
int ds_lynxnet2_forward_cached_condition_parallel_f32_avx2(
    const DSAsmLynxNet2Weights *w, const float *spec_tc,
    const float *condition_projected_tc, float timestep, float *output_tc,
    float *workspace, size_t frames, DSAsmThreadPool *pool);
int ds_lynxnet2_forward_parallel_f32_avx2(
    const DSAsmLynxNet2Weights *w, const float *spec_tc, const float *condition_tc,
    float timestep, float *output_tc, float *workspace, size_t frames,
    DSAsmThreadPool *pool);

#ifdef __cplusplus
}
#endif
#endif
