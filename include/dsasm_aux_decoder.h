#ifndef DSASM_AUX_DECODER_H
#define DSASM_AUX_DECODER_H
#include <stddef.h>
#include <stdint.h>
#include "dsasm_threadpool.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const float *dw_weight_tap_major; /* [7,C] */
    const float *dw_bias;             /* [C] */
    const float *ln_gamma;            /* [C] */
    const float *ln_beta;             /* [C] */
    const float *pw1_weight_m4n16;    /* packed16 [4C,C] */
    const float *pw1_bias;             /* [4C] */
    const float *pw2_weight_m4n16;    /* packed16 [C,4C] */
    const float *pw2_bias;             /* [C] */
    const float *gamma;                /* [C], layer scale */
} DSAsmAuxConvNeXtBlock;

typedef struct {
    uint32_t input_dim;
    uint32_t channels;
    uint32_t output_dim;
    uint32_t num_layers;
    uint32_t kernel_size;              /* current upstream: 7 */
    const float *in_weight_m4n16;      /* packed16 [C,7*input_dim] */
    const float *in_bias;              /* [C] */
    const DSAsmAuxConvNeXtBlock *blocks;
    const float *out_weight_m4n16;     /* packed16 [D,7*C] */
    const float *out_bias;             /* [D] */
} DSAsmAuxConvNeXtWeights;

size_t ds_aux_convnext_workspace_floats(const DSAsmAuxConvNeXtWeights *w,size_t frames);

/* Decoder output in normalized mel domain, matching ConvNeXtDecoder.forward(). */
int ds_aux_convnext_forward_norm_f32_avx2(
    const DSAsmAuxConvNeXtWeights *w,const float *condition_tc,float *out_tc,
    float *workspace,size_t frames,DSAsmThreadPool *pool);

/* AuxDecoderAdaptor(infer=True): decoder + spec denormalization. */
int ds_aux_convnext_infer_f32_avx2(
    const DSAsmAuxConvNeXtWeights *w,const float *condition_tc,
    const float *spec_min,const float *spec_max,size_t range_dims,float *out_mel_tc,
    float *workspace,size_t frames,DSAsmThreadPool *pool);

#ifdef __cplusplus
}
#endif
#endif
