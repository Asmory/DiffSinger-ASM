#ifndef DSASM_FULL_ACOUSTIC_H
#define DSASM_FULL_ACOUSTIC_H
#include <stddef.h>
#include <stdint.h>
#include "dsasm_fs2_encoder.h"
#include "dsasm_aux_decoder.h"
#include "dsasm_acoustic.h"
#ifdef __cplusplus
extern "C" {
#endif
size_t ds_full_acoustic_workspace_floats(
    const DSAsmFS2AcousticWeights *fs2,const DSAsmAuxConvNeXtWeights *aux,
    const DSAsmLynxNet2Weights *rf,size_t text_tokens,size_t mel_frames);

/* Current-default B=1 acoustic inference chain after preprocessing.
   mel2ph is 1-based with 0 padding, matching DiffSinger. Caller supplies noise
   so tests and frontends can control RNG/seed exactly. */
int ds_full_acoustic_infer_f32_avx2(
    const DSAsmFS2AcousticWeights *fs2,const DSAsmAuxConvNeXtWeights *aux,
    const DSAsmLynxNet2Weights *rf,
    const int32_t *token_ids,size_t text_tokens,
    const int32_t *mel2ph,const float *f0,size_t mel_frames,
    const float *noise_tc,const float *spec_min,const float *spec_max,size_t range_dims,
    float t_start,float time_scale_factor,size_t steps,
    float *output_mel_tc,float *workspace,DSAsmThreadPool *pool);
/* M22 full-chain variant using the normalized-aux fast path. */
size_t ds_full_acoustic_normfast_workspace_floats(
    const DSAsmFS2AcousticWeights *fs2,const DSAsmAuxConvNeXtWeights *aux,
    const DSAsmLynxNet2Weights *rf,size_t text_tokens,size_t mel_frames);
int ds_full_acoustic_infer_normfast_f32_avx2(
    const DSAsmFS2AcousticWeights *fs2,const DSAsmAuxConvNeXtWeights *aux,
    const DSAsmLynxNet2Weights *rf,
    const int32_t *token_ids,size_t text_tokens,
    const int32_t *mel2ph,const float *f0,size_t mel_frames,
    const float *noise_tc,const float *spec_min,const float *spec_max,size_t range_dims,
    float t_start,float time_scale_factor,size_t steps,
    float *output_mel_tc,float *workspace,DSAsmThreadPool *pool);

int ds_full_acoustic_infer_deploy_normfast_f32_avx2(
    const DSAsmFS2AcousticWeights *fs2,const DSAsmFS2DeploymentExtras *extras,
    const DSAsmFS2DeploymentInputs *inputs,
    const DSAsmAuxConvNeXtWeights *aux,const DSAsmLynxNet2Weights *rf,
    const int32_t *token_ids,size_t text_tokens,
    const int32_t *mel2ph,const float *f0,size_t mel_frames,
    const float *noise_tc,const float *spec_min,const float *spec_max,size_t range_dims,
    float t_start,float time_scale_factor,size_t steps,
    float *output_mel_tc,float *workspace,DSAsmThreadPool *pool);
int ds_full_acoustic_infer_normfast_cancel_f32_avx2(
    const DSAsmFS2AcousticWeights *fs2,const DSAsmAuxConvNeXtWeights *aux,
    const DSAsmLynxNet2Weights *rf,
    const int32_t *token_ids,size_t text_tokens,
    const int32_t *mel2ph,const float *f0,size_t mel_frames,
    const float *noise_tc,const float *spec_min,const float *spec_max,size_t range_dims,
    float t_start,float time_scale_factor,size_t steps,
    float *output_mel_tc,float *workspace,DSAsmThreadPool *pool,
    DSAsmCancelCheck cancel_check,void *cancel_userdata);
int ds_full_acoustic_infer_deploy_normfast_cancel_f32_avx2(
    const DSAsmFS2AcousticWeights *fs2,const DSAsmFS2DeploymentExtras *extras,
    const DSAsmFS2DeploymentInputs *inputs,
    const DSAsmAuxConvNeXtWeights *aux,const DSAsmLynxNet2Weights *rf,
    const int32_t *token_ids,size_t text_tokens,
    const int32_t *mel2ph,const float *f0,size_t mel_frames,
    const float *noise_tc,const float *spec_min,const float *spec_max,size_t range_dims,
    float t_start,float time_scale_factor,size_t steps,
    float *output_mel_tc,float *workspace,DSAsmThreadPool *pool,
    DSAsmCancelCheck cancel_check,void *cancel_userdata);

#ifdef __cplusplus
}
#endif
#endif
