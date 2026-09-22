#ifndef DSASM_ACOUSTIC_H
#define DSASM_ACOUSTIC_H

#include <stddef.h>
#include "dsasm_reflow.h"
#include "dsasm_aux_decoder.h"

#ifdef __cplusplus
extern "C" {
#endif

/* M19: native acoustic-decoder envelope for current DiffSinger shallow
   Rectified Flow inference.  This is the boundary after FastSpeech2 + aux
   decoder and before the vocoder.

   Inputs use frames-major layouts matching upstream logical tensors:
     condition_tc : [T,Q]    (DiffSingerAcoustic condition [B,T,H])
     aux_mel_tc   : [T,D]    (aux decoder mel, raw/denormalized domain)
     noise_tc     : [T,D]    (caller-selected RNG output in normalized domain)
     frame_mask_t : [T]      (normally (mel2ph > 0).float(); NULL => all ones)

   The wrapper reproduces the upstream ordering:
     aux_mel *= frame_mask
     src_norm = norm_spec(aux_mel)
     x = shallow RectifiedFlow Euler(condition, src_norm, noise)
     mel = denorm_spec(x)
     mel *= frame_mask

   spec_min/spec_max may be scalar-broadcast (range_dims=1, official default
   [-12,0]) or one value per mel bin (range_dims=D). */

size_t ds_acoustic_reflow_workspace_floats(
    const DSAsmLynxNet2Weights *w, size_t frames);

int ds_acoustic_reflow_decode_f32_avx2(
    const DSAsmLynxNet2Weights *w,
    const float *condition_tc,
    const float *aux_mel_tc,
    const float *noise_tc,
    const float *frame_mask_t,
    const float *spec_min,
    const float *spec_max,
    size_t range_dims,
    float t_start,
    float time_scale_factor,
    size_t steps,
    float *output_mel_tc,
    float *workspace,
    size_t frames,
    DSAsmThreadPool *pool);

/* M20: complete post-FS2 acoustic path. The same condition feeds the shallow
   ConvNeXt aux decoder and the Rectified Flow conditioner. */
size_t ds_acoustic_post_fs2_workspace_floats(
    const DSAsmAuxConvNeXtWeights *aux,const DSAsmLynxNet2Weights *rf,size_t frames);
int ds_acoustic_post_fs2_f32_avx2(
    const DSAsmAuxConvNeXtWeights *aux,const DSAsmLynxNet2Weights *rf,
    const float *condition_tc,const float *noise_tc,const float *frame_mask_t,
    const float *spec_min,const float *spec_max,size_t range_dims,
    float t_start,float time_scale_factor,size_t steps,float *output_mel_tc,
    float *workspace,size_t frames,DSAsmThreadPool *pool);

/* M22: fast path when the shallow aux decoder output is already in the
   normalized [-1,1]-style spec domain.  It preserves upstream semantics for
   frame masking by applying the mask algebraically in normalized space,
   avoiding aux denorm -> immediate re-norm. */
size_t ds_acoustic_reflow_normsrc_workspace_floats(
    const DSAsmLynxNet2Weights *w, size_t frames);
int ds_acoustic_reflow_decode_normsrc_f32_avx2(
    const DSAsmLynxNet2Weights *w,
    const float *condition_tc,const float *aux_norm_tc,const float *noise_tc,
    const float *frame_mask_t,const float *spec_min,const float *spec_max,size_t range_dims,
    float t_start,float time_scale_factor,size_t steps,float *output_mel_tc,
    float *workspace,size_t frames,DSAsmThreadPool *pool);
int ds_acoustic_reflow_decode_normsrc_cancel_f32_avx2(
    const DSAsmLynxNet2Weights *w,
    const float *condition_tc,const float *aux_norm_tc,const float *noise_tc,
    const float *frame_mask_t,const float *spec_min,const float *spec_max,size_t range_dims,
    float t_start,float time_scale_factor,size_t steps,float *output_mel_tc,
    float *workspace,size_t frames,DSAsmThreadPool *pool,
    DSAsmCancelCheck cancel_check,void *cancel_userdata);

size_t ds_acoustic_post_fs2_normfast_workspace_floats(
    const DSAsmAuxConvNeXtWeights *aux,const DSAsmLynxNet2Weights *rf,size_t frames);
int ds_acoustic_post_fs2_normfast_f32_avx2(
    const DSAsmAuxConvNeXtWeights *aux,const DSAsmLynxNet2Weights *rf,
    const float *condition_tc,const float *noise_tc,const float *frame_mask_t,
    const float *spec_min,const float *spec_max,size_t range_dims,
    float t_start,float time_scale_factor,size_t steps,float *output_mel_tc,
    float *workspace,size_t frames,DSAsmThreadPool *pool);
int ds_acoustic_post_fs2_normfast_cancel_f32_avx2(
    const DSAsmAuxConvNeXtWeights *aux,const DSAsmLynxNet2Weights *rf,
    const float *condition_tc,const float *noise_tc,const float *frame_mask_t,
    const float *spec_min,const float *spec_max,size_t range_dims,
    float t_start,float time_scale_factor,size_t steps,float *output_mel_tc,
    float *workspace,size_t frames,DSAsmThreadPool *pool,
    DSAsmCancelCheck cancel_check,void *cancel_userdata);

#ifdef __cplusplus
}
#endif
#endif
