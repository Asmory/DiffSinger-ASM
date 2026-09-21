#include "dsasm_acoustic.h"

size_t ds_acoustic_post_fs2_workspace_floats(
    const DSAsmAuxConvNeXtWeights *aux,const DSAsmLynxNet2Weights *rf,size_t frames){
    if(!aux||!rf||!frames||aux->output_dim!=rf->input_dim||aux->input_dim!=rf->condition_dim)return 0;
    size_t aw=ds_aux_convnext_workspace_floats(aux,frames);
    size_t rw=ds_acoustic_reflow_workspace_floats(rf,frames);
    if(!aw||!rw)return 0;
    return aw + frames*aux->output_dim + rw;
}

int ds_acoustic_post_fs2_f32_avx2(
    const DSAsmAuxConvNeXtWeights *aux,const DSAsmLynxNet2Weights *rf,
    const float *condition_tc,const float *noise_tc,const float *frame_mask_t,
    const float *spec_min,const float *spec_max,size_t range_dims,
    float t_start,float time_scale_factor,size_t steps,float *output_mel_tc,
    float *workspace,size_t frames,DSAsmThreadPool *pool){
    if(!aux||!rf||!condition_tc||!noise_tc||!spec_min||!spec_max||!output_mel_tc||!workspace||!frames)return -1;
    if(aux->output_dim!=rf->input_dim||aux->input_dim!=rf->condition_dim)return -2;
    size_t aw=ds_aux_convnext_workspace_floats(aux,frames);if(!aw)return -3;
    float*aux_ws=workspace;
    float*aux_mel=workspace+aw;
    float*rf_ws=aux_mel+frames*aux->output_dim;
    int rc=ds_aux_convnext_infer_f32_avx2(aux,condition_tc,spec_min,spec_max,range_dims,aux_mel,aux_ws,frames,pool);
    if(rc)return rc;
    return ds_acoustic_reflow_decode_f32_avx2(rf,condition_tc,aux_mel,noise_tc,frame_mask_t,
        spec_min,spec_max,range_dims,t_start,time_scale_factor,steps,output_mel_tc,rf_ws,frames,pool);
}

size_t ds_acoustic_post_fs2_normfast_workspace_floats(
    const DSAsmAuxConvNeXtWeights *aux,const DSAsmLynxNet2Weights *rf,size_t frames){
    if(!aux||!rf||!frames||aux->output_dim!=rf->input_dim||aux->input_dim!=rf->condition_dim)return 0;
    size_t aw=ds_aux_convnext_workspace_floats(aux,frames);
    size_t rw=ds_acoustic_reflow_normsrc_workspace_floats(rf,frames);
    if(!aw||!rw)return 0;
    return aw + frames*aux->output_dim + rw;
}

int ds_acoustic_post_fs2_normfast_f32_avx2(
    const DSAsmAuxConvNeXtWeights *aux,const DSAsmLynxNet2Weights *rf,
    const float *condition_tc,const float *noise_tc,const float *frame_mask_t,
    const float *spec_min,const float *spec_max,size_t range_dims,
    float t_start,float time_scale_factor,size_t steps,float *output_mel_tc,
    float *workspace,size_t frames,DSAsmThreadPool *pool){
    if(!aux||!rf||!condition_tc||!noise_tc||!spec_min||!spec_max||!output_mel_tc||!workspace||!frames)return -1;
    if(aux->output_dim!=rf->input_dim||aux->input_dim!=rf->condition_dim)return -2;
    size_t aw=ds_aux_convnext_workspace_floats(aux,frames);if(!aw)return -3;
    float*aux_ws=workspace;
    float*aux_norm=workspace+aw;
    float*rf_ws=aux_norm+frames*aux->output_dim;
    int rc=ds_aux_convnext_forward_norm_f32_avx2(aux,condition_tc,aux_norm,aux_ws,frames,pool);
    if(rc)return rc;
    return ds_acoustic_reflow_decode_normsrc_f32_avx2(rf,condition_tc,aux_norm,noise_tc,frame_mask_t,
        spec_min,spec_max,range_dims,t_start,time_scale_factor,steps,output_mel_tc,rf_ws,frames,pool);
}
