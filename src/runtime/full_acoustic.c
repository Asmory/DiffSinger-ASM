#include "dsasm_full_acoustic.h"
#include <stddef.h>

size_t ds_full_acoustic_workspace_floats(
    const DSAsmFS2AcousticWeights*fs2,const DSAsmAuxConvNeXtWeights*aux,
    const DSAsmLynxNet2Weights*rf,size_t P,size_t T){
    if(!fs2||!aux||!rf||!P||!T)return 0;
    if(fs2->encoder.hidden_size!=aux->input_dim||aux->input_dim!=rf->condition_dim||aux->output_dim!=rf->input_dim)return 0;
    size_t fw=ds_fs2_acoustic_condition_workspace_floats(fs2,P,T);
    size_t pw=ds_acoustic_post_fs2_workspace_floats(aux,rf,T);
    if(!fw||!pw)return 0;
    return fw + T*fs2->encoder.hidden_size + T + pw;
}

int ds_full_acoustic_infer_f32_avx2(
    const DSAsmFS2AcousticWeights*fs2,const DSAsmAuxConvNeXtWeights*aux,const DSAsmLynxNet2Weights*rf,
    const int32_t*tok,size_t P,const int32_t*mel2ph,const float*f0,size_t T,
    const float*noise,const float*lo,const float*hi,size_t rd,float t0,float scale,size_t steps,
    float*out,float*ws,DSAsmThreadPool*pool){
    if(!fs2||!aux||!rf||!tok||!mel2ph||!f0||!noise||!lo||!hi||!out||!ws||!P||!T)return -1;
    if(fs2->encoder.hidden_size!=aux->input_dim||aux->input_dim!=rf->condition_dim||aux->output_dim!=rf->input_dim)return -2;
    size_t fw=ds_fs2_acoustic_condition_workspace_floats(fs2,P,T);if(!fw)return -3;
    float*cond=ws+fw;
    float*mask=cond+T*fs2->encoder.hidden_size;
    float*postws=mask+T;
    int rc=ds_fs2_acoustic_condition_f32_avx2(fs2,tok,P,mel2ph,f0,T,cond,ws,pool);if(rc)return rc;
    for(size_t t=0;t<T;t++)mask[t]=mel2ph[t]>0?1.0f:0.0f;
    return ds_acoustic_post_fs2_f32_avx2(aux,rf,cond,noise,mask,lo,hi,rd,t0,scale,steps,out,postws,T,pool);
}

size_t ds_full_acoustic_normfast_workspace_floats(
    const DSAsmFS2AcousticWeights*fs2,const DSAsmAuxConvNeXtWeights*aux,
    const DSAsmLynxNet2Weights*rf,size_t P,size_t T){
    if(!fs2||!aux||!rf||!P||!T)return 0;
    if(fs2->encoder.hidden_size!=aux->input_dim||aux->input_dim!=rf->condition_dim||aux->output_dim!=rf->input_dim)return 0;
    size_t fw=ds_fs2_acoustic_condition_workspace_floats(fs2,P,T);
    size_t pw=ds_acoustic_post_fs2_normfast_workspace_floats(aux,rf,T);
    if(!fw||!pw)return 0;
    return fw + T*fs2->encoder.hidden_size + T + pw;
}

int ds_full_acoustic_infer_normfast_f32_avx2(
    const DSAsmFS2AcousticWeights*fs2,const DSAsmAuxConvNeXtWeights*aux,const DSAsmLynxNet2Weights*rf,
    const int32_t*tok,size_t P,const int32_t*mel2ph,const float*f0,size_t T,
    const float*noise,const float*lo,const float*hi,size_t rd,float t0,float scale,size_t steps,
    float*out,float*ws,DSAsmThreadPool*pool){
    return ds_full_acoustic_infer_normfast_cancel_f32_avx2(
        fs2,aux,rf,tok,P,mel2ph,f0,T,noise,lo,hi,rd,t0,scale,steps,out,ws,pool,
        NULL,NULL);
}

int ds_full_acoustic_infer_normfast_cancel_f32_avx2(
    const DSAsmFS2AcousticWeights*fs2,const DSAsmAuxConvNeXtWeights*aux,const DSAsmLynxNet2Weights*rf,
    const int32_t*tok,size_t P,const int32_t*mel2ph,const float*f0,size_t T,
    const float*noise,const float*lo,const float*hi,size_t rd,float t0,float scale,size_t steps,
    float*out,float*ws,DSAsmThreadPool*pool,DSAsmCancelCheck cancel_check,void*cancel_userdata){
    if(!fs2||!aux||!rf||!tok||!mel2ph||!f0||!noise||!lo||!hi||!out||!ws||!P||!T)return -1;
    if(fs2->encoder.hidden_size!=aux->input_dim||aux->input_dim!=rf->condition_dim||aux->output_dim!=rf->input_dim)return -2;
    size_t fw=ds_fs2_acoustic_condition_workspace_floats(fs2,P,T);if(!fw)return -3;
    float*cond=ws+fw;
    float*mask=cond+T*fs2->encoder.hidden_size;
    float*postws=mask+T;
    int rc=ds_fs2_acoustic_condition_f32_avx2(fs2,tok,P,mel2ph,f0,T,cond,ws,pool);if(rc)return rc;
    if(cancel_check&&cancel_check(cancel_userdata))return -1;
    for(size_t t=0;t<T;t++)mask[t]=mel2ph[t]>0?1.0f:0.0f;
    return ds_acoustic_post_fs2_normfast_cancel_f32_avx2(aux,rf,cond,noise,mask,lo,hi,rd,t0,scale,steps,out,postws,T,pool,
        cancel_check,cancel_userdata);
}

int ds_full_acoustic_infer_deploy_normfast_f32_avx2(
    const DSAsmFS2AcousticWeights*fs2,const DSAsmFS2DeploymentExtras*extras,const DSAsmFS2DeploymentInputs*inputs,
    const DSAsmAuxConvNeXtWeights*aux,const DSAsmLynxNet2Weights*rf,
    const int32_t*tok,size_t P,const int32_t*mel2ph,const float*f0,size_t T,
    const float*noise,const float*lo,const float*hi,size_t rd,float t0,float scale,size_t steps,
    float*out,float*ws,DSAsmThreadPool*pool){
    return ds_full_acoustic_infer_deploy_normfast_cancel_f32_avx2(
        fs2,extras,inputs,aux,rf,tok,P,mel2ph,f0,T,noise,lo,hi,rd,t0,scale,
        steps,out,ws,pool,NULL,NULL);
}

int ds_full_acoustic_infer_deploy_normfast_cancel_f32_avx2(
    const DSAsmFS2AcousticWeights*fs2,const DSAsmFS2DeploymentExtras*extras,const DSAsmFS2DeploymentInputs*inputs,
    const DSAsmAuxConvNeXtWeights*aux,const DSAsmLynxNet2Weights*rf,
    const int32_t*tok,size_t P,const int32_t*mel2ph,const float*f0,size_t T,
    const float*noise,const float*lo,const float*hi,size_t rd,float t0,float scale,size_t steps,
    float*out,float*ws,DSAsmThreadPool*pool,DSAsmCancelCheck cancel_check,void*cancel_userdata){
    if(!fs2||!aux||!rf||!tok||!mel2ph||!f0||!noise||!lo||!hi||!out||!ws||!P||!T)return -1;
    if(fs2->encoder.hidden_size!=aux->input_dim||aux->input_dim!=rf->condition_dim||aux->output_dim!=rf->input_dim)return -2;
    size_t fw=ds_fs2_acoustic_condition_workspace_floats(fs2,P,T);if(!fw)return -3;
    float*cond=ws+fw;
    float*mask=cond+T*fs2->encoder.hidden_size;
    float*postws=mask+T;
    int rc=ds_fs2_acoustic_condition_deploy_f32_avx2(fs2,extras,inputs,tok,P,mel2ph,f0,T,cond,ws,pool);if(rc)return rc;
    if(cancel_check&&cancel_check(cancel_userdata))return -1;
    for(size_t t=0;t<T;t++)mask[t]=mel2ph[t]>0?1.0f:0.0f;
    return ds_acoustic_post_fs2_normfast_cancel_f32_avx2(aux,rf,cond,noise,mask,lo,hi,rd,t0,scale,steps,out,postws,T,pool,
        cancel_check,cancel_userdata);
}
