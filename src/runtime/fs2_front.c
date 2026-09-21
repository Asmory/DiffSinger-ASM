#include "dsasm_fs2_front.h"
#include <math.h>
#include <string.h>
int ds_fs2_mel2ph_to_dur_i32(const int32_t*m,size_t T,size_t P,int32_t*d){
    if(!m||!d||!P)return -1;
    memset(d,0,P*sizeof(*d));
    for(size_t t=0;t<T;t++){int32_t q=m[t];if(q<0||(size_t)q>P)return -2;if(q>0)d[q-1]++;}
    return 0;
}
int ds_fs2_stretch_f32(const int32_t*m,size_t T,const int32_t*d,size_t P,float*out){
    if(!m||!d||!out)return -1;
    int32_t accum=0;
    for(size_t t=0;t<T;t++){
        int32_t q=m[t];if(q<0||(size_t)q>P)return -2;
        if(q==0){out[t]=0;accum=0;continue;}
        int32_t md=d[q-1];if(md<=0){out[t]=0;continue;}
        if(t==0||m[t]!=m[t-1])accum=0;else accum++;
        out[t]=(float)accum/(float)md;
    }
    return 0;
}
int ds_fs2_gather_encoder_f32(const float*e,const int32_t*m,size_t T,size_t P,size_t H,float*c){
    if(!e||!m||!c||!H)return -1;
    for(size_t t=0;t<T;t++){
        int32_t q=m[t];float*dst=c+t*H;if(q<0||(size_t)q>P)return -2;
        if(q==0){memset(dst,0,H*sizeof(float));continue;}
        memcpy(dst,e+(size_t)(q-1)*H,H*sizeof(float));
    }
    return 0;
}
void ds_fs2_pitch_input_f32(const float*f0,size_t T,float*out){for(size_t i=0;i<T;i++)out[i]=log1pf(f0[i]/700.0f);}
void ds_fs2_duration_input_f32(const int32_t*d,size_t P,float*out){for(size_t i=0;i<P;i++)out[i]=log1pf((float)d[i]);}
