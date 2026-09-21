#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "dsasm_kernels.h"

static uint32_t rng_state=1;
static float frand_small(void){
    rng_state=rng_state*1664525u+1013904223u;
    return ((int32_t)(rng_state>>8))/(float)(1<<23)*0.1f;
}

static int check(size_t C,size_t T){
    const size_t K=16,Tout=8*T;
    float *x=aligned_alloc(32,C*T*sizeof(float));
    float *w=aligned_alloc(32,C*K*sizeof(float));
    float *ref=aligned_alloc(32,Tout*sizeof(float));
    float *got=aligned_alloc(32,Tout*sizeof(float));
    if(!x||!w||!ref||!got) return 99;
    for(size_t i=0;i<C*T;i++)x[i]=frand_small();
    for(size_t i=0;i<C*K;i++)w[i]=frand_small();
    const float bias=0.03125f;
    ds_convtranspose1d_oc_f32_avx2(x,w,bias,ref,C,K,T,Tout,4,8);
    ds_convtranspose1d_s8k16_oc_f32_avx2(x,w,bias,got,C,T);
    float max_abs=0.0f,rmse=0.0f;
    for(size_t i=0;i<Tout;i++){
        const float d=fabsf(ref[i]-got[i]);
        if(d>max_abs)max_abs=d;
        rmse+=d*d;
    }
    rmse=sqrtf(rmse/(float)Tout);
    printf("M42 s8k16 parity C=%zu T=%zu max_abs=%.9g rmse=%.9g\n",C,T,max_abs,rmse);
    free(x);free(w);free(ref);free(got);
    return max_abs<=2.0e-5f?0:1;
}

int main(void){
    int rc=0;
    rc|=check(1,1);
    rc|=check(3,2);
    rc|=check(16,17);
    rc|=check(512,48);
    rc|=check(256,384);
    return rc;
}
