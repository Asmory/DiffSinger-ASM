#include "dsasm_kernels.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float val(size_t i){ unsigned x=(unsigned)i*1664525u+1013904223u; return ((int)((x>>8)%2001)-1000)/1000.0f; }
int main(void){
    const size_t M=13,N=64,K=31;
    float *x=malloc(M*K*sizeof(float)),*b=malloc(N*sizeof(float)),*w=malloc(N*K*sizeof(float));
    float *r=malloc(M*N*sizeof(float)),*full=calloc(M*N,sizeof(float)),*tile=calloc(M*N,sizeof(float));
    float *fullr=calloc(M*N,sizeof(float)),*tiler=calloc(M*N,sizeof(float));
    if(!x||!b||!w||!r||!full||!tile||!fullr||!tiler)return 2;
    for(size_t i=0;i<M*K;i++) x[i]=val(i);
    for(size_t i=0;i<N;i++) b[i]=val(i+7);
    for(size_t i=0;i<M*N;i++) r[i]=val(i+99);
    for(size_t nb=0;nb<N/16;nb++)for(size_t k=0;k<K;k++)for(size_t j=0;j<16;j++)w[(nb*K+k)*16+j]=val(nb*K*16+k*16+j+21);
    ds_linear_f32_avx2_m4n16(x,w,b,full,M,N,K);
    ds_linear_residual_f32_avx2_m4n16(x,w,b,r,fullr,M,N,K);
    for(size_t n0=0;n0<N;n0+=16){const float *wt=w+(n0/16)*K*16;
        ds_linear_f32_avx2_m4n16_strided(x,wt,b+n0,tile+n0,M,16,K,N);
        ds_linear_residual_f32_avx2_m4n16_strided(x,wt,b+n0,r+n0,tiler+n0,M,16,K,N);
    }
    float e=0,er=0;for(size_t i=0;i<M*N;i++){float d=fabsf(full[i]-tile[i]);if(d>e)e=d;d=fabsf(fullr[i]-tiler[i]);if(d>er)er=d;}
    printf("M8 strided Linear max_abs=%g; residual=%g %s\n",e,er,(e<1e-6f&&er<1e-6f)?"OK":"FAIL");
    free(x);free(b);free(w);free(r);free(full);free(tile);free(fullr);free(tiler);return (e<1e-6f&&er<1e-6f)?0:1;
}
