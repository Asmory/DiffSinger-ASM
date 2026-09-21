#include "dsasm_kernels.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float rndf(void){ return ((float)rand()/(float)RAND_MAX)*6.0f-3.0f; }
int main(void){
    srand(9);
    const size_t M=13,N=128,YS=320;
    float *x=(float*)malloc(M*2*N*sizeof(float));
    float *ref=(float*)malloc(M*N*sizeof(float));
    float *y=(float*)malloc(M*YS*sizeof(float));
    if(!x||!ref||!y)return 2;
    for(size_t i=0;i<M*2*N;i++)x[i]=rndf();
    for(size_t i=0;i<M*YS;i++)y[i]=1234.0f;
    ds_atan_glu_f32_avx2(x,ref,M,N);
    ds_atan_glu_f32_avx2_ystrided(x,y,M,N,YS);
    float mx=0.0f;
    for(size_t m=0;m<M;m++)for(size_t n=0;n<N;n++){
        float e=fabsf(ref[m*N+n]-y[m*YS+n]); if(e>mx)mx=e;
    }
    printf("M9 ATanGLU y-strided max_abs=%g %s\n",mx,mx==0.0f?"OK":"FAIL");
    free(x);free(ref);free(y);return mx==0.0f?0:1;
}
