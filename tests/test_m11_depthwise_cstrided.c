#include "dsasm_kernels.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static uint32_t st=1;
static float rf(void){st=1664525u*st+1013904223u;return ((st>>8)&0xffff)/32768.0f-1.0f;}
int main(void){
    const size_t T=64,C=1024,P=8;
    float *x=malloc(T*C*4),*w=malloc(31*C*4),*b=malloc(C*4),*a=malloc(T*C*4),*z=malloc(T*C*4);
    if(!x||!w||!b||!a||!z)return 2;
    for(size_t i=0;i<T*C;i++) x[i]=rf();
    for(size_t i=0;i<31*C;i++) w[i]=rf()*.02f;
    for(size_t i=0;i<C;i++) b[i]=rf()*.01f;
    ds_depthwise_conv1d_k31_tc_f32_avx2(x,w,b,a,T,C);
    for(size_t q=0;q<P;q++){
        size_t blocks=(C+7)/8,b0=blocks*q/P,b1=blocks*(q+1)/P,c0=b0*8,c1=b1*8;if(c1>C)c1=C;
        ds_depthwise_conv1d_k31_tc_f32_avx2_cstrided(x+c0,w+c0,b+c0,z+c0,T,c1-c0,C);
    }
    float e=0;for(size_t i=0;i<T*C;i++){float d=fabsf(a[i]-z[i]);if(d>e)e=d;}
    printf("M11 depthwise C-strided 8-way max_abs=%g %s\n",e,e==0.0f?"OK":"FAIL");
    free(x);free(w);free(b);free(a);free(z);return e==0.0f?0:1;
}
