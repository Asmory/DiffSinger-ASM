#include "dsasm_threadpool.h"
#include "dsasm_kernels.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double ms(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1000.0+t.tv_nsec/1e6;}
static uint32_t rng=1; static float frand1(void){rng=rng*1664525u+1013904223u;return ((int32_t)(rng>>8))/8388608.0f;}
int main(void){
    const size_t C=128,T=24576,pad=15,Tp=T+2*pad,n=C*T,np=C*Tp;
    float *x=aligned_alloc(64,(n*sizeof(float)+63)&~63ull);
    float *a=aligned_alloc(64,(np*sizeof(float)+63)&~63ull);
    float *b=aligned_alloc(64,(np*sizeof(float)+63)&~63ull);
    if(!x||!a||!b)return 2;
    for(size_t i=0;i<n;i++)x[i]=frand1();
    DSAsmThreadPool *p=ds_threadpool_create(8); if(!p)return 3;
    ds_leaky_copy_nct_f32_avx2(x,a,C,T,Tp,pad,0.1f);
    ds_threadpool_leaky_copy_nct_f32(p,x,b,C,T,Tp,pad,0.1f);
    size_t diff=0; float ma=0; for(size_t i=0;i<np;i++){float d=fabsf(a[i]-b[i]);if(d>ma)ma=d;if(memcmp(a+i,b+i,4))diff++;}
    const int R=20; double t0=ms();for(int r=0;r<R;r++)ds_leaky_copy_nct_f32_avx2(x,a,C,T,Tp,pad,0.1f);double t1=ms();
    double t2=ms();for(int r=0;r<R;r++)ds_threadpool_leaky_copy_nct_f32(p,x,b,C,T,Tp,pad,0.1f);double t3=ms();
    printf("M55 leaky-copy C=%zu T=%zu max_abs=%g bitdiff=%zu serial=%.3fms parallel=%.3fms speedup=%.3fx\
",C,T,ma,diff,(t1-t0)/R,(t3-t2)/R,(t1-t0)/(t3-t2));
    ds_threadpool_destroy(p);free(x);free(a);free(b);return diff?1:0;
}
