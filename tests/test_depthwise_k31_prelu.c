#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

extern void ds_depthwise_conv1d_k31_prelu_f32_avx2(
    const float *x_padded, const float *weight, const float *bias,
    const float *slope, float *y, size_t C, size_t T
);

static uint32_t rng_state=0x9e3779b9u;
static float frand_small(void){
    rng_state^=rng_state<<13; rng_state^=rng_state>>17; rng_state^=rng_state<<5;
    return ((rng_state&0xffffu)/65535.0f-0.5f)*0.2f;
}
static double now_sec(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}
static void *xmalloc(size_t n){size_t p=(n+63u)&~63u;if(!p)p=64;void *q=aligned_alloc(64,p);if(!q){perror("aligned_alloc");exit(1);}return q;}

static void pad15(const float *x,float *xp,size_t C,size_t T){
    const size_t TP=T+30;
    memset(xp,0,C*TP*sizeof(float));
    for(size_t c=0;c<C;c++) memcpy(xp+c*TP+15,x+c*T,T*sizeof(float));
}

static void reference(const float *xp,const float *w,const float *b,const float *a,
                      float *y,size_t C,size_t T){
    const size_t TP=T+30;
    for(size_t c=0;c<C;c++){
        for(size_t t=0;t<T;t++){
            float v=b[c];
            for(size_t j=0;j<31;j++) v += xp[c*TP+t+j]*w[c*31+j];
            y[c*T+t]=v>=0.0f?v:a[c]*v;
        }
    }
}

static int run_case(size_t C,size_t T){
    printf("dwconv case C=%zu T=%zu\n",C,T);
    const size_t TP=T+30;
    float *x=xmalloc(C*T*4),*xp=xmalloc(C*TP*4),*w=xmalloc(C*31*4);
    float *b=xmalloc(C*4),*a=xmalloc(C*4),*ref=xmalloc(C*T*4),*got=xmalloc(C*T*4);
    for(size_t i=0;i<C*T;i++)x[i]=frand_small();
    for(size_t i=0;i<C*31;i++)w[i]=frand_small();
    for(size_t c=0;c<C;c++){b[c]=frand_small();a[c]=0.1f+fabsf(frand_small());}
    pad15(x,xp,C,T); reference(xp,w,b,a,ref,C,T);
    ds_depthwise_conv1d_k31_prelu_f32_avx2(xp,w,b,a,got,C,T);
    float ma=0,mr=0;size_t wi=0;
    for(size_t i=0;i<C*T;i++){float ae=fabsf(ref[i]-got[i]);float re=ae/fmaxf(fabsf(ref[i]),1e-6f);if(ae>ma){ma=ae;wi=i;}if(re>mr)mr=re;}
    printf("  max_abs=%g max_rel=%g",ma,mr);
    int fail=ma>2e-5f;
    if(fail)printf(" FAIL @%zu ref=%g got=%g\n",wi,ref[wi],got[wi]); else puts(" OK");
    free(x);free(xp);free(w);free(b);free(a);free(ref);free(got);return fail;
}

static void benchmark(size_t C,size_t T,int iters){
    const size_t TP=T+30;
    float *x=xmalloc(C*T*4),*xp=xmalloc(C*TP*4),*w=xmalloc(C*31*4),*b=xmalloc(C*4),*a=xmalloc(C*4),*y=xmalloc(C*T*4);
    for(size_t i=0;i<C*T;i++)x[i]=frand_small();
    for(size_t i=0;i<C*31;i++)w[i]=frand_small();
    for(size_t c=0;c<C;c++){b[c]=frand_small();a[c]=0.1f+fabsf(frand_small());}
    pad15(x,xp,C,T);
    ds_depthwise_conv1d_k31_prelu_f32_avx2(xp,w,b,a,y,C,T);
    double t0=now_sec();
    for(int i=0;i<iters;i++) ds_depthwise_conv1d_k31_prelu_f32_avx2(xp,w,b,a,y,C,T);
    double t1=now_sec();
    // 31 mul + 31 add per output. PReLU excluded from FLOP figure.
    double flops=62.0*(double)C*(double)T*(double)iters;
    printf("dwconv bench C=%zu T=%zu iters=%d : %.3f ms/iter, %.2f GFLOP/s\n",
           C,T,iters,(t1-t0)*1e3/iters,flops/(t1-t0)/1e9);
    volatile float sink=y[C*T/2];(void)sink;
    free(x);free(xp);free(w);free(b);free(a);free(y);
}

int main(void){
#if defined(__x86_64__)
    if(!__builtin_cpu_supports("avx2")||!__builtin_cpu_supports("fma")){fprintf(stderr,"AVX2 + FMA3 required\n");return 2;}
#endif
    int fail=0;
    fail|=run_case(1,1);
    fail|=run_case(2,7);
    fail|=run_case(3,8);
    fail|=run_case(5,17);
    fail|=run_case(16,257);
    if(fail)return 1;
    benchmark(1024,256,20);
    benchmark(1024,1024,10);
    return 0;
}
