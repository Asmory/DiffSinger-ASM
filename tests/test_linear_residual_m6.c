#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void ds_linear_residual_f32_avx2_packed4(const float*, const float*, const float*, const float*, float*, size_t, size_t, size_t);
void ds_linear_residual_f32_avx2_m4n16(const float*, const float*, const float*, const float*, float*, size_t, size_t, size_t);

static uint32_t st = 7;
static float rnd(void) { st=st*1664525u+1013904223u; return ((st>>8)&0xffffff)/8388607.5f-1.0f; }
static double now_sec(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static float maxdiff(const float*a,const float*b,size_t n){float d=0;for(size_t i=0;i<n;i++){float x=fabsf(a[i]-b[i]);if(x>d)d=x;}return d;}

static void pack4(const float*w,float*p,size_t N,size_t K){
    size_t z=0;
    for(size_t n=0;n<N;n+=4) for(size_t k=0;k<K;k+=8)
        for(size_t q=0;q<4;q++) for(size_t j=0;j<8;j++) p[z++]=w[(n+q)*K+k+j];
}
static void pack16(const float*w,float*p,size_t N,size_t K){
    size_t z=0;
    for(size_t n=0;n<N;n+=16) for(size_t k=0;k<K;k++){
        for(size_t q=0;q<8;q++) p[z++]=w[(n+q)*K+k];
        for(size_t q=8;q<16;q++) p[z++]=w[(n+q)*K+k];
    }
}
static void reference(const float*x,const float*w,const float*b,const float*r,float*y,size_t M,size_t N,size_t K){
    for(size_t m=0;m<M;m++) for(size_t n=0;n<N;n++){
        float a=b[n]+r[m*N+n];
        for(size_t k=0;k<K;k++) a+=x[m*K+k]*w[n*K+k];
        y[m*N+n]=a;
    }
}
static int one(size_t M,size_t N,size_t K){
    size_t xc=M*K,wc=N*K,yc=M*N;
    float*x=malloc(xc*4),*w=malloc(wc*4),*b=malloc(N*4),*r=malloc(yc*4),*ref=malloc(yc*4),*y=malloc(yc*4),*p=malloc(wc*4);
    for(size_t i=0;i<xc;i++) x[i]=rnd()*.2f;
    for(size_t i=0;i<wc;i++) w[i]=rnd()*.1f;
    for(size_t i=0;i<N;i++) b[i]=rnd()*.1f;
    for(size_t i=0;i<yc;i++) r[i]=rnd()*.2f;
    pack16(w,p,N,K); reference(x,w,b,r,ref,M,N,K);
    ds_linear_residual_f32_avx2_m4n16(x,p,b,r,y,M,N,K);
    float d=maxdiff(ref,y,yc);
    printf("M6 Linear+Residual M=%zu N=%zu K=%zu max_abs=%g %s\n",M,N,K,d,d<3e-4f?"OK":"FAIL");
    free(x);free(w);free(b);free(r);free(ref);free(y);free(p); return d>=3e-4f;
}
static void bench(void){
    const size_t M=128,N=512,K=512,xc=M*K,wc=N*K,yc=M*N;
    const int it=40;
    float*x=aligned_alloc(64,xc*4),*w=aligned_alloc(64,wc*4),*b=aligned_alloc(64,N*4),*r=aligned_alloc(64,yc*4),*y=aligned_alloc(64,yc*4),*p4=aligned_alloc(64,wc*4),*p16=aligned_alloc(64,wc*4);
    for(size_t i=0;i<xc;i++) x[i]=rnd()*.2f;
    for(size_t i=0;i<wc;i++) w[i]=rnd()*.1f;
    for(size_t i=0;i<N;i++) b[i]=rnd()*.1f;
    for(size_t i=0;i<yc;i++) r[i]=rnd()*.2f;
    pack4(w,p4,N,K); pack16(w,p16,N,K);
    ds_linear_residual_f32_avx2_packed4(x,p4,b,r,y,M,N,K);
    double a=now_sec(); for(int i=0;i<it;i++) ds_linear_residual_f32_avx2_packed4(x,p4,b,r,y,M,N,K); double bb=now_sec();
    ds_linear_residual_f32_avx2_m4n16(x,p16,b,r,y,M,N,K);
    double c=now_sec(); for(int i=0;i<it;i++) ds_linear_residual_f32_avx2_m4n16(x,p16,b,r,y,M,N,K); double d=now_sec();
    double t4=(bb-a)/it,t16=(d-c)/it,flops=2.0*M*N*K;
    printf("Linear+Residual bench\n  packed4 %.3f ms %.1f GF/s\n  M6 4x16 %.3f ms %.1f GF/s speedup %.2fx\n",t4*1e3,flops/t4/1e9,t16*1e3,flops/t16/1e9,t4/t16);
    free(x);free(w);free(b);free(r);free(y);free(p4);free(p16);
}
int main(void){
    int fail=0;
    fail|=one(1,16,7); fail|=one(3,32,31); fail|=one(4,16,64); fail|=one(5,48,65); fail|=one(37,64,64);
    if(fail) return 1;
    bench();
    return 0;
}
