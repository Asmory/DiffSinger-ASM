#include "dsasm_kernels.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

extern void ds_linear_f32_avx2_m4n16_idxstrided(const float*,const float*,const float*,float*,size_t,size_t,size_t,size_t);
extern void ds_linear_residual_f32_avx2_m4n16_idxstrided(const float*,const float*,const float*,const float*,float*,size_t,size_t,size_t,size_t);

static float val(size_t i){ unsigned x=(unsigned)i*1664525u+1013904223u; return ((int)((x>>8)%2001)-1000)/1000.0f; }
static double now_s(void){ struct timespec ts; timespec_get(&ts,TIME_UTC); return ts.tv_sec+ts.tv_nsec*1e-9; }

static int check(size_t M,size_t N,size_t K,size_t stride){
    size_t xs=M*K, ws=(N/16)*K*16, ys=M*stride;
    float *x=aligned_alloc(64,(xs*sizeof(float)+63)&~63ULL);
    float *w=aligned_alloc(64,(ws*sizeof(float)+63)&~63ULL);
    float *b=aligned_alloc(64,(N*sizeof(float)+63)&~63ULL);
    float *r=aligned_alloc(64,(ys*sizeof(float)+63)&~63ULL);
    float *a=aligned_alloc(64,(ys*sizeof(float)+63)&~63ULL);
    float *z=aligned_alloc(64,(ys*sizeof(float)+63)&~63ULL);
    float *ar=aligned_alloc(64,(ys*sizeof(float)+63)&~63ULL);
    float *zr=aligned_alloc(64,(ys*sizeof(float)+63)&~63ULL);
    if(!x||!w||!b||!r||!a||!z||!ar||!zr)return 2;
    for(size_t i=0;i<xs;i++)x[i]=val(i+1);
    for(size_t i=0;i<ws;i++)w[i]=val(i+11);
    for(size_t i=0;i<N;i++)b[i]=val(i+19);
    for(size_t i=0;i<ys;i++)r[i]=val(i+31);
    memset(a,0,ys*sizeof(float));memset(z,0,ys*sizeof(float));memset(ar,0,ys*sizeof(float));memset(zr,0,ys*sizeof(float));
    ds_linear_f32_avx2_m4n16_strided(x,w,b,a,M,N,K,stride);
    ds_linear_f32_avx2_m4n16_idxstrided(x,w,b,z,M,N,K,stride);
    ds_linear_residual_f32_avx2_m4n16_strided(x,w,b,r,ar,M,N,K,stride);
    ds_linear_residual_f32_avx2_m4n16_idxstrided(x,w,b,r,zr,M,N,K,stride);
    float e=0,er=0; for(size_t m=0;m<M;m++)for(size_t n=0;n<N;n++){size_t i=m*stride+n;float d=fabsf(a[i]-z[i]);if(d>e)e=d;d=fabsf(ar[i]-zr[i]);if(d>er)er=d;}
    printf("M13 indexed M=%zu N=%zu K=%zu stride=%zu linear=%g residual=%g %s\n",M,N,K,stride,e,er,(e==0&&er==0)?"OK":"FAIL");
    free(x);free(w);free(b);free(r);free(a);free(z);free(ar);free(zr);return (e==0&&er==0)?0:1;
}

static void bench(size_t M,size_t N,size_t K,int iters){
    size_t stride=N, xs=M*K, ws=(N/16)*K*16, ys=M*N;
    float *x=aligned_alloc(64,(xs*sizeof(float)+63)&~63ULL),*w=aligned_alloc(64,(ws*sizeof(float)+63)&~63ULL),*b=aligned_alloc(64,(N*sizeof(float)+63)&~63ULL),*y=aligned_alloc(64,(ys*sizeof(float)+63)&~63ULL);
    for(size_t i=0;i<xs;i++) x[i]=val(i+1);
    for(size_t i=0;i<ws;i++) w[i]=val(i+11);
    for(size_t i=0;i<N;i++) b[i]=val(i+19);
    for(int i=0;i<3;i++){ds_linear_f32_avx2_m4n16_strided(x,w,b,y,M,N,K,stride);ds_linear_f32_avx2_m4n16_idxstrided(x,w,b,y,M,N,K,stride);}    
    double t0=now_s();for(int i=0;i<iters;i++)ds_linear_f32_avx2_m4n16_strided(x,w,b,y,M,N,K,stride);double t1=now_s();
    double t2=now_s();for(int i=0;i<iters;i++)ds_linear_f32_avx2_m4n16_idxstrided(x,w,b,y,M,N,K,stride);double t3=now_s();
    double old=(t1-t0)*1000/iters, neu=(t3-t2)*1000/iters;
    printf("M13 bench M=%zu N=%zu K=%zu old=%.3f ms idx=%.3f ms speedup=%.3fx\n",M,N,K,old,neu,old/neu);
    free(x);free(w);free(b);free(y);
}
int main(void){
    int rc=0;rc|=check(1,16,7,32);rc|=check(3,32,31,48);rc|=check(13,64,65,96);rc|=check(32,64,1024,64);rc|=check(64,128,1024,128);
    bench(32,64,1024,30);bench(32,128,1024,20);bench(64,64,1024,20);return rc;
}
