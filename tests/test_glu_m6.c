#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void ds_fused_linear_softsign_glu_f32_avx2_packed4(const float*,const float*,const float*,const float*,float*,size_t,size_t,size_t);
void ds_fused_linear_softsign_glu_f32_avx2_m4n8(const float*,const float*,const float*,const float*,float*,size_t,size_t,size_t);

static uint32_t st=1;
static float rnd(void){st=st*1664525u+1013904223u;return ((st>>8)&0xffffff)/8388607.5f-1.0f;}
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static float maxdiff(const float*a,const float*b,size_t n){float d=0;for(size_t i=0;i<n;i++){float x=fabsf(a[i]-b[i]);if(x>d)d=x;}return d;}
static void ref(const float*x,const float*wl,const float*wg,const float*bl,const float*bg,float*y,size_t M,size_t N,size_t K){
 for(size_t m=0;m<M;m++)for(size_t n=0;n<N;n++){float l=bl[n],g=bg[n];for(size_t k=0;k<K;k++){float v=x[m*K+k];l+=v*wl[n*K+k];g+=v*wg[n*K+k];}y[m*N+n]=l*(g/(1.0f+fabsf(g)));}}
static void pack4(const float*wl,const float*wg,float*p,size_t N,size_t K){size_t z=0;for(size_t n0=0;n0<N;n0+=4)for(size_t k0=0;k0<K;k0+=8){for(size_t q=0;q<4;q++)for(size_t j=0;j<8;j++)p[z++]=wl[(n0+q)*K+k0+j];for(size_t q=0;q<4;q++)for(size_t j=0;j<8;j++)p[z++]=wg[(n0+q)*K+k0+j];}}
static void pack8(const float*wl,const float*wg,float*p,size_t N,size_t K){size_t z=0;for(size_t n0=0;n0<N;n0+=8)for(size_t k=0;k<K;k++){for(size_t q=0;q<8;q++)p[z++]=wl[(n0+q)*K+k];for(size_t q=0;q<8;q++)p[z++]=wg[(n0+q)*K+k];}}
static int one(size_t M,size_t N,size_t K){
 size_t xc=M*K,wc=N*K,yc=M*N;float*x=malloc(xc*4),*wl=malloc(wc*4),*wg=malloc(wc*4),*bl=malloc(N*4),*bg=malloc(N*4),*r=malloc(yc*4),*y=malloc(yc*4),*p=malloc(2*wc*4);
 for(size_t i=0;i<xc;i++) x[i]=rnd()*.2f;
 for(size_t i=0;i<wc;i++){ wl[i]=rnd()*.1f; wg[i]=rnd()*.1f; }
 for(size_t i=0;i<N;i++){ bl[i]=rnd()*.1f; bg[i]=rnd()*.1f; }
 pack8(wl,wg,p,N,K);ref(x,wl,wg,bl,bg,r,M,N,K);ds_fused_linear_softsign_glu_f32_avx2_m4n8(x,p,bl,bg,y,M,N,K);float d=maxdiff(r,y,yc);printf("M6 GLU M=%zu N=%zu K=%zu max_abs=%g %s\n",M,N,K,d,d<3e-4f?"OK":"FAIL");
 free(x);free(wl);free(wg);free(bl);free(bg);free(r);free(y);free(p);return d>=3e-4f;}
static void bench(size_t M,size_t N,size_t K,int iters){
 size_t xc=M*K,wc=N*K,yc=M*N;float*x=aligned_alloc(64,xc*4),*wl=aligned_alloc(64,wc*4),*wg=aligned_alloc(64,wc*4),*bl=aligned_alloc(64,N*4),*bg=aligned_alloc(64,N*4),*y=aligned_alloc(64,yc*4),*p4=aligned_alloc(64,2*wc*4),*p8=aligned_alloc(64,2*wc*4);
 for(size_t i=0;i<xc;i++) x[i]=rnd()*.2f;
 for(size_t i=0;i<wc;i++){ wl[i]=rnd()*.1f; wg[i]=rnd()*.1f; }
 for(size_t i=0;i<N;i++){ bl[i]=rnd()*.1f; bg[i]=rnd()*.1f; }
 pack4(wl,wg,p4,N,K);pack8(wl,wg,p8,N,K);
 ds_fused_linear_softsign_glu_f32_avx2_packed4(x,p4,bl,bg,y,M,N,K);double a=now();for(int i=0;i<iters;i++)ds_fused_linear_softsign_glu_f32_avx2_packed4(x,p4,bl,bg,y,M,N,K);double b=now();
 ds_fused_linear_softsign_glu_f32_avx2_m4n8(x,p8,bl,bg,y,M,N,K);double c=now();for(int i=0;i<iters;i++)ds_fused_linear_softsign_glu_f32_avx2_m4n8(x,p8,bl,bg,y,M,N,K);double d=now();
 double t4=(b-a)/iters,t8=(d-c)/iters,fl=4.0*M*N*K;printf("GLU bench M=%zu N=%zu K=%zu\n  packed4 %.3f ms %.1f GF/s\n  M6 4x8 %.3f ms %.1f GF/s speedup %.2fx\n",M,N,K,t4*1e3,fl/t4/1e9,t8*1e3,fl/t8/1e9,t4/t8);
 free(x);free(wl);free(wg);free(bl);free(bg);free(y);free(p4);free(p8);}
int main(){int f=0;f|=one(1,8,7);f|=one(3,16,31);f|=one(4,8,64);f|=one(5,24,65);f|=one(37,64,64);if(f)return 1;bench(128,512,512,30);return 0;}
