#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
void ds_linear_f32_avx2_packed4(const float*,const float*,const float*,float*,size_t,size_t,size_t);
void ds_linear_f32_avx2_m4n16(const float*,const float*,const float*,float*,size_t,size_t,size_t);
static uint32_t s=11;static float rnd(){s=s*1664525u+1013904223u;return ((s>>8)&0xffffff)/8388607.5f-1;}static double tm(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}static float md(float*a,float*b,size_t n){float m=0;for(size_t i=0;i<n;i++){float d=fabsf(a[i]-b[i]);if(d>m)m=d;}return m;}
static void p4(float*w,float*p,size_t N,size_t K){size_t z=0;for(size_t n=0;n<N;n+=4)for(size_t k=0;k<K;k+=8)for(int q=0;q<4;q++)for(int j=0;j<8;j++)p[z++]=w[(n+q)*K+k+j];}
static void p16(float*w,float*p,size_t N,size_t K){size_t z=0;for(size_t n=0;n<N;n+=16)for(size_t k=0;k<K;k++){for(int q=0;q<8;q++)p[z++]=w[(n+q)*K+k];for(int q=8;q<16;q++)p[z++]=w[(n+q)*K+k];}}
static void ref(float*x,float*w,float*b,float*y,size_t M,size_t N,size_t K){for(size_t m=0;m<M;m++)for(size_t n=0;n<N;n++){float a=b[n];for(size_t k=0;k<K;k++)a+=x[m*K+k]*w[n*K+k];y[m*N+n]=a;}}
int main(){int fail=0;size_t Ms[]={1,3,4,5,37};size_t Ks[]={7,31,64,65,64};for(int z=0;z<5;z++){size_t M=Ms[z],N=32,K=Ks[z],xc=M*K,wc=N*K,yc=M*N;float*x=malloc(xc*4),*w=malloc(wc*4),*b=malloc(N*4),*a=malloc(yc*4),*y=malloc(yc*4),*p=malloc(wc*4);for(size_t i=0;i<xc;i++)x[i]=rnd()*.2f;for(size_t i=0;i<wc;i++)w[i]=rnd()*.1f;for(size_t i=0;i<N;i++)b[i]=rnd()*.1f;p16(w,p,N,K);ref(x,w,b,a,M,N,K);ds_linear_f32_avx2_m4n16(x,p,b,y,M,N,K);float d=md(a,y,yc);printf("M6 linear M=%zu K=%zu max=%g %s\n",M,K,d,d<3e-4?"OK":"FAIL");fail|=d>=3e-4;free(x);free(w);free(b);free(a);free(y);free(p);}if(fail)return 1;size_t M=128,N=512,K=512,xc=M*K,wc=N*K,yc=M*N;float*x=aligned_alloc(64,xc*4),*w=aligned_alloc(64,wc*4),*b=aligned_alloc(64,N*4),*y=aligned_alloc(64,yc*4),*a=aligned_alloc(64,wc*4),*c=aligned_alloc(64,wc*4);for(size_t i=0;i<xc;i++)x[i]=rnd();for(size_t i=0;i<wc;i++)w[i]=rnd()*.1f;for(size_t i=0;i<N;i++)b[i]=rnd();p4(w,a,N,K);p16(w,c,N,K);int it=40;ds_linear_f32_avx2_packed4(x,a,b,y,M,N,K);double t0=tm();for(int i=0;i<it;i++)ds_linear_f32_avx2_packed4(x,a,b,y,M,N,K);double t1=tm();for(int i=0;i<it;i++)ds_linear_f32_avx2_m4n16(x,c,b,y,M,N,K);double t2=tm();double f=2.0*M*N*K;printf("linear packed4 %.3f ms %.1f GF/s; M6 4x16 %.3f ms %.1f GF/s speed %.2fx\n",(t1-t0)*1e3/it,f/((t1-t0)/it)/1e9,(t2-t1)*1e3/it,f/((t2-t1)/it)/1e9,(t1-t0)/(t2-t1));return 0;}
