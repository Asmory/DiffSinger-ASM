#define _POSIX_C_SOURCE 200809L
#include "dsasm_kernels.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint32_t st=0x12345678u;
static float frand1(void){st=st*1664525u+1013904223u;return ((st>>8)&0xffffu)/32768.0f-1.0f;}
static double now_ms(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec*1e3+ts.tv_nsec*1e-6;}

typedef void (*rrfn)(const float*,const float*,const float*,float*,size_t,size_t,size_t,size_t,size_t,size_t,size_t,const float*);
static int one(size_t K,size_t C,size_t T, rrfn fn){
  const size_t pad=(K-1)/2,Tp=T+2*pad,tile=2016,O=8;
  const size_t nx=C*Tp,nw=C*K*O,ny=O*T;
  float *x=aligned_alloc(64,(nx*4+63)&~63ull),*w=aligned_alloc(64,(nw*4+63)&~63ull);
  float *b=aligned_alloc(64,64),*r=aligned_alloc(64,(ny*4+63)&~63ull);
  float *a=aligned_alloc(64,(ny*4+63)&~63ull),*n=aligned_alloc(64,(ny*4+63)&~63ull);
  if(!x||!w||!b||!r||!a||!n)return 2;
  for(size_t c=0;c<C;c++){float*d=x+c*Tp;for(size_t i=0;i<Tp;i++)d[i]=(i<pad||i>=pad+T)?0.0f:frand1();}
  for(size_t i=0;i<nw;i++) w[i]=frand1()*0.05f;
  for(size_t i=0;i<O;i++) b[i]=frand1()*0.1f;
  for(size_t i=0;i<ny;i++) r[i]=frand1();
  memset(a,0,ny*4);memset(n,0,ny*4);
  for(size_t t0=0;t0<T;t0+=tile){size_t tc=T-t0;if(tc>tile)tc=tile;ds_conv1d_nct_f32_avx2_oc8_t8_range_residual(x,w,b,a,C,K,t0,tc,Tp,1,T,r);fn(x,w,b,n,C,K,t0,tc,Tp,1,T,r);}
  float ma=0;size_t diff=0;for(size_t i=0;i<ny;i++){float d=fabsf(a[i]-n[i]);if(d>ma)ma=d;if(memcmp(a+i,n+i,4))diff++;}
  const int it=9;double to=0,tn=0;
  for(int z=0;z<it;z++){
    double q;
    if((z&1)==0){
      q=now_ms();for(size_t t0=0;t0<T;t0+=tile){size_t tc=T-t0;if(tc>tile)tc=tile;ds_conv1d_nct_f32_avx2_oc8_t8_range_residual(x,w,b,a,C,K,t0,tc,Tp,1,T,r);}to+=now_ms()-q;
      q=now_ms();for(size_t t0=0;t0<T;t0+=tile){size_t tc=T-t0;if(tc>tile)tc=tile;fn(x,w,b,n,C,K,t0,tc,Tp,1,T,r);}tn+=now_ms()-q;
    }else{
      q=now_ms();for(size_t t0=0;t0<T;t0+=tile){size_t tc=T-t0;if(tc>tile)tc=tile;fn(x,w,b,n,C,K,t0,tc,Tp,1,T,r);}tn+=now_ms()-q;
      q=now_ms();for(size_t t0=0;t0<T;t0+=tile){size_t tc=T-t0;if(tc>tile)tc=tile;ds_conv1d_nct_f32_avx2_oc8_t8_range_residual(x,w,b,a,C,K,t0,tc,Tp,1,T,r);}to+=now_ms()-q;
    }
  }
  printf("M58 range-residual K%zu C=%zu T=%zu tile=%zu max_abs=%.9g bitdiff=%zu old=%.3fms new=%.3fms speedup=%.3fx\n",K,C,T,tile,ma,diff,to/it,tn/it,to/tn);
  free(x);free(w);free(b);free(r);free(a);free(n);return (ma==0.0f&&diff==0)?0:1;
}
int main(void){int rc=0;rc|=one(3,128,24576,ds_conv1d_nct_f32_avx2_oc4_t24_range_k3_residual);rc|=one(7,128,24576,ds_conv1d_nct_f32_avx2_oc4_t24_range_k7_residual);rc|=one(11,128,24576,ds_conv1d_nct_f32_avx2_oc4_t24_range_k11_residual);rc|=one(3,64,49152,ds_conv1d_nct_f32_avx2_oc4_t24_range_k3_residual);rc|=one(3,32,98304,ds_conv1d_nct_f32_avx2_oc4_t24_range_k3_residual);rc|=one(7,32,98304,ds_conv1d_nct_f32_avx2_oc4_t24_range_k7_residual);rc|=one(11,32,98304,ds_conv1d_nct_f32_avx2_oc4_t24_range_k11_residual);return rc;}
