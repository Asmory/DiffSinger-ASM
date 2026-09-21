#define _POSIX_C_SOURCE 200809L
#include "dsasm_kernels.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint32_t rng=1;
static float frandv(void){rng=rng*1664525u+1013904223u; return ((int32_t)(rng>>8)/(float)0x7fffff)*0.2f;}
static double now_sec(void){struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}
static void *amalloc(size_t n){void*p=0;if(posix_memalign(&p,64,n))return 0;return p;}

static void pack16(const float *w,float *p,size_t N,size_t K){
    for(size_t nb=0;nb<N/16;nb++) for(size_t k=0;k<K;k++) for(size_t j=0;j<16;j++)
        p[(nb*K+k)*16+j]=w[(nb*16+j)*K+k];
}
static float maxdiff(const float*a,const float*b,size_t n){float m=0;for(size_t i=0;i<n;i++){float d=fabsf(a[i]-b[i]);if(d>m)m=d;}return m;}

static int one(size_t M,size_t N,size_t K,size_t kb){
    float*x=amalloc(M*K*4),*wr=amalloc(N*K*4),*wp=amalloc(N*K*4),*b=amalloc(N*4),*a=amalloc(M*N*4),*z=amalloc(M*N*4);
    if(!x||!wr||!wp||!b||!a||!z)return 1;
    for(size_t i=0;i<M*K;i++)x[i]=frandv();for(size_t i=0;i<N*K;i++)wr[i]=frandv();for(size_t i=0;i<N;i++)b[i]=frandv();pack16(wr,wp,N,K);
    ds_linear_f32_avx2_m4n16_strided(x,wp,b,a,M,N,K,N);
    memset(z,0,M*N*4);
    for(size_t nb=0;nb<N;nb+=16){
        const float *wn=wp+(nb/16)*K*16;
        for(size_t k0=0;k0<K;k0+=kb){size_t kl=K-k0;if(kl>kb)kl=kb;
            ds_linear_f32_avx2_n16_kblock_accum(x+k0,wn+k0*16,b+nb,z+nb,M,K,kl,N,k0==0);
        }
    }
    float d=maxdiff(a,z,M*N); printf("M15 kblock M=%zu N=%zu K=%zu kb=%zu max_abs=%g %s\n",M,N,K,kb,d,d==0?"OK":"FAIL");
    free(x);free(wr);free(wp);free(b);free(a);free(z);return d!=0;
}
static void bench(size_t M,size_t N,size_t K,size_t kb){
    float*x=amalloc(M*K*4),*wr=amalloc(N*K*4),*wp=amalloc(N*K*4),*b=amalloc(N*4),*a=amalloc(M*N*4),*z=amalloc(M*N*4);
    for(size_t i=0;i<M*K;i++)x[i]=frandv();for(size_t i=0;i<N*K;i++)wr[i]=frandv();for(size_t i=0;i<N;i++)b[i]=frandv();pack16(wr,wp,N,K);
    const int it=80;double t0=now_sec();for(int q=0;q<it;q++)ds_linear_f32_avx2_m4n16_strided(x,wp,b,a,M,N,K,N);double t1=now_sec();
    double t2=now_sec();for(int q=0;q<it;q++)for(size_t nb=0;nb<N;nb+=16){const float*wn=wp+(nb/16)*K*16;for(size_t k0=0;k0<K;k0+=kb){size_t kl=K-k0;if(kl>kb)kl=kb;ds_linear_f32_avx2_n16_kblock_accum(x+k0,wn+k0*16,b+nb,z+nb,M,K,kl,N,k0==0);}}double t3=now_sec();
    double old=(t1-t0)*1e3/it,blk=(t3-t2)*1e3/it;printf("M15 bench M=%zu N=%zu K=%zu kb=%zu old=%.3f ms kblock=%.3f ms speedup=%.3fx\n",M,N,K,kb,old,blk,old/blk);
    free(x);free(wr);free(wp);free(b);free(a);free(z);
}
int main(void){
    int bad=0;bad|=one(1,16,7,4);bad|=one(3,32,31,8);bad|=one(13,64,65,16);bad|=one(32,64,1024,128);bad|=one(64,64,1024,256);bad|=one(64,64,1024,512);
    bench(32,64,1024,128);bench(32,64,1024,256);bench(32,64,1024,512);bench(64,64,1024,128);bench(64,64,1024,256);bench(64,64,1024,512);return bad?1:0;
}
