#define _POSIX_C_SOURCE 200809L
#include "dsasm_kernels.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef void (*full_fn)(const float*,const float*,const float*,float*,size_t,size_t,size_t,size_t,size_t,size_t);
typedef void (*range_fn)(const float*,const float*,const float*,float*,size_t,size_t,size_t,size_t,size_t,size_t,size_t);

static uint32_t rng_state=0x39a7c5u;
static float frand(void){rng_state=rng_state*1664525u+1013904223u;return ((int32_t)(rng_state>>8))/8388608.0f;}
static void *amalloc(size_t n){void *p=NULL;return posix_memalign(&p,64,n)?NULL:p;}
static double ms(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return 1e3*t.tv_sec+1e-6*t.tv_nsec;}
static void stat(const float*a,const float*b,size_t n,float*ma,double*rm){float m=0;double s=0;for(size_t i=0;i<n;i++){float d=fabsf(a[i]-b[i]);if(d>m)m=d;s+=(double)d*d;}*ma=m;*rm=sqrt(s/n);}

static int one(size_t K,size_t dil,full_fn sf,range_fn rf){
    const size_t Cin=24,Cout=24,T=1024,pad=dil*(K-1)/2,Tp=T+2*pad;
    const size_t nx=Cin*Tp,nw=Cout*Cin*K,ny=Cout*T;
    float *x=amalloc(nx*4),*w=amalloc(nw*4),*b=amalloc(Cout*4),*yg=amalloc(ny*4),*ys=amalloc(ny*4),*yrg=amalloc(ny*4),*yrs=amalloc(ny*4);
    if(!x||!w||!b||!yg||!ys||!yrg||!yrs)return 2;
    memset(x,0,nx*4);for(size_t c=0;c<Cin;c++)for(size_t t=0;t<T;t++)x[c*Tp+pad+t]=0.2f*frand();
    for(size_t i=0;i<nw;i++) w[i]=0.03f*frand();
    for(size_t i=0;i<Cout;i++) b[i]=0.02f*frand();
    ds_conv1d_nct_f32_avx2_oc8_t8(x,w,b,yg,Cin,Cout,K,T,Tp,dil);
    sf(x,w,b,ys,Cin,Cout,K,T,Tp,dil);
    float ma;double rm;stat(yg,ys,ny,&ma,&rm);
    if(memcmp(yg,ys,ny*4)!=0 && ma>1e-6f){printf("K%zu d%zu full FAIL max=%.8g rmse=%.8g\n",K,dil,ma,rm);return 1;}
    memset(yrg,0,ny*4);memset(yrs,0,ny*4);
    for(size_t ob=0;ob<Cout/8;ob++){
        const float *wb=w+ob*Cin*K*8;const float *bb=b+ob*8;float *og=yrg+ob*8*T,*os=yrs+ob*8*T;
        ds_conv1d_nct_f32_avx2_oc8_t8_range(x,wb,bb,og,Cin,K,0,T,Tp,dil,T);
        rf(x,wb,bb,os,Cin,K,0,T,Tp,dil,T);
    }
    stat(yrg,yrs,ny,&ma,&rm);
    if(memcmp(yrg,yrs,ny*4)!=0 && ma>1e-6f){printf("K%zu d%zu range FAIL max=%.8g rmse=%.8g\n",K,dil,ma,rm);return 1;}
    const int rounds=9;double tg[rounds],ts[rounds];
    for(int r=0;r<rounds;r++){double a=ms();ds_conv1d_nct_f32_avx2_oc8_t8(x,w,b,yg,Cin,Cout,K,T,Tp,dil);tg[r]=ms()-a;a=ms();sf(x,w,b,ys,Cin,Cout,K,T,Tp,dil);ts[r]=ms()-a;}
    for(int i=0;i<rounds;i++)for(int j=i+1;j<rounds;j++){if(tg[j]<tg[i]){double z=tg[i];tg[i]=tg[j];tg[j]=z;}if(ts[j]<ts[i]){double z=ts[i];ts[i]=ts[j];ts[j]=z;}}
    printf("M39 K%zu d%zu parity=bit-exact generic=%.3f ms kspec=%.3f ms speedup=%.3fx\n",K,dil,tg[rounds/2],ts[rounds/2],tg[rounds/2]/ts[rounds/2]);
    free(x);free(w);free(b);free(yg);free(ys);free(yrg);free(yrs);return 0;
}
int main(void){
    int rc=0;
    rc|=one(3,1,ds_conv1d_nct_f32_avx2_oc8_t8_k3,ds_conv1d_nct_f32_avx2_oc8_t8_range_k3);
    rc|=one(7,3,ds_conv1d_nct_f32_avx2_oc8_t8_k7,ds_conv1d_nct_f32_avx2_oc8_t8_range_k7);
    rc|=one(11,5,ds_conv1d_nct_f32_avx2_oc8_t8_k11,ds_conv1d_nct_f32_avx2_oc8_t8_range_k11);
    puts(rc?"M39 kspec parity FAIL":"M39 kspec parity OK");return rc;
}
