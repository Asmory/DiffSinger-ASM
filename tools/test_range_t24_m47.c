#define _POSIX_C_SOURCE 200809L
#include "dsasm_kernels.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static uint32_t st=0x47123456u;
static float fr(void){st=st*1664525u+1013904223u;return ((int32_t)(st>>8))/8388608.0f;}
static void *am(size_t n){void *p=0;return posix_memalign(&p,64,n)?0:p;}
static double ms(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e3+t.tv_nsec*1e-6;}
typedef void(*rf)(const float*,const float*,const float*,float*,size_t,size_t,size_t,size_t,size_t,size_t,size_t);
static int one(size_t K,size_t Cin,size_t T,size_t d,size_t t0,size_t tc,int bench){
 const size_t pad=(K/2)*d,Tp=T+2*pad,nx=Cin*Tp,nw=Cin*K*8,ny=8*T;
 float*x=am(nx*4),*w=am(nw*4),*b=am(8*4),*a=am(ny*4),*c=am(ny*4); if(!x||!w||!b||!a||!c)return 2;
 memset(x,0,nx*4); for(size_t ci=0;ci<Cin;ci++)for(size_t t=0;t<T;t++)x[ci*Tp+pad+t]=.1f*fr();
 for(size_t i=0;i<nw;i++)w[i]=.02f*fr(); for(size_t i=0;i<8;i++)b[i]=.01f*fr();
 for(size_t i=0;i<ny;i++)a[i]=c[i]=-123.0f;
 rf old=K==7?ds_conv1d_nct_f32_avx2_oc8_t8_range_k7:ds_conv1d_nct_f32_avx2_oc8_t8_range_k11;
 rf neo=K==7?ds_conv1d_nct_f32_avx2_oc4_t24_range_k7:ds_conv1d_nct_f32_avx2_oc4_t24_range_k11;
 old(x,w,b,a,Cin,K,t0,tc,Tp,d,T); neo(x,w,b,c,Cin,K,t0,tc,Tp,d,T);
 float ma=0;double ss=0;size_t bit=0,n=0; for(size_t oc=0;oc<8;oc++)for(size_t t=t0;t<t0+tc;t++){size_t i=oc*T+t;float z=fabsf(a[i]-c[i]);if(z>ma)ma=z;ss+=(double)z*z;if(memcmp(a+i,c+i,4))bit++;n++;}
 printf("M47 range K=%zu C=%zu T=%zu d=%zu t0=%zu tc=%zu max_abs=%.9g rmse=%.9g bitdiff=%zu\n",K,Cin,T,d,t0,tc,ma,sqrt(ss/n),bit);
 if(ma>1e-6f||bit)return 1;
 if(bench){enum{R=31};double ao[R],an[R];for(int r=0;r<R;r++){double q=ms();for(int z=0;z<8;z++)old(x,w,b,a,Cin,K,t0,tc,Tp,d,T);ao[r]=ms()-q;q=ms();for(int z=0;z<8;z++)neo(x,w,b,c,Cin,K,t0,tc,Tp,d,T);an[r]=ms()-q;}for(int i=0;i<R;i++)for(int j=i+1;j<R;j++){if(ao[j]<ao[i]){double z=ao[i];ao[i]=ao[j];ao[j]=z;}if(an[j]<an[i]){double z=an[i];an[i]=an[j];an[j]=z;}}printf("M47 range micro K=%zu old=%.3f ms t24=%.3f ms speedup=%.3fx\n",K,ao[R/2],an[R/2],ao[R/2]/an[R/2]);}
 free(x);free(w);free(b);free(a);free(c);return 0;
}
int main(void){int r=0;r|=one(7,32,12288,1,0,504,1);r|=one(7,16,24576,5,1008,504,1);r|=one(11,32,12288,3,504,504,1);r|=one(11,16,24576,5,1008,504,1);puts(r?"M47 range t24 FAIL":"M47 range t24 parity OK");return r;}
