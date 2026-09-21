#define _POSIX_C_SOURCE 200809L
#include "dsasm_kernels.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static uint32_t st=0x46123456u;
static float fr(void){st=st*1664525u+1013904223u;return ((int32_t)(st>>8))/8388608.0f;}
static void *am(size_t n){void *p=0;return posix_memalign(&p,64,n)?0:p;}
static double now_ms(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e3+t.tv_nsec*1e-6;}
static int one(size_t Cin,size_t Cout,size_t T,size_t d,int bench){
 const size_t K=11,pad=5*d,Tp=T+2*pad,nx=Cin*Tp,nw=Cout*Cin*K,ny=Cout*T;
 float*x=am(nx*4),*w=am(nw*4),*b=am(Cout*4),*a=am(ny*4),*c=am(ny*4);if(!x||!w||!b||!a||!c)return 2;
 memset(x,0,nx*4);for(size_t ci=0;ci<Cin;ci++)for(size_t t=0;t<T;t++)x[ci*Tp+pad+t]=.1f*fr();
 for(size_t i=0;i<nw;i++)w[i]=.02f*fr();for(size_t i=0;i<Cout;i++)b[i]=.01f*fr();
 ds_conv1d_nct_f32_avx2_oc8_t8_k11(x,w,b,a,Cin,Cout,K,T,Tp,d);
 ds_conv1d_nct_f32_avx2_oc4_t24_k11(x,w,b,c,Cin,Cout,K,T,Tp,d);
 float ma=0;double ss=0;size_t bitdiff=0;for(size_t i=0;i<ny;i++){float z=fabsf(a[i]-c[i]);if(z>ma)ma=z;ss+=(double)z*z;if(memcmp(a+i,c+i,4))bitdiff++;}
 printf("M46 K11 C=%zu O=%zu T=%zu d=%zu max_abs=%.9g rmse=%.9g bitdiff=%zu\n",Cin,Cout,T,d,ma,sqrt(ss/ny),bitdiff);
 if(ma>1e-6f)return 1;
 if(bench){enum{R=7};double old[R],neo[R];for(int r=0;r<R;r++){double q=now_ms();ds_conv1d_nct_f32_avx2_oc8_t8_k11(x,w,b,a,Cin,Cout,K,T,Tp,d);old[r]=now_ms()-q;q=now_ms();ds_conv1d_nct_f32_avx2_oc4_t24_k11(x,w,b,c,Cin,Cout,K,T,Tp,d);neo[r]=now_ms()-q;}for(int i=0;i<R;i++)for(int j=i+1;j<R;j++){if(old[j]<old[i]){double z=old[i];old[i]=old[j];old[j]=z;}if(neo[j]<neo[i]){double z=neo[i];neo[i]=neo[j];neo[j]=z;}}printf("M46 K11 micro old=%.3f ms t24=%.3f ms speedup=%.3fx\n",old[R/2],neo[R/2],old[R/2]/neo[R/2]);}
 free(x);free(w);free(b);free(a);free(c);return 0;
}
int main(void){int r=0;r|=one(8,8,48,1,0);r|=one(64,64,6144,1,1);r|=one(64,64,6144,3,1);r|=one(64,64,6144,5,1);r|=one(256,256,384,1,1);puts(r?"M46 K11 t24 FAIL":"M46 K11 t24 parity OK");return r;}
