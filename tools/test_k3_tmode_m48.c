#define _POSIX_C_SOURCE 200809L
#include "dsasm_kernels.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static uint32_t st=0x48123456u;
static float fr(void){st=st*1664525u+1013904223u;return ((int32_t)(st>>8))/8388608.0f;}
static void *am(size_t n){void *p=0;return posix_memalign(&p,64,n)?0:p;}
static double now_ms(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e3+t.tv_nsec*1e-6;}
static void sort7(double *a){for(int i=0;i<7;i++)for(int j=i+1;j<7;j++)if(a[j]<a[i]){double z=a[i];a[i]=a[j];a[j]=z;}}
static int one(size_t Cin,size_t Cout,size_t T,size_t d,int bench){
 const size_t K=3,pad=d,Tp=T+2*pad,nx=Cin*Tp,nw=Cout*Cin*K,ny=Cout*T;
 float*x=am(nx*4),*w=am(nw*4),*b=am(Cout*4),*old=am(ny*4),*t16=am(ny*4),*t24=am(ny*4);if(!x||!w||!b||!old||!t16||!t24)return 2;
 memset(x,0,nx*4);for(size_t ci=0;ci<Cin;ci++)for(size_t t=0;t<T;t++)x[ci*Tp+pad+t]=.1f*fr();
 for(size_t i=0;i<nw;i++)w[i]=.02f*fr();for(size_t i=0;i<Cout;i++)b[i]=.01f*fr();
 ds_conv1d_nct_f32_avx2_oc8_t8_k3(x,w,b,old,Cin,Cout,K,T,Tp,d);
 ds_conv1d_nct_f32_avx2_oc4_t16_k3(x,w,b,t16,Cin,Cout,K,T,Tp,d);
 ds_conv1d_nct_f32_avx2_oc4_t24_k3(x,w,b,t24,Cin,Cout,K,T,Tp,d);

 float ma16=0,ma24=0;double ss16=0,ss24=0;size_t bd16=0,bd24=0;
 for(size_t i=0;i<ny;i++){float z=fabsf(old[i]-t16[i]);if(z>ma16)ma16=z;ss16+=(double)z*z;if(memcmp(old+i,t16+i,4))bd16++;z=fabsf(old[i]-t24[i]);if(z>ma24)ma24=z;ss24+=(double)z*z;if(memcmp(old+i,t24+i,4))bd24++;}
 printf("M48 K3 C=%zu O=%zu T=%zu d=%zu t16[max=%.9g bitdiff=%zu] t24[max=%.9g bitdiff=%zu]\n",Cin,Cout,T,d,ma16,bd16,ma24,bd24);
 if(ma16>1e-6f||ma24>1e-6f)return 1;
 if(bench){double a[7],c[7],e[7];for(int r=0;r<7;r++){double q=now_ms();ds_conv1d_nct_f32_avx2_oc8_t8_k3(x,w,b,old,Cin,Cout,K,T,Tp,d);a[r]=now_ms()-q;q=now_ms();ds_conv1d_nct_f32_avx2_oc4_t16_k3(x,w,b,t16,Cin,Cout,K,T,Tp,d);c[r]=now_ms()-q;q=now_ms();ds_conv1d_nct_f32_avx2_oc4_t24_k3(x,w,b,t24,Cin,Cout,K,T,Tp,d);e[r]=now_ms()-q;}sort7(a);sort7(c);sort7(e);printf("M48 K3 micro old=%.3f t16=%.3f (%.3fx) t24=%.3f (%.3fx) ms\n",a[3],c[3],a[3]/c[3],e[3],a[3]/e[3]);}
 free(x);free(w);free(b);free(old);free(t16);free(t24);return 0;
}
int main(void){int r=0;r|=one(8,8,48,1,0);r|=one(128,128,3072,1,1);r|=one(128,128,3072,3,1);r|=one(128,128,3072,5,1);r|=one(64,64,6144,1,1);puts(r?"M48 K3 FAIL":"M48 K3 parity OK");return r;}
