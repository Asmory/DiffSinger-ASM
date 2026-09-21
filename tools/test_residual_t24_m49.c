#define _POSIX_C_SOURCE 200809L
#include "dsasm_kernels.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static uint32_t st=0x49123456u;
static float fr(void){st=st*1664525u+1013904223u;return ((int32_t)(st>>8))/8388608.0f;}
static void *am(size_t n){void *p=0;return posix_memalign(&p,64,n)?0:p;}
static double now_ms(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e3+t.tv_nsec*1e-6;}
static void sort7(double*a){for(int i=0;i<7;i++)for(int j=i+1;j<7;j++)if(a[j]<a[i]){double z=a[i];a[i]=a[j];a[j]=z;}}
static int one(int K,size_t C,size_t T,size_t d,int bench){
 const size_t pad=(size_t)(K/2)*d,Tp=T+2*pad,nx=C*Tp,nw=C*C*(size_t)K,ny=C*T;
 float*x=am(nx*4),*w=am(nw*4),*b=am(C*4),*r=am(ny*4),*sep=am(ny*4),*fus=am(ny*4);if(!x||!w||!b||!r||!sep||!fus)return 2;
 memset(x,0,nx*4);for(size_t c=0;c<C;c++)for(size_t t=0;t<T;t++)x[c*Tp+pad+t]=.1f*fr();
 for(size_t i=0;i<nw;i++)w[i]=.02f*fr();for(size_t i=0;i<C;i++)b[i]=.01f*fr();for(size_t i=0;i<ny;i++)r[i]=.03f*fr();
 if(K==3)ds_conv1d_nct_f32_avx2_oc4_t24_k3(x,w,b,sep,C,C,K,T,Tp,d);
 else if(K==7)ds_conv1d_nct_f32_avx2_oc4_t24_k7(x,w,b,sep,C,C,K,T,Tp,d);
 else ds_conv1d_nct_f32_avx2_oc4_t24_k11(x,w,b,sep,C,C,K,T,Tp,d);
 ds_add_f32_avx2(sep,r,sep,ny);
 if(K==3)ds_conv1d_nct_f32_avx2_oc4_t24_k3_residual(x,w,b,fus,C,C,K,T,Tp,d,r);
 else if(K==7)ds_conv1d_nct_f32_avx2_oc4_t24_k7_residual(x,w,b,fus,C,C,K,T,Tp,d,r);
 else ds_conv1d_nct_f32_avx2_oc4_t24_k11_residual(x,w,b,fus,C,C,K,T,Tp,d,r);
 float ma=0;size_t bd=0;for(size_t i=0;i<ny;i++){float z=fabsf(sep[i]-fus[i]);if(z>ma)ma=z;if(memcmp(sep+i,fus+i,4))bd++;}
 printf("M49 residual K=%d C=%zu T=%zu d=%zu max_abs=%.9g bitdiff=%zu\n",K,C,T,d,ma,bd);if(ma>1e-6f)return 1;
 if(bench){double a[7],f[7];for(int q=0;q<7;q++){double z=now_ms();if(K==3)ds_conv1d_nct_f32_avx2_oc4_t24_k3(x,w,b,sep,C,C,K,T,Tp,d);else if(K==7)ds_conv1d_nct_f32_avx2_oc4_t24_k7(x,w,b,sep,C,C,K,T,Tp,d);else ds_conv1d_nct_f32_avx2_oc4_t24_k11(x,w,b,sep,C,C,K,T,Tp,d);ds_add_f32_avx2(sep,r,sep,ny);a[q]=now_ms()-z;z=now_ms();if(K==3)ds_conv1d_nct_f32_avx2_oc4_t24_k3_residual(x,w,b,fus,C,C,K,T,Tp,d,r);else if(K==7)ds_conv1d_nct_f32_avx2_oc4_t24_k7_residual(x,w,b,fus,C,C,K,T,Tp,d,r);else ds_conv1d_nct_f32_avx2_oc4_t24_k11_residual(x,w,b,fus,C,C,K,T,Tp,d,r);f[q]=now_ms()-z;}sort7(a);sort7(f);printf("M49 micro K=%d separate=%.3f fused=%.3f speedup=%.3fx\n",K,a[3],f[3],a[3]/f[3]);}
 free(x);free(w);free(b);free(r);free(sep);free(fus);return 0;
}
int main(void){int q=0;
 q|=one(3,256,384,1,1);q|=one(3,128,3072,1,1);q|=one(3,64,6144,1,1);
 q|=one(7,256,384,1,1);q|=one(7,128,3072,1,1);q|=one(7,64,6144,1,1);
 q|=one(11,256,384,1,1);q|=one(11,128,3072,1,1);q|=one(11,64,6144,1,1);
 puts(q?"M49 residual FAIL":"M49 residual parity OK");return q;}
