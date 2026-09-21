#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void ds_atan_glu_f32_avx2(const float*,float*,size_t,size_t);

static uint32_t st=1;
static float rnd(void){st=st*1664525u+1013904223u; return ((st>>8)*(1.0f/16777216.0f)*2.0f-1.0f);}
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}

static int one(size_t M,size_t N){
 float *x=malloc(M*2*N*4),*y=malloc(M*N*4),*r=malloc(M*N*4);
 for(size_t i=0;i<M;i++)for(size_t j=0;j<N;j++){
   float l=rnd()*3.0f;
   float g;
   switch((i*N+j)%7){case 0:g=rnd()*.3f;break;case 1:g=rnd()*1.2f;break;case 2:g=rnd()*4.f;break;case 3:g=rnd()*30.f;break;case 4:g=0.f;break;case 5:g=1000.f*rnd();break;default:g=rnd()*2.5f;}
   x[i*2*N+j]=l;x[i*2*N+N+j]=g;r[i*N+j]=l*atanf(g);
 }
 ds_atan_glu_f32_avx2(x,y,M,N);
 float ma=0,mr=0; for(size_t i=0;i<M*N;i++){float d=fabsf(y[i]-r[i]);if(d>ma)ma=d;float q=d/fmaxf(fabsf(r[i]),1e-5f);if(q>mr)mr=q;}
 printf("M7 ATanGLU M=%zu N=%zu max_abs=%g max_rel=%g %s\n",M,N,ma,mr,ma<2e-5f?"OK":"FAIL");
 free(x);free(y);free(r);return ma>=2e-5f;
}
int main(void){
#if defined(__x86_64__)
 if(!__builtin_cpu_supports("avx2")||!__builtin_cpu_supports("fma")){puts("AVX2+FMA required");return 2;}
#endif
 int f=0;f|=one(1,8);f|=one(3,64);f|=one(37,1024);
 size_t M=128,N=1024;float*x=malloc(M*2*N*4),*y=malloc(M*N*4);for(size_t i=0;i<M*2*N;i++)x[i]=rnd()*2;
 double a=now();for(int i=0;i<200;i++)ds_atan_glu_f32_avx2(x,y,M,N);double b=now();printf("ATanGLU activation only: %.4f ms/iter\n",(b-a)*5.0);
 free(x);free(y);return f?1:0;
}
