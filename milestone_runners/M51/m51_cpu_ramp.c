#define _GNU_SOURCE
#include <immintrin.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct { long ms; int id; } Arg;
static volatile float sinkv;
static double now_ms(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec*1000.0 + ts.tv_nsec/1e6; }
static void *worker(void *vp){
    Arg *a=(Arg*)vp; double end=now_ms()+a->ms;
    __m256 x=_mm256_set1_ps(1.00001f+(float)a->id*1e-7f);
    __m256 y=_mm256_set1_ps(0.99991f);
    __m256 z=_mm256_set1_ps(0.00003f);
    while(now_ms()<end){
        for(int i=0;i<2048;i++){
            x=_mm256_fmadd_ps(x,y,z); x=_mm256_fmadd_ps(x,y,z);
            x=_mm256_fmadd_ps(x,y,z); x=_mm256_fmadd_ps(x,y,z);
            x=_mm256_fmadd_ps(x,y,z); x=_mm256_fmadd_ps(x,y,z);
            x=_mm256_fmadd_ps(x,y,z); x=_mm256_fmadd_ps(x,y,z);
        }
    }
    float v[8]; _mm256_storeu_ps(v,x); sinkv += v[a->id&7]; return NULL;
}
int main(int argc,char **argv){
    long ms=argc>1?strtol(argv[1],0,10):250; int n=argc>2?atoi(argv[2]):4;
    if(ms<1)ms=1; if(n<1)n=1; if(n>64)n=64;
    pthread_t *th=calloc((size_t)n,sizeof(*th)); Arg *args=calloc((size_t)n,sizeof(*args));
    double t0=now_ms();
    for(int i=0;i<n;i++){args[i]=(Arg){ms,i}; if(pthread_create(&th[i],0,worker,&args[i])) return 2;}
    for(int i=0;i<n;i++) pthread_join(th[i],0);
    double dt=now_ms()-t0; printf("M51 CPU RAMP: threads=%d requested_ms=%ld elapsed_ms=%.3f sink=%g\n",n,ms,dt,(double)sinkv);
    free(th); free(args); return 0;
}
