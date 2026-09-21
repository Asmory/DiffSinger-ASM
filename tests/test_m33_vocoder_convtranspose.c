#define _POSIX_C_SOURCE 200809L
#include "dsasm_vocoder.h"
#include "dsasm_threadpool.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct { char magic[8]; uint32_t version,Cin,Cout,K,Tin,Tout,pad,dilation,stride,group; unsigned char reserved[16]; } Header;
static double now_ms(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return 1e3*t.tv_sec+1e-6*t.tv_nsec;}
static void *amalloc(size_t n){void*p=0;if(posix_memalign(&p,64,n))return 0;return p;}
static void stats(const float*a,const float*b,size_t n,float*ma,double*rm){float m=0;double ss=0;for(size_t i=0;i<n;i++){float d=fabsf(a[i]-b[i]);if(d>m)m=d;ss+=(double)d*d;}*ma=m;*rm=sqrt(ss/n);}
int main(int argc,char**argv){
    if(argc<2){fprintf(stderr,"usage: %s layer.dsvct33 [rounds]\n",argv[0]);return 2;}int rounds=argc>2?atoi(argv[2]):5;if(rounds<1)rounds=1;
    FILE*f=fopen(argv[1],"rb");if(!f){perror("fopen");return 2;}Header h;if(fread(&h,1,sizeof(h),f)!=sizeof(h)||memcmp(h.magic,"DSVCT33",7)||h.version!=1||h.dilation!=1||h.group!=1){puts("bad header");return 2;}
    size_t nb=h.Cout,nw=(size_t)h.Cout*h.Cin*h.K,nx=(size_t)h.Cin*h.Tin,ny=(size_t)h.Cout*h.Tout;
    float*b=amalloc(nb*4),*w=amalloc(nw*4),*x=amalloc(nx*4),*ref=amalloc(ny*4),*y=amalloc(ny*4);if(!b||!w||!x||!ref||!y){puts("oom");return 2;}
    if(fread(b,4,nb,f)!=nb||fread(w,4,nw,f)!=nw||fread(x,4,nx,f)!=nx||fread(ref,4,ny,f)!=ny){puts("short data");return 2;}fclose(f);
    printf("M33 PURE-ASM ConvTranspose1d: Cin=%u Cout=%u K=%u Tin=%u Tout=%u stride=%u pad=%u nominal_MAC=%.3f M\n",h.Cin,h.Cout,h.K,h.Tin,h.Tout,h.stride,h.pad,(double)h.Cin*h.Tin*h.Cout*h.K/1e6);
    const int ths[]={1,2,4,8};float worst=0;double wrm=0;
    for(size_t q=0;q<4;q++){
        DSAsmThreadPool*pool=ds_threadpool_create(ths[q]);if(!pool){puts("pool fail");return 2;}
        ds_vocoder_convtranspose1d_f32_avx2(x,w,b,y,h.Cin,h.Cout,h.K,h.Tin,h.pad,h.stride,pool);float ma;double rm;stats(y,ref,ny,&ma,&rm);if(ma>worst){worst=ma;wrm=rm;}
        double*v=calloc(rounds,sizeof(*v));for(int r=0;r<rounds;r++){double t0=now_ms();ds_vocoder_convtranspose1d_f32_avx2(x,w,b,y,h.Cin,h.Cout,h.K,h.Tin,h.pad,h.stride,pool);v[r]=now_ms()-t0;}
        for(int i=0;i<rounds;i++){ for(int j=i+1;j<rounds;j++){ if(v[j]<v[i]){double z=v[i];v[i]=v[j];v[j]=z;} } }
        double med=v[rounds/2],g=2.0*(double)h.Cin*h.Tin*h.Cout*h.K/(med*1e6);
        printf("  ASM threads=%d median=%.3f ms %.2f nominal-GFLOP/s max_abs=%.8g rmse=%.8g cpus=",ths[q],med,g,ma,rm);for(size_t i=0;i<ds_threadpool_threads(pool);i++)printf("%s%d",i?",":"[",ds_threadpool_cpu_at(pool,i));puts("]");free(v);ds_threadpool_destroy(pool);
    }
    printf("M33 ConvTranspose parity: %s (max_abs=%.8g rmse=%.8g)\n",worst<1e-4?"OK":"FAIL",worst,wrm);return worst<1e-4?0:1;
}
