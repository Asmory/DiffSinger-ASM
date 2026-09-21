#define _POSIX_C_SOURCE 200809L
#include "dsasm_vocoder.h"
#include "dsasm_threadpool.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct { char magic[8]; uint32_t version,C,K1,K2,T,pad1,dil1,pad2,dil2; float alpha; unsigned char reserved[16]; } Header;
static double now_ms(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return 1e3*t.tv_sec+1e-6*t.tv_nsec;}
static void *amalloc(size_t n){void *p=0;if(posix_memalign(&p,64,n))return 0;return p;}
static void stats(const float*a,const float*b,size_t n,float*ma,double*rm){float m=0;double ss=0;for(size_t i=0;i<n;i++){float d=fabsf(a[i]-b[i]);if(d>m)m=d;ss+=(double)d*d;}*ma=m;*rm=sqrt(ss/n);}
int main(int argc,char**argv){
    if(argc<2){fprintf(stderr,"usage: %s unit.dsvru33 [rounds]\n",argv[0]);return 2;}int rounds=argc>2?atoi(argv[2]):5;if(rounds<1)rounds=1;
    FILE*f=fopen(argv[1],"rb");if(!f){perror("fopen");return 2;}Header h;if(fread(&h,1,sizeof(h),f)!=sizeof(h)||memcmp(h.magic,"DSVRU33",7)||h.version!=1||h.C%4){puts("bad header");return 2;}
    size_t C=h.C,T=h.T,n1=C*C*h.K1,n2=C*C*h.K2,nx=C*T;
    float*b1=amalloc(C*4),*w1=amalloc(n1*4),*b2=amalloc(C*4),*w2=amalloc(n2*4),*x=amalloc(nx*4),*ref=amalloc(nx*4),*y=amalloc(nx*4);
    size_t wsn=ds_vocoder_resunit_workspace_floats(C,T,h.pad1,h.pad2);float*ws=amalloc(wsn*4);
    if(!b1||!w1||!b2||!w2||!x||!ref||!y||!ws){puts("oom");return 2;}
    if(fread(b1,4,C,f)!=C||fread(w1,4,n1,f)!=n1||fread(b2,4,C,f)!=C||fread(w2,4,n2,f)!=n2||fread(x,4,nx,f)!=nx||fread(ref,4,nx,f)!=nx){puts("short data");return 2;}fclose(f);
    printf("M33 PURE-ASM HiFi-GAN residual unit: C=%u T=%u k1=%u/d%u k2=%u/d%u alpha=%.4g nominal_MAC=%.3f M\n",h.C,h.T,h.K1,h.dil1,h.K2,h.dil2,h.alpha,(double)C*T*C*(h.K1+h.K2)/1e6);
    const int ths[]={1,2,4,8};float worst=0;double wrm=0;
    for(size_t q=0;q<4;q++){
        DSAsmThreadPool*pool=ds_threadpool_create(ths[q]);if(!pool){puts("pool fail");return 2;}
        ds_vocoder_resunit_f32_avx2(x,w1,b1,h.K1,h.pad1,h.dil1,w2,b2,h.K2,h.pad2,h.dil2,h.alpha,y,C,T,ws,pool);
        float ma;double rm;stats(y,ref,nx,&ma,&rm);if(ma>worst){worst=ma;wrm=rm;}
        double*v=calloc(rounds,sizeof(*v));for(int r=0;r<rounds;r++){double t0=now_ms();ds_vocoder_resunit_f32_avx2(x,w1,b1,h.K1,h.pad1,h.dil1,w2,b2,h.K2,h.pad2,h.dil2,h.alpha,y,C,T,ws,pool);v[r]=now_ms()-t0;}
        for(int i=0;i<rounds;i++){ for(int j=i+1;j<rounds;j++){ if(v[j]<v[i]){double z=v[i];v[i]=v[j];v[j]=z;} } }
        double med=v[rounds/2],g=2.0*(double)C*T*C*(h.K1+h.K2)/(med*1e6);
        printf("  ASM threads=%d median=%.3f ms %.2f nominal-GFLOP/s max_abs=%.8g rmse=%.8g cpus=",ths[q],med,g,ma,rm);for(size_t i=0;i<ds_threadpool_threads(pool);i++)printf("%s%d",i?",":"[",ds_threadpool_cpu_at(pool,i));puts("]");
        free(v);ds_threadpool_destroy(pool);
    }
    printf("M33 resunit parity: %s (max_abs=%.8g rmse=%.8g)\n",worst<1e-4?"OK":"FAIL",worst,wrm);return worst<1e-4?0:1;
}
