#define _POSIX_C_SOURCE 200809L
#include "dsasm_vocoder.h"
#include "dsasm_threadpool.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    char magic[8]; uint32_t version,Cin,Cout,K,Tin,Tout,pad,dilation,stride,group;
    unsigned char reserved[16];
} Header;

static double now_ms(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return 1e3*t.tv_sec+1e-6*t.tv_nsec;}
static void *amalloc(size_t n){void *p=0;if(posix_memalign(&p,64,n))return 0;return p;}
static float w_at(const float *wp,size_t oc,size_t ci,size_t k,size_t Cin,size_t K){
    size_t ob=oc/4,j=oc&3;return wp[(((ob*Cin+ci)*K+k)*4)+j];
}
static void scalar_conv(const float *x,const float *wp,const float *b,float *y,const Header *h){
    for(size_t oc=0;oc<h->Cout;oc++) for(size_t t=0;t<h->Tout;t++){
        float s=b[oc];
        for(size_t ci=0;ci<h->Cin;ci++) for(size_t k=0;k<h->K;k++){
            long st=(long)t+(long)k*(long)h->dilation-(long)h->pad;
            if(st>=0 && st<(long)h->Tin) s += x[ci*h->Tin+(size_t)st]*w_at(wp,oc,ci,k,h->Cin,h->K);
        }
        y[oc*h->Tout+t]=s;
    }
}
static void stats(const float *a,const float *b,size_t n,float *ma,double *rmse){
    float m=0;double ss=0;for(size_t i=0;i<n;i++){float d=fabsf(a[i]-b[i]);if(d>m)m=d;ss+=(double)d*d;}*ma=m;*rmse=sqrt(ss/(double)n);
}
int main(int argc,char **argv){
    if(argc<2){fprintf(stderr,"usage: %s layer.dsv32 [rounds]\n",argv[0]);return 2;}int rounds=argc>2?atoi(argv[2]):5;if(rounds<1)rounds=1;
    FILE *f=fopen(argv[1],"rb");if(!f){perror("fopen");return 2;}Header h;if(fread(&h,1,sizeof(h),f)!=sizeof(h)){puts("short header");return 2;}
    if(memcmp(h.magic,"DSVOC32",7)||h.version!=1||h.stride!=1||h.group!=1||h.Cout%4){puts("unsupported bundle");return 2;}
    size_t nb=h.Cout, nw=(size_t)h.Cout*h.Cin*h.K, nx=(size_t)h.Cin*h.Tin, ny=(size_t)h.Cout*h.Tout;
    float *b=amalloc(nb*4),*w=amalloc(nw*4),*x=amalloc(nx*4),*ref=amalloc(ny*4),*y=amalloc(ny*4),*ys=amalloc(ny*4);
    size_t ws_n=ds_vocoder_conv1d_workspace_floats(h.Cin,h.Tin,h.pad);float *ws=amalloc(ws_n*4);
    if(!b||!w||!x||!ref||!y||!ys||!ws){puts("oom");return 2;}
    if(fread(b,4,nb,f)!=nb||fread(w,4,nw,f)!=nw||fread(x,4,nx,f)!=nx||fread(ref,4,ny,f)!=ny){puts("short data");return 2;}fclose(f);
    printf("M32 PURE-ASM Conv1d: Cin=%u Cout=%u K=%u Tin=%u Tout=%u pad=%u dil=%u MAC=%.3f M workspace=%.2f MiB\n",h.Cin,h.Cout,h.K,h.Tin,h.Tout,h.pad,h.dilation,(double)h.Cout*h.Tout*h.Cin*h.K/1e6,(double)ws_n*4/(1024*1024));
    const int ths[]={1,2,4,8};
    float best_ma=0;double best_rm=0;
    for(size_t q=0;q<sizeof(ths)/sizeof(ths[0]);q++){
        DSAsmThreadPool *pool=ds_threadpool_create((size_t)ths[q]);if(!pool){puts("pool create failed");return 2;}
        ds_vocoder_conv1d_f32_avx2(x,w,b,y,h.Cin,h.Cout,h.K,h.Tin,h.pad,h.dilation,ws,pool);
        float ma;double rm;stats(y,ref,ny,&ma,&rm); if(q==0){best_ma=ma;best_rm=rm;}
        double *v=calloc((size_t)rounds,sizeof(double));
        for(int r=0;r<rounds;r++){double t0=now_ms();ds_vocoder_conv1d_f32_avx2(x,w,b,y,h.Cin,h.Cout,h.K,h.Tin,h.pad,h.dilation,ws,pool);v[r]=now_ms()-t0;}
        for(int i=0;i<rounds;i++)for(int j=i+1;j<rounds;j++)if(v[j]<v[i]){double z=v[i];v[i]=v[j];v[j]=z;}
        double med=v[rounds/2],gflops=2.0*(double)h.Cout*h.Tout*h.Cin*h.K/(med*1e6);
        printf("  ASM threads=%d median=%.3f ms  %.2f GFLOP/s  max_abs=%.8g rmse=%.8g cpus=",ths[q],med,gflops,ma,rm);
        for(size_t i=0;i<ds_threadpool_threads(pool);i++)printf("%s%d",i?",":"[",ds_threadpool_cpu_at(pool,i));puts("]");
        free(v);ds_threadpool_destroy(pool);
    }
    const double macs=(double)h.Cout*h.Tout*h.Cin*h.K;
    if(macs<=3.5e8){
        double t0=now_ms();scalar_conv(x,w,b,ys,&h);double ms=now_ms()-t0;float ma;double rm;stats(ys,ref,ny,&ma,&rm);
        printf("  scalar-C one-shot=%.3f ms  %.2f GFLOP/s  max_abs=%.8g rmse=%.8g\n",ms,2.0*macs/(ms*1e6),ma,rm);
    } else printf("  scalar-C skipped (%.1f M MAC is intentionally too large for the one-shot reference benchmark)\n",macs/1e6);
    printf("M32 parity: %s (max_abs=%.8g rmse=%.8g)\n",best_ma<5e-4?"OK":"FAIL",best_ma,best_rm);
    return best_ma<5e-4?0:1;
}
