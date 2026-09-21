#define _POSIX_C_SOURCE 200809L
#include "dsasm_kernels.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct{char magic[8];uint32_t version,Cin,Cout,K,Tin,Tout,pad,dilation,K4,Tblocks,r0,r1;unsigned char reserved[8];} Header;

static int cpu_has_avx_vnni(void){
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    return __builtin_cpu_supports("avxvnni") != 0;
#else
    return 0;
#endif
}
static double now_ms(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return 1e3*t.tv_sec+1e-6*t.tv_nsec;}
static void *amalloc(size_t n){void*p=0;return posix_memalign(&p,64,n)?0:p;}
static int cmpd(const void*a,const void*b){double x=*(const double*)a,y=*(const double*)b;return (x>y)-(x<y);}
static float quant_pack(const float*x,uint8_t*qx,uint8_t*xpack,const Header*h,const float*wscale,float*scale){
    const size_t nx=(size_t)h->Cin*h->Tin,kred=(size_t)h->Cin*h->K;
    float mx=0;for(size_t i=0;i<nx;i++){float a=fabsf(x[i]);if(a>mx)mx=a;}float sx=mx>0?mx/127.0f:1.0f,inv=1.0f/sx;
    for(size_t i=0;i<nx;i++){int q=(int)lrintf(x[i]*inv);if(q>127)q=127;if(q<-127)q=-127;qx[i]=(uint8_t)(q+128);}
    for(size_t oc=0;oc<h->Cout;oc++)scale[oc]=sx*wscale[oc];
    for(size_t tb=0;tb<h->Tblocks;tb++)for(size_t kb=0;kb<h->K4;kb++){
        uint8_t*d=xpack+(tb*h->K4+kb)*32;
        for(size_t j=0;j<4;j++){
            size_t r=kb*4+j;int valid=r<kred;size_t ci=valid?r/h->K:0,kk=valid?r%h->K:0;
            for(size_t lane=0;lane<8;lane++){
                size_t t=tb*8+lane;long st=(long)t+(long)kk*(long)h->dilation-(long)h->pad;
                d[lane*4+j]=(valid && t<h->Tout && st>=0 && st<(long)h->Tin)?qx[ci*h->Tin+(size_t)st]:128u;
            }
        }
    }
    return sx;
}
static void stats(const float*y,const float*r,size_t n,float*ma,double*rm,double*cosv,double*snr){
    float m=0;double se=0,yy=0,rr=0,yr=0;
    for(size_t i=0;i<n;i++){double d=(double)y[i]-r[i];float ad=fabsf((float)d);if(ad>m)m=ad;se+=d*d;yy+=(double)y[i]*y[i];rr+=(double)r[i]*r[i];yr+=(double)y[i]*r[i];}
    *ma=m;*rm=sqrt(se/n);*cosv=yr/(sqrt(yy*rr)+1e-30);*snr=10.0*log10((rr+1e-30)/(se+1e-30));
}
int main(int argc,char**argv){
    if(!cpu_has_avx_vnni()){puts("M39.1 AVX-VNNI: SKIP (CPU/OS does not expose AVX-VNNI)");return 0;}
    if(argc<2){fprintf(stderr,"usage: %s hot.dsvn39 [rounds]\n",argv[0]);return 2;}int rounds=argc>2?atoi(argv[2]):7;if(rounds<3)rounds=3;
    FILE*f=fopen(argv[1],"rb");if(!f){perror("fopen");return 2;}Header h;if(fread(&h,1,sizeof(h),f)!=sizeof(h)||memcmp(h.magic,"DSVN39",6)||h.version!=1||h.Cout%8||h.Tout%8){puts("bad bundle");return 2;}
    size_t nb=h.Cout,nwq=(size_t)h.Cout*h.K4*4,nwf=(size_t)h.Cout*h.Cin*h.K,nx=(size_t)h.Cin*h.Tin,ny=(size_t)h.Cout*h.Tout;
    float*b=amalloc(nb*4),*ws=amalloc(nb*4),*fpw=amalloc(nwf*4),*x=amalloc(nx*4),*ref=amalloc(ny*4),*yv=amalloc(ny*4),*yf=amalloc(ny*4),*sc=amalloc(nb*4);
    int32_t*corr=amalloc(nb*4);int8_t*wq=amalloc(nwq);uint8_t*qx=amalloc(nx),*xp=amalloc((size_t)h.Tblocks*h.K4*32);float*xpad=amalloc((size_t)h.Cin*(h.Tin+2*h.pad)*4);
    if(!b||!ws||!corr||!wq||!fpw||!x||!ref||!yv||!yf||!sc||!qx||!xp||!xpad){puts("oom");return 2;}
    if(fread(b,4,nb,f)!=nb||fread(ws,4,nb,f)!=nb||fread(corr,4,nb,f)!=nb||fread(wq,1,nwq,f)!=nwq||fread(fpw,4,nwf,f)!=nwf||fread(x,4,nx,f)!=nx||fread(ref,4,ny,f)!=ny){puts("short data");return 2;}fclose(f);
    size_t Tp=h.Tin+2*h.pad;memset(xpad,0,(size_t)h.Cin*Tp*4);for(size_t c=0;c<h.Cin;c++)memcpy(xpad+c*Tp+h.pad,x+c*h.Tin,h.Tin*4);
    quant_pack(x,qx,xp,&h,ws,sc);ds_vnni_conv1d_u8s8_t8_oc8(xp,(const signed char*)wq,corr,sc,b,yv,h.Tblocks,h.K4,h.Cout,h.Tout);
    ds_conv1d_nct_f32_avx2_oc8_t8(xpad,fpw,b,yf,h.Cin,h.Cout,h.K,h.Tout,Tp,h.dilation);
    float ma;double rm,co,snr;stats(yv,ref,ny,&ma,&rm,&co,&snr);
    float maf;double rmf,cof,snrf;stats(yf,ref,ny,&maf,&rmf,&cof,&snrf);
    double*tp=calloc(rounds,sizeof(double)),*tk=calloc(rounds,sizeof(double)),*tf=calloc(rounds,sizeof(double));
    for(int r=0;r<rounds;r++){
        double a=now_ms();quant_pack(x,qx,xp,&h,ws,sc);tp[r]=now_ms()-a;
        a=now_ms();ds_vnni_conv1d_u8s8_t8_oc8(xp,(const signed char*)wq,corr,sc,b,yv,h.Tblocks,h.K4,h.Cout,h.Tout);tk[r]=now_ms()-a;
        a=now_ms();ds_conv1d_nct_f32_avx2_oc8_t8(xpad,fpw,b,yf,h.Cin,h.Cout,h.K,h.Tout,Tp,h.dilation);tf[r]=now_ms()-a;
    }
    qsort(tp,rounds,sizeof(double),cmpd);qsort(tk,rounds,sizeof(double),cmpd);qsort(tf,rounds,sizeof(double),cmpd);double p=tp[rounds/2],k=tk[rounds/2],ff=tf[rounds/2],mac=(double)h.Cout*h.Tout*h.Cin*h.K;
    printf("M39 AVX-VNNI Conv: Cin=%u Cout=%u K=%u T=%u d=%u MAC=%.1fM\n",h.Cin,h.Cout,h.K,h.Tout,h.dilation,mac/1e6);
    printf("  FP32 ASM kernel       %.3f ms  %.1f GFLOP/s  max_abs=%.7g\n",ff,2*mac/(ff*1e6),maf);
    printf("  VNNI pack/quant       %.3f ms\n",p);
    printf("  VNNI ASM kernel       %.3f ms  nominal %.1f GOP/s\n",k,2*mac/(k*1e6));
    printf("  VNNI total            %.3f ms  speedup-vs-FP32=%.3fx\n",p+k,ff/(p+k));
    printf("  quant parity          max_abs=%.7g rmse=%.7g cosine=%.9f SNR=%.2f dB\n",ma,rm,co,snr);
    printf("  FP32 reference parity max_abs=%.7g cosine=%.9f\n",maf,cof);
    free(b);free(ws);free(corr);free(wq);free(fpw);free(x);free(ref);free(yv);free(yf);free(sc);free(qx);free(xp);free(xpad);free(tp);free(tk);free(tf);return 0;
}
