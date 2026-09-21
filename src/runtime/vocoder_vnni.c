#define _GNU_SOURCE
#include "dsasm_vocoder.h"
#include "dsasm_kernels.h"
#include "threadpool_internal.h"
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <alloca.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static inline double vnni_now_ms(void){
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC_RAW,&ts);
    return 1000.0*(double)ts.tv_sec + 1e-6*(double)ts.tv_nsec;
}

int ds_vocoder_vnni_available(void){
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    return __builtin_cpu_supports("avxvnni") != 0;
#else
    return 0;
#endif
}

static int vnni_asym_enabled(void){
    static int init=0,enabled=0;
    if(!init){const char *e=getenv("DSASM_VNNI_ASYM");enabled=e&&strcmp(e,"0")!=0;init=1;}
    return enabled;
}

static float quantize_u8_padded_avx2(
    const float *x,unsigned char *q,size_t C,size_t T,size_t pad,int leaky,float alpha,
    int asym,int *zero_point)
{
    const size_t n=C*T,Tp=T+2u*pad;
    const __m256 zero=_mm256_setzero_ps(), av=_mm256_set1_ps(alpha);
    float sx=1.0f,inv=1.0f; int zp=128;
    if(asym){
        /* LeakyReLU activations are strongly asymmetric.  The old M40 path
           spent half of U8's dynamic range on a negative tail that is scaled
           by alpha.  Track [min,max] (including exact zero for pad) and use
           the full 0..255 code space instead. */
        __m256 vmin=zero,vmax=zero;
        size_t i=0;
        for(;i+8<=n;i+=8){
            __m256 v=_mm256_loadu_ps(x+i);
            if(leaky){__m256 pos=_mm256_max_ps(v,zero),neg=_mm256_min_ps(v,zero);v=_mm256_add_ps(pos,_mm256_mul_ps(neg,av));}
            vmin=_mm256_min_ps(vmin,v);vmax=_mm256_max_ps(vmax,v);
        }
        float lo8[8],hi8[8];_mm256_storeu_ps(lo8,vmin);_mm256_storeu_ps(hi8,vmax);
        float lo=0.0f,hi=0.0f;for(int j=0;j<8;j++){if(lo8[j]<lo)lo=lo8[j];if(hi8[j]>hi)hi=hi8[j];}
        for(;i<n;i++){float v=x[i];if(leaky&&v<0)v*=alpha;if(v<lo)lo=v;if(v>hi)hi=v;}
        const float span=hi-lo;
        if(span>0.0f){sx=span/255.0f;inv=1.0f/sx;long z=lrintf(-lo*inv);if(z<0)z=0;if(z>255)z=255;zp=(int)z;}
    }else{
        const __m256 sign=_mm256_set1_ps(-0.0f);__m256 vmax=zero;
        size_t i=0;
        for(;i+8<=n;i+=8){
            __m256 v=_mm256_loadu_ps(x+i);
            if(leaky){__m256 pos=_mm256_max_ps(v,zero),neg=_mm256_min_ps(v,zero);v=_mm256_add_ps(pos,_mm256_mul_ps(neg,av));}
            vmax=_mm256_max_ps(vmax,_mm256_andnot_ps(sign,v));
        }
        float tmp[8];_mm256_storeu_ps(tmp,vmax);float mx=0.0f;for(int j=0;j<8;j++)if(tmp[j]>mx)mx=tmp[j];
        for(;i<n;i++){float v=x[i];if(leaky&&v<0)v*=alpha;float a=fabsf(v);if(a>mx)mx=a;}
        sx=mx>0.0f?mx/127.0f:1.0f;inv=1.0f/sx;zp=128;
    }
    const __m256 vinv=_mm256_set1_ps(inv);
    const __m256i vlo=asym?_mm256_setzero_si256():_mm256_set1_epi32(-127);
    const __m256i vhi=asym?_mm256_set1_epi32(255):_mm256_set1_epi32(127);
    const __m256i vzp=_mm256_set1_epi32(zp);
    for(size_t c=0;c<C;c++){
        unsigned char *qc=q+c*Tp; const float *xc=x+c*T;
        if(pad)memset(qc,zp,pad);
        size_t t=0;
        for(;t+8<=T;t+=8){
            __m256 v=_mm256_loadu_ps(xc+t);
            if(leaky){__m256 pos=_mm256_max_ps(v,zero),neg=_mm256_min_ps(v,zero);v=_mm256_add_ps(pos,_mm256_mul_ps(neg,av));}
            __m256i z=_mm256_cvtps_epi32(_mm256_mul_ps(v,vinv));
            if(asym){
                z=_mm256_add_epi32(z,vzp);
                z=_mm256_max_epi32(z,vlo);z=_mm256_min_epi32(z,vhi);
            }else{
                z=_mm256_max_epi32(z,vlo);z=_mm256_min_epi32(z,vhi);
                z=_mm256_add_epi32(z,vzp);
            }
            __m128i a=_mm256_castsi256_si128(z),b=_mm256_extracti128_si256(z,1);
            __m128i w=_mm_packs_epi32(a,b);__m128i bytes=_mm_packus_epi16(w,_mm_setzero_si128());
            _mm_storel_epi64((__m128i*)(qc+pad+t),bytes);
        }
        for(;t<T;t++){
            float v=xc[t];if(leaky&&v<0)v*=alpha;
            long z=lrintf(v*inv)+zp;
            if(asym){if(z<0)z=0;if(z>255)z=255;}else{if(z<1)z=1;if(z>255)z=255;}
            qc[pad+t]=(unsigned char)z;
        }
        if(pad)memset(qc+pad+T,zp,pad);
    }
    if(zero_point)*zero_point=zp;
    return sx;
}

int ds_vocoder_conv1d_vnni_u8s8(
    const float *x,const unsigned char *blob,const float *bias,float *y,
    size_t Cin,size_t Cout,size_t K,size_t Tin,size_t pad,size_t dilation,
    int fuse_leaky,float alpha,unsigned char *qx,unsigned char *xpack,float *scales,
    DSAsmThreadPool *pool,double *pack_ms,double *kernel_ms)
{
    if(!x||!blob||!bias||!y||!qx||!xpack||!scales||!Cin||!Cout||!K||!Tin||!pool)return -1;
    if((Cout&7u)!=0||!ds_vocoder_vnni_available())return -2;
    const size_t receptive=dilation*(K-1u)+1u;
    if(Tin+2u*pad<receptive)return -3;
    const size_t Tout=Tin+2u*pad-receptive+1u;
    if(Tout&7u)return -4;
    const size_t K4=(Cin*K+3u)/4u,Tblocks=Tout/8u,Tp=Tin+2u*pad;
    const float *wscale=(const float*)blob;
    const int *corr_base=(const int*)(blob+Cout*sizeof(float));
    const signed char *wpack=(const signed char*)(blob+2u*Cout*sizeof(float));
    const int asym=vnni_asym_enabled(); int zp=128;
    double a=vnni_now_ms();
    const float sx=quantize_u8_padded_avx2(x,qx,Cin,Tin,pad,fuse_leaky,alpha,asym,&zp);
    for(size_t oc=0;oc<Cout;oc++)scales[oc]=sx*wscale[oc];
    const int *corr=corr_base; int *corr_dyn=NULL;
    if(asym && zp!=128){
        corr_dyn=(int*)alloca(Cout*sizeof(int));
        for(size_t oc=0;oc<Cout;oc++)corr_dyn[oc]=(corr_base[oc]/128)*zp;
        corr=corr_dyn;
    }
    /* M40.1: precompute flattened reduction offsets once, then let the ASM
       packer transpose four contiguous 8-byte source vectors into one
       32-byte VNNI block. qx already includes 128-valued zero-point halos. */
    int *offs=(int*)alloca(K4*4u*sizeof(int));
    const size_t kred=Cin*K;
    for(size_t r=0;r<K4*4u;r++){
        if(r<kred){size_t ci=r/K,kk=r-ci*K;offs[r]=(int)(ci*Tp+kk*dilation);}
        else offs[r]=0;
    }
    DSAsmJob pj={0};pj.kind=DS_JOB_VOCODER_VNNI_PACK;pj.x=(const float*)qx;pj.y=(float*)xpack;pj.b0=(const float*)offs;
    pj.M=Tblocks;pj.N=K4;
    ds_threadpool_run(pool,&pj);
    double b=vnni_now_ms();
    DSAsmJob cj={0};cj.kind=DS_JOB_VOCODER_VNNI_CONV;cj.x=(const float*)xpack;cj.w=(const float*)wpack;cj.b0=(const float*)corr;cj.b1=scales;cj.residual=bias;cj.y=y;
    cj.M=Tblocks;cj.N=Cout;cj.K=K4;cj.P=Tout;cj.vnni_fn=ds_vnni_conv1d_u8s8_t8_oc8;
    ds_threadpool_run(pool,&cj);
    double c=vnni_now_ms();
    if(pack_ms) *pack_ms += b-a;
    if(kernel_ms) *kernel_ms += c-b;
    return 0;
}
