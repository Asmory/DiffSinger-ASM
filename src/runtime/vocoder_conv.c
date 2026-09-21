#include "dsasm_vocoder.h"
#include "dsasm_kernels.h"
#include "threadpool_internal.h"
#include <string.h>
#include <stdlib.h>


static int m39_full_memset_enabled(void){
    static int mode=-1;
    if(mode<0){const char *e=getenv("DSASM_FULL_MEMSET");mode=(e && strcmp(e,"0")!=0)?1:0;}
    return mode;
}

/* M50 release defaults, derived from M45-M49 perf-guided A/B. */
static int env_bool_default(const char *name,int def){const char *e=getenv(name);if(!e||!*e)return def;return strcmp(e,"0")!=0 && strcmp(e,"off")!=0;}
static int kspec_enabled(void){static int v=-1;if(v<0)v=env_bool_default("DSASM_KSPEC",0);return v;}
static int range_t24_enabled(void){static int v=-1;if(v<0)v=env_bool_default("DSASM_RANGE_T24",0);return v;}
static int range_residual_t24_enabled(void){static int v=-1;if(v<0)v=env_bool_default("DSASM_RANGE_RESIDUAL_T24",1);return v;}
static int k7_t24_enabled(void){static int v=-1;if(v<0)v=env_bool_default("DSASM_K7_T24",1);return v;}
static int k11_t24_enabled(void){static int v=-1;if(v<0)v=env_bool_default("DSASM_K11_T24",1);return v;}
static int k3_tmode(void){static int v=-1;if(v>=0)return v;const char *e=getenv("DSASM_K3_TMODE");if(!e||!*e)return v=24;int x=atoi(e);return v=(x==16||x==24)?x:0;}

static int parallel_leaky_copy_enabled(void){
    static int v=-1;
    if(v<0)v=env_bool_default("DSASM_PARALLEL_LEAKY_COPY",0);
    return v;
}
static size_t parallel_leaky_copy_min_floats(void){
    static size_t v=0;
    if(v)return v;
    const char *e=getenv("DSASM_PARALLEL_LEAKY_MIN");
    unsigned long x=e?strtoul(e,NULL,10):262144ul;
    if(x<16384ul)x=16384ul;
    return v=(size_t)x;
}

static void run_padded_conv(
    const float *x_padded, const float *w_packed, const float *bias,
    const float *residual, float *y,
    size_t Cin, size_t Cout, size_t K, size_t Tout, size_t Tp, size_t dilation,
    size_t pack_width, DSAsmThreadPool *pool)
{
    if(pool){
        DSAsmJob j={0};
        j.kind=DS_JOB_VOCODER_CONV1D;
        j.x=x_padded;j.w=w_packed;j.b0=bias;j.residual=residual;j.y=y;
        j.M=Tout;j.N=Cout;j.K=Cin;
        j.P=K;j.Q=Tp;j.R=dilation;j.S=pack_width;
        j.T=(kspec_enabled()?1u:0u) | (range_t24_enabled()?2u:0u) | (range_residual_t24_enabled()?4u:0u);
        ds_threadpool_run(pool,&j);
    }else if(pack_width==8){
        const int k3mode=k3_tmode();
        const int ks=kspec_enabled() && !residual && (Tout%8u)==0;
        const int k7t24=k7_t24_enabled() && !residual && (Tout%24u)==0;
        const int k11t24=k11_t24_enabled() && !residual && (Tout%24u)==0;
        if(residual) ds_conv1d_nct_f32_avx2_oc8_t8_residual(x_padded,w_packed,bias,y,Cin,Cout,K,Tout,Tp,dilation,residual);
        else if(k3mode==24 && (Tout%24u)==0 && K==3u) ds_conv1d_nct_f32_avx2_oc4_t24_k3(x_padded,w_packed,bias,y,Cin,Cout,K,Tout,Tp,dilation);
        else if(k3mode==16 && (Tout%16u)==0 && K==3u) ds_conv1d_nct_f32_avx2_oc4_t16_k3(x_padded,w_packed,bias,y,Cin,Cout,K,Tout,Tp,dilation);
        else if(k7t24 && K==7u) ds_conv1d_nct_f32_avx2_oc4_t24_k7(x_padded,w_packed,bias,y,Cin,Cout,K,Tout,Tp,dilation);
        else if(k11t24 && K==11u) ds_conv1d_nct_f32_avx2_oc4_t24_k11(x_padded,w_packed,bias,y,Cin,Cout,K,Tout,Tp,dilation);
        else if(ks && K==3u) ds_conv1d_nct_f32_avx2_oc8_t8_k3(x_padded,w_packed,bias,y,Cin,Cout,K,Tout,Tp,dilation);
        else if(ks && K==7u) ds_conv1d_nct_f32_avx2_oc8_t8_k7(x_padded,w_packed,bias,y,Cin,Cout,K,Tout,Tp,dilation);
        else if(ks && K==11u) ds_conv1d_nct_f32_avx2_oc8_t8_k11(x_padded,w_packed,bias,y,Cin,Cout,K,Tout,Tp,dilation);
        else ds_conv1d_nct_f32_avx2_oc8_t8(x_padded,w_packed,bias,y,Cin,Cout,K,Tout,Tp,dilation);
    }else{
        ds_conv1d_nct_f32_avx2_oc4_t8(x_padded,w_packed,bias,y,Cin,Cout,K,Tout,Tp,dilation);
    }
}

size_t ds_vocoder_conv1d_workspace_floats(size_t Cin, size_t Tin, size_t pad){
    return Cin * (Tin + 2u*pad);
}

int ds_vocoder_conv1d_ex_residual_f32_avx2(
    const float *x, const float *w_packed, const float *bias,
    const float *residual, float *y,
    size_t Cin, size_t Cout, size_t K, size_t Tin,
    size_t pad, size_t dilation, size_t pack_width,
    int fuse_leaky, float alpha, float *workspace,
    DSAsmThreadPool *pool)
{
    if(!x||!w_packed||!bias||!y||!workspace||!Cin||!Cout||!K||!Tin||!dilation) return -1;
    if(pack_width!=4u && pack_width!=8u) return -2;
    if(residual && pack_width!=8u) return -5;
    if(Cout % pack_width) return -3;
    const size_t receptive=dilation*(K-1u)+1u;
    if(Tin + 2u*pad < receptive) return -4;
    const size_t Tout=Tin + 2u*pad - receptive + 1u;
    const size_t Tp=Tin + 2u*pad;
    /* M39: the old path memset the entire [Cin,Tp] workspace and immediately
       overwrote the Tin-wide center. HiFi-GAN pads are tiny (<=25 samples)
       while Tin reaches 24576, so clear only the left/right halos. */
    if(fuse_leaky){
        if(m39_full_memset_enabled()) memset(workspace,0,Cin*Tp*sizeof(float));
        /* M39 leaky-copy clears both halos itself, so the default path writes
           every workspace byte exactly once instead of memset+overwrite. */
        if(pool && parallel_leaky_copy_enabled() && Cin*Tin>=parallel_leaky_copy_min_floats())
            ds_threadpool_leaky_copy_nct_f32(pool,x,workspace,Cin,Tin,Tp,pad,alpha);
        else
            ds_leaky_copy_nct_f32_avx2(x,workspace,Cin,Tin,Tp,pad,alpha);
    }else{
        if(m39_full_memset_enabled()) memset(workspace,0,Cin*Tp*sizeof(float));
        else if(pad){
            for(size_t c=0;c<Cin;c++){
                float *d=workspace+c*Tp;
                memset(d,0,pad*sizeof(float));
                memset(d+pad+Tin,0,pad*sizeof(float));
            }
        }
        for(size_t c=0;c<Cin;c++) memcpy(workspace+c*Tp+pad,x+c*Tin,Tin*sizeof(float));
    }
    run_padded_conv(workspace,w_packed,bias,residual,y,Cin,Cout,K,Tout,Tp,dilation,pack_width,pool);
    return 0;
}

int ds_vocoder_conv1d_ex_f32_avx2(
    const float *x, const float *w_packed, const float *bias, float *y,
    size_t Cin, size_t Cout, size_t K, size_t Tin,
    size_t pad, size_t dilation, size_t pack_width,
    int fuse_leaky, float alpha, float *workspace,
    DSAsmThreadPool *pool)
{
    return ds_vocoder_conv1d_ex_residual_f32_avx2(
        x,w_packed,bias,NULL,y,Cin,Cout,K,Tin,pad,dilation,pack_width,
        fuse_leaky,alpha,workspace,pool);
}

int ds_vocoder_conv1d_f32_avx2(
    const float *x, const float *w_packed4, const float *bias, float *y,
    size_t Cin, size_t Cout, size_t K, size_t Tin,
    size_t pad, size_t dilation, float *workspace,
    DSAsmThreadPool *pool)
{
    return ds_vocoder_conv1d_ex_f32_avx2(x,w_packed4,bias,y,Cin,Cout,K,Tin,pad,dilation,4,0,0.0f,workspace,pool);
}

int ds_vocoder_convtranspose1d_f32_avx2(
    const float *x, const float *w_oc_major, const float *bias, float *y,
    size_t Cin, size_t Cout, size_t K, size_t Tin,
    size_t pad, size_t stride, DSAsmThreadPool *pool)
{
    if(!x||!w_oc_major||!bias||!y||!Cin||!Cout||!K||!Tin||!stride) return -1;
    const size_t Tout=(Tin-1u)*stride - 2u*pad + K;
    if(pool){
        DSAsmJob j={0};
        j.kind=DS_JOB_VOCODER_CONVTRANSPOSE;
        j.x=x;j.w=w_oc_major;j.b0=bias;j.y=y;
        j.M=Tout;j.N=Cout;j.K=Cin;j.P=K;j.Q=Tin;j.R=pad;
        j.S=stride;
        ds_threadpool_run(pool,&j);
    }else{
        for(size_t oc=0;oc<Cout;oc++)
            ds_convtranspose1d_oc_f32_avx2(x,w_oc_major+oc*Cin*K,bias[oc],y+oc*Tout,Cin,K,Tin,Tout,pad,stride);
    }
    return 0;
}

size_t ds_vocoder_resunit_workspace_floats(size_t C, size_t T, size_t pad1, size_t pad2){
    size_t p=pad1>pad2?pad1:pad2;
    return 2u*C*T + ds_vocoder_conv1d_workspace_floats(C,T,p);
}

int ds_vocoder_resunit_f32_avx2(
    const float *x,
    const float *w1_packed4, const float *b1, size_t K1, size_t pad1, size_t dil1,
    const float *w2_packed4, const float *b2, size_t K2, size_t pad2, size_t dil2,
    float alpha, float *y, size_t C, size_t T, float *workspace,
    DSAsmThreadPool *pool)
{
    if(!x||!w1_packed4||!b1||!w2_packed4||!b2||!y||!workspace||!C||!T) return -1;
    if(C&3u) return -2;
    if(T+2u*pad1 != T+dil1*(K1-1u) || T+2u*pad2 != T+dil2*(K2-1u)) return -3;
    float *tmp0=workspace;
    float *tmp1=tmp0+C*T;
    float *cws=tmp1+C*T;
    ds_leaky_relu_f32_avx2(x,tmp0,C*T,alpha);
    int rc=ds_vocoder_conv1d_f32_avx2(tmp0,w1_packed4,b1,tmp1,C,C,K1,T,pad1,dil1,cws,pool);
    if(rc) return rc;
    ds_leaky_relu_f32_avx2(tmp1,tmp0,C*T,alpha);
    rc=ds_vocoder_conv1d_f32_avx2(tmp0,w2_packed4,b2,y,C,C,K2,T,pad2,dil2,cws,pool);
    if(rc) return rc;
    ds_add_f32_avx2(y,x,y,C*T);
    return 0;
}
