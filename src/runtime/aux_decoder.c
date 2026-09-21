#include "dsasm_aux_decoder.h"
#include "dsasm_kernels.h"
#include "threadpool_internal.h"
#include <math.h>
#include <stddef.h>

static int valid(const DSAsmAuxConvNeXtWeights*w){
    return w&&w->blocks&&w->input_dim&&w->channels&&w->output_dim&&w->num_layers&&
           w->kernel_size==7&&(w->channels%16u)==0&&(w->output_dim%16u)==0;
}

size_t ds_aux_convnext_workspace_floats(const DSAsmAuxConvNeXtWeights*w,size_t t){
    if(!valid(w)||!t)return 0;
    size_t flat_in=7u*w->input_dim, flat_out=7u*w->channels;
    size_t im2=t*(flat_in>flat_out?flat_in:flat_out);
    return im2 + 3u*t*w->channels + t*(4u*w->channels);
}

static void p_linear(DSAsmThreadPool*p,const float*x,const float*w,const float*b,float*y,
                     size_t M,size_t N,size_t K){
    if(!p){ds_linear_f32_avx2_m4n16(x,w,b,y,M,N,K);return;}
    DSAsmJob j={0}; j.kind=DS_JOB_LINEAR; j.x=x; j.w=w; j.b0=b; j.y=y; j.M=M; j.N=N; j.K=K;
    ds_threadpool_run(p,&j);
}

static void im2col7(const float*x,float*col,size_t T,size_t C){
    const size_t K=7*C;
    for(size_t t=0;t<T;t++){
        float*dst=col+t*K;
        for(size_t q=0;q<7;q++){
            long it=(long)t+(long)q-3;
            float*d=dst+q*C;
            if(it<0||(size_t)it>=T){for(size_t c=0;c<C;c++)d[c]=0.0f;}
            else {const float*s=x+(size_t)it*C;for(size_t c=0;c<C;c++)d[c]=s[c];}
        }
    }
}

static void gelu_exact(float*x,size_t n){
    const float k=0.7071067811865475244f;
    for(size_t i=0;i<n;i++){float v=x[i];x[i]=0.5f*v*(1.0f+erff(v*k));}
}

static void gamma_residual(const float*x,const float*gamma,const float*res,float*y,size_t T,size_t C){
    for(size_t t=0;t<T;t++)for(size_t c=0;c<C;c++)y[t*C+c]=res[t*C+c]+x[t*C+c]*gamma[c];
}

int ds_aux_convnext_forward_norm_f32_avx2(
    const DSAsmAuxConvNeXtWeights*w,const float*cond,float*out,float*ws,size_t T,DSAsmThreadPool*pool){
    if(!valid(w)||!cond||!out||!ws||!T)return -1;
    const size_t C=w->channels,D=w->output_dim,H4=4*C;
    size_t flat_in=7u*w->input_dim, flat_out=7u*C;
    size_t im2n=T*(flat_in>flat_out?flat_in:flat_out);
    float*im2=ws; ws+=im2n;
    float*a=ws; ws+=T*C;
    float*b=ws; ws+=T*C;
    float*norm=ws; ws+=T*C;
    float*h=ws;

    im2col7(cond,im2,T,w->input_dim);
    p_linear(pool,im2,w->in_weight_m4n16,w->in_bias,a,T,C,flat_in);
    float*cur=a,*tmp=b;
    for(size_t li=0;li<w->num_layers;li++){
        const DSAsmAuxConvNeXtBlock*q=&w->blocks[li];
        ds_depthwise_conv1d_k7_tc_f32_avx2(cur,q->dw_weight_tap_major,q->dw_bias,tmp,T,C);
        ds_layernorm_f32_avx2(tmp,q->ln_gamma,q->ln_beta,norm,T,C,1e-6f);
        p_linear(pool,norm,q->pw1_weight_m4n16,q->pw1_bias,h,T,H4,C);
        gelu_exact(h,T*H4);
        p_linear(pool,h,q->pw2_weight_m4n16,q->pw2_bias,tmp,T,C,H4);
        gamma_residual(tmp,q->gamma,cur,tmp,T,C);
        float*sw=cur;cur=tmp;tmp=sw;
    }
    im2col7(cur,im2,T,C);
    p_linear(pool,im2,w->out_weight_m4n16,w->out_bias,out,T,D,flat_out);
    return 0;
}

int ds_aux_convnext_infer_f32_avx2(
    const DSAsmAuxConvNeXtWeights*w,const float*cond,const float*lo,const float*hi,size_t rd,
    float*out,float*ws,size_t T,DSAsmThreadPool*pool){
    if(!valid(w)||!lo||!hi||(rd!=1&&rd!=w->output_dim))return -1;
    int rc=ds_aux_convnext_forward_norm_f32_avx2(w,cond,out,ws,T,pool);if(rc)return rc;
    size_t D=w->output_dim;
    for(size_t t=0;t<T;t++)for(size_t d=0;d<D;d++){
        size_t j=rd==1?0:d;float k=(hi[j]-lo[j])*0.5f,b=(hi[j]+lo[j])*0.5f;
        out[t*D+d]=out[t*D+d]*k+b;
    }
    return 0;
}
