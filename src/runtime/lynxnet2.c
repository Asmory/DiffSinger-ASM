#include "dsasm_lynxnet2.h"
#include "dsasm_kernels.h"
#include "threadpool_internal.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int valid(const DSAsmLynxNet2Weights *w) {
    if (!w || !w->blocks || !w->input_dim || !w->condition_dim || !w->channels ||
        !w->hidden_dim || !w->num_layers || w->kernel_size != 31) return 0;
    if ((w->channels & 15u) || (w->input_dim & 15u) || (w->hidden_dim & 7u)) return 0;
    if (w->glu_type != DSASM_GLU_ATAN && w->glu_type != DSASM_GLU_SOFTSIGN) return 0;
    return 1;
}

size_t ds_lynxnet2_workspace_floats(const DSAsmLynxNet2Weights *w, size_t t) {
    if (!valid(w) || !t) return 0;
    const size_t c=w->channels, h=w->hidden_dim;
    /* x0,x1,norm,dw,condition = 5*T*C; h1,h2,tmp2h = 4*T*H;
       sinusoidal C + time hidden 4C + time output C = 6C. */
    return 5*t*c + 4*t*h + 6*c;
}


static void p_linear(DSAsmThreadPool *p,const float*x,const float*w,const float*b,float*y,size_t M,size_t N,size_t K){
    if(!p){ds_linear_f32_avx2_m4n16(x,w,b,y,M,N,K);return;}
    DSAsmJob j={0};j.kind=DS_JOB_LINEAR;j.x=x;j.w=w;j.b0=b;j.y=y;j.M=M;j.N=N;j.K=K;ds_threadpool_run(p,&j);
}
static void p_linear_res(DSAsmThreadPool *p,const float*x,const float*w,const float*b,const float*r,float*y,size_t M,size_t N,size_t K){
    if(!p){ds_linear_residual_f32_avx2_m4n16(x,w,b,r,y,M,N,K);return;}
    DSAsmJob j={0};j.kind=DS_JOB_LINEAR_RESIDUAL;j.x=x;j.w=w;j.b0=b;j.residual=r;j.y=y;j.M=M;j.N=N;j.K=K;ds_threadpool_run(p,&j);
}
static void p_softsign(DSAsmThreadPool *p,const float*x,const float*w,const float*bl,const float*bg,float*y,size_t M,size_t N,size_t K){
    if(!p){ds_fused_linear_softsign_glu_f32_avx2_m4n8(x,w,bl,bg,y,M,N,K);return;}
    DSAsmJob j={0};j.kind=DS_JOB_SOFTSIGN_GLU;j.x=x;j.w=w;j.b0=bl;j.b1=bg;j.y=y;j.M=M;j.N=N;j.K=K;ds_threadpool_run(p,&j);
}
static void p_depthwise(DSAsmThreadPool *p,const float*x,const float*w,const float*b,float*y,size_t T,size_t C){
    if(p && ds_threadpool_get_parallel_depthwise(p) && T>=31 && C>=512){
        DSAsmJob j={0};j.kind=DS_JOB_DEPTHWISE_K31;j.x=x;j.w=w;j.b0=b;j.y=y;j.M=T;j.N=C;
        ds_threadpool_run(p,&j);
    } else {
        ds_depthwise_conv1d_k31_tc_f32_avx2(x,w,b,y,T,C);
    }
}
static void p_atan_linear(DSAsmThreadPool *p,const float*x,const float*w,const float*b,
                          float*tmp2n,float*y,size_t M,size_t N,size_t K){
    if(p && ds_threadpool_get_2d(p) && ds_threadpool_get_atan_pipeline(p) &&
       M>=8 && N>=128 && K>=128 && (N%16)==0) {
        DSAsmJob j={0};j.kind=DS_JOB_ATAN_GLU_LINEAR;j.x=x;j.w=w;j.b0=b;j.y=y;j.M=M;j.N=N;j.K=K;
        ds_threadpool_run(p,&j);
        return;
    }
    p_linear(p,x,w,b,tmp2n,M,2*N,K);
    ds_atan_glu_f32_avx2(tmp2n,y,M,N);
}

int ds_lynxnet2_prepare_condition_f32_avx2(
    const DSAsmLynxNet2Weights *w, const float *cond, float *out, size_t t) {
    if (!valid(w) || !cond || !out || !t) return -1;
    ds_linear_f32_avx2_m4n16(cond, w->condition_weight_m4n16, w->condition_bias,
                             out, t, w->channels, w->condition_dim);
    return 0;
}

int ds_lynxnet2_prepare_condition_parallel_f32_avx2(
    const DSAsmLynxNet2Weights *w,const float *cond,float *out,size_t t,DSAsmThreadPool *pool){
    if(!valid(w)||!cond||!out||!t)return -1;
    p_linear(pool,cond,w->condition_weight_m4n16,w->condition_bias,out,t,w->channels,w->condition_dim);
    return 0;
}

static void sinusoidal(float x, float *out, size_t c) {
    const size_t half=c/2;
    const float scale = (half > 1) ? -logf(10000.0f)/(float)(half-1) : 0.0f;
    for (size_t i=0;i<half;i++) {
        float v=x*expf((float)i*scale);
        out[i]=sinf(v);
        out[half+i]=cosf(v);
    }
    if (c & 1u) out[c-1]=0.0f;
}

static void gelu_exact(float *x, size_t n) {
    const float inv_sqrt2=0.7071067811865475244f;
    for(size_t i=0;i<n;i++) {
        float v=x[i];
        x[i]=0.5f*v*(1.0f+erff(v*inv_sqrt2));
    }
}

static int forward_impl(
    const DSAsmLynxNet2Weights *w,
    const float *spec, const float *cond_projected,
    float timestep, float *out, float *ws, size_t t, DSAsmThreadPool *pool) {
    if (!valid(w) || !spec || !cond_projected || !out || !ws || !t) return -1;
    const size_t c=w->channels,h=w->hidden_dim;

    float *x0=ws; ws += t*c;
    float *x1=ws; ws += t*c;
    float *norm=ws; ws += t*c;
    float *dw=ws; ws += t*c;
    /* The full-forward wrapper owns this slot for conditioner projection. It is
       intentionally skipped here so cached/full workspace layouts are identical. */
    ws += t*c;
    float *h1=ws; ws += t*h;
    float *h2=ws; ws += t*h;
    float *tmp2h=ws; ws += t*(2*h);
    float *sinemb=ws; ws += c;
    float *time_hidden=ws; ws += 4*c;
    float *time_out=ws;

    p_linear(pool,spec,w->input_weight_m4n16,w->input_bias,x0,t,c,w->input_dim);

    sinusoidal(timestep, sinemb, c);
    ds_linear_f32_avx2_m4n16(sinemb, w->time1_weight_m4n16, w->time1_bias,
                             time_hidden, 1, 4*c, c);
    gelu_exact(time_hidden, 4*c);
    ds_linear_f32_avx2_m4n16(time_hidden, w->time2_weight_m4n16, w->time2_bias,
                             time_out, 1, c, 4*c);

    ds_add3_broadcast_f32_avx2(x0, cond_projected, time_out, x0, t, c);

    float *cur=x0, *next=x1;
    for(size_t li=0; li<w->num_layers; li++) {
        const DSAsmLynxNet2Block *b=&w->blocks[li];
        ds_layernorm_f32_avx2(cur,b->ln_gamma,b->ln_beta,norm,t,c,1e-5f);
        p_depthwise(pool,norm,b->dw_weight_tap_major,b->dw_bias,dw,t,c);
        if(w->glu_type==DSASM_GLU_ATAN) {
            p_atan_linear(pool,dw,b->glu1_weight,b->glu1_bias,tmp2h,h1,t,h,c);
            p_atan_linear(pool,h1,b->glu2_weight,b->glu2_bias,tmp2h,h2,t,h,h);
        } else {
            p_softsign(pool,dw,b->glu1_weight,b->glu1_bias,b->glu1_bias+h,h1,t,h,c);
            p_softsign(pool,h1,b->glu2_weight,b->glu2_bias,b->glu2_bias+h,h2,t,h,h);
        }
        p_linear_res(pool,h2,b->out_weight_m4n16,b->out_bias,cur,next,t,c,h);
        float *swap=cur;cur=next;next=swap;
    }
    ds_layernorm_f32_avx2(cur,w->post_norm_gamma,w->post_norm_beta,norm,t,c,1e-5f);
    p_linear(pool,norm,w->output_weight_m4n16,w->output_bias,out,t,w->input_dim,c);
    return 0;
}

int ds_lynxnet2_forward_cached_condition_f32_avx2(
    const DSAsmLynxNet2Weights *w, const float *spec, const float *cond_projected,
    float timestep, float *out, float *workspace, size_t t) {
    return forward_impl(w,spec,cond_projected,timestep,out,workspace,t,NULL);
}

int ds_lynxnet2_forward_f32_avx2(
    const DSAsmLynxNet2Weights *w, const float *spec, const float *cond,
    float timestep, float *out, float *workspace, size_t t) {
    if(!valid(w)||!workspace) return -1;
    const size_t c=w->channels;
    float *cond_slot=workspace + 4*t*c; /* after x0,x1,norm,dw */
    int rc=ds_lynxnet2_prepare_condition_f32_avx2(w,cond,cond_slot,t);
    if(rc) return rc;
    return forward_impl(w,spec,cond_slot,timestep,out,workspace,t,NULL);
}


int ds_lynxnet2_forward_cached_condition_parallel_f32_avx2(
    const DSAsmLynxNet2Weights *w,const float *spec,const float *cond_projected,
    float timestep,float *out,float *workspace,size_t t,DSAsmThreadPool *pool){
    return forward_impl(w,spec,cond_projected,timestep,out,workspace,t,pool);
}
int ds_lynxnet2_forward_parallel_f32_avx2(
    const DSAsmLynxNet2Weights *w,const float *spec,const float *cond,float timestep,
    float *out,float *workspace,size_t t,DSAsmThreadPool *pool){
    if(!valid(w)||!workspace)return -1;
    const size_t c=w->channels;
    float *cond_slot=workspace+4*t*c;
    int rc=ds_lynxnet2_prepare_condition_parallel_f32_avx2(w,cond,cond_slot,t,pool);if(rc)return rc;
    return forward_impl(w,spec,cond_slot,timestep,out,workspace,t,pool);
}
