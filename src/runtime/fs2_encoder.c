#include "dsasm_fs2_encoder.h"
#include "dsasm_kernels.h"
#include "threadpool_internal.h"
#include <immintrin.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

static int fs2_precise_ln_enabled(void){
    const char *e=getenv("DSASM_FS2_PRECISE_LN");
    return e && e[0] && e[0] != '0';
}

static void fs2_layernorm(const float*x,const float*g,const float*b,float*y,size_t M,size_t K,float eps){
    if(fs2_precise_ln_enabled()) ds_layernorm_f32_avx2_precise(x,g,b,y,M,K,eps);
    else ds_layernorm_f32_avx2(x,g,b,y,M,K,eps);
}

static int valid(const DSAsmFS2EncoderWeights *w){
    if(!w||!w->layers||!w->token_embedding||!w->dur_weight||!w->dur_bias||
       !w->final_ln_gamma||!w->final_ln_beta)return 0;
    if(!w->vocab_size||!w->hidden_size||!w->num_layers||!w->num_heads)return 0;
    if((w->hidden_size%16u)!=0||(w->hidden_size%w->num_heads)!=0)return 0;
    if(w->ffn_kernel_size!=3u)return 0;
    if(!(w->rope_theta>0.0f))return 0;
    const size_t C=w->hidden_size;
    for(size_t i=0;i<w->num_layers;i++){
        const DSAsmFS2EncoderLayer *q=&w->layers[i];
        if(!q->ln1_gamma||!q->ln1_beta||!q->qkv_weight_m4n16||!q->qkv_bias||
           !q->out_weight_m4n16||!q->out_bias||!q->ln2_gamma||!q->ln2_beta||
           !q->ffn1_weight_m4n16||!q->ffn1_bias||!q->ffn2_weight_m4n16||!q->ffn2_bias)return 0;
    }
    return C>0;
}

size_t ds_fs2_encoder_workspace_floats(const DSAsmFS2EncoderWeights *w,size_t T){
    if(!valid(w)||!T)return 0;
    const size_t C=w->hidden_size,hd=C/w->num_heads,half=hd/2u;
    /* x,tmp,norm,qkv,attn,im2,ffn,rope cos+sin,scores */
    return T*C*3u + T*(3u*C) + T*C + T*(3u*C) + T*(4u*C) + 2u*T*half + T;
}

static void p_linear(DSAsmThreadPool *p,const float*x,const float*w,const float*b,float*y,
                     size_t M,size_t N,size_t K){
    if(!p){ds_linear_f32_avx2_m4n16(x,w,b,y,M,N,K);return;}
    DSAsmJob j={0}; j.kind=DS_JOB_LINEAR; j.x=x; j.w=w; j.b0=b; j.y=y; j.M=M; j.N=N; j.K=K;
    ds_threadpool_run(p,&j);
}

static void im2col3(const float*x,float*col,size_t T,size_t C){
    const size_t K=3u*C;
    for(size_t t=0;t<T;t++){
        float*dst=col+t*K;
        for(size_t q=0;q<3;q++){
            long it=(long)t+(long)q-1;
            float*d=dst+q*C;
            if(it<0||(size_t)it>=T)memset(d,0,C*sizeof(float));
            else memcpy(d,x+(size_t)it*C,C*sizeof(float));
        }
    }
}

static void gelu_exact_scaled(float*x,size_t n,float scale){
    const float k=0.7071067811865475244f;
    for(size_t i=0;i<n;i++){
        float v=x[i]*scale;
        x[i]=0.5f*v*(1.0f+erff(v*k));
    }
}

static void mask_rows(float*x,const int32_t*tok,size_t T,size_t C){
    for(size_t t=0;t<T;t++)if(tok[t]==0)memset(x+t*C,0,C*sizeof(float));
}

static void add_residual_mask(const float*a,const float*b,float*y,const int32_t*tok,size_t T,size_t C){
    for(size_t t=0;t<T;t++){
        float*o=y+t*C;
        if(tok[t]==0){memset(o,0,C*sizeof(float));continue;}
        const float*x=a+t*C,*z=b+t*C;
        size_t c=0;
        for(;c+8<=C;c+=8){
            __m256 vx=_mm256_loadu_ps(x+c),vz=_mm256_loadu_ps(z+c);
            _mm256_storeu_ps(o+c,_mm256_add_ps(vx,vz));
        }
        for(;c<C;c++)o[c]=x[c]+z[c];
    }
}

static void build_rope(float*cosv,float*sinv,size_t T,size_t hd,float theta){
    const size_t half=hd/2u;
    const float ltheta=logf(theta);
    for(size_t t=0;t<T;t++)for(size_t j=0;j<half;j++){
        const float exponent=(float)(2u*j)/(float)hd;
        const float inv=expf(-ltheta*exponent);
        const float a=(float)t*inv;
        cosv[t*half+j]=cosf(a);
        sinv[t*half+j]=sinf(a);
    }
}

static void rope_one(float*x,const float*c,const float*s,size_t hd,int interleaved){
    if(interleaved){
        for(size_t j=0;j<hd/2u;j++){
            const size_t i=2u*j;
            const float x1=x[i],x2=x[i+1],co=c[j],si=s[j];
            x[i]=x1*co-x2*si;
            x[i+1]=x2*co+x1*si;
        }
    }else{
        const size_t half=hd/2u;
        for(size_t j=0;j<half;j++){
            const float x1=x[j],x2=x[j+half],co=c[j],si=s[j];
            x[j]=x1*co-x2*si;
            x[j+half]=x2*co+x1*si;
        }
    }
}

static void apply_rope(float*qkv,const float*cosv,const float*sinv,size_t T,size_t C,size_t H,int interleaved){
    const size_t hd=C/H,half=hd/2u;
    for(size_t t=0;t<T;t++){
        float*row=qkv+t*(3u*C);
        for(size_t h=0;h<H;h++){
            const float*c=cosv+t*half,*s=sinv+t*half;
            rope_one(row+h*hd,c,s,hd,interleaved);
            rope_one(row+C+h*hd,c,s,hd,interleaved);
        }
    }
}

static void weighted_v(const float*qkv,const float*p,size_t T,size_t C,size_t h,size_t H,float*out){
    const size_t hd=C/H;
    memset(out,0,hd*sizeof(float));
    for(size_t k=0;k<T;k++){
        const float*w=qkv+k*(3u*C)+2u*C+h*hd;
        __m256 vp=_mm256_set1_ps(p[k]);
        size_t d=0;
        for(;d+8<=hd;d+=8){
            __m256 a=_mm256_loadu_ps(out+d),b=_mm256_loadu_ps(w+d);
            a=_mm256_fmadd_ps(vp,b,a);_mm256_storeu_ps(out+d,a);
        }
        for(;d<hd;d++)out[d]+=p[k]*w[d];
    }
}

static void attention(const float*qkv,const int32_t*tok,float*out,float*scores,
                      size_t T,size_t C,size_t H){
    const size_t hd=C/H;
    const float scale=1.0f/sqrtf((float)hd);
    memset(out,0,T*C*sizeof(float));
    for(size_t h=0;h<H;h++){
        for(size_t i=0;i<T;i++){
            if(tok[i]==0)continue;
            const float*q=qkv+i*(3u*C)+h*hd;
            float mx=-INFINITY;
            size_t validn=0;
            for(size_t j=0;j<T;j++){
                if(tok[j]==0){scores[j]=-INFINITY;continue;}
                const float*k=qkv+j*(3u*C)+C+h*hd;
                float z=ds_dot_f32_avx2_fma(q,k,hd)*scale;
                scores[j]=z;if(z>mx)mx=z;validn++;
            }
            if(!validn)continue;
            float sum=0.0f;
            for(size_t j=0;j<T;j++){
                if(tok[j]==0){scores[j]=0.0f;continue;}
                float e=expf(scores[j]-mx);scores[j]=e;sum+=e;
            }
            const float inv=1.0f/sum;
            for(size_t j=0;j<T;j++)scores[j]*=inv;
            weighted_v(qkv,scores,T,C,h,H,out+i*C+h*hd);
        }
    }
}

static int encoder_forward_impl(
    const DSAsmFS2EncoderWeights *w,const int32_t*tok,const int32_t*dur,size_t T,
    const int32_t*lang,const float*lang_emb,size_t nlang,const float*lang_token_mask,
    float*out,float*ws,DSAsmThreadPool*pool){
    if(!valid(w)||!tok||!dur||!out||!ws||!T)return -1;
    const size_t C=w->hidden_size,H=w->num_heads,hd=C/H,half=hd/2u,F4=4u*C;
    if((hd&1u)!=0)return -2;
    for(size_t t=0;t<T;t++){
        if(tok[t]<0||(uint32_t)tok[t]>=w->vocab_size||dur[t]<0)return -3;
        if(lang&&tok[t]!=0&&(lang[t]<0||(size_t)lang[t]>=nlang))return -4;
    }

    float*x=ws; ws+=T*C;
    float*tmp=ws; ws+=T*C;
    float*norm=ws; ws+=T*C;
    float*qkv=ws; ws+=T*(3u*C);
    float*attn=ws; ws+=T*C;
    float*im2=ws; ws+=T*(3u*C);
    float*ffn=ws; ws+=T*F4;
    float*cosv=ws; ws+=T*half;
    float*sinv=ws; ws+=T*half;
    float*scores=ws;

    const float emb_scale=sqrtf((float)C);
    for(size_t t=0;t<T;t++){
        float*dst=x+t*C;
        if(tok[t]==0){memset(dst,0,C*sizeof(float));continue;}
        const float*e=w->token_embedding+(size_t)tok[t]*C;
        size_t lid=(lang&&lang_emb)?(size_t)lang[t]:0u;
        if(lang_token_mask && (size_t)tok[t] < w->vocab_size && lang_token_mask[tok[t]] < 0.5f) lid=0u;
        const float*le=(lang&&lang_emb)?lang_emb+lid*C:NULL;
        const float di=log1pf((float)dur[t]);
        for(size_t c=0;c<C;c++)dst[c]=emb_scale*e[c] + di*w->dur_weight[c] + w->dur_bias[c] + (le?le[c]:0.0f);
    }
    build_rope(cosv,sinv,T,hd,w->rope_theta);

    for(size_t li=0;li<w->num_layers;li++){
        const DSAsmFS2EncoderLayer*q=&w->layers[li];
        fs2_layernorm(x,q->ln1_gamma,q->ln1_beta,norm,T,C,1e-5f);
        p_linear(pool,norm,q->qkv_weight_m4n16,q->qkv_bias,qkv,T,3u*C,C);
        apply_rope(qkv,cosv,sinv,T,C,H,(int)w->rope_interleaved);
        attention(qkv,tok,attn,scores,T,C,H);
        p_linear(pool,attn,q->out_weight_m4n16,q->out_bias,tmp,T,C,C);
        add_residual_mask(x,tmp,x,tok,T,C);

        fs2_layernorm(x,q->ln2_gamma,q->ln2_beta,norm,T,C,1e-5f);
        im2col3(norm,im2,T,C);
        p_linear(pool,im2,q->ffn1_weight_m4n16,q->ffn1_bias,ffn,T,F4,3u*C);
        gelu_exact_scaled(ffn,T*F4,0.5773502691896257645f);
        p_linear(pool,ffn,q->ffn2_weight_m4n16,q->ffn2_bias,tmp,T,C,F4);
        add_residual_mask(x,tmp,x,tok,T,C);
    }
    fs2_layernorm(x,w->final_ln_gamma,w->final_ln_beta,out,T,C,1e-5f);
    mask_rows(out,tok,T,C);
    return 0;
}

int ds_fs2_encoder_forward_f32_avx2(
    const DSAsmFS2EncoderWeights *w,const int32_t*tok,const int32_t*dur,size_t T,
    float*out,float*ws,DSAsmThreadPool*pool){
    return encoder_forward_impl(w,tok,dur,T,NULL,NULL,0,NULL,out,ws,pool);
}

static int valid_acoustic(const DSAsmFS2AcousticWeights*w){
    if(!w||!valid(&w->encoder))return 0;
    return w->stretch_w1_m4n16&&w->stretch_b1&&w->stretch_w2_m4n16&&w->stretch_b2&&
           w->gru_w_ih_m4n16&&w->gru_b_ih&&w->gru_w_hh_m4n16&&w->gru_b_hh&&
           w->pitch_weight&&w->pitch_bias;
}

size_t ds_fs2_acoustic_condition_workspace_floats(const DSAsmFS2AcousticWeights*w,size_t P,size_t T){
    if(!valid_acoustic(w)||!P||!T)return 0;
    const size_t C=w->encoder.hidden_size;
    return P + ds_fs2_encoder_workspace_floats(&w->encoder,P) + P*C + T + T*C + T*(4u*C) + T*C + T*(3u*C) + 4u*C;
}

static void stretch_values(const int32_t*m,size_t T,const int32_t*d,size_t P,float*out){
    int32_t accum=0;
    for(size_t t=0;t<T;t++){
        int32_t q=m[t];
        if(q<=0||(size_t)q>P){out[t]=0.0f;accum=0;continue;}
        int32_t md=d[q-1];
        if(t==0||m[t]!=m[t-1])accum=0; else accum++;
        out[t]=md>0?(float)accum/(float)md:0.0f;
    }
}

static void stretch_sinusoidal(const float*stretch,float*out,size_t T,size_t C){
    const size_t half=C/2u;
    const float a=-logf(10000.0f)/(float)(half-1u);
    for(size_t t=0;t<T;t++){
        float v=nearbyintf(1000.0f*stretch[t]);
        float*dst=out+t*C;
        for(size_t j=0;j<half;j++){
            float z=v*expf((float)j*a);
            dst[j]=sinf(z);dst[j+half]=cosf(z);
        }
    }
}

static inline float sigmoidf_local(float x){return 1.0f/(1.0f+expf(-x));}

static void gru_forward(const DSAsmFS2AcousticWeights*w,const float*x,float*out,float*proj,float*tmp,size_t T){
    const size_t C=w->encoder.hidden_size,N3=3u*C;
    /* Input half for all frames amortizes one large packed GEMM. */
    ds_linear_f32_avx2_m4n16(x,w->gru_w_ih_m4n16,w->gru_b_ih,proj,T,N3,C);
    float*rec=tmp,*h=tmp+N3;
    memset(h,0,C*sizeof(float));
    for(size_t t=0;t<T;t++){
        ds_linear_f32_avx2_m4n16(h,w->gru_w_hh_m4n16,w->gru_b_hh,rec,1,N3,C);
        const float*ip=proj+t*N3;
        float*dst=out+t*C;
        for(size_t c=0;c<C;c++){
            float r=sigmoidf_local(ip[c]+rec[c]);
            float z=sigmoidf_local(ip[C+c]+rec[C+c]);
            float n=tanhf(ip[2u*C+c] + r*rec[2u*C+c]);
            float nh=(1.0f-z)*n + z*h[c];
            dst[c]=nh;
        }
        memcpy(h,dst,C*sizeof(float));
    }
}

int ds_fs2_acoustic_condition_f32_avx2(
    const DSAsmFS2AcousticWeights*w,const int32_t*tok,size_t P,
    const int32_t*mel2ph,const float*f0,size_t T,float*cond,float*ws,DSAsmThreadPool*pool){
    if(!valid_acoustic(w)||!tok||!mel2ph||!f0||!cond||!ws||!P||!T)return -1;
    const size_t C=w->encoder.hidden_size,F4=4u*C;
    int32_t*dur=(int32_t*)ws; ws+=P;
    memset(dur,0,P*sizeof(int32_t));
    for(size_t t=0;t<T;t++){
        int32_t q=mel2ph[t];if(q<0||(size_t)q>P)return -3;if(q>0)dur[q-1]++;
    }
    size_t encn=ds_fs2_encoder_workspace_floats(&w->encoder,P);
    float*encws=ws;ws+=encn;
    float*enc=ws;ws+=P*C;
    float*stretch=ws;ws+=T;
    float*semb=ws;ws+=T*C;
    float*h4=ws;ws+=T*F4;
    float*tmp=ws;ws+=T*C;
    float*gru_proj=ws;ws+=T*(3u*C);
    float*gru_tmp=ws; /* 3C recurrent projection + C hidden */

    int rc=ds_fs2_encoder_forward_f32_avx2(&w->encoder,tok,dur,P,enc,encws,pool);
    if(rc)return rc;
    for(size_t t=0;t<T;t++){
        int32_t q=mel2ph[t];float*dst=cond+t*C;
        if(q==0)memset(dst,0,C*sizeof(float));
        else memcpy(dst,enc+(size_t)(q-1)*C,C*sizeof(float));
    }
    stretch_values(mel2ph,T,dur,P,stretch);
    stretch_sinusoidal(stretch,semb,T,C);
    p_linear(pool,semb,w->stretch_w1_m4n16,w->stretch_b1,h4,T,F4,C);
    gelu_exact_scaled(h4,T*F4,1.0f);
    p_linear(pool,h4,w->stretch_w2_m4n16,w->stretch_b2,tmp,T,C,F4);
    for(size_t i=0;i<T*C;i++)cond[i]+=tmp[i];

    gru_forward(w,cond,tmp,gru_proj,gru_tmp,T);
    for(size_t i=0;i<T*C;i++)cond[i]+=tmp[i];

    for(size_t t=0;t<T;t++){
        float pv=log1pf(f0[t]/700.0f);float*dst=cond+t*C;
        for(size_t c=0;c<C;c++)dst[c]+=pv*w->pitch_weight[c]+w->pitch_bias[c];
    }
    return 0;
}

static inline float clipf_local(float v,float lo,float hi){
    if(v<lo)v=lo;
    if(v>hi)v=hi;
    return v;
}

static int valid_deploy_extras(const DSAsmFS2AcousticWeights*w,const DSAsmFS2DeploymentExtras*x){
    if(!x)return 1;
    const uint32_t f=x->flags;
    if((f&DSASM_FS2_FEAT_LANGUAGE) && (!x->language_embedding||x->num_languages<2))return 0;
    if((f&DSASM_FS2_FEAT_LANGUAGE_MASK) && (!(f&DSASM_FS2_FEAT_LANGUAGE)||!x->language_token_mask))return 0;
    if((f&DSASM_FS2_FEAT_BREATH) && (!x->breath_weight||!x->breath_bias))return 0;
    if((f&DSASM_FS2_FEAT_VOICING) && (!x->voicing_weight||!x->voicing_bias))return 0;
    if((f&DSASM_FS2_FEAT_TENSION) && (!x->tension_weight||!x->tension_bias))return 0;
    if((f&DSASM_FS2_FEAT_KEY_SHIFT) && (!x->key_shift_weight||!x->key_shift_bias))return 0;
    if((f&DSASM_FS2_FEAT_SPEED) && (!x->speed_weight||!x->speed_bias))return 0;
    if((f&DSASM_FS2_FEAT_STRETCH_TABLE) && !x->stretch_table)return 0;
    (void)w;return 1;
}

static inline void dbg_copy(float *dst,const float *src,size_t n){if(dst)memcpy(dst,src,n*sizeof(float));}

static int ds_fs2_acoustic_condition_deploy_impl_f32_avx2(
    const DSAsmFS2AcousticWeights*w,const DSAsmFS2DeploymentExtras*x,const DSAsmFS2DeploymentInputs*in,
    const int32_t*tok,size_t P,const int32_t*mel2ph,const float*f0,size_t T,
    float*cond,float*ws,DSAsmThreadPool*pool,DSAsmFS2DeploymentDebug*dbg){
    if(!valid_acoustic(w)||!valid_deploy_extras(w,x)||!tok||!mel2ph||!f0||!cond||!ws||!P||!T)return -1;
    const uint32_t flags=x?x->flags:0u;
    if((flags&DSASM_FS2_FEAT_LANGUAGE)&&(!in||!in->languages))return -5;
    if((flags&DSASM_FS2_FEAT_SPEAKER)&&(!in||!in->speaker_embedding_tc))return -6;
    const size_t C=w->encoder.hidden_size,F4=4u*C;
    int32_t*dur=(int32_t*)ws; ws+=P;
    memset(dur,0,P*sizeof(int32_t));
    for(size_t t=0;t<T;t++){
        int32_t q=mel2ph[t];if(q<0||(size_t)q>P)return -3;if(q>0)dur[q-1]++;
    }
    size_t encn=ds_fs2_encoder_workspace_floats(&w->encoder,P);
    float*encws=ws;ws+=encn;
    float*enc=ws;ws+=P*C;
    float*stretch=ws;ws+=T;
    float*semb=ws;ws+=T*C;
    float*h4=ws;ws+=T*F4;
    float*tmp=ws;ws+=T*C;
    float*gru_proj=ws;ws+=T*(3u*C);
    float*gru_tmp=ws;

    const int32_t*langs=(flags&DSASM_FS2_FEAT_LANGUAGE)?in->languages:NULL;
    const float*langemb=(flags&DSASM_FS2_FEAT_LANGUAGE)?x->language_embedding:NULL;
    size_t nlang=(flags&DSASM_FS2_FEAT_LANGUAGE)?x->num_languages:0;
    const float*langmask=((flags&DSASM_FS2_FEAT_LANGUAGE_MASK)&&x)?x->language_token_mask:NULL;
    int rc=encoder_forward_impl(&w->encoder,tok,dur,P,langs,langemb,nlang,langmask,enc,encws,pool);
    if(rc)return rc;
    if(dbg)dbg_copy(dbg->encoder_txt_pc,enc,P*C);
    for(size_t t=0;t<T;t++){
        int32_t q=mel2ph[t];float*dst=cond+t*C;
        if(q==0)memset(dst,0,C*sizeof(float));
        else memcpy(dst,enc+(size_t)(q-1)*C,C*sizeof(float));
    }
    if(dbg)dbg_copy(dbg->gathered_tc,cond,T*C);
    stretch_values(mel2ph,T,dur,P,stretch);
    if((flags&DSASM_FS2_FEAT_STRETCH_TABLE) && x->stretch_table){
        for(size_t t=0;t<T;t++){
            long qi=lrintf(1000.0f*stretch[t]);
            if(qi<0)qi=0;else if(qi>1000)qi=1000;
            memcpy(tmp+t*C,x->stretch_table+(size_t)qi*C,C*sizeof(float));
        }
    }else{
        stretch_sinusoidal(stretch,semb,T,C);
        p_linear(pool,semb,w->stretch_w1_m4n16,w->stretch_b1,h4,T,F4,C);
        gelu_exact_scaled(h4,T*F4,1.0f);
        p_linear(pool,h4,w->stretch_w2_m4n16,w->stretch_b2,tmp,T,C,F4);
    }
    for(size_t i=0;i<T*C;i++)cond[i]+=tmp[i];
    if(dbg)dbg_copy(dbg->stretch_tc,cond,T*C);
    gru_forward(w,cond,tmp,gru_proj,gru_tmp,T);
    for(size_t i=0;i<T*C;i++)cond[i]+=tmp[i];
    if(dbg)dbg_copy(dbg->gru_tc,cond,T*C);

    for(size_t t=0;t<T;t++){
        float*dst=cond+t*C;
        float pv=log1pf(f0[t]/700.0f);
        for(size_t c=0;c<C;c++)dst[c]+=pv*w->pitch_weight[c]+w->pitch_bias[c];
    }
    if(dbg)dbg_copy(dbg->pitch_tc,cond,T*C);

    for(size_t t=0;t<T;t++){
        float*dst=cond+t*C;
        if(flags&DSASM_FS2_FEAT_BREATH){
            float v=(in&&in->breathiness?in->breathiness[t]:0.0f)*x->breath_scale;
            for(size_t c=0;c<C;c++)dst[c]+=v*x->breath_weight[c]+x->breath_bias[c];
        }
        if(flags&DSASM_FS2_FEAT_VOICING){
            float v=(in&&in->voicing?in->voicing[t]:0.0f)*x->voicing_scale;
            for(size_t c=0;c<C;c++)dst[c]+=v*x->voicing_weight[c]+x->voicing_bias[c];
        }
        if(flags&DSASM_FS2_FEAT_TENSION){
            float v=(in&&in->tension?in->tension[t]:0.0f)*x->tension_scale;
            for(size_t c=0;c<C;c++)dst[c]+=v*x->tension_weight[c]+x->tension_bias[c];
        }
    }
    if(dbg)dbg_copy(dbg->variance_tc,cond,T*C);

    for(size_t t=0;t<T;t++){
        float*dst=cond+t*C;
        if(flags&DSASM_FS2_FEAT_KEY_SHIFT){
            float g=in&&in->gender?in->gender[t]:0.0f;
            g=clipf_local(g,x->gender_clip_min,x->gender_clip_max)*x->gender_pre_scale*x->key_shift_scale;
            for(size_t c=0;c<C;c++)dst[c]+=g*x->key_shift_weight[c]+x->key_shift_bias[c];
        }
    }
    if(dbg)dbg_copy(dbg->key_shift_tc,cond,T*C);

    for(size_t t=0;t<T;t++){
        float*dst=cond+t*C;
        if(flags&DSASM_FS2_FEAT_SPEED){
            float v=in&&in->velocity?in->velocity[t]:1.0f;
            v=clipf_local(v,x->speed_clip_min,x->speed_clip_max)*x->speed_scale;
            for(size_t c=0;c<C;c++)dst[c]+=v*x->speed_weight[c]+x->speed_bias[c];
        }
    }
    if(dbg)dbg_copy(dbg->speed_tc,cond,T*C);

    for(size_t t=0;t<T;t++){
        float*dst=cond+t*C;
        if(flags&DSASM_FS2_FEAT_SPEAKER){
            const float*sp=in->speaker_embedding_tc+t*C;
            for(size_t c=0;c<C;c++)dst[c]+=sp[c];
        }
    }
    if(dbg)dbg_copy(dbg->speaker_tc,cond,T*C);
    return 0;
}

int ds_fs2_acoustic_condition_deploy_f32_avx2(
    const DSAsmFS2AcousticWeights*w,const DSAsmFS2DeploymentExtras*x,const DSAsmFS2DeploymentInputs*in,
    const int32_t*tok,size_t P,const int32_t*mel2ph,const float*f0,size_t T,
    float*cond,float*ws,DSAsmThreadPool*pool){
    return ds_fs2_acoustic_condition_deploy_impl_f32_avx2(w,x,in,tok,P,mel2ph,f0,T,cond,ws,pool,NULL);
}

int ds_fs2_acoustic_condition_deploy_debug_f32_avx2(
    const DSAsmFS2AcousticWeights*w,const DSAsmFS2DeploymentExtras*x,const DSAsmFS2DeploymentInputs*in,
    const int32_t*tok,size_t P,const int32_t*mel2ph,const float*f0,size_t T,
    float*cond,float*ws,DSAsmThreadPool*pool,DSAsmFS2DeploymentDebug*dbg){
    return ds_fs2_acoustic_condition_deploy_impl_f32_avx2(w,x,in,tok,P,mel2ph,f0,T,cond,ws,pool,dbg);
}
