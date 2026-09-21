#include "dsasm_model.h"
#include "dsasm_full_acoustic.h"

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static size_t al64(size_t x){ return (x + 63u) & ~(size_t)63u; }

static int map_ro(const char *path, void **base, size_t *bytes){
    if(!path||!base||!bytes)return -1;
    int fd=open(path,O_RDONLY);
    if(fd<0)return -2;
    struct stat st;
    if(fstat(fd,&st)!=0||st.st_size<64){close(fd);return -3;}
    void *p=mmap(NULL,(size_t)st.st_size,PROT_READ,MAP_PRIVATE,fd,0);
    close(fd);
    if(p==MAP_FAILED)return -4;
    *base=p;*bytes=(size_t)st.st_size;return 0;
}

static void unmap_one(void **p,size_t *n){
    if(p&&*p&&n&&*n)munmap(*p,*n);
    if(p) *p=NULL;
    if(n) *n=0;
}

static int take_f32(const uint8_t *base,size_t bytes,size_t *off,size_t nf,const float **out){
    if(!base||!off||!out)return -1;
    size_t o=al64(*off);
    if(nf > (SIZE_MAX/sizeof(float)))return -1;
    size_t nb=nf*sizeof(float);
    if(o>bytes||nb>bytes-o)return -1;
    *out=(const float*)(base+o);*off=o+nb;return 0;
}

static int load_fs2(DSAsmAcousticModel *m,const char *path){
    int rc=map_ro(path,&m->fs2_map,&m->fs2_map_bytes);if(rc)return rc;
    const uint8_t*b=(const uint8_t*)m->fs2_map;size_t n=m->fs2_map_bytes;
    if(n<64)return -10;
    const int is25=memcmp(b,"DSFS25\0\0",8)==0;
    const int is21=memcmp(b,"DSFS21\0\0",8)==0;
    if(!is25&&!is21)return -10;
    uint32_t ver=0,V=0,C=0,L=0,H=0,K=0,inter=0,nlang=0,flags=0;float theta=0.0f;
    float breath_scale=1.0f,voicing_scale=1.0f,tension_scale=1.0f;
    if(is21){
        uint32_t h[7];memcpy(h,b+8,sizeof(h));memcpy(&theta,b+8+sizeof(h),sizeof(theta));
        ver=h[0];V=h[1];C=h[2];L=h[3];H=h[4];K=h[5];inter=h[6];
        if(ver!=1)return -11;
    }else{
        uint32_t h[9];float f[4];memcpy(h,b+8,sizeof(h));memcpy(f,b+8+sizeof(h),sizeof(f));
        ver=h[0];V=h[1];C=h[2];L=h[3];H=h[4];K=h[5];inter=h[6];nlang=h[7];flags=h[8];
        theta=f[0];breath_scale=f[1];voicing_scale=f[2];tension_scale=f[3];
        if(ver!=1)return -11;
    }
    if(!V||!C||!L||!H||K!=3||C%16u||C%H)return -11;
    DSAsmFS2EncoderLayer*ls=(DSAsmFS2EncoderLayer*)calloc(L,sizeof(*ls));if(!ls)return -12;
    m->fs2_layers_owned=ls;
    size_t off=64;
    DSAsmFS2AcousticWeights*w=&m->fs2;
    memset(&m->fs2_extras,0,sizeof(m->fs2_extras));
    w->encoder.vocab_size=V;w->encoder.hidden_size=C;w->encoder.num_layers=L;w->encoder.num_heads=H;
    w->encoder.ffn_kernel_size=K;w->encoder.rope_interleaved=inter;w->encoder.rope_theta=theta;w->encoder.layers=ls;
#define TAKE(dst,count) do{ if(take_f32(b,n,&off,(count),&(dst))!=0)return -13; }while(0)
    TAKE(w->encoder.token_embedding,(size_t)V*C);TAKE(w->encoder.dur_weight,C);TAKE(w->encoder.dur_bias,C);
    for(uint32_t i=0;i<L;i++){
        DSAsmFS2EncoderLayer*q=&ls[i];
        TAKE(q->ln1_gamma,C);TAKE(q->ln1_beta,C);TAKE(q->qkv_weight_m4n16,(size_t)3*C*C);TAKE(q->qkv_bias,3u*C);
        TAKE(q->out_weight_m4n16,(size_t)C*C);TAKE(q->out_bias,C);TAKE(q->ln2_gamma,C);TAKE(q->ln2_beta,C);
        TAKE(q->ffn1_weight_m4n16,(size_t)12*C*C);TAKE(q->ffn1_bias,4u*C);TAKE(q->ffn2_weight_m4n16,(size_t)4*C*C);TAKE(q->ffn2_bias,C);
    }
    TAKE(w->encoder.final_ln_gamma,C);TAKE(w->encoder.final_ln_beta,C);
    TAKE(w->stretch_w1_m4n16,(size_t)4*C*C);TAKE(w->stretch_b1,4u*C);TAKE(w->stretch_w2_m4n16,(size_t)4*C*C);TAKE(w->stretch_b2,C);
    TAKE(w->gru_w_ih_m4n16,(size_t)3*C*C);TAKE(w->gru_b_ih,3u*C);TAKE(w->gru_w_hh_m4n16,(size_t)3*C*C);TAKE(w->gru_b_hh,3u*C);
    TAKE(w->pitch_weight,C);TAKE(w->pitch_bias,C);
    if(is25){
        DSAsmFS2DeploymentExtras*x=&m->fs2_extras;x->num_languages=nlang;x->flags=flags;
        x->breath_scale=breath_scale;x->voicing_scale=voicing_scale;x->tension_scale=tension_scale;
        if(flags&DSASM_FS2_FEAT_LANGUAGE)TAKE(x->language_embedding,(size_t)nlang*C);
        if(flags&DSASM_FS2_FEAT_LANGUAGE_MASK)TAKE(x->language_token_mask,V);
        if(flags&DSASM_FS2_FEAT_BREATH){TAKE(x->breath_weight,C);TAKE(x->breath_bias,C);}
        if(flags&DSASM_FS2_FEAT_VOICING){TAKE(x->voicing_weight,C);TAKE(x->voicing_bias,C);}
        if(flags&DSASM_FS2_FEAT_TENSION){TAKE(x->tension_weight,C);TAKE(x->tension_bias,C);}
        if(flags&DSASM_FS2_FEAT_KEY_SHIFT){TAKE(x->key_shift_weight,C);TAKE(x->key_shift_bias,C);}
        if(flags&DSASM_FS2_FEAT_SPEED){TAKE(x->speed_weight,C);TAKE(x->speed_bias,C);}
        const float*fc=NULL;TAKE(fc,7u);
        x->gender_clip_min=fc[0];x->gender_clip_max=fc[1];x->gender_pre_scale=fc[2];x->key_shift_scale=fc[3];
        x->speed_clip_min=fc[4];x->speed_clip_max=fc[5];x->speed_scale=fc[6];
        if(flags&DSASM_FS2_FEAT_STRETCH_TABLE)TAKE(x->stretch_table,(size_t)1001u*C);
    }
#undef TAKE
    return 0;
}

static int load_aux(DSAsmAcousticModel *m,const char *path){
    int rc=map_ro(path,&m->aux_map,&m->aux_map_bytes);if(rc)return rc;
    const uint8_t*b=(const uint8_t*)m->aux_map;size_t n=m->aux_map_bytes;
    if(n<64||memcmp(b,"DSAUX20\0",8)!=0)return -20;
    uint32_t h[8];memcpy(h,b+8,sizeof(h));
    uint32_t ver=h[0],I=h[1],C=h[2],D=h[3],L=h[4],K=h[5];
    if(ver!=1||!I||!C||!D||!L||K!=7||C%16u||D%16u)return -21;
    DSAsmAuxConvNeXtBlock*bs=(DSAsmAuxConvNeXtBlock*)calloc(L,sizeof(*bs));if(!bs)return -22;
    m->aux_blocks_owned=bs;
    DSAsmAuxConvNeXtWeights*w=&m->aux;w->input_dim=I;w->channels=C;w->output_dim=D;w->num_layers=L;w->kernel_size=K;w->blocks=bs;
    size_t off=64;
#define TAKE(dst,count) do{ if(take_f32(b,n,&off,(count),&(dst))!=0)return -23; }while(0)
    TAKE(w->in_weight_m4n16,(size_t)C*7u*I);TAKE(w->in_bias,C);
    for(uint32_t i=0;i<L;i++){
        DSAsmAuxConvNeXtBlock*q=&bs[i];
        TAKE(q->dw_weight_tap_major,7u*C);TAKE(q->dw_bias,C);TAKE(q->ln_gamma,C);TAKE(q->ln_beta,C);
        TAKE(q->pw1_weight_m4n16,(size_t)4*C*C);TAKE(q->pw1_bias,4u*C);TAKE(q->pw2_weight_m4n16,(size_t)4*C*C);TAKE(q->pw2_bias,C);TAKE(q->gamma,C);
    }
    TAKE(w->out_weight_m4n16,(size_t)D*7u*C);TAKE(w->out_bias,D);
#undef TAKE
    return 0;
}

static int load_rf(DSAsmAcousticModel *m,const char *path){
    int rc=map_ro(path,&m->rf_map,&m->rf_map_bytes);if(rc)return rc;
    const uint8_t*b=(const uint8_t*)m->rf_map;size_t n=m->rf_map_bytes;
    if(n<64||memcmp(b,"DSLYNX7\0",8)!=0)return -30;
    uint32_t h[12];memcpy(h,b+8,sizeof(h));
    uint32_t ver=h[0],I=h[1],Q=h[2],C=h[3],H=h[4],L=h[5],K=h[6],glu=h[7];
    if(ver!=1||!I||!Q||!C||!H||!L||K!=31||C%16u||I%16u||H%8u||(glu!=DSASM_GLU_ATAN&&glu!=DSASM_GLU_SOFTSIGN))return -31;
    DSAsmLynxNet2Block*bs=(DSAsmLynxNet2Block*)calloc(L,sizeof(*bs));if(!bs)return -32;
    m->rf_blocks_owned=bs;
    DSAsmLynxNet2Weights*w=&m->rf;w->input_dim=I;w->condition_dim=Q;w->channels=C;w->hidden_dim=H;w->num_layers=L;w->kernel_size=K;w->glu_type=glu;w->blocks=bs;
    size_t off=64;
#define TAKE(dst,count) do{ if(take_f32(b,n,&off,(count),&(dst))!=0)return -33; }while(0)
    TAKE(w->input_weight_m4n16,(size_t)C*I);TAKE(w->input_bias,C);TAKE(w->condition_weight_m4n16,(size_t)C*Q);TAKE(w->condition_bias,C);
    TAKE(w->time1_weight_m4n16,(size_t)4*C*C);TAKE(w->time1_bias,4u*C);TAKE(w->time2_weight_m4n16,(size_t)4*C*C);TAKE(w->time2_bias,C);
    for(uint32_t i=0;i<L;i++){
        DSAsmLynxNet2Block*q=&bs[i];
        TAKE(q->ln_gamma,C);TAKE(q->ln_beta,C);TAKE(q->dw_weight_tap_major,31u*C);TAKE(q->dw_bias,C);
        TAKE(q->glu1_weight,(size_t)2*H*C);TAKE(q->glu1_bias,2u*H);TAKE(q->glu2_weight,(size_t)2*H*H);TAKE(q->glu2_bias,2u*H);
        TAKE(q->out_weight_m4n16,(size_t)C*H);TAKE(q->out_bias,C);
    }
    TAKE(w->post_norm_gamma,C);TAKE(w->post_norm_beta,C);TAKE(w->output_weight_m4n16,(size_t)I*C);TAKE(w->output_bias,I);
#undef TAKE
    return 0;
}

int ds_acoustic_model_valid(const DSAsmAcousticModel*m){
    if(!m)return 0;
    return m->fs2.encoder.hidden_size && m->aux.input_dim && m->rf.condition_dim &&
           m->fs2.encoder.hidden_size==m->aux.input_dim &&
           m->aux.input_dim==m->rf.condition_dim &&
           m->aux.output_dim==m->rf.input_dim;
}

void ds_acoustic_model_unload(DSAsmAcousticModel*m){
    if(!m)return;
    free(m->fs2_layers_owned);free(m->aux_blocks_owned);free(m->rf_blocks_owned);
    m->fs2_layers_owned=NULL;m->aux_blocks_owned=NULL;m->rf_blocks_owned=NULL;
    unmap_one(&m->fs2_map,&m->fs2_map_bytes);unmap_one(&m->aux_map,&m->aux_map_bytes);unmap_one(&m->rf_map,&m->rf_map_bytes);
    memset(&m->fs2,0,sizeof(m->fs2));memset(&m->fs2_extras,0,sizeof(m->fs2_extras));memset(&m->aux,0,sizeof(m->aux));memset(&m->rf,0,sizeof(m->rf));
}

int ds_acoustic_model_load(DSAsmAcousticModel*m,const char*fs2_path,const char*aux_path,const char*rf_path){
    if(!m) return -1;
    memset(m,0,sizeof(*m));
    int rc=load_fs2(m,fs2_path);if(rc){ds_acoustic_model_unload(m);return rc;}
    rc=load_aux(m,aux_path);if(rc){ds_acoustic_model_unload(m);return rc;}
    rc=load_rf(m,rf_path);if(rc){ds_acoustic_model_unload(m);return rc;}
    if(!ds_acoustic_model_valid(m)){ds_acoustic_model_unload(m);return -40;}
    return 0;
}

size_t ds_acoustic_model_workspace_floats(const DSAsmAcousticModel*m,size_t P,size_t T){
    if(!ds_acoustic_model_valid(m))return 0;
    return ds_full_acoustic_normfast_workspace_floats(&m->fs2,&m->aux,&m->rf,P,T);
}

int ds_acoustic_model_infer_f32_avx2(
    const DSAsmAcousticModel*m,const int32_t*tok,size_t P,const int32_t*mel2ph,const float*f0,size_t T,
    const float*noise,const float*lo,const float*hi,size_t rd,float t0,float scale,size_t steps,
    float*out,float*ws,DSAsmThreadPool*pool){
    if(!ds_acoustic_model_valid(m))return -1;
    return ds_full_acoustic_infer_normfast_f32_avx2(&m->fs2,&m->aux,&m->rf,tok,P,mel2ph,f0,T,
        noise,lo,hi,rd,t0,scale,steps,out,ws,pool);
}

int ds_acoustic_model_infer_deploy_f32_avx2(
    const DSAsmAcousticModel*m,const DSAsmFS2DeploymentInputs*inputs,
    const int32_t*tok,size_t P,const int32_t*mel2ph,const float*f0,size_t T,
    const float*noise,const float*lo,const float*hi,size_t rd,float t0,float scale,size_t steps,
    float*out,float*ws,DSAsmThreadPool*pool){
    if(!ds_acoustic_model_valid(m))return -1;
    return ds_full_acoustic_infer_deploy_normfast_f32_avx2(&m->fs2,&m->fs2_extras,inputs,&m->aux,&m->rf,
        tok,P,mel2ph,f0,T,noise,lo,hi,rd,t0,scale,steps,out,ws,pool);
}
