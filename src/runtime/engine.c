#define _POSIX_C_SOURCE 200809L
#include "dsasm_engine.h"
#include "dsasm_model.h"
#include "dsasm_threadpool.h"
#include "dsasm_vocoder_graph.h"

#include <dirent.h>
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

typedef struct {
    DSAsmVocoderGraph *graph;
    size_t frames;
    size_t samples;
} vocoder_bucket;

struct dsasm_engine {
    DSAsmAcousticModel acoustic;
    DSAsmThreadPool *pool;
    vocoder_bucket *buckets;
    size_t bucket_count;
    float *spec_min;
    float *spec_max;
    size_t spec_dims;
    float t_start;
    float time_scale;
    size_t steps;
    size_t sample_rate;
    size_t hop_size;
    size_t mel_bins;
    pthread_mutex_t render_mutex;
    atomic_uint_fast64_t cancel_epoch;
    char error[256];
    int32_t *mel2ph_scratch;
    size_t mel2ph_capacity;
    float *noise_scratch,*mel_scratch,*workspace_scratch,*speaker_scratch;
    float *vocoder_mel_scratch,*vocoder_f0_scratch,*wave_scratch,*emit_scratch,*tail_scratch;
    size_t noise_capacity,mel_capacity,workspace_capacity,speaker_capacity;
    size_t vocoder_mel_capacity,vocoder_f0_capacity,wave_capacity,emit_capacity,tail_capacity;
};

static _Thread_local char create_error[256];

static void set_error(char *dst,size_t cap,const char *fmt,...){
    va_list ap;va_start(ap,fmt);vsnprintf(dst,cap,fmt,ap);va_end(ap);
}
static int fail(dsasm_engine *e,int rc,const char *fmt,...){
    va_list ap;va_start(ap,fmt);vsnprintf(e->error,sizeof(e->error),fmt,ap);va_end(ap);return rc;
}
static int mul_overflow(size_t a,size_t b,size_t *out){if(a&&b>SIZE_MAX/a)return 1;*out=a*b;return 0;}
static void *alloc64(size_t count,size_t elem){
    size_t bytes;if(mul_overflow(count,elem,&bytes))return NULL;if(!bytes)bytes=1;
    size_t rounded=(bytes+63u)&~(size_t)63u;if(rounded<bytes)return NULL;
    return aligned_alloc(64,rounded);
}
static int ensure_buffer(void **ptr,size_t *capacity,size_t count,size_t elem){
    if(count<=*capacity)return 0;
    void *next=alloc64(count,elem);if(!next)return -1;free(*ptr);*ptr=next;*capacity=count;return 0;
}
static uint64_t xs64(uint64_t *s){uint64_t x=*s;x^=x>>12;x^=x<<25;x^=x>>27;*s=x;return x*2685821657736338717ULL;}
static float uni01(uint64_t *s){return ((xs64(s)>>40)+1.0f)*(1.0f/16777217.0f);}
static void gaussian(float *x,size_t n,uint64_t seed){
    uint64_t s=seed?seed:UINT64_C(0x9e3779b97f4a7c15);size_t i=0;
    while(i<n){float u1=uni01(&s),u2=uni01(&s);float r=sqrtf(-2.f*logf(u1));float a=6.2831853071795864769f*u2;x[i++]=r*cosf(a);if(i<n)x[i++]=r*sinf(a);}
}
static int set_vec(float **dst,size_t *dims,const char *text){
    size_t cap=16,n=0;float *v=malloc(cap*sizeof(*v));if(!v)return -1;const char *p=text;
    while(*p){while(*p&&(*p==' '||*p=='\t'||*p==','))p++;if(!*p)break;char *end=NULL;errno=0;float x=strtof(p,&end);if(end==p||errno||!isfinite(x)){free(v);return -1;}if(n==cap){cap*=2;float *q=realloc(v,cap*sizeof(*v));if(!q){free(v);return -1;}v=q;}v[n++]=x;p=end;}
    if(!n){free(v);return -1;}free(*dst);*dst=v;*dims=n;return 0;
}
static int load_config(dsasm_engine *e,const char *dir){
    e->spec_min=malloc(sizeof(float));e->spec_max=malloc(sizeof(float));if(!e->spec_min||!e->spec_max)return -1;
    e->spec_min[0]=-12.f;e->spec_max[0]=0.f;e->spec_dims=1;e->t_start=.4f;e->time_scale=1000.f;e->steps=20;e->sample_rate=44100;e->hop_size=512;
    char path[PATH_MAX];int n=snprintf(path,sizeof(path),"%s/model.conf",dir);if(n<0||(size_t)n>=sizeof(path))return -2;
    FILE *f=fopen(path,"r");if(!f)return errno==ENOENT?0:-2;char line[32768];size_t lo_dims=1,hi_dims=1;
    while(fgets(line,sizeof(line),f)){
        char *nl=strchr(line,'\n');if(nl)*nl=0;char *eq=strchr(line,'=');if(!eq||line[0]=='#')continue;*eq++=0;
        if(!strcmp(line,"spec_min")){if(set_vec(&e->spec_min,&lo_dims,eq)){fclose(f);return -3;}}
        else if(!strcmp(line,"spec_max")){if(set_vec(&e->spec_max,&hi_dims,eq)){fclose(f);return -3;}}
        else if(!strcmp(line,"t_start"))e->t_start=strtof(eq,NULL);
        else if(!strcmp(line,"time_scale_factor"))e->time_scale=strtof(eq,NULL);
        else if(!strcmp(line,"steps"))e->steps=(size_t)strtoull(eq,NULL,10);
        else if(!strcmp(line,"sample_rate"))e->sample_rate=(size_t)strtoull(eq,NULL,10);
        else if(!strcmp(line,"hop_size"))e->hop_size=(size_t)strtoull(eq,NULL,10);
    }
    fclose(f);if(lo_dims!=hi_dims)return -3;e->spec_dims=lo_dims;
    if(!e->steps||!e->sample_rate||!e->hop_size||e->t_start<0.f||e->t_start>1.f||e->time_scale<=0.f)return -3;
    return 0;
}
static int bucket_cmp(const void *a,const void *b){
    const vocoder_bucket *x=a,*y=b;return (x->frames>y->frames)-(x->frames<y->frames);
}
static int append_bucket(dsasm_engine *e,const char *path){
    DSAsmVocoderGraph *g=ds_vocoder_graph_load_with_pool(path,e->pool);if(!g)return -1;
    size_t frames=ds_vocoder_graph_frames(g),mel=ds_vocoder_graph_mel_bins(g),samples=ds_vocoder_graph_samples(g);
    if(!frames||!samples||samples%frames||mel!=e->mel_bins||samples/frames!=e->hop_size){ds_vocoder_graph_free(g);return -2;}
    vocoder_bucket *q=realloc(e->buckets,(e->bucket_count+1)*sizeof(*q));if(!q){ds_vocoder_graph_free(g);return -3;}
    e->buckets=q;e->buckets[e->bucket_count++]=(vocoder_bucket){g,frames,samples};return 0;
}
static int numeric_bundle_name(const char *name){
    const char *suffix=".dsv35";size_t n=strlen(name),s=strlen(suffix);if(n<=s||strcmp(name+n-s,suffix))return 0;
    for(size_t i=0;i<n-s;i++)if(name[i]<'0'||name[i]>'9')return 0;
    return 1;
}
static int load_buckets(dsasm_engine *e,const char *path){
    struct stat st;if(stat(path,&st))return -1;
    if(S_ISREG(st.st_mode))return append_bucket(e,path);
    if(!S_ISDIR(st.st_mode))return -1;
    DIR *d=opendir(path);if(!d)return -1;struct dirent *de;int rc=0;
    while((de=readdir(d))){
        if(!numeric_bundle_name(de->d_name))continue;
        char full[PATH_MAX];int n=snprintf(full,sizeof(full),"%s/%s",path,de->d_name);if(n<0||(size_t)n>=sizeof(full)){rc=-1;break;}
        int q=append_bucket(e,full);if(q){rc=q;break;}
    }
    closedir(d);if(rc)return rc;if(!e->bucket_count)return -1;qsort(e->buckets,e->bucket_count,sizeof(*e->buckets),bucket_cmp);
    for(size_t i=1;i<e->bucket_count;i++)if(e->buckets[i-1].frames==e->buckets[i].frames)return -2;
    return 0;
}
static void free_engine(dsasm_engine *e){
    if(!e)return;
    for(size_t i=0;i<e->bucket_count;i++)ds_vocoder_graph_free(e->buckets[i].graph);
    free(e->buckets);
    if(e->pool)ds_threadpool_destroy(e->pool);
    ds_acoustic_model_unload(&e->acoustic);free(e->spec_min);free(e->spec_max);
    free(e->mel2ph_scratch);free(e->noise_scratch);free(e->mel_scratch);free(e->workspace_scratch);free(e->speaker_scratch);
    free(e->vocoder_mel_scratch);free(e->vocoder_f0_scratch);free(e->wave_scratch);free(e->emit_scratch);free(e->tail_scratch);
    pthread_mutex_destroy(&e->render_mutex);free(e);
}

uint32_t dsasm_engine_abi_version(void){return DSASM_ENGINE_ABI_VERSION;}
int dsasm_engine_is_supported(char *reason,size_t reason_size){
#if !defined(__linux__) || !defined(__x86_64__)
    if(reason&&reason_size)snprintf(reason,reason_size,"requires Linux x86-64");
    return 0;
#elif defined(__GNUC__) || defined(__clang__)
    __builtin_cpu_init();
    if(!__builtin_cpu_supports("avx2")||!__builtin_cpu_supports("fma")){if(reason&&reason_size)snprintf(reason,reason_size,"requires AVX2 and FMA");return 0;}
    if(reason&&reason_size)reason[0]=0;
    return 1;
#else
    if(reason&&reason_size)snprintf(reason,reason_size,"CPU feature detection unavailable");
    return 0;
#endif
}
dsasm_engine *dsasm_engine_create(const char *acoustic_dir,const char *vocoder_dir,int workers){
    create_error[0]=0;if(!acoustic_dir||!vocoder_dir||workers<0){set_error(create_error,sizeof(create_error),"invalid engine arguments");return NULL;}
    if(!dsasm_engine_is_supported(create_error,sizeof(create_error)))return NULL;
    dsasm_engine *e=calloc(1,sizeof(*e));if(!e){set_error(create_error,sizeof(create_error),"engine allocation failed");return NULL;}
    if(pthread_mutex_init(&e->render_mutex,NULL)){set_error(create_error,sizeof(create_error),"render mutex creation failed");free(e);return NULL;}
    atomic_init(&e->cancel_epoch,0);
    int rc=load_config(e,acoustic_dir);if(rc){set_error(create_error,sizeof(create_error),"invalid model.conf (%d)",rc);free_engine(e);return NULL;}
    char fs[PATH_MAX],aux[PATH_MAX],rf[PATH_MAX];
    int fsn=snprintf(fs,sizeof(fs),"%s/fs2_acoustic.dsfs",acoustic_dir);
    int auxn=snprintf(aux,sizeof(aux),"%s/aux_convnext.dsa",acoustic_dir);
    int rfn=snprintf(rf,sizeof(rf),"%s/lynxnet2.dsn",acoustic_dir);
    if(fsn<0||auxn<0||rfn<0||(size_t)fsn>=sizeof(fs)||(size_t)auxn>=sizeof(aux)||(size_t)rfn>=sizeof(rf)){set_error(create_error,sizeof(create_error),"acoustic path is too long");free_engine(e);return NULL;}
    rc=ds_acoustic_model_load(&e->acoustic,fs,aux,rf);if(rc){set_error(create_error,sizeof(create_error),"acoustic model load failed (%d)",rc);free_engine(e);return NULL;}
    e->mel_bins=e->acoustic.rf.input_dim;if(e->spec_dims!=1&&e->spec_dims!=e->mel_bins){set_error(create_error,sizeof(create_error),"spec range has %zu values, expected 1 or %zu",e->spec_dims,e->mel_bins);free_engine(e);return NULL;}
    e->pool=ds_threadpool_create((size_t)workers);if(!e->pool){set_error(create_error,sizeof(create_error),"thread pool creation failed");free_engine(e);return NULL;}
    rc=load_buckets(e,vocoder_dir);if(rc){set_error(create_error,sizeof(create_error),"vocoder bucket load failed (%d); expected N.dsv35 files",rc);free_engine(e);return NULL;}
    e->error[0]=0;return e;
}
static vocoder_bucket *choose_bucket(dsasm_engine *e,size_t remaining,int first){
    if(first)return &e->buckets[0];
    for(size_t i=0;i<e->bucket_count;i++)if(e->buckets[i].frames>=remaining)return &e->buckets[i];
    return &e->buckets[e->bucket_count-1];
}
static int cancelled(dsasm_engine *e,uint64_t epoch){return atomic_load_explicit(&e->cancel_epoch,memory_order_relaxed)!=epoch;}
int dsasm_engine_render(dsasm_engine *e,const dsasm_request *r,dsasm_pcm_callback callback,void *userdata){
    if(!e||!r||!callback)return DSASM_E_INVALID;
    pthread_mutex_lock(&e->render_mutex);e->error[0]=0;uint64_t epoch=atomic_load_explicit(&e->cancel_epoch,memory_order_relaxed);int rc=DSASM_OK;
    float *noise=NULL,*mel=NULL,*workspace=NULL,*vin=NULL,*vf0=NULL,*wave=NULL,*emit=NULL,*tail=NULL;
#define RFAIL(code,...) do{rc=fail(e,(code),__VA_ARGS__);goto done;}while(0)
    if(r->abi_version!=DSASM_ENGINE_ABI_VERSION||r->struct_size<sizeof(dsasm_request))RFAIL(DSASM_E_INVALID,"request ABI mismatch");
    if(!r->token_ids||!r->text_tokens||!r->f0||!r->mel_frames)RFAIL(DSASM_E_INVALID,"tokens, f0, and nonzero dimensions are required");
    const size_t P=r->text_tokens,T=r->mel_frames,M=e->mel_bins,C=e->acoustic.fs2.encoder.hidden_size;size_t NM;
    if(mul_overflow(T,M,&NM))RFAIL(DSASM_E_INVALID,"request dimensions overflow");
    for(size_t i=0;i<P;i++)if(r->token_ids[i]<0||(size_t)r->token_ids[i]>=e->acoustic.fs2.encoder.vocab_size)RFAIL(DSASM_E_INVALID,"token %zu is out of range",i);
    const int32_t *mel2ph=r->mel2ph;
    if(!mel2ph){
        if(!r->durations)RFAIL(DSASM_E_INVALID,"mel2ph or durations is required");
        if(ensure_buffer((void **)&e->mel2ph_scratch,&e->mel2ph_capacity,T,sizeof(*e->mel2ph_scratch)))RFAIL(DSASM_E_NOMEM,"mel2ph allocation failed");
        size_t z=0;for(size_t i=0;i<P;i++){if(r->durations[i]<0||(size_t)r->durations[i]>T-z)RFAIL(DSASM_E_INVALID,"durations do not match mel_frames");for(int32_t j=0;j<r->durations[i];j++)e->mel2ph_scratch[z++]=(int32_t)i+1;}if(z!=T)RFAIL(DSASM_E_INVALID,"durations sum to %zu, expected %zu",z,T);mel2ph=e->mel2ph_scratch;
    }else for(size_t i=0;i<T;i++)if(mel2ph[i]<0||(size_t)mel2ph[i]>P)RFAIL(DSASM_E_INVALID,"mel2ph[%zu] is out of range",i);
    uint32_t features=e->acoustic.fs2_extras.flags;if((features&DSASM_FS2_FEAT_LANGUAGE)&&!r->language_ids)RFAIL(DSASM_E_INVALID,"model requires language_ids");
    if(features&DSASM_FS2_FEAT_LANGUAGE)for(size_t i=0;i<P;i++)if(r->language_ids[i]<0||(uint32_t)r->language_ids[i]>=e->acoustic.fs2_extras.num_languages)RFAIL(DSASM_E_INVALID,"language_ids[%zu] is out of range",i);
    const float *speaker=r->speaker_embedding;
    if(features&DSASM_FS2_FEAT_SPEAKER){
        if(!speaker||(r->speaker_embedding_frames!=1&&r->speaker_embedding_frames!=T))RFAIL(DSASM_E_INVALID,"speaker embedding must contain 1 or mel_frames rows");
        if(r->speaker_embedding_frames==1){size_t TC;if(mul_overflow(T,C,&TC))RFAIL(DSASM_E_INVALID,"speaker dimensions overflow");if(ensure_buffer((void **)&e->speaker_scratch,&e->speaker_capacity,TC,sizeof(float)))RFAIL(DSASM_E_NOMEM,"speaker allocation failed");for(size_t t=0;t<T;t++)memcpy(e->speaker_scratch+t*C,speaker,C*sizeof(float));speaker=e->speaker_scratch;}
    }
    float t_start=e->t_start;if(r->flags&DSASM_REQUEST_USE_DEPTH){if(!isfinite(r->depth)||r->depth<0.f||r->depth>1.f)RFAIL(DSASM_E_INVALID,"depth must be in [0,1]");t_start=fmaxf(1.f-r->depth,0.f);}if(r->flags&DSASM_REQUEST_USE_T_START)t_start=r->t_start;
    float scale=(r->flags&DSASM_REQUEST_USE_TIME_SCALE)?r->time_scale_factor:e->time_scale;size_t steps=(r->flags&DSASM_REQUEST_USE_STEPS)?r->steps:e->steps;
    if(!isfinite(t_start)||t_start<0.f||t_start>1.f||!isfinite(scale)||scale<=0.f||!steps)RFAIL(DSASM_E_INVALID,"invalid sampling parameters");
    const float *lo=e->spec_min,*hi=e->spec_max;size_t dims=e->spec_dims;if(r->flags&DSASM_REQUEST_USE_SPEC_RANGE){lo=r->spec_min;hi=r->spec_max;dims=r->spec_range_dims;if(!lo||!hi||(dims!=1&&dims!=M))RFAIL(DSASM_E_INVALID,"invalid spec range");}
    size_t wsn=ds_acoustic_model_workspace_floats(&e->acoustic,P,T);
    if(ensure_buffer((void **)&e->noise_scratch,&e->noise_capacity,NM,sizeof(float))||ensure_buffer((void **)&e->mel_scratch,&e->mel_capacity,NM,sizeof(float))||ensure_buffer((void **)&e->workspace_scratch,&e->workspace_capacity,wsn,sizeof(float)))RFAIL(DSASM_E_NOMEM,"acoustic buffer allocation failed");
    noise=e->noise_scratch;mel=e->mel_scratch;workspace=e->workspace_scratch;if(r->noise)memcpy(noise,r->noise,NM*sizeof(float));else gaussian(noise,NM,r->noise_seed);
    if(cancelled(e,epoch))RFAIL(DSASM_E_CANCELLED,"render cancelled");
    DSAsmFS2DeploymentInputs inputs={r->language_ids,r->breathiness,r->voicing,r->tension,r->gender,r->velocity,speaker};
    if(features)rc=ds_acoustic_model_infer_deploy_f32_avx2(&e->acoustic,&inputs,r->token_ids,P,mel2ph,r->f0,T,noise,lo,hi,dims,t_start,scale,steps,mel,workspace,e->pool);
    else rc=ds_acoustic_model_infer_f32_avx2(&e->acoustic,r->token_ids,P,mel2ph,r->f0,T,noise,lo,hi,dims,t_start,scale,steps,mel,workspace,e->pool);
    if(rc)RFAIL(DSASM_E_INTERNAL,"acoustic inference failed (%d)",rc);
    if(cancelled(e,epoch))RFAIL(DSASM_E_CANCELLED,"render cancelled");
    size_t max_frames=e->buckets[e->bucket_count-1].frames,max_samples=e->buckets[e->bucket_count-1].samples;
    size_t max_mel;if(mul_overflow(max_frames,M,&max_mel))RFAIL(DSASM_E_INVALID,"vocoder dimensions overflow");
    if(ensure_buffer((void **)&e->vocoder_mel_scratch,&e->vocoder_mel_capacity,max_mel,sizeof(float))||ensure_buffer((void **)&e->vocoder_f0_scratch,&e->vocoder_f0_capacity,max_frames,sizeof(float))||ensure_buffer((void **)&e->wave_scratch,&e->wave_capacity,max_samples,sizeof(float))||ensure_buffer((void **)&e->emit_scratch,&e->emit_capacity,max_samples,sizeof(float))||ensure_buffer((void **)&e->tail_scratch,&e->tail_capacity,max_samples,sizeof(float)))RFAIL(DSASM_E_NOMEM,"vocoder buffer allocation failed");
    vin=e->vocoder_mel_scratch;vf0=e->vocoder_f0_scratch;wave=e->wave_scratch;emit=e->emit_scratch;tail=e->tail_scratch;
    size_t frame=0,tail_n=0;uint64_t sample_offset=0;int first=1;
    while(frame<T){
        if(cancelled(e,epoch))RFAIL(DSASM_E_CANCELLED,"render cancelled");
        size_t remaining=T-frame;vocoder_bucket *b=choose_bucket(e,remaining,first);size_t valid=remaining<b->frames?remaining:b->frames;
        memset(vin,0,b->frames*M*sizeof(float));memset(vf0,0,b->frames*sizeof(float));memcpy(vin,mel+frame*M,valid*M*sizeof(float));memcpy(vf0,r->f0+frame,valid*sizeof(float));
        int vr=ds_vocoder_graph_infer(b->graph,vin,vf0,wave,0);if(vr)RFAIL(DSASM_E_INTERNAL,"vocoder inference failed (%d)",vr);if(cancelled(e,epoch))RFAIL(DSASM_E_CANCELLED,"render cancelled");
        size_t spf=b->samples/b->frames,valid_samples=valid*spf;if(tail_n>valid_samples)RFAIL(DSASM_E_INTERNAL,"invalid bucket overlap");memcpy(emit,wave,valid_samples*sizeof(float));
        if(tail_n){for(size_t i=0;i<tail_n;i++){float a=(float)(i+1)/(float)(tail_n+1);emit[i]=tail[i]*(1.f-a)+wave[i]*a;}}
        int has_future=frame+valid<T;size_t keep_frames=0;if(has_future){keep_frames=r->overlap_frames?r->overlap_frames:8u;if(keep_frames>=valid)keep_frames=valid/2;if(!keep_frames)RFAIL(DSASM_E_UNSUPPORTED,"vocoder bucket is too short for streaming overlap");}
        size_t keep=keep_frames*spf,emit_n=valid_samples-keep;if(keep)memcpy(tail,wave+emit_n,keep*sizeof(float));int is_final=!has_future;
        if(emit_n&&callback(userdata,sample_offset,emit,emit_n,is_final))RFAIL(DSASM_E_CALLBACK,"PCM callback stopped rendering");
        sample_offset+=emit_n;tail_n=keep;frame+=valid-keep_frames;first=0;
    }
done:
    pthread_mutex_unlock(&e->render_mutex);return rc;
#undef RFAIL
}
void dsasm_engine_cancel(dsasm_engine *e){if(e)atomic_fetch_add_explicit(&e->cancel_epoch,1,memory_order_relaxed);}
void dsasm_engine_destroy(dsasm_engine *e){free_engine(e);}
const char *dsasm_engine_last_error(const dsasm_engine *e){return e?e->error:"invalid engine";}
const char *dsasm_engine_last_create_error(void){return create_error;}
uint32_t dsasm_engine_sample_rate(const dsasm_engine *e){return e?(uint32_t)e->sample_rate:0;}
uint32_t dsasm_engine_hop_size(const dsasm_engine *e){return e?(uint32_t)e->hop_size:0;}
uint32_t dsasm_engine_mel_bins(const dsasm_engine *e){return e?(uint32_t)e->mel_bins:0;}
size_t dsasm_engine_bucket_count(const dsasm_engine *e){return e?e->bucket_count:0;}
uint32_t dsasm_engine_bucket_frames(const dsasm_engine *e,size_t i){return e&&i<e->bucket_count?(uint32_t)e->buckets[i].frames:0;}
