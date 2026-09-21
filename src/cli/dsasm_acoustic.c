#define _POSIX_C_SOURCE 200809L
#include "dsasm_model.h"
#include "dsasm_acoustic.h"
#include "dsasm_threadpool.h"
#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

typedef struct {
    float *lo,*hi; size_t lo_dims,hi_dims,range_dims;
    float t_start,scale; size_t steps; size_t sample_rate,hop_size;
} ModelCfg;

static double now_ms(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1000.0+t.tv_nsec/1e6;}
static void usage(const char *p){
    fprintf(stderr,
      "DiffSinger-ASM M27 native acoustic CLI\n"
      "usage:\n"
      "  %s inspect MODEL_DIR\n"
      "  %s infer MODEL_DIR --tokens TOKENS.txt --durations DUR.txt --f0 F0.txt --out MEL.f32 [options]\n"
      "base options:\n"
      "  --seed N | --noise NOISE.f32   --steps N   --t-start X   --depth X   --time-scale X\n"
      "  --spec-min CSV --spec-max CSV  --threads N   --profile-stages\n"
      "deployment-condition options (DSFS25):\n"
      "  --language-id N | --languages LANG.txt\n"
      "  --speaker-emb FILE              raw/text C floats, or frame-wise T*C floats\n"
      "  --breathiness FILE --voicing FILE --tension FILE --gender FILE --velocity FILE\n"
      "debug parity options:\n"
      "  --dump-condition FILE.f32 --dump-aux-mel FILE.f32 --dump-fs2-stages DIR\n"
      "text vectors accept whitespace and/or commas. durations are integer frame counts.\n"
      "--depth is normalized [0,1] and maps to t_start=max(1-depth,0).\n",p,p);
}
static char *slurp(const char *path,size_t *n){FILE*f=fopen(path,"rb");if(!f)return NULL;if(fseek(f,0,SEEK_END)){fclose(f);return NULL;}long z=ftell(f);if(z<0){fclose(f);return NULL;}rewind(f);char*b=malloc((size_t)z+1);if(!b){fclose(f);return NULL;}size_t got=fread(b,1,(size_t)z,f);fclose(f);b[got]=0;if(n)*n=got;return b;}
static int parse_i32_text(const char*path,int32_t**out,size_t*n){size_t nb;char*b=slurp(path,&nb);if(!b)return -1;size_t cap=32,k=0;int32_t*a=malloc(cap*sizeof(*a));if(!a){free(b);return -2;}char*p=b;while(*p){while(*p&&(isspace((unsigned char)*p)||*p==','))p++;if(!*p)break;errno=0;char*e;long v=strtol(p,&e,10);if(e==p||errno||v<INT32_MIN||v>INT32_MAX){free(a);free(b);return -3;}if(k==cap){cap*=2;void*q=realloc(a,cap*sizeof(*a));if(!q){free(a);free(b);return -2;}a=q;}a[k++]=(int32_t)v;p=e;}free(b);*out=a;*n=k;return 0;}
static int parse_f32_text(const char*path,float**out,size_t*n){size_t nb;char*b=slurp(path,&nb);if(!b)return -1;size_t cap=64,k=0;float*a=malloc(cap*sizeof(*a));if(!a){free(b);return -2;}char*p=b;while(*p){while(*p&&(isspace((unsigned char)*p)||*p==','))p++;if(!*p)break;errno=0;char*e;float v=strtof(p,&e);if(e==p||errno||!isfinite(v)){free(a);free(b);return -3;}if(k==cap){cap*=2;void*q=realloc(a,cap*sizeof(*a));if(!q){free(a);free(b);return -2;}a=q;}a[k++]=v;p=e;}free(b);*out=a;*n=k;return 0;}
static int parse_csv(const char*s,float**out,size_t*n){char*tmp=strdup(s);if(!tmp)return -1;size_t cap=16,k=0;float*a=malloc(cap*sizeof(*a));if(!a){free(tmp);return -1;}char*p=tmp;while(*p){while(*p&&(isspace((unsigned char)*p)||*p==','))p++;if(!*p)break;char*e;float v=strtof(p,&e);if(e==p||!isfinite(v)){free(a);free(tmp);return -1;}if(k==cap){cap*=2;void*q=realloc(a,cap*sizeof(*a));if(!q){free(a);free(tmp);return -1;}a=q;}a[k++]=v;p=e;}free(tmp);*out=a;*n=k;return 0;}
static int cfg_defaults(ModelCfg*c){memset(c,0,sizeof(*c));c->lo=malloc(sizeof(float));c->hi=malloc(sizeof(float));if(!c->lo||!c->hi)return -1;c->lo[0]=-12.f;c->hi[0]=0.f;c->lo_dims=c->hi_dims=c->range_dims=1;c->t_start=.4f;c->scale=1000.f;c->steps=20;c->sample_rate=44100;c->hop_size=512;return 0;}
static void cfg_free(ModelCfg*c){free(c->lo);free(c->hi);memset(c,0,sizeof(*c));}
static int cfg_set_vec(float**dst,size_t*dn,const char*s){float*x=NULL;size_t n=0;if(parse_csv(s,&x,&n)||!n)return -1;free(*dst);*dst=x;*dn=n;return 0;}
static int load_conf(const char*dir,ModelCfg*c){if(cfg_defaults(c))return -1;char path[PATH_MAX];snprintf(path,sizeof(path),"%s/model.conf",dir);FILE*f=fopen(path,"r");if(!f)return 0;char line[32768];size_t nlo=1,nhi=1;while(fgets(line,sizeof(line),f)){char*nl=strchr(line,'\n');if(nl)*nl=0;char*eq=strchr(line,'=');if(!eq||line[0]=='#')continue;*eq++=0;if(!strcmp(line,"spec_min")){if(cfg_set_vec(&c->lo,&nlo,eq)){fclose(f);return -2;}}else if(!strcmp(line,"spec_max")){if(cfg_set_vec(&c->hi,&nhi,eq)){fclose(f);return -2;}}else if(!strcmp(line,"t_start"))c->t_start=strtof(eq,NULL);else if(!strcmp(line,"time_scale_factor"))c->scale=strtof(eq,NULL);else if(!strcmp(line,"steps"))c->steps=(size_t)strtoull(eq,NULL,10);else if(!strcmp(line,"sample_rate"))c->sample_rate=(size_t)strtoull(eq,NULL,10);else if(!strcmp(line,"hop_size"))c->hop_size=(size_t)strtoull(eq,NULL,10);}fclose(f);if(nlo!=nhi)return -3;c->lo_dims=nlo;c->hi_dims=nhi;c->range_dims=nlo;return 0;}
static uint64_t xs64(uint64_t*s){uint64_t x=*s;x^=x>>12;x^=x<<25;x^=x>>27;*s=x;return x*2685821657736338717ULL;}
static float uni01(uint64_t*s){uint64_t x=xs64(s);return ((x>>40)+1.0f)*(1.0f/16777217.0f);}
static void gaussian(float*x,size_t n,uint64_t seed){uint64_t s=seed?seed:0x9e3779b97f4a7c15ULL;size_t i=0;while(i<n){float u1=uni01(&s),u2=uni01(&s);float r=sqrtf(-2.f*logf(u1)),a=6.2831853071795864769f*u2;x[i++]=r*cosf(a);if(i<n)x[i++]=r*sinf(a);}}
static int read_noise(const char*path,float*x,size_t n){FILE*f=fopen(path,"rb");if(!f)return -1;size_t k=fread(x,sizeof(float),n,f);int extra=fgetc(f);fclose(f);return (k==n&&extra==EOF)?0:-2;}
static int write_f32(const char*path,const float*x,size_t n){FILE*f=fopen(path,"wb");if(!f)return -1;int ok=fwrite(x,sizeof(float),n,f)==n;fclose(f);return ok?0:-2;}
static float* fvec_default(size_t n,float v){float*x=malloc(n*sizeof(float));if(!x)return NULL;for(size_t i=0;i<n;i++)x[i]=v;return x;}
static int load_curve(const char*path,size_t n,float def,float**out){if(!path){*out=fvec_default(n,def);return *out?0:-1;}size_t got=0;float*x=NULL;if(parse_f32_text(path,&x,&got)||got!=n){free(x);return -2;}*out=x;return 0;}
static int load_speaker(const char*path,size_t T,size_t C,float**out){
    FILE*f=fopen(path,"rb");if(!f)return -1;if(fseek(f,0,SEEK_END)){fclose(f);return -1;}long z=ftell(f);rewind(f);
    size_t one=C*sizeof(float),all=T*C*sizeof(float);float*x=NULL;
    if(z==(long)one||z==(long)all){size_t n=(z==(long)one)?C:T*C;float*raw=malloc(n*sizeof(float));if(!raw){fclose(f);return -2;}if(fread(raw,sizeof(float),n,f)!=n){free(raw);fclose(f);return -3;}fclose(f);if(n==T*C){*out=raw;return 0;}x=malloc(all);if(!x){free(raw);return -2;}for(size_t t=0;t<T;t++)memcpy(x+t*C,raw,one);free(raw);*out=x;return 0;}
    fclose(f);size_t n=0;float*txt=NULL;if(parse_f32_text(path,&txt,&n))return -4;if(n!=C&&n!=T*C){free(txt);return -5;}if(n==T*C){*out=txt;return 0;}x=malloc(all);if(!x){free(txt);return -2;}for(size_t t=0;t<T;t++)memcpy(x+t*C,txt,one);free(txt);*out=x;return 0;
}
static int model_paths(const char*dir,char*fs,char*aux,char*rf){snprintf(fs,PATH_MAX,"%s/fs2_acoustic.dsfs",dir);snprintf(aux,PATH_MAX,"%s/aux_convnext.dsa",dir);snprintf(rf,PATH_MAX,"%s/lynxnet2.dsn",dir);return 0;}
static void print_features(uint32_t f){
    struct P{uint32_t b;const char*n;}p[]={{DSASM_FS2_FEAT_LANGUAGE,"language"},{DSASM_FS2_FEAT_BREATH,"breathiness"},{DSASM_FS2_FEAT_VOICING,"voicing"},{DSASM_FS2_FEAT_TENSION,"tension"},{DSASM_FS2_FEAT_KEY_SHIFT,"gender/key-shift"},{DSASM_FS2_FEAT_SPEED,"velocity/speed"},{DSASM_FS2_FEAT_SPEAKER,"speaker"},{DSASM_FS2_FEAT_STRETCH_TABLE,"stretch-table"},{DSASM_FS2_FEAT_LANGUAGE_MASK,"language-token-mask"}};
    printf("features: flags=0x%x [",f);int first=1;for(size_t i=0;i<sizeof(p)/sizeof(p[0]);i++)if(f&p[i].b){printf("%s%s",first?"":",",p[i].n);first=0;}printf("]\n");
}
static int inspect(const char*dir){char fs[PATH_MAX],au[PATH_MAX],rf[PATH_MAX];model_paths(dir,fs,au,rf);DSAsmAcousticModel m;int rc=ds_acoustic_model_load(&m,fs,au,rf);if(rc){fprintf(stderr,"load failed rc=%d\n",rc);return 2;}ModelCfg c;if(load_conf(dir,&c)){fprintf(stderr,"bad model.conf\n");ds_acoustic_model_unload(&m);return 2;}printf("M25 model: vocab=%u hidden=%u fs2L=%u H=%u | aux=%u->%u C=%u L=%u | rf I=%u Q=%u C=%u H=%u L=%u glu=%u\n",m.fs2.encoder.vocab_size,m.fs2.encoder.hidden_size,m.fs2.encoder.num_layers,m.fs2.encoder.num_heads,m.aux.input_dim,m.aux.output_dim,m.aux.channels,m.aux.num_layers,m.rf.input_dim,m.rf.condition_dim,m.rf.channels,m.rf.hidden_dim,m.rf.num_layers,m.rf.glu_type);print_features(m.fs2_extras.flags);if(m.fs2_extras.flags&DSASM_FS2_FEAT_LANGUAGE)printf("languages: rows=%u (0 is padding)\n",m.fs2_extras.num_languages);if((m.fs2_extras.flags&DSASM_FS2_FEAT_LANGUAGE_MASK)&&m.fs2_extras.language_token_mask){size_t nx=0;for(uint32_t i=0;i<m.fs2.encoder.vocab_size;i++)if(m.fs2_extras.language_token_mask[i]>0.5f)nx++;printf("cross-lingual tokens: %zu/%u\n",nx,m.fs2.encoder.vocab_size);}printf("sampling: t_start=%g scale=%g steps=%zu spec_range_dims=%zu sr=%zu hop=%zu\n",c.t_start,c.scale,c.steps,c.range_dims,c.sample_rate,c.hop_size);cfg_free(&c);ds_acoustic_model_unload(&m);return 0;}

int main(int argc,char**argv){
    if(argc<3){usage(argv[0]);return 2;}const char*cmd=argv[1],*dir=argv[2];if(!strcmp(cmd,"inspect"))return inspect(dir);if(strcmp(cmd,"infer")){usage(argv[0]);return 2;}
    const char *tokens_p=NULL,*dur_p=NULL,*f0_p=NULL,*out_p=NULL,*noise_p=NULL,*langs_p=NULL,*spk_p=NULL;
    const char *breath_p=NULL,*voice_p=NULL,*tension_p=NULL,*gender_p=NULL,*velocity_p=NULL,*dump_cond_p=NULL,*dump_aux_p=NULL,*dump_stages_dir=NULL;
    int have_lang_id=0;int32_t lang_id=0;int have_depth=0;float depth=0.0f;uint64_t seed=123456789ULL;size_t threads=0;int profile_stages=0;
    ModelCfg cfg;if(load_conf(dir,&cfg)){fprintf(stderr,"bad model.conf\n");return 2;}
#define NEED() do{if(i+1>=argc){usage(argv[0]);cfg_free(&cfg);return 2;}}while(0)
    for(int i=3;i<argc;i++){
        if(!strcmp(argv[i],"--tokens")){NEED();tokens_p=argv[++i];}
        else if(!strcmp(argv[i],"--durations")){NEED();dur_p=argv[++i];}
        else if(!strcmp(argv[i],"--f0")){NEED();f0_p=argv[++i];}
        else if(!strcmp(argv[i],"--out")){NEED();out_p=argv[++i];}
        else if(!strcmp(argv[i],"--noise")){NEED();noise_p=argv[++i];}
        else if(!strcmp(argv[i],"--seed")){NEED();seed=strtoull(argv[++i],NULL,0);}
        else if(!strcmp(argv[i],"--threads")){NEED();threads=(size_t)strtoull(argv[++i],NULL,10);}
        else if(!strcmp(argv[i],"--profile-stages")){profile_stages=1;}
        else if(!strcmp(argv[i],"--steps")){NEED();cfg.steps=(size_t)strtoull(argv[++i],NULL,10);}
        else if(!strcmp(argv[i],"--t-start")){NEED();cfg.t_start=strtof(argv[++i],NULL);have_depth=0;}
        else if(!strcmp(argv[i],"--depth")){NEED();depth=strtof(argv[++i],NULL);have_depth=1;}
        else if(!strcmp(argv[i],"--time-scale")){NEED();cfg.scale=strtof(argv[++i],NULL);}
        else if(!strcmp(argv[i],"--spec-min")){NEED();size_t n=0;if(cfg_set_vec(&cfg.lo,&n,argv[++i])){fprintf(stderr,"bad --spec-min\n");cfg_free(&cfg);return 2;}cfg.lo_dims=n;cfg.range_dims=n;}
        else if(!strcmp(argv[i],"--spec-max")){NEED();size_t n=0;if(cfg_set_vec(&cfg.hi,&n,argv[++i])){fprintf(stderr,"bad --spec-max\n");cfg_free(&cfg);return 2;}cfg.hi_dims=n;}
        else if(!strcmp(argv[i],"--languages")){NEED();langs_p=argv[++i];}
        else if(!strcmp(argv[i],"--language-id")){NEED();long v=strtol(argv[++i],NULL,10);if(v<0||v>INT32_MAX){fprintf(stderr,"bad language id\n");cfg_free(&cfg);return 2;}lang_id=(int32_t)v;have_lang_id=1;}
        else if(!strcmp(argv[i],"--speaker-emb")){NEED();spk_p=argv[++i];}
        else if(!strcmp(argv[i],"--breathiness")){NEED();breath_p=argv[++i];}
        else if(!strcmp(argv[i],"--voicing")){NEED();voice_p=argv[++i];}
        else if(!strcmp(argv[i],"--tension")){NEED();tension_p=argv[++i];}
        else if(!strcmp(argv[i],"--gender")){NEED();gender_p=argv[++i];}
        else if(!strcmp(argv[i],"--velocity")){NEED();velocity_p=argv[++i];}
        else if(!strcmp(argv[i],"--dump-condition")){NEED();dump_cond_p=argv[++i];}
        else if(!strcmp(argv[i],"--dump-aux-mel")){NEED();dump_aux_p=argv[++i];}
        else if(!strcmp(argv[i],"--dump-fs2-stages")){NEED();dump_stages_dir=argv[++i];}
        else {fprintf(stderr,"unknown option: %s\n",argv[i]);usage(argv[0]);cfg_free(&cfg);return 2;}
    }
    if(!tokens_p||!dur_p||!f0_p||!out_p|| (langs_p&&have_lang_id)){usage(argv[0]);cfg_free(&cfg);return 2;}
    if(have_depth){if(!isfinite(depth)||depth<0.f||depth>1.f){fprintf(stderr,"--depth must be normalized [0,1]\n");cfg_free(&cfg);return 2;}cfg.t_start=fmaxf(1.0f-depth,0.0f);}
    if(cfg.steps<1||cfg.scale<=0.f||cfg.t_start<0.f||cfg.t_start>1.f||!cfg.sample_rate||!cfg.hop_size){fprintf(stderr,"invalid sampling/audio config\n");cfg_free(&cfg);return 2;}
    if(cfg.lo_dims!=cfg.hi_dims){fprintf(stderr,"spec-min/spec-max dimension mismatch: %zu vs %zu\n",cfg.lo_dims,cfg.hi_dims);cfg_free(&cfg);return 2;}cfg.range_dims=cfg.lo_dims;

    char fs[PATH_MAX],au[PATH_MAX],rf[PATH_MAX];model_paths(dir,fs,au,rf);DSAsmAcousticModel m;int rc=ds_acoustic_model_load(&m,fs,au,rf);if(rc){fprintf(stderr,"model load failed rc=%d\n",rc);cfg_free(&cfg);return 3;}
    int32_t *tok=NULL,*dur=NULL,*langs=NULL;size_t P=0,Pd=0,Pl=0;float*f0=NULL;size_t Tf=0;
    float *breath=NULL,*voice=NULL,*tension=NULL,*gender=NULL,*velocity=NULL,*speaker=NULL;
    if(parse_i32_text(tokens_p,&tok,&P)||parse_i32_text(dur_p,&dur,&Pd)||parse_f32_text(f0_p,&f0,&Tf)||!P||Pd!=P){fprintf(stderr,"bad tokens/durations/f0 input\n");rc=4;goto done;}
    size_t T=0;for(size_t i=0;i<P;i++){if(dur[i]<0||tok[i]<0||(size_t)tok[i]>=m.fs2.encoder.vocab_size){fprintf(stderr,"invalid token/duration at %zu\n",i);rc=4;goto done;}if((size_t)dur[i]>SIZE_MAX-T){rc=4;goto done;}T+=(size_t)dur[i];}
    if(T!=Tf||!T){fprintf(stderr,"sum(durations)=%zu but f0 count=%zu\n",T,Tf);rc=4;goto done;}if(cfg.range_dims!=1&&cfg.range_dims!=m.rf.input_dim){fprintf(stderr,"spec range dims=%zu, expected 1 or %u\n",cfg.range_dims,m.rf.input_dim);rc=4;goto done;}

    uint32_t flags=m.fs2_extras.flags;const size_t C=m.fs2.encoder.hidden_size;
    if(flags&DSASM_FS2_FEAT_LANGUAGE){
        if(langs_p){if(parse_i32_text(langs_p,&langs,&Pl)||Pl!=P){fprintf(stderr,"bad languages input: expected %zu ids\n",P);rc=4;goto done;}}
        else if(have_lang_id){langs=malloc(P*sizeof(*langs));if(!langs){rc=5;goto done;}Pl=P;for(size_t i=0;i<P;i++)langs[i]=(tok[i]==0)?0:lang_id;}
        else {fprintf(stderr,"model requires language ids: use --language-id N or --languages FILE\n");rc=4;goto done;}
        for(size_t i=0;i<P;i++)if(langs[i]<0||(uint32_t)langs[i]>=m.fs2_extras.num_languages){fprintf(stderr,"language id out of range at %zu: %d (rows=%u)\n",i,langs[i],m.fs2_extras.num_languages);rc=4;goto done;}
    }
    if((flags&DSASM_FS2_FEAT_SPEAKER)&&!spk_p){fprintf(stderr,"model requires --speaker-emb FILE\n");rc=4;goto done;}
    if((flags&DSASM_FS2_FEAT_SPEAKER)&&load_speaker(spk_p,T,C,&speaker)){fprintf(stderr,"bad speaker embedding: expected %zu or %zu float32 values\n",C,T*C);rc=4;goto done;}
    if((flags&DSASM_FS2_FEAT_BREATH)&&load_curve(breath_p,T,0.f,&breath)){fprintf(stderr,"bad breathiness curve: expected %zu floats\n",T);rc=4;goto done;}
    if((flags&DSASM_FS2_FEAT_VOICING)&&load_curve(voice_p,T,0.f,&voice)){fprintf(stderr,"bad voicing curve: expected %zu floats\n",T);rc=4;goto done;}
    if((flags&DSASM_FS2_FEAT_TENSION)&&load_curve(tension_p,T,0.f,&tension)){fprintf(stderr,"bad tension curve: expected %zu floats\n",T);rc=4;goto done;}
    if((flags&DSASM_FS2_FEAT_KEY_SHIFT)&&load_curve(gender_p,T,0.f,&gender)){fprintf(stderr,"bad gender curve: expected %zu floats\n",T);rc=4;goto done;}
    if((flags&DSASM_FS2_FEAT_SPEED)&&load_curve(velocity_p,T,1.f,&velocity)){fprintf(stderr,"bad velocity curve: expected %zu floats\n",T);rc=4;goto done;}

    int32_t*mel2ph=malloc(T*sizeof(*mel2ph));size_t M=m.rf.input_dim,N=T*M;float*noise=aligned_alloc(64,((N*sizeof(float)+63)/64)*64);float*out=aligned_alloc(64,((N*sizeof(float)+63)/64)*64);size_t wsn=ds_acoustic_model_workspace_floats(&m,P,T);float*ws=aligned_alloc(64,((wsn*sizeof(float)+63)/64)*64);
    if(!mel2ph||!noise||!out||!ws){fprintf(stderr,"allocation failed\n");free(mel2ph);free(noise);free(out);free(ws);rc=5;goto done;}
    size_t z=0;for(size_t i=0;i<P;i++)for(int32_t j=0;j<dur[i];j++)mel2ph[z++]=(int32_t)i+1;if(noise_p){if(read_noise(noise_p,noise,N)){fprintf(stderr,"bad noise file, expected %zu float32 values\n",N);rc=4;goto memdone;}}else gaussian(noise,N,seed);
    DSAsmThreadPool*pool=ds_threadpool_create(threads);if(!pool){fprintf(stderr,"threadpool create failed\n");rc=5;goto memdone;}
    if(dump_cond_p||dump_aux_p||dump_stages_dir){
        size_t fw=ds_fs2_acoustic_condition_workspace_floats(&m.fs2,P,T);
        size_t aw=ds_aux_convnext_workspace_floats(&m.aux,T);
        float*dcond=aligned_alloc(64,((T*C*sizeof(float)+63)/64)*64);
        float*daux=aligned_alloc(64,((T*M*sizeof(float)+63)/64)*64);
        float*dws=aligned_alloc(64,(((fw>aw?fw:aw)*sizeof(float)+63)/64)*64);
        float *st_enc=NULL,*st_gather=NULL,*st_stretch=NULL,*st_gru=NULL,*st_pitch=NULL,*st_var=NULL,*st_key=NULL,*st_speed=NULL,*st_spk=NULL;
        if(dump_stages_dir){
            st_enc=malloc(P*C*sizeof(float));
            st_gather=malloc(T*C*sizeof(float));st_stretch=malloc(T*C*sizeof(float));st_gru=malloc(T*C*sizeof(float));
            st_pitch=malloc(T*C*sizeof(float));st_var=malloc(T*C*sizeof(float));st_key=malloc(T*C*sizeof(float));
            st_speed=malloc(T*C*sizeof(float));st_spk=malloc(T*C*sizeof(float));
        }
        if(!dcond||!daux||!dws||(dump_stages_dir&&(!st_enc||!st_gather||!st_stretch||!st_gru||!st_pitch||!st_var||!st_key||!st_speed||!st_spk))){
            free(dcond);free(daux);free(dws);free(st_enc);free(st_gather);free(st_stretch);free(st_gru);free(st_pitch);free(st_var);free(st_key);free(st_speed);free(st_spk);
            fprintf(stderr,"debug dump allocation failed\n");ds_threadpool_destroy(pool);rc=5;goto memdone;
        }
        if(flags){
            DSAsmFS2DeploymentInputs di={langs,breath,voice,tension,gender,velocity,speaker};
            if(dump_stages_dir){
                DSAsmFS2DeploymentDebug dbg={st_enc,st_gather,st_stretch,st_gru,st_pitch,st_var,st_key,st_speed,st_spk};
                rc=ds_fs2_acoustic_condition_deploy_debug_f32_avx2(&m.fs2,&m.fs2_extras,&di,tok,P,mel2ph,f0,T,dcond,dws,pool,&dbg);
            }else rc=ds_fs2_acoustic_condition_deploy_f32_avx2(&m.fs2,&m.fs2_extras,&di,tok,P,mel2ph,f0,T,dcond,dws,pool);
        } else rc=ds_fs2_acoustic_condition_f32_avx2(&m.fs2,tok,P,mel2ph,f0,T,dcond,dws,pool);
        if(!rc&&dump_cond_p)rc=write_f32(dump_cond_p,dcond,T*C);
        if(!rc&&dump_aux_p){rc=ds_aux_convnext_infer_f32_avx2(&m.aux,dcond,cfg.lo,cfg.hi,cfg.range_dims,daux,dws,T,pool);if(!rc)rc=write_f32(dump_aux_p,daux,T*M);}
        if(!rc&&dump_stages_dir){
            char spath[PATH_MAX];
#define WSTAGE(name,ptr,count) do{snprintf(spath,sizeof(spath),"%s/%s.f32",dump_stages_dir,(name));rc=write_f32(spath,(ptr),(count));}while(0)
            WSTAGE("encoder_txt",st_enc,P*C);
            if(!rc)WSTAGE("gathered",st_gather,T*C);
            if(!rc)WSTAGE("stretch",st_stretch,T*C);
            if(!rc)WSTAGE("gru",st_gru,T*C);
            if(!rc)WSTAGE("pitch",st_pitch,T*C);
            if(!rc)WSTAGE("variance",st_var,T*C);
            if(!rc)WSTAGE("key_shift",st_key,T*C);
            if(!rc)WSTAGE("speed",st_speed,T*C);
            if(!rc)WSTAGE("speaker",st_spk,T*C);
#undef WSTAGE
        }
        free(dcond);free(daux);free(dws);free(st_enc);free(st_gather);free(st_stretch);free(st_gru);free(st_pitch);free(st_var);free(st_key);free(st_speed);free(st_spk);
        if(rc){fprintf(stderr,"debug dump failed rc=%d\n",rc);ds_threadpool_destroy(pool);goto memdone;}
    }
    double fs2_ms=0.0,aux_ms=0.0,rf_ms=0.0,ms=0.0;
    if(profile_stages){
        const size_t fw=ds_fs2_acoustic_condition_workspace_floats(&m.fs2,P,T);
        const size_t aw=ds_aux_convnext_workspace_floats(&m.aux,T);
        const size_t rw=ds_acoustic_reflow_normsrc_workspace_floats(&m.rf,T);
        float*cond=aligned_alloc(64,((T*C*sizeof(float)+63)/64)*64);
        float*aux_norm=aligned_alloc(64,((N*sizeof(float)+63)/64)*64);
        float*mask=aligned_alloc(64,((T*sizeof(float)+63)/64)*64);
        float*fsws=aligned_alloc(64,((fw*sizeof(float)+63)/64)*64);
        float*auws=aligned_alloc(64,((aw*sizeof(float)+63)/64)*64);
        float*rfws=aligned_alloc(64,((rw*sizeof(float)+63)/64)*64);
        if(!cond||!aux_norm||!mask||!fsws||!auws||!rfws){
            fprintf(stderr,"stage profiler allocation failed\n");
            free(cond);free(aux_norm);free(mask);free(fsws);free(auws);free(rfws);
            ds_threadpool_destroy(pool);rc=5;goto memdone;
        }
        for(size_t t=0;t<T;t++)mask[t]=mel2ph[t]>0?1.0f:0.0f;
        double q0=now_ms();
        if(flags){DSAsmFS2DeploymentInputs di={langs,breath,voice,tension,gender,velocity,speaker};rc=ds_fs2_acoustic_condition_deploy_f32_avx2(&m.fs2,&m.fs2_extras,&di,tok,P,mel2ph,f0,T,cond,fsws,pool);}
        else rc=ds_fs2_acoustic_condition_f32_avx2(&m.fs2,tok,P,mel2ph,f0,T,cond,fsws,pool);
        double q1=now_ms();
        if(!rc)rc=ds_aux_convnext_forward_norm_f32_avx2(&m.aux,cond,aux_norm,auws,T,pool);
        double q2=now_ms();
        if(!rc)rc=ds_acoustic_reflow_decode_normsrc_f32_avx2(&m.rf,cond,aux_norm,noise,mask,cfg.lo,cfg.hi,cfg.range_dims,cfg.t_start,cfg.scale,cfg.steps,out,rfws,T,pool);
        double q3=now_ms();
        fs2_ms=q1-q0;aux_ms=q2-q1;rf_ms=q3-q2;ms=q3-q0;
        free(cond);free(aux_norm);free(mask);free(fsws);free(auws);free(rfws);
    }else{
        double t0=now_ms();
        if(flags){DSAsmFS2DeploymentInputs di={langs,breath,voice,tension,gender,velocity,speaker};rc=ds_acoustic_model_infer_deploy_f32_avx2(&m,&di,tok,P,mel2ph,f0,T,noise,cfg.lo,cfg.hi,cfg.range_dims,cfg.t_start,cfg.scale,cfg.steps,out,ws,pool);}
        else rc=ds_acoustic_model_infer_f32_avx2(&m,tok,P,mel2ph,f0,T,noise,cfg.lo,cfg.hi,cfg.range_dims,cfg.t_start,cfg.scale,cfg.steps,out,ws,pool);
        ms=now_ms()-t0;
    }
    if(rc){fprintf(stderr,"inference failed rc=%d\n",rc);ds_threadpool_destroy(pool);goto memdone;}if(write_f32(out_p,out,N)){fprintf(stderr,"write failed: %s\n",out_p);rc=6;ds_threadpool_destroy(pool);goto memdone;}
    double sum=0;float mn=out[0],mx=out[0];for(size_t i=0;i<N;i++){sum+=out[i];if(out[i]<mn)mn=out[i];if(out[i]>mx)mx=out[i];}double audio_ms=T*(double)cfg.hop_size/(double)cfg.sample_rate*1000.0;printf("M25 infer: P=%zu T=%zu mel=%zu workers=%zu steps=%zu t_start=%g flags=0x%x\n",P,T,M,ds_threadpool_threads(pool),cfg.steps,cfg.t_start,flags);printf("  time=%.3f ms audio@%zu/hop%zu=%.3f ms RTF=%.3f speed=%.3fx\n",ms,cfg.sample_rate,cfg.hop_size,audio_ms,ms/audio_ms,audio_ms/ms);if(profile_stages)printf("  stages: fs2=%.3f ms aux=%.3f ms rf=%.3f ms rf_per_step=%.3f ms sum=%.3f ms\n",fs2_ms,aux_ms,rf_ms,cfg.steps?rf_ms/(double)cfg.steps:0.0,fs2_ms+aux_ms+rf_ms);printf("  output=%s floats=%zu checksum=%.9g min=%g max=%g\n",out_p,N,sum,mn,mx);ds_threadpool_destroy(pool);
memdone: free(mel2ph);free(noise);free(out);free(ws);
done: free(tok);free(dur);free(f0);free(langs);free(breath);free(voice);free(tension);free(gender);free(velocity);free(speaker);ds_acoustic_model_unload(&m);cfg_free(&cfg);return rc?1:0;
}
