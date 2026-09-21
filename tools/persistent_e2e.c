#define _POSIX_C_SOURCE 200809L
#include "dsasm_model.h"
#include "dsasm_vocoder_graph.h"
#include "dsasm_threadpool.h"
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
    float *lo,*hi; size_t lo_n,hi_n;
    float t_start, scale; size_t steps;
    size_t sample_rate, hop_size;
} Cfg;

static double now_ms(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1000.0+t.tv_nsec/1e6; }
static size_t align64(size_t z){ return (z+63u)&~63u; }
static void *a64(size_t z){ return aligned_alloc(64,align64(z)); }
static char *slurp(const char *p,size_t*n){ FILE*f=fopen(p,"rb"); if(!f)return NULL; if(fseek(f,0,SEEK_END)){fclose(f);return NULL;} long z=ftell(f); if(z<0){fclose(f);return NULL;} rewind(f); char*b=malloc((size_t)z+1); if(!b){fclose(f);return NULL;} size_t g=fread(b,1,(size_t)z,f); fclose(f); b[g]=0; if(n)*n=g; return b; }
static int parse_i32(const char*p,int32_t**out,size_t*n){ size_t nb=0; char*b=slurp(p,&nb); if(!b)return -1; size_t cap=64,k=0; int32_t*a=malloc(cap*sizeof(*a)); if(!a){free(b);return -2;} char*s=b; while(*s){ while(*s&&((*s>0&&*s<=' ')||*s==','))s++; if(!*s)break; errno=0; char*e; long v=strtol(s,&e,10); if(e==s||errno||v<INT32_MIN||v>INT32_MAX){free(a);free(b);return -3;} if(k==cap){cap*=2;void*q=realloc(a,cap*sizeof(*a));if(!q){free(a);free(b);return -2;}a=q;} a[k++]=(int32_t)v;s=e;} free(b);*out=a;*n=k;return 0; }
static int parse_f32(const char*p,float**out,size_t*n){ size_t nb=0; char*b=slurp(p,&nb); if(!b)return -1; size_t cap=128,k=0; float*a=malloc(cap*sizeof(*a)); if(!a){free(b);return -2;} char*s=b; while(*s){ while(*s&&((*s>0&&*s<=' ')||*s==','))s++; if(!*s)break; errno=0; char*e; float v=strtof(s,&e); if(e==s||errno||!isfinite(v)){free(a);free(b);return -3;} if(k==cap){cap*=2;void*q=realloc(a,cap*sizeof(*a));if(!q){free(a);free(b);return -2;}a=q;} a[k++]=v;s=e;} free(b);*out=a;*n=k;return 0; }
static int parse_csv(const char*s,float**out,size_t*n){ char*t=strdup(s); if(!t)return -1; size_t cap=16,k=0;float*a=malloc(cap*sizeof(*a));if(!a){free(t);return -1;}char*p=t;while(*p){while(*p&&((*p>0&&*p<=' ')||*p==','))p++;if(!*p)break;char*e;float v=strtof(p,&e);if(e==p||!isfinite(v)){free(a);free(t);return -1;}if(k==cap){cap*=2;void*q=realloc(a,cap*sizeof(*a));if(!q){free(a);free(t);return -1;}a=q;}a[k++]=v;p=e;}free(t);*out=a;*n=k;return 0; }
static int cfg_vec(float **dst,size_t*n,const char*s){float*x=NULL;size_t k=0;if(parse_csv(s,&x,&k)||!k)return -1;free(*dst);*dst=x;*n=k;return 0;}
static int cfg_load(const char*dir,Cfg*c){ memset(c,0,sizeof(*c));c->lo=malloc(sizeof(float));c->hi=malloc(sizeof(float));if(!c->lo||!c->hi)return -1;c->lo[0]=-12.f;c->hi[0]=0.f;c->lo_n=c->hi_n=1;c->t_start=.4f;c->scale=1000.f;c->steps=4;c->sample_rate=44100;c->hop_size=512;char p[PATH_MAX];snprintf(p,sizeof(p),"%s/model.conf",dir);FILE*f=fopen(p,"r");if(!f)return 0;char line[32768];while(fgets(line,sizeof(line),f)){char*nl=strchr(line,'\n');if(nl)*nl=0;char*eq=strchr(line,'=');if(!eq||line[0]=='#')continue;*eq++=0;if(!strcmp(line,"spec_min")){if(cfg_vec(&c->lo,&c->lo_n,eq)){fclose(f);return -2;}}else if(!strcmp(line,"spec_max")){if(cfg_vec(&c->hi,&c->hi_n,eq)){fclose(f);return -2;}}else if(!strcmp(line,"t_start"))c->t_start=strtof(eq,NULL);else if(!strcmp(line,"time_scale_factor"))c->scale=strtof(eq,NULL);else if(!strcmp(line,"steps"))c->steps=(size_t)strtoull(eq,NULL,10);else if(!strcmp(line,"sample_rate"))c->sample_rate=(size_t)strtoull(eq,NULL,10);else if(!strcmp(line,"hop_size"))c->hop_size=(size_t)strtoull(eq,NULL,10);}fclose(f);return c->lo_n==c->hi_n?0:-3; }
static void cfg_free(Cfg*c){free(c->lo);free(c->hi);memset(c,0,sizeof(*c));}
static float *read_bin(const char*p,size_t n){FILE*f=fopen(p,"rb");if(!f)return NULL;float*x=a64(n*sizeof(float));if(!x){fclose(f);return NULL;}size_t g=fread(x,sizeof(float),n,f);int e=fgetc(f);fclose(f);if(g!=n||e!=EOF){free(x);return NULL;}return x;}
static int write_bin(const char*p,const float*x,size_t n){FILE*f=fopen(p,"wb");if(!f)return -1;int ok=fwrite(x,sizeof(float),n,f)==n;fclose(f);return ok?0:-1;}
static int load_curve(const char*fixture,const char*name,size_t T,float def,float**out){char p[PATH_MAX];snprintf(p,sizeof(p),"%s/%s.txt",fixture,name);float*x=NULL;size_t n=0;if(parse_f32(p,&x,&n)){x=malloc(T*sizeof(float));if(!x)return -1;for(size_t i=0;i<T;i++)x[i]=def;*out=x;return 0;}if(n!=T){free(x);return -2;}*out=x;return 0;}
static int load_speaker(const char*path,size_t T,size_t C,float**out){FILE*f=fopen(path,"rb");if(!f)return -1;if(fseek(f,0,SEEK_END)){fclose(f);return -1;}long z=ftell(f);rewind(f);size_t one=C*sizeof(float),all=T*C*sizeof(float);float*x=NULL;if(z==(long)one||z==(long)all){size_t n=z==(long)one?C:T*C;float*r=malloc(n*sizeof(float));if(!r){fclose(f);return -2;}if(fread(r,sizeof(float),n,f)!=n){free(r);fclose(f);return -3;}fclose(f);if(n==T*C){*out=r;return 0;}x=malloc(all);if(!x){free(r);return -2;}for(size_t t=0;t<T;t++)memcpy(x+t*C,r,one);free(r);*out=x;return 0;}fclose(f);float*r=NULL;size_t n=0;if(parse_f32(path,&r,&n))return -4;if(n!=C&&n!=T*C){free(r);return -5;}if(n==T*C){*out=r;return 0;}x=malloc(all);if(!x){free(r);return -2;}for(size_t t=0;t<T;t++)memcpy(x+t*C,r,one);free(r);*out=x;return 0;}
static int cmp_mel(const float*a,const float*b,size_t n){size_t diff=0;float ma=0;for(size_t i=0;i<n;i++){uint32_t x,y;memcpy(&x,a+i,4);memcpy(&y,b+i,4);if(x!=y)diff++;float d=fabsf(a[i]-b[i]);if(d>ma)ma=d;}printf("E2E acoustic parity: max_abs=%.9g bitdiff=%zu\n",ma,diff);return diff?1:0;}
static int cmp_wave(const float*x,const float*g,size_t n){double se=0,sg=0,sx=0,dot=0;float ma=0;for(size_t i=0;i<n;i++){double d=(double)x[i]-g[i];se+=d*d;sg+=(double)g[i]*g[i];sx+=(double)x[i]*x[i];dot+=(double)x[i]*g[i];float ad=fabsf(x[i]-g[i]);if(ad>ma)ma=ad;}double rmse=sqrt(se/(double)n),cos=dot/(sqrt(sg*sx)+1e-300),snr=10.0*log10((sg+1e-300)/(se+1e-300));printf("E2E waveform parity: max_abs=%.9g rmse=%.9g cosine=%.9f SNR=%.2f dB\n",ma,rmse,cos,snr);return !(cos>=0.999&&snr>=25.0);}
static int dblcmp(const void*a,const void*b){double x=*(const double*)a,y=*(const double*)b;return (x>y)-(x<y);}
static const char *envs(const char *k){const char *v=getenv(k);return v?v:"(default)";}
static void usage(const char*p){fprintf(stderr,"usage: %s --model DIR --vocoder BUNDLE --fixture DIR --speaker-emb FILE [--workers 8] [--warmup 1] [--requests 5] [--steps 4] [--depth .6] [--out-wave FILE]\n",p);}

int main(int argc,char**argv){
    const char *model=NULL,*voc=NULL,*fixture=NULL,*spk=NULL,*outwave=NULL;size_t workers=8,warmup=1,requests=5,steps=4;float depth=.6f;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--model")){if(i+1>=argc){usage(argv[0]);return 2;}model=argv[++i];}
        else if(!strcmp(argv[i],"--vocoder")){if(i+1>=argc){usage(argv[0]);return 2;}voc=argv[++i];}
        else if(!strcmp(argv[i],"--fixture")){if(i+1>=argc){usage(argv[0]);return 2;}fixture=argv[++i];}
        else if(!strcmp(argv[i],"--speaker-emb")){if(i+1>=argc){usage(argv[0]);return 2;}spk=argv[++i];}
        else if(!strcmp(argv[i],"--workers")){if(i+1>=argc){usage(argv[0]);return 2;}workers=(size_t)strtoull(argv[++i],NULL,10);}
        else if(!strcmp(argv[i],"--warmup")){if(i+1>=argc){usage(argv[0]);return 2;}warmup=(size_t)strtoull(argv[++i],NULL,10);}
        else if(!strcmp(argv[i],"--requests")){if(i+1>=argc){usage(argv[0]);return 2;}requests=(size_t)strtoull(argv[++i],NULL,10);}
        else if(!strcmp(argv[i],"--steps")){if(i+1>=argc){usage(argv[0]);return 2;}steps=(size_t)strtoull(argv[++i],NULL,10);}
        else if(!strcmp(argv[i],"--depth")){if(i+1>=argc){usage(argv[0]);return 2;}depth=strtof(argv[++i],NULL);}
        else if(!strcmp(argv[i],"--out-wave")){if(i+1>=argc){usage(argv[0]);return 2;}outwave=argv[++i];}
        else {usage(argv[0]);return 2;}
    }
    if(!model||!voc||!fixture||!spk||!workers||!requests||steps<1||depth<0||depth>1){usage(argv[0]);return 2;}
    Cfg cfg;if(cfg_load(model,&cfg)){fprintf(stderr,"bad model.conf\n");return 3;}cfg.steps=steps;cfg.t_start=fmaxf(1.f-depth,0.f);
    char p[PATH_MAX],fs[PATH_MAX],au[PATH_MAX],rf[PATH_MAX];snprintf(fs,sizeof(fs),"%s/fs2_acoustic.dsfs",model);snprintf(au,sizeof(au),"%s/aux_convnext.dsa",model);snprintf(rf,sizeof(rf),"%s/lynxnet2.dsn",model);
    double load0=now_ms();DSAsmAcousticModel am;int rc=ds_acoustic_model_load(&am,fs,au,rf);if(rc){fprintf(stderr,"acoustic load rc=%d\n",rc);return 4;}DSAsmVocoderGraph*vg=ds_vocoder_graph_load(voc,workers);if(!vg){fprintf(stderr,"vocoder load failed\n");ds_acoustic_model_unload(&am);return 4;}DSAsmThreadPool*ap=ds_threadpool_create(workers);if(!ap){fprintf(stderr,"acoustic pool failed\n");ds_vocoder_graph_free(vg);ds_acoustic_model_unload(&am);return 4;}double loadms=now_ms()-load0;
    int32_t *tok=NULL,*dur=NULL,*langs=NULL;size_t P=0,Pd=0,Pl=0;snprintf(p,sizeof(p),"%s/tokens.txt",fixture);if(parse_i32(p,&tok,&P)){fprintf(stderr,"tokens read failed\n");return 5;}snprintf(p,sizeof(p),"%s/durations.txt",fixture);if(parse_i32(p,&dur,&Pd)||Pd!=P){fprintf(stderr,"durations read failed\n");return 5;}snprintf(p,sizeof(p),"%s/languages.txt",fixture);if(parse_i32(p,&langs,&Pl)||Pl!=P){fprintf(stderr,"languages read failed\n");return 5;}size_t T=0;for(size_t i=0;i<P;i++){if(dur[i]<0){fprintf(stderr,"negative duration\n");return 5;}T+=(size_t)dur[i];}
    size_t M=am.rf.input_dim;if(T!=ds_vocoder_graph_frames(vg)||M!=ds_vocoder_graph_mel_bins(vg)){fprintf(stderr,"shape mismatch acoustic T=%zu M=%zu vocoder T=%zu M=%zu\n",T,M,ds_vocoder_graph_frames(vg),ds_vocoder_graph_mel_bins(vg));return 5;}
    snprintf(p,sizeof(p),"%s/f0.f32",fixture);float*f0=read_bin(p,T);snprintf(p,sizeof(p),"%s/noise.f32",fixture);float*noise=read_bin(p,T*M);snprintf(p,sizeof(p),"%s/reference_mel.f32",fixture);float*refmel=read_bin(p,T*M);size_t S=ds_vocoder_graph_samples(vg);snprintf(p,sizeof(p),"%s/golden_wave.f32",fixture);float*gold=read_bin(p,S);if(!f0||!noise||!refmel||!gold){fprintf(stderr,"fixture binary read failed\n");return 5;}
    float *breath=NULL,*voice=NULL,*tension=NULL,*gender=NULL,*velocity=NULL,*speaker=NULL;if(load_curve(fixture,"breathiness",T,0,&breath)||load_curve(fixture,"voicing",T,0,&voice)||load_curve(fixture,"tension",T,0,&tension)||load_curve(fixture,"gender",T,0,&gender)||load_curve(fixture,"velocity",T,1,&velocity)||load_speaker(spk,T,am.fs2.encoder.hidden_size,&speaker)){fprintf(stderr,"deployment input load failed\n");return 5;}
    int32_t*mel2ph=malloc(T*sizeof(*mel2ph));if(!mel2ph)return 5;size_t q=0;for(size_t i=0;i<P;i++)for(int32_t j=0;j<dur[i];j++)mel2ph[q++]=(int32_t)i+1;
    size_t wsn=ds_acoustic_model_workspace_floats(&am,P,T);float*ws=a64(wsn*sizeof(float)),*mel=a64(T*M*sizeof(float)),*wave=a64(S*sizeof(float));if(!ws||!mel||!wave){fprintf(stderr,"workspace alloc failed\n");return 5;}
    DSAsmFS2DeploymentInputs di={langs,breath,voice,tension,gender,velocity,speaker};
    printf("E2E persistent load: P=%zu T=%zu M=%zu samples=%zu workers=%zu load=%.3fms warmup=%zu requests=%zu\n",P,T,M,S,workers,loadms,warmup,requests);
    printf("E2E env: ADD=%s LEAKY=%s TILE=%s RANGE_T24=%s 2D=%s VNNI=%s\n",envs("DSASM_PARALLEL_ADD"),envs("DSASM_PARALLEL_LEAKY_COPY"),envs("DSASM_VOCODER_T_TILE"),envs("DSASM_RANGE_T24"),envs("DSASM_2D"),envs("DSASM_VNNI"));
    for(size_t r=0;r<warmup;r++){double a=now_ms();rc=ds_acoustic_model_infer_deploy_f32_avx2(&am,&di,tok,P,mel2ph,f0,T,noise,cfg.lo,cfg.hi,cfg.lo_n,cfg.t_start,cfg.scale,cfg.steps,mel,ws,ap);double b=now_ms();if(!rc)rc=ds_vocoder_graph_infer(vg,mel,f0,wave,0);double c=now_ms();if(rc){fprintf(stderr,"warmup rc=%d\n",rc);return 6;}printf("E2E warmup%zu acoustic=%.3f vocoder=%.3f total=%.3f RTF=%.3f\n",r+1,b-a,c-b,c-a,(c-a)/(1000.0*S/(double)cfg.sample_rate));}
    double *at=calloc(requests,sizeof(double)),*vt=calloc(requests,sizeof(double)),*tt=calloc(requests,sizeof(double)),*rtf=calloc(requests,sizeof(double));if(!at||!vt||!tt||!rtf)return 5;double audio=1000.0*S/(double)cfg.sample_rate;
    for(size_t r=0;r<requests;r++){double a=now_ms();rc=ds_acoustic_model_infer_deploy_f32_avx2(&am,&di,tok,P,mel2ph,f0,T,noise,cfg.lo,cfg.hi,cfg.lo_n,cfg.t_start,cfg.scale,cfg.steps,mel,ws,ap);double b=now_ms();if(!rc)rc=ds_vocoder_graph_infer(vg,mel,f0,wave,0);double c=now_ms();if(rc){fprintf(stderr,"request rc=%d\n",rc);return 6;}at[r]=b-a;vt[r]=c-b;tt[r]=c-a;rtf[r]=tt[r]/audio;printf("E2E request%zu acoustic=%.3f vocoder=%.3f total=%.3f audio=%.3f RTF=%.3f speed=%.3fx\n",r+1,at[r],vt[r],tt[r],audio,rtf[r],audio/tt[r]);}
    int bad=cmp_mel(mel,refmel,T*M)|cmp_wave(wave,gold,S);if(outwave&&write_bin(outwave,wave,S)){fprintf(stderr,"wave write failed\n");bad=1;}
    double*tmp=malloc(requests*sizeof(double));memcpy(tmp,rtf,requests*sizeof(double));qsort(tmp,requests,sizeof(double),dblcmp);double med=tmp[requests/2],worst=rtf[0],mean=0;for(size_t i=0;i<requests;i++){if(rtf[i]>worst)worst=rtf[i];mean+=rtf[i];}mean/=requests;printf("E2E FINAL persistent_requests=%zu warmup=%zu mean_RTF=%.3f median_RTF=%.3f worst_RTF=%.3f realtime_all=%s\n",requests,warmup,mean,med,worst,worst<1.0?"PASS":"FAIL");
    free(tmp);free(at);free(vt);free(tt);free(rtf);free(ws);free(mel);free(wave);free(mel2ph);free(tok);free(dur);free(langs);free(f0);free(noise);free(refmel);free(gold);free(breath);free(voice);free(tension);free(gender);free(velocity);free(speaker);ds_threadpool_destroy(ap);ds_vocoder_graph_free(vg);ds_acoustic_model_unload(&am);cfg_free(&cfg);return bad?7:0;
}
