#define _POSIX_C_SOURCE 200809L
#include "dsasm_lynxnet2.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { unsigned char *p; size_t n; } Bytes;
static Bytes load(const char *p){FILE*f=fopen(p,"rb");if(!f){perror(p);exit(2);}fseek(f,0,SEEK_END);long n=ftell(f);rewind(f);unsigned char*b=malloc((size_t)n);if(!b||fread(b,1,(size_t)n,f)!=(size_t)n){perror("read");exit(2);}fclose(f);return (Bytes){b,(size_t)n};}
static size_t al64(size_t x){return (x+63u)&~63u;}
static const float *take(Bytes b,size_t *off,size_t nf){*off=al64(*off);size_t n=nf*4;if(*off+n>b.n){fprintf(stderr,"bundle truncated\n");exit(2);}const float*p=(const float*)(b.p+*off);*off+=n;return p;}
static float maxerr(const float*a,const float*b,size_t n){float m=0;for(size_t i=0;i<n;i++){float d=fabsf(a[i]-b[i]);if(d>m)m=d;}return m;}
int main(int ac,char**av){
 if(ac<6){fprintf(stderr,"usage: %s model.dsn spec.f32 condition.f32 timestep.f32 output.f32 [threads]\n",av[0]);return 2;}
 Bytes m=load(av[1]);if(m.n<64||memcmp(m.p,"DSLYNX7\0",8)){fprintf(stderr,"bad DSLYNX7 magic\n");return 2;}uint32_t*v=(uint32_t*)(m.p+8);uint32_t ver=v[0],I=v[1],Q=v[2],C=v[3],H=v[4],L=v[5],ks=v[6],glu=v[7];if(ver!=1||ks!=31||!L){fprintf(stderr,"unsupported header\n");return 2;}
 size_t off=64;DSAsmLynxNet2Weights w={0};w.input_dim=I;w.condition_dim=Q;w.channels=C;w.hidden_dim=H;w.num_layers=L;w.kernel_size=31;w.glu_type=glu;
 w.input_weight_m4n16=take(m,&off,(size_t)C*I);w.input_bias=take(m,&off,C);w.condition_weight_m4n16=take(m,&off,(size_t)C*Q);w.condition_bias=take(m,&off,C);w.time1_weight_m4n16=take(m,&off,(size_t)4*C*C);w.time1_bias=take(m,&off,4*C);w.time2_weight_m4n16=take(m,&off,(size_t)4*C*C);w.time2_bias=take(m,&off,C);
 DSAsmLynxNet2Block *bs=calloc(L,sizeof(*bs));if(!bs)return 2;w.blocks=bs;
 for(uint32_t i=0;i<L;i++){DSAsmLynxNet2Block*b=&bs[i];b->ln_gamma=take(m,&off,C);b->ln_beta=take(m,&off,C);b->dw_weight_tap_major=take(m,&off,(size_t)31*C);b->dw_bias=take(m,&off,C);b->glu1_weight=take(m,&off,(size_t)2*H*C);b->glu1_bias=take(m,&off,2*H);b->glu2_weight=take(m,&off,(size_t)2*H*H);b->glu2_bias=take(m,&off,2*H);b->out_weight_m4n16=take(m,&off,(size_t)C*H);b->out_bias=take(m,&off,C);}
 w.post_norm_gamma=take(m,&off,C);w.post_norm_beta=take(m,&off,C);w.output_weight_m4n16=take(m,&off,(size_t)I*C);w.output_bias=take(m,&off,I);
 Bytes sp=load(av[2]),co=load(av[3]),ti=load(av[4]),rf=load(av[5]);if(sp.n%(I*4)||ti.n<4){fprintf(stderr,"bad vector files\n");return 2;}size_t T=sp.n/(I*4);if(co.n!=T*Q*4||rf.n!=T*I*4){fprintf(stderr,"shape mismatch\n");return 2;}float*out=malloc(T*I*4),*ws=malloc(ds_lynxnet2_workspace_floats(&w,T)*4);if(!out||!ws)return 2;int rc=ds_lynxnet2_forward_f32_avx2(&w,(float*)sp.p,(float*)co.p,*(float*)ti.p,out,ws,T);if(rc){fprintf(stderr,"forward failed %d\n",rc);return 2;}float e=maxerr(out,(float*)rf.p,T*I);printf("DSLYNX7 serial PyTorch reference: T=%zu I=%u Q=%u C=%u H=%u L=%u glu=%u max_abs=%g %s\n",T,I,Q,C,H,L,glu,e,e<2e-4f?"OK":"FAIL");int fail=e>=2e-4f;
 if(ac>6){size_t nt=strtoul(av[6],0,10);DSAsmThreadPool*p=ds_threadpool_create(nt);float*cache=malloc(T*C*4);if(!p||!cache)return 2;ds_lynxnet2_prepare_condition_parallel_f32_avx2(&w,(float*)co.p,cache,T,p);ds_lynxnet2_forward_cached_condition_parallel_f32_avx2(&w,(float*)sp.p,cache,*(float*)ti.p,out,ws,T,p);float ep=maxerr(out,(float*)rf.p,T*I);printf("DSLYNX7 parallel %zuT max_abs=%g %s\n",nt,ep,ep<2e-4f?"OK":"FAIL");fail|=ep>=2e-4f;ds_threadpool_destroy(p);free(cache);}
 free(out);free(ws);free(bs);free(m.p);free(sp.p);free(co.p);free(ti.p);free(rf.p);return fail?1:0;
}
