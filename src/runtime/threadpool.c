#define _GNU_SOURCE
#include "threadpool_internal.h"
#include "dsasm_kernels.h"
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <immintrin.h>

#define DSASM_MAX_M_TILE 64u
#define DSASM_MAX_N_TILE 256u
#define DSASM_DEFAULT_M_TILE 16u
#define DSASM_DEFAULT_N_TILE 128u
#define DSASM_VOCODER_T_TILE_DEFAULT 504u

static size_t vocoder_t_tile(void){
    static size_t cached=0;
    if(cached) return cached;
    const char *e=getenv("DSASM_VOCODER_T_TILE");
    if(!e||!*e) return cached=DSASM_VOCODER_T_TILE_DEFAULT;
    char *end=0; unsigned long v=strtoul(e,&end,10);
    if(end==e || *end || v<24ul || (v%8ul)!=0ul) return cached=DSASM_VOCODER_T_TILE_DEFAULT;
    return cached=(size_t)v;
}

/* M50 release profile: the perf-guided M45-M49 winners become defaults.
   Environment variables remain explicit escape hatches for A/B and fallback. */
static int env_bool_default(const char *name,int def){
    const char *e=getenv(name);
    if(!e||!*e)return def;
    return strcmp(e,"0")!=0 && strcmp(e,"off")!=0;
}
static int vocoder_residual_t24_enabled(void){static int v=-1;if(v<0)v=env_bool_default("DSASM_RESIDUAL_T24",1);return v;}
static int vocoder_k7_t24_enabled(void){static int v=-1;if(v<0)v=env_bool_default("DSASM_K7_T24",1);return v;}
static int vocoder_k11_t24_enabled(void){static int v=-1;if(v<0)v=env_bool_default("DSASM_K11_T24",1);return v;}
static int vocoder_k3_tmode(void){
    static int v=-1;if(v>=0)return v;
    const char *e=getenv("DSASM_K3_TMODE");
    if(!e||!*e)return v=24;
    int x=atoi(e);return v=(x==16||x==24)?x:0;
}

struct DSAsmThreadPool;
typedef struct { struct DSAsmThreadPool *pool; size_t id; } WorkerArg;

typedef struct { int cpu, package_id, core_id; } CpuInfo;
typedef struct { int package_id, core_id, cpus[8]; size_t count; } CoreGroup;

struct DSAsmThreadPool {
    size_t n;
    pthread_t *threads;
    WorkerArg *args;
    int *cpus;
    int affinity_enabled;
    int use_2d;
    int atan_pipeline;
    int auto_tiles;
    int parallel_depthwise;
    int indexed_linear;
    int n_owner;
    int kblocked_atan;
    size_t k_block;
    int spin_dispatch;
    atomic_uint_fast64_t spin_generation;
    atomic_size_t spin_done;
    atomic_size_t spin_ready;
    atomic_int spin_stop;
    size_t m_tile, n_tile;
    float *scratch;
    size_t scratch_stride; /* floats per worker */
    pthread_mutex_t mu;
    pthread_cond_t start_cv, done_cv;
    uint64_t generation;
    size_t pending;
    int stop;
    DSAsmJob job;
    atomic_size_t next_task;
    size_t task_count;
};

static int read_int_file(const char *path, int fallback) {
    FILE *f=fopen(path,"r"); if(!f) return fallback;
    int v=fallback; if(fscanf(f,"%d",&v)!=1) v=fallback; fclose(f); return v;
}
static int cmp_cpu(const void *a,const void *b){
    const CpuInfo *x=(const CpuInfo*)a,*y=(const CpuInfo*)b;
    if(x->package_id!=y->package_id) return x->package_id-y->package_id;
    if(x->core_id!=y->core_id) return x->core_id-y->core_id;
    return x->cpu-y->cpu;
}

/* Prefer SMT-capable physical cores when a hybrid topology contains both SMT
   and single-thread physical cores. On i5-13420H this selects CPUs 0..7 (P
   cores) and leaves E cores out of the synchronous heavy AVX2 pool. */
static int *select_worker_cpus(size_t requested, size_t *out_n, int *out_affinity) {
    *out_n=0; *out_affinity=0;
#ifdef __linux__
    cpu_set_t allowed; CPU_ZERO(&allowed);
    if(sched_getaffinity(0,sizeof(allowed),&allowed)!=0) return NULL;
    long conf=sysconf(_SC_NPROCESSORS_CONF); if(conf<1) return NULL;
    CpuInfo *infos=calloc((size_t)conf,sizeof(*infos)); if(!infos) return NULL;
    size_t ni=0;
    for(int cpu=0;cpu<conf && cpu<CPU_SETSIZE;cpu++) if(CPU_ISSET(cpu,&allowed)) {
        char p[256];
        snprintf(p,sizeof(p),"/sys/devices/system/cpu/cpu%d/topology/core_id",cpu);
        int core=read_int_file(p,cpu);
        snprintf(p,sizeof(p),"/sys/devices/system/cpu/cpu%d/topology/physical_package_id",cpu);
        int pkg=read_int_file(p,0);
        infos[ni++]=(CpuInfo){cpu,pkg,core};
    }
    if(!ni){free(infos);return NULL;}
    qsort(infos,ni,sizeof(*infos),cmp_cpu);
    CoreGroup *groups=calloc(ni,sizeof(*groups)); if(!groups){free(infos);return NULL;}
    size_t ng=0;
    for(size_t i=0;i<ni;i++) {
        if(!ng || groups[ng-1].package_id!=infos[i].package_id || groups[ng-1].core_id!=infos[i].core_id) {
            groups[ng].package_id=infos[i].package_id; groups[ng].core_id=infos[i].core_id; ng++;
        }
        CoreGroup *g=&groups[ng-1]; if(g->count<8) g->cpus[g->count++]=infos[i].cpu;
    }
    size_t smt_groups=0,single_groups=0,pref_count=0;
    for(size_t g=0;g<ng;g++){if(groups[g].count>1){smt_groups++;pref_count+=groups[g].count;}else single_groups++;}
    int hybrid=(smt_groups>0 && single_groups>0);
    size_t available=hybrid?pref_count:ni;
    size_t want=requested?requested:available; if(want>ni) want=ni;
    int *sel=malloc(want*sizeof(*sel)); if(!sel){free(groups);free(infos);return NULL;}
    size_t nsel=0;
    for(size_t g=0;g<ng && nsel<want;g++) if(!hybrid || groups[g].count>1) sel[nsel++]=groups[g].cpus[0];
    for(size_t level=1;nsel<want;level++) {
        size_t before=nsel;
        for(size_t g=0;g<ng && nsel<want;g++) if((!hybrid || groups[g].count>1) && groups[g].count>level) sel[nsel++]=groups[g].cpus[level];
        if(nsel==before) break;
    }
    for(size_t g=0;g<ng && nsel<want;g++) if(hybrid && groups[g].count==1) sel[nsel++]=groups[g].cpus[0];
    free(groups); free(infos);
    if(!nsel){free(sel);return NULL;}
    *out_n=nsel; *out_affinity=1; return sel;
#else
    (void)requested; return NULL;
#endif
}

static inline void leaky_copy_channels(const float *x,float *dst,size_t C,size_t Tin,size_t Tp,size_t pad,float alpha){
    const __m256 z=_mm256_setzero_ps(), av=_mm256_set1_ps(alpha);
    for(size_t c=0;c<C;c++){
        float *d=dst+c*Tp; const float *s=x+c*Tin;
        if(pad){memset(d,0,pad*sizeof(float));memset(d+pad+Tin,0,pad*sizeof(float));}
        size_t i=0;
        for(;i+8u<=Tin;i+=8u){
            __m256 v=_mm256_loadu_ps(s+i);
            __m256 pos=_mm256_max_ps(v,z),neg=_mm256_min_ps(v,z);
            _mm256_storeu_ps(d+pad+i,_mm256_add_ps(pos,_mm256_mul_ps(neg,av)));
        }
        for(;i<Tin;i++){
            __m128 v=_mm_load_ss(s+i),zz=_mm_setzero_ps(),aa=_mm_set_ss(alpha);
            __m128 pos=_mm_max_ss(v,zz),neg=_mm_min_ss(v,zz);
            _mm_store_ss(d+pad+i,_mm_add_ss(pos,_mm_mul_ss(neg,aa)));
        }
    }
}

static void run_m_slice(const DSAsmJob *j,size_t id,size_t nthreads){
    if(j->kind==DS_JOB_LEAKY_COPY_NCT){
        const size_t c0=(j->M*id)/nthreads, c1=(j->M*(id+1))/nthreads;
        if(c1>c0) leaky_copy_channels(
            j->x+c0*j->N,j->y+c0*j->K,c1-c0,j->N,j->K,j->P,j->b0[0]);
        return;
    }
    if(j->kind==DS_JOB_ADD_F32){
        const size_t s=(j->M*id)/nthreads, e=(j->M*(id+1))/nthreads;
        size_t i=s;
        for(;i+8u<=e;i+=8u){
            __m256 a=_mm256_loadu_ps(j->x+i),b=_mm256_loadu_ps(j->w+i);
            _mm256_storeu_ps(j->y+i,_mm256_add_ps(a,b));
        }
        for(;i<e;i++)j->y[i]=j->x[i]+j->w[i];
        return;
    }
    if(j->kind==DS_JOB_VOCODER_VNNI_PACK){
        const unsigned char *qx=(const unsigned char*)j->x;
        unsigned char *xp=(unsigned char*)j->y;
        const int *offs=(const int*)j->b0;
        const size_t tb0=(j->M*id)/nthreads,tb1=(j->M*(id+1))/nthreads;
        if(tb1>tb0) ds_vnni_pack_u8_4x8_avx2(qx,xp,offs,j->N,tb0,tb1);
        return;
    }
    if(j->kind==DS_JOB_VOCODER_VNNI_CONV){
        const size_t blocks=j->N/8u,b0=(blocks*id)/nthreads,b1=(blocks*(id+1))/nthreads;
        if(b1>b0){
            const size_t c0=b0*8u,ct=(b1-b0)*8u;
            const unsigned char *xp=(const unsigned char*)j->x;
            const signed char *wp=(const signed char*)j->w + b0*j->K*32u;
            const int *corr=(const int*)j->b0 + c0;
            const float *scale=j->b1+c0,*bias=j->residual+c0;
            if(j->vnni_fn)j->vnni_fn(xp,wp,corr,scale,bias,j->y+c0*j->P,j->M,j->K,ct,j->P);
        }
        return;
    }
    if(j->kind==DS_JOB_VOCODER_CONVTRANSPOSE){
        const size_t c0=(j->N*id)/nthreads, c1=(j->N*(id+1))/nthreads;
        for(size_t oc=c0;oc<c1;oc++){
            const float *woc=j->w+oc*j->K*j->P;
            float *yoc=j->y+oc*j->M;
            const char *ct8=getenv("DSASM_CONVT_S8K16");
            const int use_s8k16=!(ct8 && !strcmp(ct8,"0"));
            if(use_s8k16 && j->P==16u && j->S==8u && j->R==4u && j->M==8u*j->Q)
                ds_convtranspose1d_s8k16_oc_f32_avx2(
                    j->x,woc,j->b0[oc],yoc,j->K,j->Q);
            else if(j->P==4u && j->S==2u && j->R==1u && j->M==2u*j->Q)
                ds_convtranspose1d_s2k4_oc_f32_avx2(
                    j->x,woc,j->b0[oc],yoc,j->K,j->Q);
            else
                ds_convtranspose1d_oc_f32_avx2(
                    j->x,woc,j->b0[oc],yoc,j->K,j->P,j->Q,j->M,j->R,j->S);
        }
        return;
    }
    if(j->kind==DS_JOB_VOCODER_CONV1D){
        const size_t pack=(j->S==8u)?8u:4u;
        const size_t blocks=j->N/pack;
        const size_t b0=(blocks*id)/nthreads, b1=(blocks*(id+1))/nthreads;
        if(b1>b0){
            const size_t c0=pack*b0, ct=pack*(b1-b0);
            const size_t block_floats=j->K*j->P*pack;
            if(pack==8u) {
                if(j->residual) {
                    const int rt24=vocoder_residual_t24_enabled() && (j->M%24u)==0;
                    const int k3mode=vocoder_k3_tmode();
                    const int k7t24=vocoder_k7_t24_enabled();
                    const int k11t24=vocoder_k11_t24_enabled();
                    if(rt24 && j->P==3u && k3mode==24)
                        ds_conv1d_nct_f32_avx2_oc4_t24_k3_residual(
                            j->x,j->w+b0*block_floats,j->b0+c0,j->y+c0*j->M,
                            j->K,ct,j->P,j->M,j->Q,j->R,j->residual+c0*j->M);
                    else if(rt24 && j->P==7u && k7t24)
                        ds_conv1d_nct_f32_avx2_oc4_t24_k7_residual(
                            j->x,j->w+b0*block_floats,j->b0+c0,j->y+c0*j->M,
                            j->K,ct,j->P,j->M,j->Q,j->R,j->residual+c0*j->M);
                    else if(rt24 && j->P==11u && k11t24)
                        ds_conv1d_nct_f32_avx2_oc4_t24_k11_residual(
                            j->x,j->w+b0*block_floats,j->b0+c0,j->y+c0*j->M,
                            j->K,ct,j->P,j->M,j->Q,j->R,j->residual+c0*j->M);
                    else
                        ds_conv1d_nct_f32_avx2_oc8_t8_residual(
                            j->x, j->w+b0*block_floats, j->b0+c0, j->y+c0*j->M,
                            j->K, ct, j->P, j->M, j->Q, j->R, j->residual+c0*j->M);
                } else {
                    const int k3mode=vocoder_k3_tmode();
                    const int k7t24=vocoder_k7_t24_enabled();
                    const int k11t24=vocoder_k11_t24_enabled();
                    if(k3mode==24 && (j->M%24u)==0 && j->P==3u)
                        ds_conv1d_nct_f32_avx2_oc4_t24_k3(
                            j->x, j->w+b0*block_floats, j->b0+c0, j->y+c0*j->M,
                            j->K, ct, j->P, j->M, j->Q, j->R);
                    else if(k3mode==16 && (j->M%16u)==0 && j->P==3u)
                        ds_conv1d_nct_f32_avx2_oc4_t16_k3(
                            j->x, j->w+b0*block_floats, j->b0+c0, j->y+c0*j->M,
                            j->K, ct, j->P, j->M, j->Q, j->R);
                    else if(k7t24 && (j->M%24u)==0 && j->P==7u)
                        ds_conv1d_nct_f32_avx2_oc4_t24_k7(
                            j->x, j->w+b0*block_floats, j->b0+c0, j->y+c0*j->M,
                            j->K, ct, j->P, j->M, j->Q, j->R);
                    else if(k11t24 && (j->M%24u)==0 && j->P==11u)
                        ds_conv1d_nct_f32_avx2_oc4_t24_k11(
                            j->x, j->w+b0*block_floats, j->b0+c0, j->y+c0*j->M,
                            j->K, ct, j->P, j->M, j->Q, j->R);
                    else if(j->T && (j->M%8u)==0 && j->P==3u) ds_conv1d_nct_f32_avx2_oc8_t8_k3(
                    j->x, j->w+b0*block_floats, j->b0+c0, j->y+c0*j->M,
                    j->K, ct, j->P, j->M, j->Q, j->R);
                else if(j->T && (j->M%8u)==0 && j->P==7u) ds_conv1d_nct_f32_avx2_oc8_t8_k7(
                    j->x, j->w+b0*block_floats, j->b0+c0, j->y+c0*j->M,
                    j->K, ct, j->P, j->M, j->Q, j->R);
                    else if(j->T && (j->M%8u)==0 && j->P==11u) ds_conv1d_nct_f32_avx2_oc8_t8_k11(
                        j->x, j->w+b0*block_floats, j->b0+c0, j->y+c0*j->M,
                        j->K, ct, j->P, j->M, j->Q, j->R);
                    else ds_conv1d_nct_f32_avx2_oc8_t8(
                        j->x, j->w+b0*block_floats, j->b0+c0, j->y+c0*j->M,
                        j->K, ct, j->P, j->M, j->Q, j->R);
                }
            }
            else ds_conv1d_nct_f32_avx2_oc4_t8(
                j->x, j->w+b0*block_floats, j->b0+c0, j->y+c0*j->M,
                j->K, ct, j->P, j->M, j->Q, j->R);
        }
        return;
    }
    if(j->kind==DS_JOB_DEPTHWISE_K31){
        const size_t blocks=(j->N+7u)/8u;
        const size_t b0=(blocks*id)/nthreads, b1=(blocks*(id+1))/nthreads;
        const size_t c0=b0*8u, c1=(b1*8u<j->N)?b1*8u:j->N;
        if(c1>c0) ds_depthwise_conv1d_k31_tc_f32_avx2_cstrided(
            j->x+c0,j->w+c0,j->b0+c0,j->y+c0,j->M,c1-c0,j->N);
        return;
    }
    size_t s=(j->M*id)/nthreads, e=(j->M*(id+1))/nthreads;
    if(e<=s) return;
    size_t m=e-s;
    switch(j->kind){
    case DS_JOB_LINEAR:
        ds_linear_f32_avx2_m4n16(j->x+s*j->K,j->w,j->b0,j->y+s*j->N,m,j->N,j->K);break;
    case DS_JOB_LINEAR_RESIDUAL:
        ds_linear_residual_f32_avx2_m4n16(j->x+s*j->K,j->w,j->b0,j->residual+s*j->N,j->y+s*j->N,m,j->N,j->K);break;
    case DS_JOB_SOFTSIGN_GLU:
        ds_fused_linear_softsign_glu_f32_avx2_m4n8(j->x+s*j->K,j->w,j->b0,j->b1,j->y+s*j->N,m,j->N,j->K);break;
    default: break;
    }
}

static int job_use_2d(const DSAsmThreadPool *p,const DSAsmJob *j){
    if(!p->use_2d) return 0;
    /* M37: packed-8 vocoder Conv becomes under-subscribed once Cout<8*n.
       Split only those long late-stage layers across output-block x time tiles.
       Large-channel layers keep M36's channel-owner path for weight-cache reuse. */
    if(j->kind==DS_JOB_VOCODER_CONV1D) {
        const size_t pack=j->S;
        const size_t blocks=pack?j->N/pack:0;
        return pack==8u && blocks>0 && blocks<p->n && j->M>=1024u;
    }
    if(j->kind==DS_JOB_ATAN_GLU_LINEAR) {
        return p->atan_pipeline && j->M>=8 && j->N>=128 && j->K>=128 && (j->N%16)==0;
    }
    if(j->kind!=DS_JOB_LINEAR && j->kind!=DS_JOB_LINEAR_RESIDUAL) return 0;
    return j->M>=16 && j->N>=256 && j->K>=128 && (j->N%16)==0;
}
static void choose_tiles(const DSAsmThreadPool *p,const DSAsmJob *j,size_t *mt,size_t *nt){
    *mt=p->m_tile; *nt=p->n_tile;
    if(!p->auto_tiles) return;

    /* M10 joint shape/worker policy. The user's i5-13420H sweep measured the
       best whole-forward setting as 8x256 for C/H=256 and 32x64 for the
       official C/H=1024 shape. Apply the selected geometry consistently to
       every eligible 2-D Linear/Residual/ATan job so the default path really
       reproduces the winning whole-forward configuration. */
    if(p->n>=8){
        if(j->N>=768){ *mt=32; *nt=64; }
        else if(j->N<=256){ *mt=8; *nt=256; }
        else { *mt=16; *nt=128; }
    } else {
        /* Development host (5 preferred workers) repeatedly favored 16x64. */
        *mt=16; *nt=64;
    }
}

static int job_use_n_owner(const DSAsmThreadPool *p,const DSAsmJob *j){
    if(!p->n_owner || !job_use_2d(p,j) || p->n<2) return 0;
    /* M14: one worker owns a complete N tile and walks all M chunks.
       This avoids loading the same 64-wide packed weight tile into multiple
       private P-core caches. Only enable when there are enough N tiles to
       keep every worker busy; small C=256 shapes deliberately stay on M10. */
    const size_t nt=(j->N>=512)?64u:128u;
    const size_t ntasks=(j->N+nt-1)/nt;
    return j->M>=32 && j->K>=128 && (j->N%16)==0 && ntasks>=p->n;
}

static size_t task_count_for(const DSAsmThreadPool *p,const DSAsmJob *j){
    if(!job_use_2d(p,j)) return p->n;
    if(j->kind==DS_JOB_VOCODER_CONV1D){
        const size_t blocks=j->N/8u;
        const size_t tile=vocoder_t_tile();
        const size_t tt=(j->M+tile-1u)/tile;
        return blocks*tt;
    }
    if(job_use_n_owner(p,j)){
        const size_t nt=(j->N>=512)?64u:128u;
        return (j->N+nt-1)/nt;
    }
    size_t mt,nt; choose_tiles(p,j,&mt,&nt);
    return ((j->M+mt-1)/mt)*((j->N+nt-1)/nt);
}

static inline int use_indexed_linear(const DSAsmThreadPool *p,const DSAsmJob *j,size_t nl){
    /* M13: perf on the official C=1024 path showed the heavy work is P-core
       Linear/ATan GEMM. The indexed-address inner loop is locally faster for
       the winning 32x64 geometry but not for wider N tiles, so keep the
       selection deliberately narrow instead of globally replacing M12. */
    return p->indexed_linear && nl<=64 && j->K>=128;
}

static void run_kblocked_linear16(const float *x,const float *w,const float *b,float *y,
                                      size_t M,size_t N,size_t K,size_t y_stride,size_t k_block){
    if(k_block<1) k_block=K;
    for(size_t n0=0;n0<N;n0+=16){
        const float *wn=w+(n0/16)*K*16;
        for(size_t k0=0;k0<K;k0+=k_block){
            size_t kl=K-k0;if(kl>k_block)kl=k_block;
            ds_linear_f32_avx2_n16_kblock_accum(x+k0,wn+k0*16,b+n0,y+n0,M,K,kl,y_stride,k0==0);
        }
    }
}

static void run_tile_region(DSAsmThreadPool *p,const DSAsmJob *j,
                            size_t m0,size_t n0,size_t ml,size_t nl,size_t worker_id){
    if(!ml||!nl) return;
    const float *xt=j->x+m0*j->K;
    if(j->kind==DS_JOB_ATAN_GLU_LINEAR) {
        float *tmp=p->scratch+worker_id*p->scratch_stride;
        const size_t tmp_stride=2*nl;
        const float *wl=j->w+(n0/16)*j->K*16;
        const float *wg=j->w+((j->N+n0)/16)*j->K*16;
        const float *bl=j->b0+n0;
        const float *bg=j->b0+j->N+n0;
        const int use_kblock = p->kblocked_atan && j->M>=32 && j->N>=512 && j->K>=512;
        if(use_kblock){
            run_kblocked_linear16(xt,wl,bl,tmp,ml,nl,j->K,tmp_stride,p->k_block);
            run_kblocked_linear16(xt,wg,bg,tmp+nl,ml,nl,j->K,tmp_stride,p->k_block);
        }else if(use_indexed_linear(p,j,nl)){
            ds_linear_f32_avx2_m4n16_idxstrided(xt,wl,bl,tmp,ml,nl,j->K,tmp_stride);
            ds_linear_f32_avx2_m4n16_idxstrided(xt,wg,bg,tmp+nl,ml,nl,j->K,tmp_stride);
        }else{
            ds_linear_f32_avx2_m4n16_strided(xt,wl,bl,tmp,ml,nl,j->K,tmp_stride);
            ds_linear_f32_avx2_m4n16_strided(xt,wg,bg,tmp+nl,ml,nl,j->K,tmp_stride);
        }
        ds_atan_glu_f32_avx2_ystrided(tmp,j->y+m0*j->N+n0,ml,nl,j->N);
        return;
    }

    const float *wt=j->w+(n0/16)*j->K*16;
    const float *bt=j->b0+n0;
    float *yt=j->y+m0*j->N+n0;
    if(j->kind==DS_JOB_LINEAR) {
        if(use_indexed_linear(p,j,nl)) ds_linear_f32_avx2_m4n16_idxstrided(xt,wt,bt,yt,ml,nl,j->K,j->N);
        else ds_linear_f32_avx2_m4n16_strided(xt,wt,bt,yt,ml,nl,j->K,j->N);
    } else {
        const float *rt=j->residual+m0*j->N+n0;
        if(use_indexed_linear(p,j,nl)) ds_linear_residual_f32_avx2_m4n16_idxstrided(xt,wt,bt,rt,yt,ml,nl,j->K,j->N);
        else ds_linear_residual_f32_avx2_m4n16_strided(xt,wt,bt,rt,yt,ml,nl,j->K,j->N);
    }
}

static void run_2d_task(DSAsmThreadPool *p,const DSAsmJob *j,size_t task,size_t worker_id){
    if(j->kind==DS_JOB_VOCODER_CONV1D){
        (void)p;(void)worker_id;
        const size_t pack=8u;
        const size_t blocks=j->N/pack;
        const size_t tile=vocoder_t_tile();
        const size_t tt=(j->M+tile-1u)/tile;
        const size_t ob=task/tt, ti=task%tt;
        if(ob>=blocks) return;
        const size_t t0=ti*tile;
        size_t tc=j->M-t0; if(tc>tile)tc=tile;
        const size_t block_floats=j->K*j->P*pack;
        if(j->residual) ds_conv1d_nct_f32_avx2_oc8_t8_range_residual(
            j->x,j->w+ob*block_floats,j->b0+ob*pack,j->y+ob*pack*j->M,
            j->K,j->P,t0,tc,j->Q,j->R,j->M,j->residual+ob*pack*j->M);
        else if((j->T&2u) && (t0%24u)==0 && (tc%24u)==0 && j->P==7u) ds_conv1d_nct_f32_avx2_oc4_t24_range_k7(
            j->x,j->w+ob*block_floats,j->b0+ob*pack,j->y+ob*pack*j->M,
            j->K,j->P,t0,tc,j->Q,j->R,j->M);
        else if((j->T&2u) && (t0%24u)==0 && (tc%24u)==0 && j->P==11u) ds_conv1d_nct_f32_avx2_oc4_t24_range_k11(
            j->x,j->w+ob*block_floats,j->b0+ob*pack,j->y+ob*pack*j->M,
            j->K,j->P,t0,tc,j->Q,j->R,j->M);
        else if((j->T&1u) && (t0%8u)==0 && (tc%8u)==0 && j->P==3u) ds_conv1d_nct_f32_avx2_oc8_t8_range_k3(
            j->x,j->w+ob*block_floats,j->b0+ob*pack,j->y+ob*pack*j->M,
            j->K,j->P,t0,tc,j->Q,j->R,j->M);
        else if((j->T&1u) && (t0%8u)==0 && (tc%8u)==0 && j->P==7u) ds_conv1d_nct_f32_avx2_oc8_t8_range_k7(
            j->x,j->w+ob*block_floats,j->b0+ob*pack,j->y+ob*pack*j->M,
            j->K,j->P,t0,tc,j->Q,j->R,j->M);
        else if((j->T&1u) && (t0%8u)==0 && (tc%8u)==0 && j->P==11u) ds_conv1d_nct_f32_avx2_oc8_t8_range_k11(
            j->x,j->w+ob*block_floats,j->b0+ob*pack,j->y+ob*pack*j->M,
            j->K,j->P,t0,tc,j->Q,j->R,j->M);
        else ds_conv1d_nct_f32_avx2_oc8_t8_range(
            j->x,j->w+ob*block_floats,j->b0+ob*pack,j->y+ob*pack*j->M,
            j->K,j->P,t0,tc,j->Q,j->R,j->M);
        return;
    }
    size_t mt,nt; choose_tiles(p,j,&mt,&nt);
    const size_t tm=(j->M+mt-1)/mt;
    const size_t ni=task/tm, mi=task%tm;
    const size_t m0=mi*mt, n0=ni*nt;
    size_t ml=j->M-m0; if(ml>mt) ml=mt;
    size_t nl=j->N-n0; if(nl>nt) nl=nt;
    run_tile_region(p,j,m0,n0,ml,nl,worker_id);
}

static void run_n_owner_task(DSAsmThreadPool *p,const DSAsmJob *j,size_t task,size_t worker_id){
    const size_t nt=(j->N>=512)?64u:128u;
    const size_t n0=task*nt;
    size_t nl=j->N-n0; if(nl>nt) nl=nt;
    /* 64 rows keep ATan private scratch <= 32 KiB for N=64 and let one
       worker/core reuse its packed weight slice across the whole phrase. */
    for(size_t m0=0;m0<j->M;m0+=DSASM_MAX_M_TILE){
        size_t ml=j->M-m0; if(ml>DSASM_MAX_M_TILE) ml=DSASM_MAX_M_TILE;
        run_tile_region(p,j,m0,n0,ml,nl,worker_id);
    }
}

static inline void run_worker_job(DSAsmThreadPool *p,const DSAsmJob *j,size_t worker_id){
    const int use2d=job_use_2d(p,j);
    if(use2d) {
        const int owner=job_use_n_owner(p,j);
        for(;;){
            size_t q=atomic_fetch_add_explicit(&p->next_task,1,memory_order_relaxed);
            if(q>=p->task_count)break;
            if(owner) run_n_owner_task(p,j,q,worker_id);
            else run_2d_task(p,j,q,worker_id);
        }
    } else {
        size_t q=atomic_fetch_add_explicit(&p->next_task,1,memory_order_relaxed);
        if(q<p->task_count) run_m_slice(j,q,p->n);
    }
}

static void pin_worker(DSAsmThreadPool *p,size_t id){
#ifdef __linux__
    if(p->affinity_enabled && p->cpus && p->cpus[id]>=0) {
        cpu_set_t set; CPU_ZERO(&set); CPU_SET(p->cpus[id],&set);
        (void)pthread_setaffinity_np(pthread_self(),sizeof(set),&set);
    }
#else
    (void)p;(void)id;
#endif
}

static void *worker(void *vp){
    WorkerArg *a=(WorkerArg*)vp; DSAsmThreadPool *p=a->pool; uint64_t seen=0;
    pin_worker(p,a->id);
    if(p->spin_dispatch){
        seen=atomic_load_explicit(&p->spin_generation,memory_order_acquire);
        atomic_fetch_add_explicit(&p->spin_ready,1,memory_order_release);
        for(;;){
            uint64_t g;
            while((g=atomic_load_explicit(&p->spin_generation,memory_order_acquire))==seen){
                if(atomic_load_explicit(&p->spin_stop,memory_order_relaxed)) return NULL;
                _mm_pause();
            }
            if(atomic_load_explicit(&p->spin_stop,memory_order_relaxed)) return NULL;
            seen=g; DSAsmJob j=p->job;
            run_worker_job(p,&j,a->id);
            size_t prev=atomic_fetch_add_explicit(&p->spin_done,1,memory_order_release);
            if(prev+1==p->n){
                pthread_mutex_lock(&p->mu);
                pthread_cond_signal(&p->done_cv);
                pthread_mutex_unlock(&p->mu);
            }
        }
    }
    pthread_mutex_lock(&p->mu);
    for(;;){
        while(!p->stop && p->generation==seen) pthread_cond_wait(&p->start_cv,&p->mu);
        if(p->stop){pthread_mutex_unlock(&p->mu);return NULL;}
        seen=p->generation; DSAsmJob j=p->job;
        pthread_mutex_unlock(&p->mu);
        run_worker_job(p,&j,a->id);
        pthread_mutex_lock(&p->mu);
        if(--p->pending==0) pthread_cond_signal(&p->done_cv);
    }
}

DSAsmThreadPool *ds_threadpool_create(size_t n){
    size_t selected=0;int affinity=0;int *cpus=select_worker_cpus(n,&selected,&affinity);
    if(cpus) n=selected; else if(n<1) n=1;
    DSAsmThreadPool *p=calloc(1,sizeof(*p)); if(!p){free(cpus);return NULL;}
    p->n=n;p->cpus=cpus;p->affinity_enabled=affinity;p->use_2d=1;p->atan_pipeline=1;p->auto_tiles=1;p->parallel_depthwise=0;p->indexed_linear=0;
    /* M18 freeze: repeated target-machine A/B showed Kblock=512 and N-owner
       are useful experiments but not stable defaults across frequency/thermal
       states.  Freeze the production CPU denoiser on dynamic full-K; public
       setters/environment variables keep both cache experiments available. */
    p->n_owner=0;p->kblocked_atan=0;p->k_block=512;p->spin_dispatch=0;
    p->m_tile=DSASM_DEFAULT_M_TILE;p->n_tile=DSASM_DEFAULT_N_TILE;
    p->scratch_stride=2u*DSASM_MAX_M_TILE*DSASM_MAX_N_TILE;
    if(posix_memalign((void**)&p->scratch,64,n*p->scratch_stride*sizeof(float))!=0)p->scratch=NULL;
    const char *ae=getenv("DSASM_AFFINITY");if(ae && !strcmp(ae,"0"))p->affinity_enabled=0;
    const char *s2=getenv("DSASM_2D");if(s2 && !strcmp(s2,"0"))p->use_2d=0;
    const char *ap=getenv("DSASM_ATAN_PIPELINE");if(ap && !strcmp(ap,"0"))p->atan_pipeline=0;
    const char *mt=getenv("DSASM_M_TILE");if(mt){size_t v=(size_t)strtoul(mt,NULL,10);if(v>=1&&v<=DSASM_MAX_M_TILE){p->m_tile=v;p->auto_tiles=0;}}
    const char *nt=getenv("DSASM_N_TILE");if(nt){size_t v=(size_t)strtoul(nt,NULL,10);if(v>=16&&v<=DSASM_MAX_N_TILE&&(v%16)==0){p->n_tile=v;p->auto_tiles=0;}}
    const char *at=getenv("DSASM_AUTO_TILES");if(at)p->auto_tiles=strcmp(at,"0")!=0;
    const char *pd=getenv("DSASM_PARALLEL_DW");if(pd)p->parallel_depthwise=strcmp(pd,"0")!=0;
    const char *il=getenv("DSASM_INDEXED_LINEAR");if(il)p->indexed_linear=strcmp(il,"0")!=0;
    const char *no=getenv("DSASM_N_OWNER");if(no)p->n_owner=strcmp(no,"0")!=0;
    const char *ka=getenv("DSASM_KBLOCK_ATAN");if(ka)p->kblocked_atan=strcmp(ka,"0")!=0;
    const char *kb=getenv("DSASM_K_BLOCK");if(kb){size_t v=(size_t)strtoul(kb,NULL,10);if(v>=16&&v<=4096)p->k_block=v;}
    const char *sp=getenv("DSASM_SPIN_POOL");if(sp)p->spin_dispatch=strcmp(sp,"0")!=0;
    p->threads=calloc(n,sizeof(*p->threads));p->args=calloc(n,sizeof(*p->args));
    if(!p->threads||!p->args||!p->scratch){free(p->threads);free(p->args);free(p->scratch);free(p->cpus);free(p);return NULL;}
    pthread_mutex_init(&p->mu,NULL);pthread_cond_init(&p->start_cv,NULL);pthread_cond_init(&p->done_cv,NULL);atomic_init(&p->next_task,0);atomic_init(&p->spin_generation,0);atomic_init(&p->spin_done,0);atomic_init(&p->spin_ready,0);atomic_init(&p->spin_stop,0);
    for(size_t i=0;i<n;i++){
        p->args[i]=(WorkerArg){p,i};
        if(pthread_create(&p->threads[i],NULL,worker,&p->args[i])){
            if(p->spin_dispatch){atomic_store_explicit(&p->spin_stop,1,memory_order_relaxed);atomic_fetch_add_explicit(&p->spin_generation,1,memory_order_release);}
            else{pthread_mutex_lock(&p->mu);p->stop=1;p->generation++;pthread_cond_broadcast(&p->start_cv);pthread_mutex_unlock(&p->mu);}
            for(size_t j=0;j<i;j++)pthread_join(p->threads[j],NULL);
            free(p->threads);free(p->args);free(p->scratch);free(p->cpus);free(p);return NULL;
        }
    }
    if(p->spin_dispatch){
        while(atomic_load_explicit(&p->spin_ready,memory_order_acquire)<n) _mm_pause();
    }
    return p;
}
DSAsmThreadPool *ds_threadpool_create_auto(void){return ds_threadpool_create(0);}
void ds_threadpool_destroy(DSAsmThreadPool *p){if(!p)return;if(p->spin_dispatch){atomic_store_explicit(&p->spin_stop,1,memory_order_relaxed);atomic_fetch_add_explicit(&p->spin_generation,1,memory_order_release);}else{pthread_mutex_lock(&p->mu);p->stop=1;p->generation++;pthread_cond_broadcast(&p->start_cv);pthread_mutex_unlock(&p->mu);}for(size_t i=0;i<p->n;i++)pthread_join(p->threads[i],NULL);pthread_mutex_destroy(&p->mu);pthread_cond_destroy(&p->start_cv);pthread_cond_destroy(&p->done_cv);free(p->threads);free(p->args);free(p->scratch);free(p->cpus);free(p);}
size_t ds_threadpool_threads(const DSAsmThreadPool *p){return p?p->n:0;}
int ds_threadpool_cpu_at(const DSAsmThreadPool *p,size_t i){return (p&&p->cpus&&i<p->n)?p->cpus[i]:-1;}
int ds_threadpool_affinity_enabled(const DSAsmThreadPool *p){return p?p->affinity_enabled:0;}
void ds_threadpool_add_f32(DSAsmThreadPool *p,const float *a,const float *b,float *y,size_t n){
    if(!n)return;
    DSAsmJob j={0};j.kind=DS_JOB_ADD_F32;j.x=a;j.w=b;j.y=y;j.M=n;
    ds_threadpool_run(p,&j);
}
void ds_threadpool_leaky_copy_nct_f32(DSAsmThreadPool *p,const float *x,float *dst,
                                      size_t C,size_t Tin,size_t Tp,size_t pad,float alpha){
    if(!C||!Tin)return;
    if(!p){leaky_copy_channels(x,dst,C,Tin,Tp,pad,alpha);return;}
    DSAsmJob j={0};j.kind=DS_JOB_LEAKY_COPY_NCT;j.x=x;j.y=dst;j.b0=&alpha;
    j.M=C;j.N=Tin;j.K=Tp;j.P=pad;
    ds_threadpool_run(p,&j);
}
void ds_threadpool_set_2d(DSAsmThreadPool *p,int enabled){if(p)p->use_2d=!!enabled;}
int ds_threadpool_get_2d(const DSAsmThreadPool *p){return p?p->use_2d:0;}
void ds_threadpool_set_atan_pipeline(DSAsmThreadPool *p,int enabled){if(p)p->atan_pipeline=!!enabled;}
int ds_threadpool_get_atan_pipeline(const DSAsmThreadPool *p){return p?p->atan_pipeline:0;}
int ds_threadpool_set_tiles(DSAsmThreadPool *p,size_t m,size_t n){
    if(!p||m<1||m>DSASM_MAX_M_TILE||n<16||n>DSASM_MAX_N_TILE||(n%16))return -1;
    p->m_tile=m;p->n_tile=n;p->auto_tiles=0;return 0;
}
size_t ds_threadpool_m_tile(const DSAsmThreadPool *p){return p?p->m_tile:0;}
size_t ds_threadpool_n_tile(const DSAsmThreadPool *p){return p?p->n_tile:0;}
void ds_threadpool_set_auto_tiles(DSAsmThreadPool *p,int enabled){if(p)p->auto_tiles=!!enabled;}
int ds_threadpool_get_auto_tiles(const DSAsmThreadPool *p){return p?p->auto_tiles:0;}
void ds_threadpool_set_parallel_depthwise(DSAsmThreadPool *p,int enabled){if(p)p->parallel_depthwise=!!enabled;}
int ds_threadpool_get_parallel_depthwise(const DSAsmThreadPool *p){return p?p->parallel_depthwise:0;}
void ds_threadpool_set_indexed_linear(DSAsmThreadPool *p,int enabled){if(p)p->indexed_linear=!!enabled;}
int ds_threadpool_get_indexed_linear(const DSAsmThreadPool *p){return p?p->indexed_linear:0;}
void ds_threadpool_set_n_owner(DSAsmThreadPool *p,int enabled){if(p)p->n_owner=!!enabled;}
int ds_threadpool_get_n_owner(const DSAsmThreadPool *p){return p?p->n_owner:0;}
void ds_threadpool_set_kblocked_atan(DSAsmThreadPool *p,int enabled){if(p)p->kblocked_atan=!!enabled;}
int ds_threadpool_get_kblocked_atan(const DSAsmThreadPool *p){return p?p->kblocked_atan:0;}
int ds_threadpool_set_k_block(DSAsmThreadPool *p,size_t k){if(!p||k<16||k>4096)return -1;p->k_block=k;return 0;}
size_t ds_threadpool_k_block(const DSAsmThreadPool *p){return p?p->k_block:0;}
void ds_threadpool_run(DSAsmThreadPool *p,const DSAsmJob *j){
    if(!p){run_m_slice(j,0,1);return;}
    if(p->n<=1){
        if(job_use_2d(p,j)){
            const size_t count=task_count_for(p,j);
            for(size_t q=0;q<count;q++)run_2d_task(p,j,q,0);
        }else run_m_slice(j,0,1);
        return;
    }
    if(p->spin_dispatch){
        pthread_mutex_lock(&p->mu);
        p->job=*j;p->task_count=task_count_for(p,j);
        atomic_store_explicit(&p->next_task,0,memory_order_relaxed);
        atomic_store_explicit(&p->spin_done,0,memory_order_relaxed);
        atomic_fetch_add_explicit(&p->spin_generation,1,memory_order_release);
        while(atomic_load_explicit(&p->spin_done,memory_order_acquire)<p->n) pthread_cond_wait(&p->done_cv,&p->mu);
        pthread_mutex_unlock(&p->mu);
        return;
    }
    pthread_mutex_lock(&p->mu);p->job=*j;p->task_count=task_count_for(p,j);atomic_store_explicit(&p->next_task,0,memory_order_relaxed);p->pending=p->n;p->generation++;pthread_cond_broadcast(&p->start_cv);while(p->pending)pthread_cond_wait(&p->done_cv,&p->mu);pthread_mutex_unlock(&p->mu);
}
