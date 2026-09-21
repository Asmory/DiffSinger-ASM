#define _GNU_SOURCE
#include "dsasm_vocoder_graph.h"
#include "dsasm_vocoder.h"
#include "dsasm_kernels.h"
#include "dsasm_threadpool.h"
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define DSV35_MAGIC "DSVOC35\0"
#define DSV35_VERSION 1u
#define DSV35_TENSOR_WORK 0u
#define DSV35_TENSOR_CONST 1u
#define DSV35_F32 1u
#define DSV36_CONV_FUSED_LEAKY 1u
#define DSV38_CONV_FUSED_RESIDUAL 2u
#define DSV40_CONV_VNNI 4u
#define DSV49_CONV_RESIDUAL_P8 8u

enum {
    DSV35_ADD=1, DSV35_SUB, DSV35_MUL, DSV35_DIV, DSV35_MOD,
    DSV35_LEAKY, DSV35_TANH, DSV35_SIN, DSV35_CUMSUM,
    DSV35_RESHAPE, DSV35_SQUEEZE, DSV35_TRANSPOSE, DSV35_SLICE,
    DSV35_PAD, DSV35_CONV, DSV35_CONVTRANSPOSE, DSV35_UNSQUEEZE
};

#pragma pack(push,1)
typedef struct {
    char magic[8];
    uint32_t version, tensor_count, op_count, input_mel, input_f0, output, reserved;
    uint64_t arena_floats, const_bytes, tensor_table_offset, op_table_offset, const_offset;
    uint64_t frames, mel_bins, samples;
    uint8_t pad[28];
} DSV35Header;

typedef struct {
    uint32_t kind, dtype, rank, reserved;
    uint64_t dims[4];
    uint64_t nelem, alloc_nelem, offset, byte_size;
} DSV35Tensor;

typedef struct {
    uint32_t type, in0, in1, in2, out, rank, flags, reserved;
    int64_t p[16];
    float f[4];
    uint8_t pad[16];
} DSV35Op;
#pragma pack(pop)

_Static_assert(sizeof(DSV35Header)==128, "DSV35Header size");
_Static_assert(sizeof(DSV35Tensor)==80, "DSV35Tensor size");
_Static_assert(sizeof(DSV35Op)==192, "DSV35Op size");

struct DSAsmVocoderGraph {
    int fd;
    void *map;
    size_t map_size;
    const DSV35Header *h;
    const DSV35Tensor *t;
    const DSV35Op *ops;
    const unsigned char *cbase;
    float *arena;
    float *conv_ws;
    size_t conv_ws_floats;
    unsigned char *vnni_qx,*vnni_xpack;
    float *vnni_scales;
    size_t vnni_qx_bytes,vnni_xpack_bytes,vnni_scale_count;
    int vnni_available;
    double vnni_pack_ms,vnni_kernel_ms;
    DSAsmThreadPool *pool;
    int owns_pool;
};

typedef struct {
    uint32_t op_index;
    uint32_t type;
    double ms;
} DSVocoderHotOp;

static int hot_op_cmp_desc(const void *a,const void *b){
    const DSVocoderHotOp *x=(const DSVocoderHotOp*)a;
    const DSVocoderHotOp *y=(const DSVocoderHotOp*)b;
    return (x->ms<y->ms)-(x->ms>y->ms);
}

static int profile_shapes_enabled(void){
    const char *e=getenv("DSASM_PROFILE_SHAPES");
    return e && strcmp(e,"0")!=0;
}

static inline double now_ms(void){
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC_RAW,&ts);
    return 1000.0*(double)ts.tv_sec + 1e-6*(double)ts.tv_nsec;
}
static inline float *wptr(DSAsmVocoderGraph *g,uint32_t id){ return g->arena + g->t[id].offset; }
static inline const unsigned char *cptr(const DSAsmVocoderGraph *g,uint32_t id){ return g->cbase+g->t[id].offset; }
static inline const float *rptr(const DSAsmVocoderGraph *g,uint32_t id){
    const DSV35Tensor *t=&g->t[id];
    return t->kind==DSV35_TENSOR_CONST ? (const float*)(g->cbase+t->offset) : g->arena+t->offset;
}
static int vnni_mode(void){
    const char *e=getenv("DSASM_VNNI");
    /* Promoted E2E default: quality-gated K7/K11 VNNI path. */
    if(!e||!*e)return 2;
    if(!strcmp(e,"0")||!strcmp(e,"off"))return 0;
    if(!strcmp(e,"k11"))return 1;
    if(!strcmp(e,"1")||!strcmp(e,"auto")||!strcmp(e,"k117")||!strcmp(e,"k7k11"))return 2;
    return 0;
}
static unsigned parse_vnni_cin_mask(const char *e,unsigned fallback){
    unsigned mask=0;
    if(!e||!*e)return fallback;
    if(!strcmp(e,"all")){mask=0x1fu;return mask;}
    const char *p=e;
    while(*p){
        char *end=NULL; unsigned long v=strtoul(p,&end,10);
        if(end==p){while(*p&&*p!=',')p++;if(*p==',')p++;continue;}
        if(v==16)mask|=1u<<0; else if(v==32)mask|=1u<<1;
        else if(v==64)mask|=1u<<2; else if(v==128)mask|=1u<<3;
        else if(v==256)mask|=1u<<4;
        p=end;while(*p&&*p!=',')p++;if(*p==',')p++;
    }
    return mask;
}
static unsigned vnni_cin_mask(void){
    /* M44: an all-k711 bundle may carry VNNI weights for every HiFi-GAN
       stage, while runtime can admit only quality-safe channel stages. */
    static int init=0; static unsigned mask=0;
    if(init)return mask;
    init=1;
    /* Cin=128 and Cin=64 passed the persistent real-model E2E quality gate.
       "all" remains an explicit lab opt-in for quantization experiments. */
    mask=parse_vnni_cin_mask(getenv("DSASM_VNNI_CIN"),(1u<<3)|(1u<<2));
    return mask;
}
static int vnni_cin_allowed(size_t cin){
    unsigned bit=cin==16?1u<<0:cin==32?1u<<1:cin==64?1u<<2:cin==128?1u<<3:cin==256?1u<<4:0u;
    return bit && (vnni_cin_mask()&bit)!=0;
}
static int vnni_k7_cin_allowed(size_t cin){
    const char *configured=getenv("DSASM_VNNI_K7_CIN");
    if(!configured||!*configured)return vnni_cin_allowed(cin);
    unsigned bit=cin==16?1u<<0:cin==32?1u<<1:cin==64?1u<<2:cin==128?1u<<3:cin==256?1u<<4:0u;
    return bit&&(parse_vnni_cin_mask(configured,0)&bit)!=0;
}
static int vnni_k7_op_allowed(const DSAsmVocoderGraph *g,const DSV35Op *op,size_t cin){
    const char *configured=getenv("DSASM_VNNI_K7_OPS");
    if(cin!=128u||!configured||!*configured)return 1;
    unsigned long wanted=(unsigned long)(op-g->ops);
    const char *p=configured;
    while(*p){
        char *end=NULL;unsigned long value=strtoul(p,&end,10);
        if(end!=p&&value==wanted)return 1;
        p=end==p?p+1:end;while(*p&&*p!=',')p++;if(*p==',')p++;
    }
    return 0;
}
static int vnni_extra_op_allowed(const DSAsmVocoderGraph *g,const DSV35Op *op){
    const char *configured=getenv("DSASM_VNNI_EXTRA_OPS");
    unsigned long wanted=(unsigned long)(op-g->ops);
    /* The 32-frame stream gate promoted these six measured hot operators.
       Other fixed shapes remain separate quality/performance strata. */
    if(!configured){
        if(g->h->frames!=32u)return 0;
        return wanted==79u||wanted==77u||wanted==80u||
               wanted==76u||wanted==74u||wanted==73u;
    }
    if(!*configured||!strcmp(configured,"off"))return 0;
    const char *p=configured;
    while(*p){
        char *end=NULL;unsigned long value=strtoul(p,&end,10);
        if(end!=p&&value==wanted)return 1;
        p=end==p?p+1:end;while(*p&&*p!=',')p++;if(*p==',')p++;
    }
    return 0;
}
static int parallel_add_enabled(void){
    static int v=-1;
    if(v<0){const char *e=getenv("DSASM_PARALLEL_ADD");v=!e||!*e||(strcmp(e,"0")!=0&&strcmp(e,"off")!=0);}
    return v;
}

static int op_uses_vnni(const DSAsmVocoderGraph*g,const DSV35Op*op){
    if(!g->vnni_available || !(op->flags&DSV40_CONV_VNNI))return 0;
    const int extra=vnni_extra_op_allowed(g,op);
    int m=vnni_mode();size_t K=(size_t)op->p[3],Cin=(size_t)op->p[0];
    const char *configured=getenv("DSASM_VNNI");
    const char *configured_cin=getenv("DSASM_VNNI_CIN");
    /* Real 32/64-frame parity rejects the default K7 quantized stages. Keep
       explicit lab overrides, but make the product default quality-safe. */
    if((!configured||!*configured)&&g->h->frames<=64u&&m>=2)m=1;
    /* Exact-request golden tests admit Cin64/K11 for the 32/64-frame stream
       buckets, but reject it for the 384-frame batch bucket. Explicit masks
       remain available for quality experiments on other model/workload strata. */
    if(!extra){
        if((!configured_cin||!*configured_cin)&&g->h->frames>64u&&Cin==64u)return 0;
        if(K==7u){if(!vnni_k7_cin_allowed(Cin)||!vnni_k7_op_allowed(g,op,Cin))return 0;}
        else if(!vnni_cin_allowed(Cin))return 0;
    }
    return extra || (m==1 && K==11u) || (m>=2 && (K==7u||K==11u));
}
static inline size_t dim4(const DSV35Tensor *t,int axis){
    int shift=4-(int)t->rank; int q=axis-shift; return q<0?1u:(size_t)t->dims[q];
}
static inline size_t out_count(const DSV35Tensor *t){ return (size_t)t->nelem; }

static void elementwise2(DSAsmVocoderGraph *g,const DSV35Op *op){
    const DSV35Tensor *ta=&g->t[op->in0], *tb=&g->t[op->in1], *to=&g->t[op->out];
    const float *a=rptr(g,op->in0), *b=rptr(g,op->in1); float *o=wptr(g,op->out);
    size_t n=out_count(to);
    if(ta->nelem==to->nelem && tb->nelem==to->nelem && op->type==DSV35_ADD){
        /* M43: residual merges are large independent elementwise work.  The
           old single-thread Add showed up as ~20-30 ms in whole vocoder
           profiles while seven other P-core logical CPUs were idle. */
        /* M55 re-tests this opt-in path on multi-second tensors, where
           dispatch overhead is amortized. */
        if(parallel_add_enabled() && g->pool && n>=16384u) ds_threadpool_add_f32(g->pool,a,b,o,n);
        else ds_add_f32_avx2(a,b,o,n);
        return;
    }
    if(ta->nelem==to->nelem && tb->nelem==1){
        float s=b[0];
        switch(op->type){
        case DSV35_ADD: for(size_t i=0;i<n;i++)o[i]=a[i]+s;break;
        case DSV35_SUB: for(size_t i=0;i<n;i++)o[i]=a[i]-s;break;
        case DSV35_MUL: for(size_t i=0;i<n;i++)o[i]=a[i]*s;break;
        case DSV35_DIV: for(size_t i=0;i<n;i++)o[i]=a[i]/s;break;
        case DSV35_MOD: for(size_t i=0;i<n;i++)o[i]=fmodf(a[i],s);break;
        } return;
    }
    if(tb->nelem==to->nelem && ta->nelem==1){
        float s=a[0];
        switch(op->type){
        case DSV35_ADD: for(size_t i=0;i<n;i++)o[i]=s+b[i];break;
        case DSV35_SUB: for(size_t i=0;i<n;i++)o[i]=s-b[i];break;
        case DSV35_MUL: for(size_t i=0;i<n;i++)o[i]=s*b[i];break;
        case DSV35_DIV: for(size_t i=0;i<n;i++)o[i]=s/b[i];break;
        case DSV35_MOD: for(size_t i=0;i<n;i++)o[i]=fmodf(s,b[i]);break;
        } return;
    }
    /* Generic NumPy-style broadcasting for rank <= 4. This is intentionally
       a thin correctness path; the large same-shape residual Add goes to ASM. */
    size_t od[4],ad[4],bd[4]; for(int d=0;d<4;d++){od[d]=dim4(to,d);ad[d]=dim4(ta,d);bd[d]=dim4(tb,d);}
    size_t as[4],bs[4]; as[3]=bs[3]=1;
    for(int d=2;d>=0;d--){as[d]=as[d+1]*ad[d+1];bs[d]=bs[d+1]*bd[d+1];}
    for(size_t idx=0;idx<n;idx++){
        size_t q=idx, ia=0,ib=0;
        for(int d=3;d>=0;d--){size_t c=q%od[d];q/=od[d];if(ad[d]!=1)ia+=c*as[d];if(bd[d]!=1)ib+=c*bs[d];}
        float x=a[ia],y=b[ib];
        switch(op->type){case DSV35_ADD:o[idx]=x+y;break;case DSV35_SUB:o[idx]=x-y;break;case DSV35_MUL:o[idx]=x*y;break;case DSV35_DIV:o[idx]=x/y;break;default:o[idx]=fmodf(x,y);break;}
    }
}

static void op_transpose(DSAsmVocoderGraph *g,const DSV35Op *op){
    const DSV35Tensor *ti=&g->t[op->in0],*to=&g->t[op->out]; const float *x=rptr(g,op->in0);float*y=wptr(g,op->out);
    size_t rank=ti->rank; size_t id[4]={1,1,1,1},od[4]={1,1,1,1};
    for(size_t i=0;i<rank;i++){id[i]=ti->dims[i];od[i]=to->dims[i];}
    size_t istr[4]={0},ostr[4]={0};istr[rank-1]=ostr[rank-1]=1;
    for(int i=(int)rank-2;i>=0;i--){istr[i]=istr[i+1]*id[i+1];ostr[i]=ostr[i+1]*od[i+1];}
    for(size_t oi=0;oi<to->nelem;oi++){
        size_t q=oi, ii=0, coord[4]={0};
        for(int d=(int)rank-1;d>=0;d--){coord[d]=q%od[d];q/=od[d];}
        for(size_t d=0;d<rank;d++) ii += coord[d]*istr[(size_t)op->p[d]];
        y[oi]=x[ii];
    }
}

static void op_slice(DSAsmVocoderGraph *g,const DSV35Op *op){
    const DSV35Tensor *ti=&g->t[op->in0],*to=&g->t[op->out];const float*x=rptr(g,op->in0);float*y=wptr(g,op->out);
    size_t rank=ti->rank; size_t istr[4]={0},od[4]={1,1,1,1}; istr[rank-1]=1;
    for(int i=(int)rank-2;i>=0;i--)istr[i]=istr[i+1]*ti->dims[i+1];
    for(size_t i=0;i<rank;i++)od[i]=to->dims[i];
    for(size_t oi=0;oi<to->nelem;oi++){
        size_t q=oi,ii=0;
        for(int d=(int)rank-1;d>=0;d--){size_t c=q%od[d];q/=od[d];int64_t s=op->p[d],step=op->p[4+d];ii+=(size_t)(s+(int64_t)c*step)*istr[d];}
        y[oi]=x[ii];
    }
}
static void op_pad(DSAsmVocoderGraph *g,const DSV35Op *op){
    const DSV35Tensor *ti=&g->t[op->in0],*to=&g->t[op->out];const float*x=rptr(g,op->in0);float*y=wptr(g,op->out);
    size_t rank=ti->rank; float v=op->f[0]; for(size_t i=0;i<to->nelem;i++)y[i]=v;
    size_t istr[4]={0},ostr[4]={0};istr[rank-1]=ostr[rank-1]=1;
    for(int i=(int)rank-2;i>=0;i--){istr[i]=istr[i+1]*ti->dims[i+1];ostr[i]=ostr[i+1]*to->dims[i+1];}
    for(size_t ii=0;ii<ti->nelem;ii++){
        size_t q=ii,oi=0;for(int d=(int)rank-1;d>=0;d--){size_t c=q%ti->dims[d];q/=ti->dims[d];oi+=(c+(size_t)op->p[d])*ostr[d];}y[oi]=x[ii];
    }
}
static void op_cumsum(DSAsmVocoderGraph *g,const DSV35Op *op){
    const DSV35Tensor *t=&g->t[op->out];const float*x=rptr(g,op->in0);float*y=wptr(g,op->out);int axis=(int)op->p[0];if(axis<0)axis+=(int)t->rank;
    size_t inner=1,alen=t->dims[axis],outer=1;for(size_t d=(size_t)axis+1;d<t->rank;d++)inner*=t->dims[d];for(int d=0;d<axis;d++)outer*=t->dims[d];
    for(size_t o=0;o<outer;o++)for(size_t i=0;i<inner;i++){float s=0;for(size_t a=0;a<alen;a++){size_t z=(o*alen+a)*inner+i;s+=x[z];y[z]=s;}}
}

static int run_op(DSAsmVocoderGraph *g,const DSV35Op *op){
    switch(op->type){
    case DSV35_ADD:case DSV35_SUB:case DSV35_MUL:case DSV35_DIV:case DSV35_MOD: elementwise2(g,op);return 0;
    case DSV35_LEAKY: ds_leaky_relu_f32_avx2(rptr(g,op->in0),wptr(g,op->out),g->t[op->out].nelem,op->f[0]);return 0;
    case DSV35_TANH:{const float*x=rptr(g,op->in0);float*y=wptr(g,op->out);for(size_t i=0;i<g->t[op->out].nelem;i++)y[i]=tanhf(x[i]);return 0;}
    case DSV35_SIN:{const float*x=rptr(g,op->in0);float*y=wptr(g,op->out);for(size_t i=0;i<g->t[op->out].nelem;i++)y[i]=sinf(x[i]);return 0;}
    case DSV35_CUMSUM: op_cumsum(g,op);return 0;
    case DSV35_RESHAPE:case DSV35_SQUEEZE:case DSV35_UNSQUEEZE:
        memcpy(wptr(g,op->out),rptr(g,op->in0),g->t[op->out].nelem*sizeof(float));return 0;
    case DSV35_TRANSPOSE: op_transpose(g,op);return 0;
    case DSV35_SLICE: op_slice(g,op);return 0;
    case DSV35_PAD: op_pad(g,op);return 0;
    case DSV35_CONV:{
        size_t Cin=(size_t)op->p[0],Cout=(size_t)op->p[1],Cpad=(size_t)op->p[2],K=(size_t)op->p[3],Tin=(size_t)op->p[4],pad=(size_t)op->p[5],dil=(size_t)op->p[6];
        size_t pack=((size_t)op->p[7]==8u)?8u:4u;
        const uint32_t residual_id=(op->flags&DSV38_CONV_FUSED_RESIDUAL)
            ? ((op->flags&DSV49_CONV_RESIDUAL_P8)?(uint32_t)op->p[8]:op->reserved) : 0xffffffffu;
        const float *residual=(residual_id!=0xffffffffu)?rptr(g,residual_id):NULL;
        if(op_uses_vnni(g,op)){
            int rc=ds_vocoder_conv1d_vnni_u8s8(
                rptr(g,op->in0),cptr(g,op->reserved),rptr(g,op->in2),wptr(g,op->out),
                Cin,Cout,K,Tin,pad,dil,(op->flags&DSV36_CONV_FUSED_LEAKY)!=0,op->f[0],
                g->vnni_qx,g->vnni_xpack,g->vnni_scales,g->pool,&g->vnni_pack_ms,&g->vnni_kernel_ms);
            if(rc)return rc;
            if(residual){
                const size_t Tout=Tin+2u*pad-dil*(K-1u), n=Cout*Tout;
                if(parallel_add_enabled() && g->pool && n>=16384u)
                    ds_threadpool_add_f32(g->pool,wptr(g,op->out),residual,wptr(g,op->out),n);
                else
                    ds_add_f32_avx2(wptr(g,op->out),residual,wptr(g,op->out),n);
            }
            return 0;
        }
        (void)Cout; return ds_vocoder_conv1d_ex_residual_f32_avx2(
            rptr(g,op->in0),rptr(g,op->in1),rptr(g,op->in2),residual,wptr(g,op->out),
            Cin,Cpad,K,Tin,pad,dil,pack,(op->flags&DSV36_CONV_FUSED_LEAKY)!=0,op->f[0],g->conv_ws,g->pool);
    }
    case DSV35_CONVTRANSPOSE:{
        return ds_vocoder_convtranspose1d_f32_avx2(rptr(g,op->in0),rptr(g,op->in1),rptr(g,op->in2),wptr(g,op->out),(size_t)op->p[0],(size_t)op->p[1],(size_t)op->p[2],(size_t)op->p[3],(size_t)op->p[4],(size_t)op->p[5],g->pool);
    }
    default:return -99;
    }
}

static DSAsmVocoderGraph *load_graph(const char *path,DSAsmThreadPool *pool,size_t workers){
    int fd=open(path,O_RDONLY);if(fd<0)return NULL;struct stat st;if(fstat(fd,&st)){close(fd);return NULL;}
    void *map=mmap(NULL,(size_t)st.st_size,PROT_READ,MAP_PRIVATE,fd,0);if(map==MAP_FAILED){close(fd);return NULL;}
    const DSV35Header*h=(const DSV35Header*)map;
    if((size_t)st.st_size<sizeof(*h)||memcmp(h->magic,DSV35_MAGIC,8)||h->version!=DSV35_VERSION){munmap(map,(size_t)st.st_size);close(fd);return NULL;}
    if(h->tensor_table_offset+h->tensor_count*sizeof(DSV35Tensor)>(uint64_t)st.st_size||h->op_table_offset+h->op_count*sizeof(DSV35Op)>(uint64_t)st.st_size||h->const_offset+h->const_bytes>(uint64_t)st.st_size){munmap(map,(size_t)st.st_size);close(fd);return NULL;}
    DSAsmVocoderGraph*g=calloc(1,sizeof(*g));if(!g){munmap(map,(size_t)st.st_size);close(fd);return NULL;}
    g->fd=fd;g->map=map;g->map_size=(size_t)st.st_size;g->h=h;g->t=(const DSV35Tensor*)((const unsigned char*)map+h->tensor_table_offset);g->ops=(const DSV35Op*)((const unsigned char*)map+h->op_table_offset);g->cbase=(const unsigned char*)map+h->const_offset;
    if(posix_memalign((void**)&g->arena,64,(size_t)h->arena_floats*sizeof(float))){ds_vocoder_graph_free(g);return NULL;}memset(g->arena,0,(size_t)h->arena_floats*sizeof(float));
    g->vnni_available=ds_vocoder_vnni_available();
    for(uint32_t i=0;i<h->op_count;i++)if(g->ops[i].type==DSV35_CONV){
        size_t z=(size_t)g->ops[i].p[0]*((size_t)g->ops[i].p[4]+2u*(size_t)g->ops[i].p[5]);if(z>g->conv_ws_floats)g->conv_ws_floats=z;
        if(g->ops[i].flags&DSV40_CONV_VNNI){
            size_t Cin=(size_t)g->ops[i].p[0],Cout=(size_t)g->ops[i].p[1],K=(size_t)g->ops[i].p[3],Tin=(size_t)g->ops[i].p[4],pad=(size_t)g->ops[i].p[5],dil=(size_t)g->ops[i].p[6];
            size_t Tout=Tin+2u*pad-dil*(K-1u),K4=(Cin*K+3u)/4u,Tb=(Tout+7u)/8u;
            size_t q=Cin*(Tin+2u*pad),xp=Tb*K4*32u;if(q>g->vnni_qx_bytes)g->vnni_qx_bytes=q;if(xp>g->vnni_xpack_bytes)g->vnni_xpack_bytes=xp;if(Cout>g->vnni_scale_count)g->vnni_scale_count=Cout;
        }
    }
    if(g->conv_ws_floats&&posix_memalign((void**)&g->conv_ws,64,g->conv_ws_floats*sizeof(float))){ds_vocoder_graph_free(g);return NULL;}
    if(g->vnni_qx_bytes&&posix_memalign((void**)&g->vnni_qx,64,g->vnni_qx_bytes)){ds_vocoder_graph_free(g);return NULL;}
    if(g->vnni_xpack_bytes&&posix_memalign((void**)&g->vnni_xpack,64,g->vnni_xpack_bytes)){ds_vocoder_graph_free(g);return NULL;}
    if(g->vnni_scale_count&&posix_memalign((void**)&g->vnni_scales,64,g->vnni_scale_count*sizeof(float))){ds_vocoder_graph_free(g);return NULL;}
    if(pool){g->pool=pool;g->owns_pool=0;}
    else {g->pool=ds_threadpool_create(workers?workers:8);g->owns_pool=1;}
    if(!g->pool){ds_vocoder_graph_free(g);return NULL;}return g;
}
DSAsmVocoderGraph *ds_vocoder_graph_load(const char *path,size_t workers){return load_graph(path,NULL,workers);}
DSAsmVocoderGraph *ds_vocoder_graph_load_with_pool(const char *path,DSAsmThreadPool *pool){if(!pool){errno=EINVAL;return NULL;}return load_graph(path,pool,0);}
void ds_vocoder_graph_free(DSAsmVocoderGraph*g){if(!g)return;if(g->pool&&g->owns_pool)ds_threadpool_destroy(g->pool);free(g->vnni_scales);free(g->vnni_xpack);free(g->vnni_qx);free(g->conv_ws);free(g->arena);if(g->map&&g->map!=MAP_FAILED)munmap(g->map,g->map_size);if(g->fd>=0)close(g->fd);free(g);}
size_t ds_vocoder_graph_frames(const DSAsmVocoderGraph*g){return g?(size_t)g->h->frames:0;}size_t ds_vocoder_graph_mel_bins(const DSAsmVocoderGraph*g){return g?(size_t)g->h->mel_bins:0;}size_t ds_vocoder_graph_samples(const DSAsmVocoderGraph*g){return g?(size_t)g->h->samples:0;}size_t ds_vocoder_graph_workers(const DSAsmVocoderGraph*g){return g&&g->pool?ds_threadpool_threads(g->pool):0;}

int ds_vocoder_graph_infer(DSAsmVocoderGraph*g,const float*mel,const float*f0,float*wave,int profile){
    if(!g||!mel||!f0||!wave)return -1;
    const DSV35Tensor*tm=&g->t[g->h->input_mel],*tf=&g->t[g->h->input_f0];
    memcpy(wptr(g,g->h->input_mel),mel,tm->nelem*sizeof(float));
    memcpy(wptr(g,g->h->input_f0),f0,tf->nelem*sizeof(float));
    double totals[18]={0};g->vnni_pack_ms=0.0;g->vnni_kernel_ms=0.0;
    const int shape_profile=profile && profile_shapes_enabled();
    DSVocoderHotOp *hot=shape_profile?calloc(g->h->op_count,sizeof(*hot)):NULL;
    size_t hot_n=0;
    for(uint32_t i=0;i<g->h->op_count;i++){
        double a=profile?now_ms():0;
        int rc=run_op(g,&g->ops[i]);
        if(rc){free(hot);return -(1000+(int)i);}
        if(profile){
            double dt=now_ms()-a;
            totals[g->ops[i].type]+=dt;
            if(hot && (g->ops[i].type==DSV35_CONV || g->ops[i].type==DSV35_CONVTRANSPOSE))
                hot[hot_n++]=(DSVocoderHotOp){i,g->ops[i].type,dt};
        }
    }
    memcpy(wave,rptr(g,g->h->output),g->h->samples*sizeof(float));
    if(profile){
        static const char*n[]={"","Add","Sub","Mul","Div","Mod","LeakyRelu","Tanh","Sin","CumSum","Reshape","Squeeze","Transpose","Slice","Pad","Conv","ConvTranspose","Unsqueeze"};
        fprintf(stderr,"DSASM native vocoder op profile:\n");
        for(int i=1;i<=17;i++)if(totals[i]>0.001)fprintf(stderr,"  %-14s %9.3f ms\n",n[i],totals[i]);
        if(g->vnni_pack_ms>0.001||g->vnni_kernel_ms>0.001)
            fprintf(stderr,"  VNNI-pack       %9.3f ms\n  VNNI-kernel     %9.3f ms\n",g->vnni_pack_ms,g->vnni_kernel_ms);
        if(hot){
            qsort(hot,hot_n,sizeof(*hot),hot_op_cmp_desc);
            fprintf(stderr,"DSASM M41 hot Conv/ConvTranspose profile (descending):\n");
            for(size_t h=0;h<hot_n;h++){
                const DSV35Op *op=&g->ops[hot[h].op_index];
                if(op->type==DSV35_CONV){
                    const size_t Cin=(size_t)op->p[0],Cout=(size_t)op->p[1],K=(size_t)op->p[3];
                    const size_t Tin=(size_t)op->p[4],pad=(size_t)op->p[5],dil=(size_t)op->p[6],pack=(size_t)op->p[7];
                    const size_t receptive=dil*(K-1u)+1u;
                    const size_t Tout=Tin+2u*pad-receptive+1u;
                    fprintf(stderr,
                        "  #%03u Conv  %9.3f ms  C=%zux%zu K=%zu T=%zu->%zu d=%zu p=%zu pack=%zu path=%s%s%s\n",
                        hot[h].op_index,hot[h].ms,Cin,Cout,K,Tin,Tout,dil,pad,pack,
                        op_uses_vnni(g,op)?"VNNI":"FP32",
                        (op->flags&DSV36_CONV_FUSED_LEAKY)?"+leaky":"",
                        (op->flags&DSV38_CONV_FUSED_RESIDUAL)?"+res":"");
                }else{
                    const size_t Cin=(size_t)op->p[0],Cout=(size_t)op->p[1],K=(size_t)op->p[2];
                    const size_t Tin=(size_t)op->p[3],pad=(size_t)op->p[4],stride=(size_t)op->p[5];
                    const size_t Tout=(Tin-1u)*stride-2u*pad+K;
                    fprintf(stderr,
                        "  #%03u ConvT %9.3f ms  C=%zux%zu K=%zu T=%zu->%zu stride=%zu p=%zu\n",
                        hot[h].op_index,hot[h].ms,Cin,Cout,K,Tin,Tout,stride,pad);
                }
            }
        }
    }
    free(hot);
    return 0;
}
