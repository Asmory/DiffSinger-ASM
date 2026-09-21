#define main dsasm_m5_embedded_main
#include "test_m5_lynxnet2_block.c"
#undef main

void ds_fused_linear_softsign_glu_f32_avx2_m4n8(const float*,const float*,const float*,const float*,float*,size_t,size_t,size_t);
void ds_linear_residual_f32_avx2_m4n8(const float*,const float*,const float*,const float*,float*,size_t,size_t,size_t);
void ds_linear_residual_f32_avx2_m4n16(const float*,const float*,const float*,const float*,float*,size_t,size_t,size_t);

static void pack_glu8_m6(const float *wl,const float *wg,float *p,size_t N,size_t K){
    size_t z=0;
    for(size_t n0=0;n0<N;n0+=8)
        for(size_t k=0;k<K;k++){
            for(size_t q=0;q<8;q++) p[z++]=wl[(n0+q)*K+k];
            for(size_t q=0;q<8;q++) p[z++]=wg[(n0+q)*K+k];
        }
}
static void pack_linear16_m6(const float *w,float *p,size_t N,size_t K){
    size_t z=0;
    for(size_t n0=0;n0<N;n0+=16)
        for(size_t k=0;k<K;k++){
            for(size_t q=0;q<8;q++) p[z++]=w[(n0+q)*K+k];
            for(size_t q=8;q<16;q++) p[z++]=w[(n0+q)*K+k];
        }
}

static void block_asm_m6(const float *x,float *y,const BlockWeights *q,
                         const float *p1,const float *p2,const float *p3,
                         float *ln,float *dw,float *h1,float *h2){
    const size_t T=q->T,D=q->D,H=q->H;
    ds_layernorm_f32_avx2(x,q->gamma,q->beta,ln,T,D,1e-5f);
    ds_depthwise_conv1d_k31_tc_f32_avx2(ln,q->dw_wp,q->dw_b,dw,T,D);
    ds_fused_linear_softsign_glu_f32_avx2_m4n8(dw,p1,q->b1l,q->b1g,h1,T,H,D);
    ds_fused_linear_softsign_glu_f32_avx2_m4n8(h1,p2,q->b2l,q->b2g,h2,T,H,H);
    ds_linear_residual_f32_avx2_m4n16(h2,p3,q->b3,x,y,T,D,H);
}

static int run_m6_block(size_t T,size_t D,size_t H,int bench){
    BlockWeights q; init_block(&q,T,D,H);
    float *p1=malloc(2*H*D*sizeof(float));
    float *p2=malloc(2*H*H*sizeof(float));
    float *p3=malloc(D*H*sizeof(float));
    pack_glu8_m6(q.w1l,q.w1g,p1,H,D);
    pack_glu8_m6(q.w2l,q.w2g,p2,H,H);
    pack_linear16_m6(q.w3,p3,D,H);

    float *x=malloc(T*D*4),*ref=malloc(T*D*4),*m5=malloc(T*D*4),*m6=malloc(T*D*4);
    float *ln=malloc(T*D*4),*dw=malloc(T*D*4),*h1=malloc(T*H*4),*h2=malloc(T*H*4);
    for(size_t i=0;i<T*D;i++)x[i]=frand_sym();
    block_ref(x,ref,&q);
    block_asm_m5(x,m5,&q,ln,dw,h1,h2);
    block_asm_m6(x,m6,&q,p1,p2,p3,ln,dw,h1,h2);
    float d5=max_abs_diff(ref,m5,T*D),d6=max_abs_diff(ref,m6,T*D),d56=max_abs_diff(m5,m6,T*D);
    printf("M6 block T=%zu D=%zu H=%zu: M5=%g M6=%g M5vsM6=%g %s\n",T,D,H,d5,d6,d56,d6<8e-4f?"OK":"FAIL");

    if(bench){
        const int it=30;
        double a=now_sec();
        for(int i=0;i<it;i++)block_asm_m5(x,m5,&q,ln,dw,h1,h2);
        double b=now_sec();
        for(int i=0;i<it;i++)block_asm_m6(x,m6,&q,p1,p2,p3,ln,dw,h1,h2);
        double c=now_sec();
        double t5=(b-a)/it,t6=(c-b)/it;
        double flops=2.0*T*((double)H*D*2.0+(double)H*H*2.0+(double)D*H)+2.0*T*D*31.0;
        printf("  M5 packed4: %.3f ms/block %.2f GF/s\n",t5*1e3,flops/t5/1e9);
        printf("  M6 mixed:   %.3f ms/block %.2f GF/s speedup %.2fx\n",t6*1e3,flops/t6/1e9,t5/t6);

        // Stage profile in the same process/data to show where time remains.
#define STAGE(lbl, stmt) do { double s0=now_sec(); for(int ii=0;ii<100;ii++){stmt;} double s1=now_sec(); printf("    %-16s %.4f ms\n",lbl,(s1-s0)*10.0); } while(0)
        puts("  M6 stage profile:");
        STAGE("layernorm", ds_layernorm_f32_avx2(x,q.gamma,q.beta,ln,T,D,1e-5f));
        STAGE("depthwise", ds_depthwise_conv1d_k31_tc_f32_avx2(ln,q.dw_wp,q.dw_b,dw,T,D));
        STAGE("glu1 4x8", ds_fused_linear_softsign_glu_f32_avx2_m4n8(dw,p1,q.b1l,q.b1g,h1,T,H,D));
        STAGE("glu2 4x8", ds_fused_linear_softsign_glu_f32_avx2_m4n8(h1,p2,q.b2l,q.b2g,h2,T,H,H));
        STAGE("linear+res 4x16", ds_linear_residual_f32_avx2_m4n16(h2,p3,q.b3,x,m6,T,D,H));
#undef STAGE
    }
    int fail=d6>=8e-4f;
    free(p1);free(p2);free(p3);free(x);free(ref);free(m5);free(m6);free(ln);free(dw);free(h1);free(h2);free_block(&q);
    return fail;
}

int main(void){
#if defined(__x86_64__)
    if(!__builtin_cpu_supports("avx2")||!__builtin_cpu_supports("fma")){fprintf(stderr,"AVX2+FMA required\n");return 2;}
#endif
    int fail=0;
    fail|=run_m6_block(37,64,64,0);
    fail|=run_m6_block(128,512,512,1);
    if(fail){fprintf(stderr,"M6 regression FAILED\n");return 1;}
    puts("M6 regression PASSED");return 0;
}
