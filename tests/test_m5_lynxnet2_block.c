#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void ds_linear_f32_avx2_packed4(const float*, const float*, const float*, float*, size_t, size_t, size_t);
void ds_layernorm_f32_avx2(const float*, const float*, const float*, float*, size_t, size_t, float);
void ds_depthwise_conv1d_k31_f32_avx2(const float*, const float*, const float*, float*, size_t, size_t);
void ds_depthwise_conv1d_k31_tc_f32_avx2(const float*, const float*, const float*, float*, size_t, size_t);
void ds_linear_residual_f32_avx2_packed4(const float*, const float*, const float*, const float*, float*, size_t, size_t, size_t);
void ds_fused_linear_softsign_glu_f32_avx2_packed4(const float*, const float*, const float*, const float*, float*, size_t, size_t, size_t);

static uint32_t rng_state = 0x12345678u;
static float frand_sym(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    uint32_t v = (rng_state >> 8) & 0x00ffffffu;
    return ((float)v / 8388607.5f) - 1.0f;
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static float max_abs_diff(const float *a, const float *b, size_t n) {
    float m = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

static void pack_linear4(const float *w, float *packed, size_t N, size_t K) {
    size_t p = 0;
    for (size_t n0 = 0; n0 < N; n0 += 4) {
        for (size_t k0 = 0; k0 < K; k0 += 8) {
            for (size_t lane = 0; lane < 4; ++lane) {
                size_t n = n0 + lane;
                for (size_t kk = 0; kk < 8; ++kk) {
                    packed[p++] = (n < N) ? w[n*K + k0 + kk] : 0.0f;
                }
            }
        }
    }
}

static void pack_depthwise_tc(const float *w_ct, float *w_tc, size_t C) {
    for (size_t k = 0; k < 31; ++k)
        for (size_t c = 0; c < C; ++c)
            w_tc[k*C + c] = w_ct[c*31 + k];
}

static void pack_glu4(const float *wl, const float *wg, float *packed, size_t N, size_t K) {
    size_t p = 0;
    for (size_t n0 = 0; n0 < N; n0 += 4) {
        for (size_t k0 = 0; k0 < K; k0 += 8) {
            for (size_t lane = 0; lane < 4; ++lane) {
                size_t n = n0 + lane;
                for (size_t kk = 0; kk < 8; ++kk)
                    packed[p++] = (n < N) ? wl[n*K + k0 + kk] : 0.0f;
            }
            for (size_t lane = 0; lane < 4; ++lane) {
                size_t n = n0 + lane;
                for (size_t kk = 0; kk < 8; ++kk)
                    packed[p++] = (n < N) ? wg[n*K + k0 + kk] : 0.0f;
            }
        }
    }
}

static void ref_linear(const float *x, const float *w, const float *b, float *y,
                       size_t M, size_t N, size_t K) {
    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            float acc = b[n];
            for (size_t k = 0; k < K; ++k) acc += x[m*K+k] * w[n*K+k];
            y[m*N+n] = acc;
        }
    }
}

static void ref_glu(const float *x, const float *wl, const float *wg,
                    const float *bl, const float *bg, float *y,
                    size_t M, size_t N, size_t K) {
    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            float l = bl[n], g = bg[n];
            for (size_t k = 0; k < K; ++k) {
                float xv = x[m*K+k];
                l += xv * wl[n*K+k];
                g += xv * wg[n*K+k];
            }
            y[m*N+n] = l * (g / (1.0f + fabsf(g)));
        }
    }
}

static void ref_layernorm(const float *x, const float *g, const float *b, float *y,
                          size_t M, size_t K, float eps) {
    for (size_t m = 0; m < M; ++m) {
        float sum = 0.0f, ss = 0.0f;
        for (size_t k = 0; k < K; ++k) {
            float v = x[m*K+k];
            sum += v;
            ss += v*v;
        }
        float mean = sum / (float)K;
        float var = ss / (float)K - mean*mean;
        if (var < 0.0f) var = 0.0f;
        float inv = 1.0f / sqrtf(var + eps);
        for (size_t k = 0; k < K; ++k)
            y[m*K+k] = (x[m*K+k] - mean) * inv * g[k] + b[k];
    }
}

static void tc_to_padded_ct(const float *x, float *xp, size_t T, size_t C) {
    const size_t S = T + 30;
    memset(xp, 0, C*S*sizeof(float));
    for (size_t c = 0; c < C; ++c)
        for (size_t t = 0; t < T; ++t)
            xp[c*S + 15 + t] = x[t*C + c];
}

static void ct_to_tc(const float *x, float *y, size_t C, size_t T) {
    for (size_t t = 0; t < T; ++t)
        for (size_t c = 0; c < C; ++c)
            y[t*C+c] = x[c*T+t];
}

static void ref_depthwise_from_tc(const float *x_tc, const float *w, const float *b,
                                  float *y_tc, size_t T, size_t C) {
    for (size_t t = 0; t < T; ++t) {
        for (size_t c = 0; c < C; ++c) {
            float acc = b[c];
            for (size_t k = 0; k < 31; ++k) {
                ptrdiff_t it = (ptrdiff_t)t + (ptrdiff_t)k - 15;
                if (it >= 0 && (size_t)it < T) acc += x_tc[(size_t)it*C+c] * w[c*31+k];
            }
            y_tc[t*C+c] = acc;
        }
    }
}

static int test_linear(void) {
    const size_t M=5, N=33, K=64;
    float *x=malloc(M*K*4), *w=malloc(N*K*4), *b=malloc(N*4), *ref=malloc(M*N*4), *got=malloc(M*N*4);
    size_t blocks=(N+3)/4;
    float *wp=malloc(blocks*K*4*4);
    for(size_t i=0;i<M*K;i++)x[i]=frand_sym();
    for(size_t i=0;i<N*K;i++)w[i]=frand_sym()*0.2f;
    for(size_t i=0;i<N;i++)b[i]=frand_sym()*0.1f;
    pack_linear4(w,wp,N,K); ref_linear(x,w,b,ref,M,N,K);
    ds_linear_f32_avx2_packed4(x,wp,b,got,M,N,K);
    float d=max_abs_diff(ref,got,M*N);
    printf("linear packed4: max_abs=%g %s\n", d, d < 2e-4f ? "OK":"FAIL");
    free(x);free(w);free(b);free(ref);free(got);free(wp); return d<2e-4f?0:1;
}

static int test_layernorm(void) {
    const size_t M=7,K=67; const float eps=1e-5f;
    float *x=malloc(M*K*4), *g=malloc(K*4), *b=malloc(K*4), *ref=malloc(M*K*4), *got=malloc(M*K*4);
    for(size_t i=0;i<M*K;i++)x[i]=frand_sym()*2.0f;
    for(size_t i=0;i<K;i++){g[i]=0.8f+frand_sym()*0.2f;b[i]=frand_sym()*0.1f;}
    ref_layernorm(x,g,b,ref,M,K,eps); ds_layernorm_f32_avx2(x,g,b,got,M,K,eps);
    float d=max_abs_diff(ref,got,M*K);
    printf("layernorm:      max_abs=%g %s\n", d, d < 3e-4f ? "OK":"FAIL");
    free(x);free(g);free(b);free(ref);free(got); return d<3e-4f?0:1;
}

static int test_depthwise(void) {
    const size_t T=37,C=64,S=T+30;
    float *x=malloc(T*C*4), *xp=malloc(C*S*4), *w=malloc(C*31*4), *b=malloc(C*4), *ref=malloc(T*C*4), *ct=malloc(C*T*4), *got=malloc(T*C*4);
    for(size_t i=0;i<T*C;i++)x[i]=frand_sym();
    for(size_t i=0;i<C*31;i++)w[i]=frand_sym()*0.05f;
    for(size_t i=0;i<C;i++)b[i]=frand_sym()*0.1f;
    ref_depthwise_from_tc(x,w,b,ref,T,C); tc_to_padded_ct(x,xp,T,C);
    ds_depthwise_conv1d_k31_f32_avx2(xp,w,b,ct,C,T); ct_to_tc(ct,got,C,T);
    float d=max_abs_diff(ref,got,T*C);
    printf("depthwise k31:  max_abs=%g %s\n", d, d < 2e-4f ? "OK":"FAIL");
    free(x);free(xp);free(w);free(b);free(ref);free(ct);free(got); return d<2e-4f?0:1;
}

typedef struct {
    size_t T, D, H;
    float *gamma,*beta,*dw_w,*dw_wp,*dw_b;
    float *w1l,*w1g,*b1l,*b1g,*p1;
    float *w2l,*w2g,*b2l,*b2g,*p2;
    float *w3,*b3,*p3;
} BlockWeights;

static void init_block(BlockWeights *q, size_t T, size_t D, size_t H) {
    q->T=T;q->D=D;q->H=H;
    q->gamma=malloc(D*4);q->beta=malloc(D*4);q->dw_w=malloc(D*31*4);q->dw_wp=malloc(D*31*4);q->dw_b=malloc(D*4);
    q->w1l=malloc(H*D*4);q->w1g=malloc(H*D*4);q->b1l=malloc(H*4);q->b1g=malloc(H*4);
    q->w2l=malloc(H*H*4);q->w2g=malloc(H*H*4);q->b2l=malloc(H*4);q->b2g=malloc(H*4);
    q->w3=malloc(D*H*4);q->b3=malloc(D*4);
    q->p1=malloc(((H+3)/4)*D*8*4); q->p2=malloc(((H+3)/4)*H*8*4); q->p3=malloc(((D+3)/4)*H*4*4);
    for(size_t i=0;i<D;i++){q->gamma[i]=0.9f+0.2f*frand_sym();q->beta[i]=0.05f*frand_sym();q->dw_b[i]=0.03f*frand_sym();}
    for(size_t i=0;i<D*31;i++)q->dw_w[i]=0.04f*frand_sym();
    for(size_t i=0;i<H*D;i++){q->w1l[i]=0.04f*frand_sym();q->w1g[i]=0.04f*frand_sym();}
    for(size_t i=0;i<H;i++){q->b1l[i]=0.03f*frand_sym();q->b1g[i]=0.03f*frand_sym();q->b2l[i]=0.03f*frand_sym();q->b2g[i]=0.03f*frand_sym();}
    for(size_t i=0;i<H*H;i++){q->w2l[i]=0.04f*frand_sym();q->w2g[i]=0.04f*frand_sym();}
    for(size_t i=0;i<D*H;i++)q->w3[i]=0.04f*frand_sym();
    for(size_t i=0;i<D;i++)q->b3[i]=0.03f*frand_sym();
    pack_depthwise_tc(q->dw_w,q->dw_wp,D); pack_glu4(q->w1l,q->w1g,q->p1,H,D); pack_glu4(q->w2l,q->w2g,q->p2,H,H); pack_linear4(q->w3,q->p3,D,H);
}
static void free_block(BlockWeights *q){
    free(q->gamma);free(q->beta);free(q->dw_w);free(q->dw_wp);free(q->dw_b);free(q->w1l);free(q->w1g);free(q->b1l);free(q->b1g);
    free(q->w2l);free(q->w2g);free(q->b2l);free(q->b2g);free(q->w3);free(q->b3);free(q->p1);free(q->p2);free(q->p3);
}

static void block_ref(const float *x, float *y, const BlockWeights *q) {
    size_t T=q->T,D=q->D,H=q->H;
    float *a=malloc(T*D*4),*b=malloc(T*D*4),*h1=malloc(T*H*4),*h2=malloc(T*H*4),*p=malloc(T*D*4);
    ref_layernorm(x,q->gamma,q->beta,a,T,D,1e-5f);
    ref_depthwise_from_tc(a,q->dw_w,q->dw_b,b,T,D);
    ref_glu(b,q->w1l,q->w1g,q->b1l,q->b1g,h1,T,H,D);
    ref_glu(h1,q->w2l,q->w2g,q->b2l,q->b2g,h2,T,H,H);
    ref_linear(h2,q->w3,q->b3,p,T,D,H);
    for(size_t i=0;i<T*D;i++)y[i]=x[i]+p[i];
    free(a);free(b);free(h1);free(h2);free(p);
}

static void block_asm_m4(const float *x, float *y, const BlockWeights *q,
                         float *ln, float *xp, float *dwct, float *dwtc, float *h1, float *h2, float *proj) {
    size_t T=q->T,D=q->D,H=q->H;
    ds_layernorm_f32_avx2(x,q->gamma,q->beta,ln,T,D,1e-5f);
    tc_to_padded_ct(ln,xp,T,D);
    ds_depthwise_conv1d_k31_f32_avx2(xp,q->dw_w,q->dw_b,dwct,D,T);
    ct_to_tc(dwct,dwtc,D,T);
    ds_fused_linear_softsign_glu_f32_avx2_packed4(dwtc,q->p1,q->b1l,q->b1g,h1,T,H,D);
    ds_fused_linear_softsign_glu_f32_avx2_packed4(h1,q->p2,q->b2l,q->b2g,h2,T,H,H);
    ds_linear_f32_avx2_packed4(h2,q->p3,q->b3,proj,T,D,H);
    for(size_t i=0;i<T*D;i++)y[i]=x[i]+proj[i];
}

static void block_asm_m5(const float *x, float *y, const BlockWeights *q,
                         float *ln, float *dw, float *h1, float *h2) {
    size_t T=q->T,D=q->D,H=q->H;
    ds_layernorm_f32_avx2(x,q->gamma,q->beta,ln,T,D,1e-5f);
    ds_depthwise_conv1d_k31_tc_f32_avx2(ln,q->dw_wp,q->dw_b,dw,T,D);
    ds_fused_linear_softsign_glu_f32_avx2_packed4(dw,q->p1,q->b1l,q->b1g,h1,T,H,D);
    ds_fused_linear_softsign_glu_f32_avx2_packed4(h1,q->p2,q->b2l,q->b2g,h2,T,H,H);
    ds_linear_residual_f32_avx2_packed4(h2,q->p3,q->b3,x,y,T,D,H);
}

static int test_block(size_t T,size_t D,size_t H,int bench) {
    BlockWeights q; init_block(&q,T,D,H);
    float *x=malloc(T*D*4),*ref=malloc(T*D*4),*got4=malloc(T*D*4),*got5=malloc(T*D*4);
    float *ln=malloc(T*D*4),*xp=malloc(D*(T+30)*4),*dwct=malloc(D*T*4),*dw=malloc(T*D*4),*h1=malloc(T*H*4),*h2=malloc(T*H*4),*proj=malloc(T*D*4);
    for(size_t i=0;i<T*D;i++)x[i]=frand_sym();
    block_ref(x,ref,&q);
    block_asm_m4(x,got4,&q,ln,xp,dwct,dw,h1,h2,proj);
    block_asm_m5(x,got5,&q,ln,dw,h1,h2);
    float d4=max_abs_diff(ref,got4,T*D);
    float d5=max_abs_diff(ref,got5,T*D);
    float d45=max_abs_diff(got4,got5,T*D);
    printf("LYNXNet2 block T=%zu D=%zu H=%zu: M4=%g M5=%g M4vsM5=%g %s\n",
           T,D,H,d4,d5,d45,(d5<7e-4f && d45<7e-4f)?"OK":"FAIL");
    if(bench){
        int iters=20;
        double flops = 2.0*T*( (double)H*D*2.0 + (double)H*H*2.0 + (double)D*H ) + 2.0*T*D*31.0;
        double t0=now_sec();
        for(int i=0;i<iters;i++) block_asm_m4(x,got4,&q,ln,xp,dwct,dw,h1,h2,proj);
        double dt4=(now_sec()-t0)/iters;
        t0=now_sec();
        for(int i=0;i<iters;i++) block_asm_m5(x,got5,&q,ln,dw,h1,h2);
        double dt5=(now_sec()-t0)/iters;
        printf("  M4 layout: %.3f ms/block, arithmetic %.2f GFLOP/s\n",dt4*1e3,flops/dt4/1e9);
        printf("  M5 native: %.3f ms/block, arithmetic %.2f GFLOP/s, speedup %.2fx\n",dt5*1e3,flops/dt5/1e9,dt4/dt5);
    }
    int fail=(d5>=7e-4f || d45>=7e-4f);
    free(x);free(ref);free(got4);free(got5);free(ln);free(xp);free(dwct);free(dw);free(h1);free(h2);free(proj);free_block(&q);
    return fail;
}

int main(void){
    int fail=0;
    fail |= test_linear();
    fail |= test_layernorm();
    fail |= test_depthwise();
    fail |= test_block(37,64,64,0);
    fail |= test_block(128,512,512,1);
    if(fail){fprintf(stderr,"M5 regression FAILED\n");return 1;}
    puts("M5 regression PASSED");
    return 0;
}
