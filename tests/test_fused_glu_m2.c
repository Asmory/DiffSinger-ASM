#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

extern void ds_fused_linear_softsign_glu_f32_avx2(
    const float *x, const float *w_left, const float *w_gate,
    const float *b_left, const float *b_gate, float *y,
    size_t M, size_t N, size_t K
);

extern void ds_fused_linear_softsign_glu_f32_avx2_packed4(
    const float *x, const float *w_packed,
    const float *b_left, const float *b_gate, float *y,
    size_t M, size_t N, size_t K
);

static uint32_t rng_state = 0x12345678u;

static float frand_small(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return ((rng_state & 0xffffu) / 65535.0f - 0.5f) * 0.2f;
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void *xmalloc(size_t bytes) {
    size_t padded = (bytes + 63u) & ~63u;
    if (padded == 0) padded = 64;
    void *p = aligned_alloc(64, padded);
    if (!p) {
        perror("aligned_alloc");
        exit(1);
    }
    return p;
}

static void reference(
    const float *x, const float *wl, const float *wg,
    const float *bl, const float *bg, float *y,
    size_t M, size_t N, size_t K
) {
    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            float l = bl[n];
            float g = bg[n];
            for (size_t k = 0; k < K; ++k) {
                l += x[m*K+k] * wl[n*K+k];
                g += x[m*K+k] * wg[n*K+k];
            }
            y[m*N+n] = l * (g / (1.0f + fabsf(g)));
        }
    }
}

// Offline-style packing used by the converter. Runtime M2 never repacks.
// K must be divisible by 8.
static float *pack4(const float *wl, const float *wg, size_t N, size_t K,
                    size_t *packed_floats) {
    if ((K & 7u) != 0) return NULL;
    const size_t blocks = (N + 3u) / 4u;
    const size_t total = blocks * K * 8u;
    float *p = xmalloc(total * sizeof(float));
    float *dst = p;

    for (size_t nb = 0; nb < blocks; ++nb) {
        const size_t n0 = nb * 4u;
        for (size_t k0 = 0; k0 < K; k0 += 8u) {
            for (size_t lane = 0; lane < 4u; ++lane) {
                const size_t n = n0 + lane;
                for (size_t j = 0; j < 8u; ++j)
                    *dst++ = (n < N) ? wl[n*K + k0+j] : 0.0f;
            }
            for (size_t lane = 0; lane < 4u; ++lane) {
                const size_t n = n0 + lane;
                for (size_t j = 0; j < 8u; ++j)
                    *dst++ = (n < N) ? wg[n*K + k0+j] : 0.0f;
            }
        }
    }
    *packed_floats = total;
    return p;
}

static int compare(const char *name, const float *ref, const float *got, size_t count) {
    float max_abs = 0.0f, max_rel = 0.0f;
    size_t worst = 0;
    for (size_t i = 0; i < count; ++i) {
        const float ae = fabsf(ref[i] - got[i]);
        const float re = ae / fmaxf(fabsf(ref[i]), 1e-6f);
        if (ae > max_abs) { max_abs = ae; worst = i; }
        if (re > max_rel) max_rel = re;
    }
    printf("  %-10s max_abs=%g max_rel=%g", name, max_abs, max_rel);
    if (max_abs > 2e-4f) {
        printf(" FAIL @%zu ref=%g got=%g\n", worst, ref[worst], got[worst]);
        return 1;
    }
    puts(" OK");
    return 0;
}

static int run_case(size_t M, size_t N, size_t K) {
    printf("case M=%zu N=%zu K=%zu\n", M, N, K);
    const size_t xc=M*K, wc=N*K, yc=M*N;
    float *x=xmalloc(xc*4), *wl=xmalloc(wc*4), *wg=xmalloc(wc*4);
    float *bl=xmalloc(N*4), *bg=xmalloc(N*4);
    float *ref=xmalloc(yc*4), *m1=xmalloc(yc*4), *m2=xmalloc(yc*4);
    for(size_t i=0;i<xc;i++) x[i]=frand_small();
    for(size_t i=0;i<wc;i++){wl[i]=frand_small();wg[i]=frand_small();}
    for(size_t i=0;i<N;i++){bl[i]=frand_small();bg[i]=frand_small();}

    size_t pf=0; float *packed=pack4(wl,wg,N,K,&pf);
    if (!packed) { fprintf(stderr,"pack4 requires K%%8==0\n"); exit(2); }
    reference(x,wl,wg,bl,bg,ref,M,N,K);
    ds_fused_linear_softsign_glu_f32_avx2(x,wl,wg,bl,bg,m1,M,N,K);
    ds_fused_linear_softsign_glu_f32_avx2_packed4(x,packed,bl,bg,m2,M,N,K);

    int fail=0;
    fail |= compare("M1",ref,m1,yc);
    fail |= compare("M2-packed4",ref,m2,yc);
    free(x);free(wl);free(wg);free(bl);free(bg);free(ref);free(m1);free(m2);free(packed);
    return fail;
}

static void benchmark(size_t M,size_t N,size_t K,int iters){
    const size_t xc=M*K,wc=N*K,yc=M*N;
    float *x=xmalloc(xc*4),*wl=xmalloc(wc*4),*wg=xmalloc(wc*4);
    float *bl=xmalloc(N*4),*bg=xmalloc(N*4),*y=xmalloc(yc*4);
    for(size_t i=0;i<xc;i++)x[i]=frand_small();
    for(size_t i=0;i<wc;i++){wl[i]=frand_small();wg[i]=frand_small();}
    for(size_t i=0;i<N;i++){bl[i]=frand_small();bg[i]=frand_small();}
    size_t pf=0; float *packed=pack4(wl,wg,N,K,&pf);

    ds_fused_linear_softsign_glu_f32_avx2(x,wl,wg,bl,bg,y,M,N,K);
    double t0=now_sec();
    for(int i=0;i<iters;i++) ds_fused_linear_softsign_glu_f32_avx2(x,wl,wg,bl,bg,y,M,N,K);
    double t1=now_sec();

    ds_fused_linear_softsign_glu_f32_avx2_packed4(x,packed,bl,bg,y,M,N,K);
    double t2=now_sec();
    for(int i=0;i<iters;i++) ds_fused_linear_softsign_glu_f32_avx2_packed4(x,packed,bl,bg,y,M,N,K);
    double t3=now_sec();

    const double flops=4.0*(double)M*(double)N*(double)K*(double)iters;
    const double m1_ms=(t1-t0)*1e3/iters;
    const double m2_ms=(t3-t2)*1e3/iters;
    printf("bench M=%zu N=%zu K=%zu iters=%d\n",M,N,K,iters);
    printf("  M1          %.3f ms/iter  %7.2f GFLOP/s\n",m1_ms,flops/(t1-t0)/1e9);
    printf("  M2-packed4  %.3f ms/iter  %7.2f GFLOP/s  speedup %.2fx\n",m2_ms,flops/(t3-t2)/1e9,m1_ms/m2_ms);
    printf("  packed weights: %.2f MiB (raw %.2f MiB)\n",
           (double)pf*4/(1024.0*1024.0),(double)(2*wc)*4/(1024.0*1024.0));
    volatile float sink=y[yc/2];(void)sink;
    free(x);free(wl);free(wg);free(bl);free(bg);free(y);free(packed);
}

int main(void){
#if defined(__x86_64__)
    if(!__builtin_cpu_supports("avx2")||!__builtin_cpu_supports("fma")){
        fprintf(stderr,"AVX2 + FMA3 required\n");return 2;
    }
#endif
    int failed=0;
    failed|=run_case(1,1,8);       // N tail 1
    failed|=run_case(2,3,16);      // N tail 3
    failed|=run_case(3,4,32);      // exact block
    failed|=run_case(4,7,64);      // full block + tail
    failed|=run_case(8,33,512);    // DiffSinger-like K, awkward N
    if(failed)return 1;
    benchmark(32,256,512,20);
    benchmark(64,512,512,10);
    return 0;
}
