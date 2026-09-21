#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

extern void ds_fused_linear_softsign_glu_f32_avx2(
    const float *x,
    const float *w_left,
    const float *w_gate,
    const float *b_left,
    const float *b_gate,
    float *y,
    size_t M,
    size_t N,
    size_t K
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
    void *p = aligned_alloc(64, (bytes + 63u) & ~63u);
    if (!p) {
        perror("aligned_alloc");
        exit(1);
    }
    return p;
}

static void reference(
    const float *x,
    const float *wl,
    const float *wg,
    const float *bl,
    const float *bg,
    float *y,
    size_t M,
    size_t N,
    size_t K
) {
    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            float l = bl[n];
            float g = bg[n];
            const float *xr = x + m * K;
            const float *wlr = wl + n * K;
            const float *wgr = wg + n * K;
            for (size_t k = 0; k < K; ++k) {
                l += xr[k] * wlr[k];
                g += xr[k] * wgr[k];
            }
            float softsign = g / (1.0f + fabsf(g));
            y[m * N + n] = l * softsign;
        }
    }
}

static int run_case(size_t M, size_t N, size_t K) {
    const size_t x_count = M * K;
    const size_t w_count = N * K;
    const size_t y_count = M * N;

    float *x  = xmalloc(x_count * sizeof(float));
    float *wl = xmalloc(w_count * sizeof(float));
    float *wg = xmalloc(w_count * sizeof(float));
    float *bl = xmalloc(N * sizeof(float));
    float *bg = xmalloc(N * sizeof(float));
    float *ref = xmalloc(y_count * sizeof(float));
    float *got = xmalloc(y_count * sizeof(float));

    for (size_t i = 0; i < x_count; ++i) x[i] = frand_small();
    for (size_t i = 0; i < w_count; ++i) {
        wl[i] = frand_small();
        wg[i] = frand_small();
    }
    for (size_t i = 0; i < N; ++i) {
        bl[i] = frand_small();
        bg[i] = frand_small();
    }

    reference(x, wl, wg, bl, bg, ref, M, N, K);
    memset(got, 0, y_count * sizeof(float));
    ds_fused_linear_softsign_glu_f32_avx2(x, wl, wg, bl, bg, got, M, N, K);

    float max_abs = 0.0f;
    float max_rel = 0.0f;
    size_t worst = 0;
    for (size_t i = 0; i < y_count; ++i) {
        const float abs_err = fabsf(ref[i] - got[i]);
        const float denom = fmaxf(fabsf(ref[i]), 1e-6f);
        const float rel_err = abs_err / denom;
        if (abs_err > max_abs) {
            max_abs = abs_err;
            worst = i;
        }
        if (rel_err > max_rel) max_rel = rel_err;
    }

    printf("case M=%zu N=%zu K=%zu : max_abs=%g max_rel=%g", M, N, K, max_abs, max_rel);
    if (max_abs > 2e-4f) {
        printf("  FAIL @%zu ref=%g got=%g\n", worst, ref[worst], got[worst]);
        free(x); free(wl); free(wg); free(bl); free(bg); free(ref); free(got);
        return 1;
    }
    puts("  OK");

    free(x); free(wl); free(wg); free(bl); free(bg); free(ref); free(got);
    return 0;
}

static void benchmark(size_t M, size_t N, size_t K, int iters) {
    const size_t x_count = M * K;
    const size_t w_count = N * K;
    const size_t y_count = M * N;
    float *x  = xmalloc(x_count * sizeof(float));
    float *wl = xmalloc(w_count * sizeof(float));
    float *wg = xmalloc(w_count * sizeof(float));
    float *bl = xmalloc(N * sizeof(float));
    float *bg = xmalloc(N * sizeof(float));
    float *y = xmalloc(y_count * sizeof(float));

    for (size_t i = 0; i < x_count; ++i) x[i] = frand_small();
    for (size_t i = 0; i < w_count; ++i) { wl[i] = frand_small(); wg[i] = frand_small(); }
    for (size_t i = 0; i < N; ++i) { bl[i] = frand_small(); bg[i] = frand_small(); }

    ds_fused_linear_softsign_glu_f32_avx2(x, wl, wg, bl, bg, y, M, N, K);

    const double t0 = now_sec();
    for (int i = 0; i < iters; ++i)
        ds_fused_linear_softsign_glu_f32_avx2(x, wl, wg, bl, bg, y, M, N, K);
    const double t1 = now_sec();

    // Two independent dot products: ~4*M*N*K FLOPs, activation excluded.
    const double flops = 4.0 * (double)M * (double)N * (double)K * (double)iters;
    printf("bench M=%zu N=%zu K=%zu iters=%d : %.3f ms/iter, %.2f GFLOP/s\n",
           M, N, K, iters, (t1 - t0) * 1e3 / iters, flops / (t1 - t0) / 1e9);

    volatile float sink = y[(M * N) / 2];
    (void)sink;

    free(x); free(wl); free(wg); free(bl); free(bg); free(y);
}

int main(void) {
#if defined(__x86_64__)
    if (!__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("fma")) {
        fprintf(stderr, "AVX2 + FMA3 are required.\n");
        return 2;
    }
#endif

    int failed = 0;
    failed |= run_case(1, 1, 1);
    failed |= run_case(3, 7, 13);       // exercises scalar K tail
    failed |= run_case(4, 16, 32);
    failed |= run_case(5, 33, 511);     // awkward shape
    failed |= run_case(8, 64, 512);     // DiffSinger-like hidden dimension
    if (failed) return 1;

    benchmark(32, 256, 512, 8);
    return 0;
}
