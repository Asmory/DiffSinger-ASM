#define _POSIX_C_SOURCE 200809L
#include "dsasm_engine.h"

#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    dsasm_engine *engine;
    dsasm_request request;
    atomic_int entering;
    int rc;
    size_t callbacks;
    size_t samples;
} RenderRun;

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return 1000.0 * (double)ts.tv_sec + 1e-6 * (double)ts.tv_nsec;
}

static void sleep_ms(double milliseconds) {
    struct timespec ts = {
        .tv_sec = (time_t)(milliseconds / 1000.0),
        .tv_nsec = (long)(fmod(milliseconds, 1000.0) * 1e6),
    };
    while (nanosleep(&ts, &ts) != 0) {}
}

static float *read_f32(const char *path, size_t *count) {
    FILE *file = fopen(path, "rb");
    if (!file || fseek(file, 0, SEEK_END)) return NULL;
    long bytes = ftell(file);
    if (bytes <= 0 || bytes % (long)sizeof(float) || fseek(file, 0, SEEK_SET)) {
        fclose(file);
        return NULL;
    }
    *count = (size_t)bytes / sizeof(float);
    float *result = malloc(*count * sizeof(*result));
    if (!result || fread(result, sizeof(*result), *count, file) != *count) {
        free(result);
        result = NULL;
    }
    fclose(file);
    return result;
}

static int on_pcm(void *userdata, uint64_t offset, const float *pcm,
        size_t count, int is_final) {
    RenderRun *run = userdata;
    (void)offset;
    (void)is_final;
    if (!pcm || !count) return 1;
    run->callbacks++;
    run->samples += count;
    return 0;
}

static void *render_thread(void *userdata) {
    RenderRun *run = userdata;
    atomic_store_explicit(&run->entering, 1, memory_order_release);
    run->rc = dsasm_engine_render(run->engine, &run->request, on_pcm, run);
    return NULL;
}

static int compare_double(const void *left, const void *right) {
    double a = *(const double *)left, b = *(const double *)right;
    return (a > b) - (a < b);
}

static double percentile(const double *values, size_t count, double p) {
    double *sorted = malloc(count * sizeof(*sorted));
    if (!sorted) return NAN;
    memcpy(sorted, values, count * sizeof(*sorted));
    qsort(sorted, count, sizeof(*sorted), compare_double);
    double position = (count - 1) * p;
    size_t low = (size_t)position;
    size_t high = low + (position > low);
    double result = sorted[low] + (sorted[high] - sorted[low]) *
        (position - (double)low);
    free(sorted);
    return result;
}

static void usage(const char *program) {
    fprintf(stderr, "usage: %s ACOUSTIC_DIR VOCODER_DIR SPEAKER_EMB "
        "--mode realtime|batch [--samples 25] [--warmup 3] "
        "[--cancel-delay-ms 10] [--gate-ms 371.52] [--steps 4]\n", program);
}

int main(int argc, char **argv) {
    if (argc < 6) { usage(argv[0]); return 2; }
    dsasm_engine_mode mode = 0;
    size_t samples = 25, warmup = 3, steps = 4;
    double cancel_delay_ms = 10.0, gate_ms = 371.52;
    for (int i = 4; i < argc; i++) {
        if (!strcmp(argv[i], "--mode") && i + 1 < argc) {
            const char *value = argv[++i];
            if (!strcmp(value, "realtime")) mode = DSASM_MODE_REALTIME_STREAMING;
            else if (!strcmp(value, "batch")) mode = DSASM_MODE_BLOCK_BATCH;
            else { usage(argv[0]); return 2; }
        } else if (!strcmp(argv[i], "--samples") && i + 1 < argc) {
            samples = (size_t)strtoull(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "--warmup") && i + 1 < argc) {
            warmup = (size_t)strtoull(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "--steps") && i + 1 < argc) {
            steps = (size_t)strtoull(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "--cancel-delay-ms") && i + 1 < argc) {
            cancel_delay_ms = strtod(argv[++i], NULL);
        } else if (!strcmp(argv[i], "--gate-ms") && i + 1 < argc) {
            gate_ms = strtod(argv[++i], NULL);
        } else { usage(argv[0]); return 2; }
    }
    if (!mode || !samples || !steps || cancel_delay_ms <= 0.0 || gate_ms <= 0.0) {
        usage(argv[0]); return 2;
    }

    dsasm_mode_config config = {
        .struct_size = sizeof(config),
        .abi_version = DSASM_ENGINE_ABI_VERSION,
    };
    if (dsasm_engine_mode_config(mode, &config) != DSASM_OK) return 3;
    dsasm_engine *engine = dsasm_engine_create_mode(argv[1], argv[2], mode);
    if (!engine) {
        fprintf(stderr, "engine create failed: %s\n", dsasm_engine_last_create_error());
        return 3;
    }

    size_t speaker_count = 0;
    float *speaker = read_f32(argv[3], &speaker_count);
    float *f0 = malloc(config.region_frames * sizeof(*f0));
    if (!speaker || !speaker_count || !f0) {
        fprintf(stderr, "input allocation failed\n");
        return 4;
    }
    for (size_t i = 0; i < config.region_frames; i++) f0[i] = 220.0f;
    int32_t token = 1;
    int32_t duration = (int32_t)config.region_frames;
    int32_t language = 1;

    RenderRun run = {0};
    run.engine = engine;
    run.request = dsasm_request_init_mode(mode);
    run.request.flags = DSASM_REQUEST_USE_DEPTH | DSASM_REQUEST_USE_STEPS;
    run.request.token_ids = &token;
    run.request.text_tokens = 1;
    run.request.durations = &duration;
    run.request.f0 = f0;
    run.request.mel_frames = config.region_frames;
    run.request.language_ids = &language;
    run.request.speaker_embedding = speaker;
    run.request.speaker_embedding_frames = 1;
    run.request.noise_seed = UINT64_C(123456789);
    run.request.depth = .6f;
    run.request.steps = (uint32_t)steps;
    atomic_init(&run.entering, 0);

    double *latencies = calloc(samples, sizeof(*latencies));
    if (!latencies) return 4;
    size_t total = warmup + samples;
    for (size_t i = 0; i < total; i++) {
        run.rc = DSASM_OK;
        run.callbacks = 0;
        run.samples = 0;
        atomic_store_explicit(&run.entering, 0, memory_order_relaxed);
        pthread_t thread;
        if (pthread_create(&thread, NULL, render_thread, &run)) return 5;
        while (!atomic_load_explicit(&run.entering, memory_order_acquire)) {}
        sleep_ms(cancel_delay_ms);
        double start = now_ms();
        dsasm_engine_cancel(engine);
        pthread_join(thread, NULL);
        double latency = now_ms() - start;
        if (run.rc != DSASM_E_CANCELLED || run.callbacks || run.samples) {
            fprintf(stderr, "cancel failed index=%zu rc=%d callbacks=%zu samples=%zu error=%s\n",
                i, run.rc, run.callbacks, run.samples, dsasm_engine_last_error(engine));
            return 6;
        }
        if (i >= warmup) {
            size_t measured = i - warmup;
            latencies[measured] = latency;
            printf("CANCEL_SAMPLE mode=%u index=%zu latency_ms=%.3f rc=%d\n",
                (unsigned)mode, measured, latency, run.rc);
        }
    }

    double worst = 0.0;
    for (size_t i = 0; i < samples; i++) if (latencies[i] > worst) worst = latencies[i];
    double p50 = percentile(latencies, samples, .50);
    double p90 = percentile(latencies, samples, .90);
    double p99 = percentile(latencies, samples, .99);
    int pass = isfinite(p99) && p99 <= gate_ms;
    printf("CANCEL_FINAL mode=%u workers=%u region_frames=%u bucket=%u "
        "samples=%zu warmup=%zu delay_ms=%.3f p50_ms=%.3f p90_ms=%.3f "
        "p99_ms=%.3f worst_ms=%.3f gate_ms=%.3f pass=%s\n",
        (unsigned)mode, config.workers, config.region_frames,
        config.vocoder_bucket_frames, samples, warmup, cancel_delay_ms,
        p50, p90, p99, worst, gate_ms, pass ? "PASS" : "FAIL");

    free(latencies);
    free(f0);
    free(speaker);
    dsasm_engine_destroy(engine);
    return pass ? 0 : 7;
}
