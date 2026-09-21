#define _GNU_SOURCE
#include "dsasm_model.h"
#include "dsasm_threadpool.h"
#include "dsasm_vocoder_graph.h"

#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { FRAMES = 32, MEL_BINS = 128, HIDDEN = 384, SAMPLES = 16384 };

typedef struct {
    DSAsmVocoderGraph *graph;
    const float *mel;
    const float *f0;
    float *wave;
    pthread_barrier_t *barrier;
    double elapsed_ms;
    int rc;
} VocoderJob;

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return 1000.0 * ts.tv_sec + 1e-6 * ts.tv_nsec;
}

static double cpu_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    return 1000.0 * ts.tv_sec + 1e-6 * ts.tv_nsec;
}

static float *read_f32(const char *path, size_t count) {
    FILE *file = fopen(path, "rb");
    if (!file) return NULL;
    float *values = aligned_alloc(64, (count * sizeof(*values) + 63u) & ~63u);
    if (!values || fread(values, sizeof(*values), count, file) != count ||
            fgetc(file) != EOF) {
        free(values);
        values = NULL;
    }
    fclose(file);
    return values;
}

static uint64_t xs64(uint64_t *state) {
    uint64_t value = *state;
    value ^= value >> 12;
    value ^= value << 25;
    value ^= value >> 27;
    *state = value;
    return value * UINT64_C(2685821657736338717);
}

static float uniform01(uint64_t *state) {
    return ((xs64(state) >> 40) + 1.0f) * (1.0f / 16777217.0f);
}

static void gaussian(float *values, size_t count, uint64_t seed) {
    uint64_t state = seed;
    size_t i = 0;
    while (i < count) {
        float u1 = uniform01(&state), u2 = uniform01(&state);
        float radius = sqrtf(-2.0f * logf(u1));
        float angle = 6.2831853071795864769f * u2;
        values[i++] = radius * cosf(angle);
        if (i < count) values[i++] = radius * sinf(angle);
    }
}

static void quality(const char *name, const float *actual, const float *golden,
        size_t count) {
    double error2 = 0.0, actual2 = 0.0, golden2 = 0.0, dot = 0.0;
    double max_abs = 0.0;
    for (size_t i = 0; i < count; i++) {
        double a = actual[i], g = golden[i], error = a - g;
        double absolute = fabs(error);
        if (absolute > max_abs) max_abs = absolute;
        error2 += error * error;
        actual2 += a * a;
        golden2 += g * g;
        dot += a * g;
    }
    double cosine = dot / (sqrt(actual2 * golden2) + 1e-300);
    double snr = 10.0 * log10((golden2 + 1e-300) / (error2 + 1e-300));
    printf("PIPELINE_QUALITY stage=%s max_abs=%.9g cosine=%.9f SNR=%.2f pass=%s\n",
        name, max_abs, cosine, snr,
        cosine >= 0.999 && snr >= 25.0 ? "PASS" : "FAIL");
}

static void *run_vocoder(void *opaque) {
    VocoderJob *job = opaque;
    pthread_barrier_wait(job->barrier);
    double begin = now_ms();
    job->rc = ds_vocoder_graph_infer(
        job->graph, job->mel, job->f0, job->wave, 0);
    job->elapsed_ms = now_ms() - begin;
    return NULL;
}

static int infer_acoustic(const DSAsmAcousticModel *model,
        DSAsmThreadPool *pool, const float *speaker, const float *f0,
        const float *noise, float *mel, float *workspace) {
    int32_t token = 1, language = 1;
    int32_t mel2ph[FRAMES];
    for (size_t i = 0; i < FRAMES; i++) mel2ph[i] = 1;
    float spec_min = -12.0f, spec_max = 0.0f;
    DSAsmFS2DeploymentInputs inputs = {0};
    inputs.languages = &language;
    inputs.speaker_embedding_tc = speaker;
    return ds_acoustic_model_infer_deploy_f32_avx2(
        model, &inputs, &token, 1, mel2ph, f0, FRAMES, noise,
        &spec_min, &spec_max, 1, 0.4f, 1000.0f, 4,
        mel, workspace, pool);
}

static int run_pair(const DSAsmAcousticModel *model, DSAsmThreadPool *acoustic_pool,
        DSAsmVocoderGraph *graph, const float *speaker, const float *f0,
        const float *noise, float *read_mel, float *write_mel, float *workspace,
        float *wave, double *acoustic_ms, double *vocoder_ms, double *wall_ms,
        double *process_ms) {
    pthread_barrier_t barrier;
    if (pthread_barrier_init(&barrier, NULL, 2)) return -1;
    VocoderJob job = {graph, read_mel, f0, wave, &barrier, 0.0, 0};
    pthread_t thread;
    if (pthread_create(&thread, NULL, run_vocoder, &job)) {
        pthread_barrier_destroy(&barrier);
        return -1;
    }
    double wall_begin = now_ms(), cpu_begin = cpu_ms();
    pthread_barrier_wait(&barrier);
    double acoustic_begin = now_ms();
    int rc = infer_acoustic(model, acoustic_pool, speaker, f0, noise,
        write_mel, workspace);
    *acoustic_ms = now_ms() - acoustic_begin;
    pthread_join(thread, NULL);
    *process_ms = cpu_ms() - cpu_begin;
    *wall_ms = now_ms() - wall_begin;
    *vocoder_ms = job.elapsed_ms;
    pthread_barrier_destroy(&barrier);
    return rc ? rc : job.rc;
}

static void usage(const char *program) {
    fprintf(stderr, "usage: %s ACOUSTIC_DIR VOCODER32 SPEAKER_EMB "
        "MEL_GOLDEN WAVE_GOLDEN [acoustic_workers=3] [vocoder_workers=3] "
        "[warmup=10] [rounds=25] [acoustic_cpu_offset=0] "
        "[vocoder_cpu_offset=acoustic_workers]\n", program);
}

int main(int argc, char **argv) {
    if (argc < 6 || argc > 12) {
        usage(argv[0]);
        return 2;
    }
    size_t acoustic_workers = argc > 6 ? strtoull(argv[6], NULL, 10) : 3;
    size_t vocoder_workers = argc > 7 ? strtoull(argv[7], NULL, 10) : 3;
    size_t warmup = argc > 8 ? strtoull(argv[8], NULL, 10) : 10;
    size_t rounds = argc > 9 ? strtoull(argv[9], NULL, 10) : 25;
    size_t acoustic_offset = argc > 10 ? strtoull(argv[10], NULL, 10) : 0;
    size_t vocoder_offset = argc > 11
        ? strtoull(argv[11], NULL, 10) : acoustic_workers;
    if (!acoustic_workers || !vocoder_workers || !rounds) return 2;

    char fs[4096], aux[4096], rf[4096];
    if (snprintf(fs, sizeof(fs), "%s/fs2_acoustic.dsfs", argv[1]) >= (int)sizeof(fs) ||
            snprintf(aux, sizeof(aux), "%s/aux_convnext.dsa", argv[1]) >= (int)sizeof(aux) ||
            snprintf(rf, sizeof(rf), "%s/lynxnet2.dsn", argv[1]) >= (int)sizeof(rf))
        return 2;
    DSAsmAcousticModel model;
    if (ds_acoustic_model_load(&model, fs, aux, rf)) {
        fprintf(stderr, "acoustic model load failed\n");
        return 3;
    }
    DSAsmThreadPool *acoustic_pool = ds_threadpool_create_range(
        acoustic_workers, acoustic_offset);
    DSAsmThreadPool *vocoder_pool = ds_threadpool_create_range(
        vocoder_workers, vocoder_offset);
    DSAsmVocoderGraph *graph = vocoder_pool
        ? ds_vocoder_graph_load_with_pool(argv[2], vocoder_pool) : NULL;
    if (!acoustic_pool || !vocoder_pool || !graph ||
            ds_vocoder_graph_frames(graph) != FRAMES ||
            ds_vocoder_graph_mel_bins(graph) != MEL_BINS ||
            ds_vocoder_graph_samples(graph) != SAMPLES) {
        fprintf(stderr, "pool or vocoder load failed\n");
        return 3;
    }

    float *speaker_one = read_f32(argv[3], HIDDEN);
    float *mel_golden = read_f32(argv[4], FRAMES * MEL_BINS);
    float *wave_golden = read_f32(argv[5], SAMPLES);
    float *speaker = aligned_alloc(64, FRAMES * HIDDEN * sizeof(*speaker));
    float *f0 = aligned_alloc(64, FRAMES * sizeof(*f0));
    float *noise = aligned_alloc(64, FRAMES * MEL_BINS * sizeof(*noise));
    float *mel[2] = {
        aligned_alloc(64, FRAMES * MEL_BINS * sizeof(float)),
        aligned_alloc(64, FRAMES * MEL_BINS * sizeof(float))
    };
    float *wave = aligned_alloc(64, SAMPLES * sizeof(*wave));
    size_t workspace_count = ds_acoustic_model_workspace_floats(&model, 1, FRAMES);
    float *workspace = aligned_alloc(64,
        (workspace_count * sizeof(*workspace) + 63u) & ~63u);
    if (!speaker_one || !mel_golden || !wave_golden || !speaker || !f0 ||
            !noise || !mel[0] || !mel[1] || !wave || !workspace) {
        fprintf(stderr, "fixture allocation failed\n");
        return 4;
    }
    for (size_t t = 0; t < FRAMES; t++) {
        memcpy(speaker + t * HIDDEN, speaker_one, HIDDEN * sizeof(float));
        f0[t] = 220.0f;
    }
    gaussian(noise, FRAMES * MEL_BINS, UINT64_C(123456789));
    if (infer_acoustic(&model, acoustic_pool, speaker, f0, noise, mel[0], workspace)) {
        fprintf(stderr, "initial acoustic inference failed\n");
        return 5;
    }

    size_t ready = 0;
    for (size_t i = 0; i < warmup + rounds; i++) {
        size_t next = ready ^ 1u;
        double acoustic_ms, vocoder_ms, wall_ms, process_ms;
        int rc = run_pair(&model, acoustic_pool, graph, speaker, f0, noise,
            mel[ready], mel[next], workspace, wave,
            &acoustic_ms, &vocoder_ms, &wall_ms, &process_ms);
        if (rc) {
            fprintf(stderr, "pipeline round %zu failed rc=%d\n", i, rc);
            return 5;
        }
        ready = next;
        if (i >= warmup) {
            size_t measured = i - warmup;
            double audio_ms = 1000.0 * SAMPLES / 44100.0;
            printf("PIPELINE_REGION index=%zu acoustic_ms=%.3f vocoder_ms=%.3f "
                "wall_ms=%.3f cpu_ms=%.3f wall_RTF=%.6f cpu_RTF=%.6f\n",
                measured, acoustic_ms, vocoder_ms, wall_ms, process_ms,
                wall_ms / audio_ms, process_ms / audio_ms);
        }
    }
    quality("acoustic", mel[ready], mel_golden, FRAMES * MEL_BINS);
    quality("vocoder", wave, wave_golden, SAMPLES);
    printf("PIPELINE_CPUS acoustic=");
    for (size_t i = 0; i < acoustic_workers; i++)
        printf("%s%d", i ? "," : "", ds_threadpool_cpu_at(acoustic_pool, i));
    printf(" vocoder=");
    for (size_t i = 0; i < vocoder_workers; i++)
        printf("%s%d", i ? "," : "", ds_threadpool_cpu_at(vocoder_pool, i));
    putchar('\n');

    free(workspace); free(wave); free(mel[1]); free(mel[0]); free(noise);
    free(f0); free(speaker); free(wave_golden); free(mel_golden); free(speaker_one);
    ds_vocoder_graph_free(graph);
    ds_threadpool_destroy(vocoder_pool);
    ds_threadpool_destroy(acoustic_pool);
    ds_acoustic_model_unload(&model);
    return 0;
}
