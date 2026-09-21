#define _POSIX_C_SOURCE 200809L
#include "dsasm_engine.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    double at_ms;
    double interval_ms;
    double playable_ms;
    double lateness_ms;
    size_t samples;
    size_t region;
} Event;

typedef struct {
    Event *events;
    size_t count;
    size_t capacity;
    size_t region;
    uint64_t next_offset;
    uint64_t checksum;
    double previous_at_ms;
    double previous_playable_ms;
    double region_start_ms;
    double sample_rate;
    const float *golden;
    size_t golden_count;
    double error_square_sum;
    double golden_square_sum;
    double output_square_sum;
    double dot_sum;
    double max_abs_error;
    size_t quality_samples;
    int record;
    int invalid;
} CallbackState;

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

static int read_u64_file(const char *path, uint64_t *value) {
    FILE *file = fopen(path, "r");
    if (!file) return -1;
    unsigned long long parsed = 0;
    int rc = fscanf(file, "%llu", &parsed) == 1 ? 0 : -1;
    fclose(file);
    *value = (uint64_t)parsed;
    return rc;
}

static double pcore_frequency_mhz(void) {
    static const int cpus[] = {0, 2, 4, 6, 1, 3, 5, 7};
    double total = 0.0;
    size_t count = 0;
    for (size_t i = 0; i < sizeof(cpus) / sizeof(cpus[0]); i++) {
        char path[128];
        snprintf(path, sizeof(path),
            "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq", cpus[i]);
        uint64_t khz;
        if (!read_u64_file(path, &khz)) { total += khz / 1000.0; count++; }
    }
    return count ? total / count : NAN;
}

static int on_pcm(void *userdata, uint64_t offset, const float *pcm,
        size_t count, int is_final) {
    CallbackState *state = userdata;
    if (!pcm || !count || offset != state->next_offset) return 1;
    state->next_offset += count;
    for (size_t i = 0; i < count; i++) {
        uint32_t bits;
        if (!isfinite(pcm[i])) state->invalid = 1;
        if (state->record && state->golden) {
            if (offset + i >= state->golden_count) {
                state->invalid = 1;
            } else {
                double output = pcm[i], golden = state->golden[offset + i];
                double error = output - golden;
                double absolute = fabs(error);
                state->error_square_sum += error * error;
                state->golden_square_sum += golden * golden;
                state->output_square_sum += output * output;
                state->dot_sum += output * golden;
                if (absolute > state->max_abs_error)
                    state->max_abs_error = absolute;
                state->quality_samples++;
            }
        }
        memcpy(&bits, pcm + i, sizeof(bits));
        state->checksum = (state->checksum ^ bits) * UINT64_C(1099511628211);
    }
    if (!state->record) return 0;
    if (state->count == state->capacity) return 1;
    double at = now_ms();
    double playable = 1000.0 * count / state->sample_rate;
    double interval = state->previous_at_ms > 0.0
        ? at - state->previous_at_ms : at - state->region_start_ms;
    double deadline = state->previous_at_ms > 0.0
        ? state->previous_playable_ms : playable;
    state->events[state->count++] = (Event){
        at, interval, playable, fmax(interval - deadline, 0.0), count,
        state->region
    };
    state->previous_at_ms = at;
    state->previous_playable_ms = playable;
    (void)is_final;
    return 0;
}

static int double_compare(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double percentile(const double *values, size_t count, double p) {
    double *copy = malloc(count * sizeof(*copy));
    if (!copy) return NAN;
    memcpy(copy, values, count * sizeof(*copy));
    qsort(copy, count, sizeof(*copy), double_compare);
    double position = (count - 1) * p;
    size_t lo = (size_t)position, hi = lo + (position > lo);
    double result = copy[lo] + (copy[hi] - copy[lo]) * (position - lo);
    free(copy);
    return result;
}

static void print_stats(const char *name, const double *values, size_t count) {
    double sum = 0.0, square_sum = 0.0, worst = 0.0;
    for (size_t i = 0; i < count; i++) {
        sum += values[i];
        square_sum += values[i] * values[i];
        if (values[i] > worst) worst = values[i];
    }
    double mean = sum / count;
    double variance = fmax(square_sum / count - mean * mean, 0.0);
    printf("%s median_ms=%.3f p90_ms=%.3f worst_ms=%.3f cv=%.4f\n",
        name, percentile(values, count, .5), percentile(values, count, .9),
        worst, mean > 0.0 ? sqrt(variance) / mean : 0.0);
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

static void usage(const char *program) {
    fprintf(stderr, "usage: %s ACOUSTIC_DIR VOCODER_DIR SPEAKER_EMB "
        "[--frames 32] [--regions 25] [--warmup 3] [--workers 8] "
        "[--bucket 32] [--steps 4] [--overlap 8] [--golden PCM.f32]\n",
        program);
}

int main(int argc, char **argv) {
    if (argc < 4) { usage(argv[0]); return 2; }
    size_t frames = 32, regions = 25, warmup = 3, workers = 8;
    size_t bucket = 32, steps = 4, overlap = 8;
    const char *golden_path = NULL;
    for (int i = 4; i < argc; i++) {
        if (i + 1 >= argc) { usage(argv[0]); return 2; }
        if (!strcmp(argv[i], "--golden")) {
            golden_path = argv[++i];
            continue;
        }
        size_t value = (size_t)strtoull(argv[++i], NULL, 10);
        if (!strcmp(argv[i - 1], "--frames")) frames = value;
        else if (!strcmp(argv[i - 1], "--regions")) regions = value;
        else if (!strcmp(argv[i - 1], "--warmup")) warmup = value;
        else if (!strcmp(argv[i - 1], "--workers")) workers = value;
        else if (!strcmp(argv[i - 1], "--bucket")) bucket = value;
        else if (!strcmp(argv[i - 1], "--steps")) steps = value;
        else if (!strcmp(argv[i - 1], "--overlap")) overlap = value;
        else { usage(argv[0]); return 2; }
    }
    if (!frames || !regions || !workers || !bucket || !steps || overlap >= bucket) {
        usage(argv[0]); return 2;
    }

    dsasm_engine *engine = dsasm_engine_create(argv[1], argv[2], (int)workers);
    if (!engine) {
        fprintf(stderr, "engine create failed: %s\n", dsasm_engine_last_create_error());
        return 3;
    }
    size_t speaker_count = 0;
    float *speaker = read_f32(argv[3], &speaker_count);
    size_t golden_count = 0;
    float *golden = golden_path ? read_f32(golden_path, &golden_count) : NULL;
    float *f0 = malloc(frames * sizeof(*f0));
    if (!speaker || !f0 || (golden_path && !golden)) {
        fprintf(stderr, "input allocation failed\n"); return 4;
    }
    size_t expected_samples = frames * dsasm_engine_hop_size(engine);
    if (golden && golden_count != expected_samples) {
        fprintf(stderr, "golden sample count=%zu expected=%zu\n",
            golden_count, expected_samples);
        return 4;
    }
    for (size_t i = 0; i < frames; i++) f0[i] = 220.0f;
    int32_t token = 1, duration = (int32_t)frames, language = 1;

    dsasm_request request = DSASM_REQUEST_INIT;
    request.flags = DSASM_REQUEST_USE_DEPTH | DSASM_REQUEST_USE_STEPS;
    request.token_ids = &token;
    request.text_tokens = 1;
    request.durations = &duration;
    request.f0 = f0;
    request.mel_frames = frames;
    request.language_ids = &language;
    request.speaker_embedding = speaker;
    request.speaker_embedding_frames = 1;
    request.noise_seed = UINT64_C(123456789);
    request.depth = .6f;
    request.steps = (uint32_t)steps;
    request.overlap_frames = (uint32_t)overlap;
    request.vocoder_bucket_frames = (uint32_t)bucket;

    size_t event_capacity = regions * (1 + (frames - 1) / (bucket - overlap));
    CallbackState state = {0};
    state.capacity = event_capacity;
    state.events = calloc(event_capacity, sizeof(*state.events));
    state.sample_rate = dsasm_engine_sample_rate(engine);
    state.golden = golden;
    state.golden_count = golden_count;
    if (!state.events) return 4;

    for (size_t i = 0; i < warmup; i++) {
        state.next_offset = 0;
        state.record = 0;
        int rc = dsasm_engine_render(engine, &request, on_pcm, &state);
        if (rc || state.next_offset != frames * dsasm_engine_hop_size(engine)) {
            fprintf(stderr, "warmup failed rc=%d error=%s\n", rc,
                dsasm_engine_last_error(engine));
            return 5;
        }
    }

    double *region_ms = calloc(regions, sizeof(*region_ms));
    int telemetry = getenv("DSASM_BENCH_TELEMETRY") != NULL;
    const char *energy_path = "/sys/class/powercap/intel-rapl:0/energy_uj";
    const char *energy_max_path = "/sys/class/powercap/intel-rapl:0/max_energy_range_uj";
    uint64_t energy_max = 0;
    if (telemetry && read_u64_file(energy_max_path, &energy_max)) telemetry = 0;
    double begin = now_ms(), cpu_begin = cpu_ms();
    state.previous_at_ms = 0.0;
    state.record = 1;
    for (size_t i = 0; i < regions; i++) {
        state.region = i;
        state.next_offset = 0;
        uint64_t energy_begin = 0, energy_end = 0;
        double frequency_begin = NAN, frequency_end = NAN;
        if (telemetry) {
            read_u64_file(energy_path, &energy_begin);
            frequency_begin = pcore_frequency_mhz();
        }
        state.region_start_ms = now_ms();
        int rc = dsasm_engine_render(engine, &request, on_pcm, &state);
        region_ms[i] = now_ms() - state.region_start_ms;
        if (telemetry) {
            frequency_end = pcore_frequency_mhz();
            read_u64_file(energy_path, &energy_end);
        }
        if (rc || state.next_offset != frames * dsasm_engine_hop_size(engine)) {
            fprintf(stderr, "region %zu failed rc=%d error=%s samples=%llu\n", i,
                rc, dsasm_engine_last_error(engine),
                (unsigned long long)state.next_offset);
            return 6;
        }
        printf("REGION index=%zu latency_ms=%.3f\n", i, region_ms[i]);
        if (telemetry) {
            uint64_t energy_delta = energy_end >= energy_begin
                ? energy_end - energy_begin : energy_max - energy_begin + energy_end;
            printf("TELEMETRY index=%zu package_j=%.6f package_w=%.3f "
                "pcore_mhz_before=%.1f pcore_mhz_after=%.1f\n", i,
                energy_delta / 1e6, energy_delta / (region_ms[i] * 1000.0),
                frequency_begin, frequency_end);
        }
    }
    double aggregate_ms = now_ms() - begin;
    double aggregate_cpu_ms = cpu_ms() - cpu_begin;
    size_t misses = 0;
    double max_lateness = 0.0;
    double *intervals = malloc(state.count * sizeof(*intervals));
    for (size_t i = 0; i < state.count; i++) {
        Event *event = state.events + i;
        intervals[i] = event->interval_ms;
        if (event->lateness_ms > 0.0) misses++;
        if (event->lateness_ms > max_lateness) max_lateness = event->lateness_ms;
        printf("CALLBACK index=%zu region=%zu samples=%zu interval_ms=%.3f "
            "playable_ms=%.3f lateness_ms=%.3f\n", i, event->region,
            event->samples, event->interval_ms, event->playable_ms,
            event->lateness_ms);
    }
    double audio_ms = 1000.0 * regions * frames * dsasm_engine_hop_size(engine)
        / state.sample_rate;
    double region_audio_ms = audio_ms / regions, worst_region_ms = 0.0;
    for (size_t i = 0; i < regions; i++)
        if (region_ms[i] > worst_region_ms) worst_region_ms = region_ms[i];
    double worst_rtf = worst_region_ms / region_audio_ms;
    double quality_rmse = NAN, quality_cosine = NAN, quality_snr = NAN;
    int quality_pass = golden == NULL;
    if (golden && state.quality_samples == regions * golden_count) {
        quality_rmse = sqrt(state.error_square_sum / state.quality_samples);
        quality_cosine = state.dot_sum /
            (sqrt(state.golden_square_sum * state.output_square_sum) + 1e-300);
        quality_snr = 10.0 * log10((state.golden_square_sum + 1e-300) /
            (state.error_square_sum + 1e-300));
        quality_pass = quality_cosine >= 0.999 && quality_snr >= 25.0;
        printf("QUALITY max_abs=%.9g rmse=%.9g cosine=%.9f SNR=%.2f "
            "samples=%zu pass=%s\n", state.max_abs_error, quality_rmse,
            quality_cosine, quality_snr, state.quality_samples,
            quality_pass ? "PASS" : "FAIL");
    } else if (!golden) {
        printf("QUALITY pass=SKIP reason=no-golden\n");
    } else {
        printf("QUALITY pass=FAIL reason=sample-count got=%zu expected=%zu\n",
            state.quality_samples, regions * golden_count);
    }
    print_stats("REGION_STATS", region_ms, regions);
    print_stats("CALLBACK_STATS", intervals, state.count);
    printf("STREAM_FINAL frames=%zu regions=%zu bucket=%zu callbacks=%zu "
        "aggregate_ms=%.3f cpu_ms=%.3f audio_ms=%.3f aggregate_RTF=%.6f "
        "cpu_RTF=%.6f average_cores=%.3f "
        "worst_RTF=%.6f first_region_ms=%.3f deadline_misses=%zu "
        "max_lateness_ms=%.3f realtime_pass=%s "
        "checksum=%016llx finite=%s quality_pass=%s\n", frames, regions, bucket, state.count,
        aggregate_ms, aggregate_cpu_ms, audio_ms, aggregate_ms / audio_ms,
        aggregate_cpu_ms / audio_ms, aggregate_cpu_ms / aggregate_ms,
        worst_rtf, region_ms[0], misses,
        max_lateness, (worst_rtf<1.0&&misses==0)?"PASS":"FAIL",
        (unsigned long long)state.checksum,
        state.invalid ? "FAIL" : "PASS",
        golden ? (quality_pass ? "PASS" : "FAIL") : "SKIP");

    free(intervals); free(region_ms); free(state.events); free(f0); free(golden);
    free(speaker);
    dsasm_engine_destroy(engine);
    return state.invalid ? 7 : quality_pass ? 0 : 8;
}
