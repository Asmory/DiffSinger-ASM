#ifndef DSASM_ENGINE_H
#define DSASM_ENGINE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#define DSASM_API __declspec(dllexport)
#else
#define DSASM_API __attribute__((visibility("default")))
#endif

#define DSASM_ENGINE_ABI_VERSION 3u

typedef struct dsasm_engine dsasm_engine;

typedef enum dsasm_engine_mode {
    DSASM_MODE_REALTIME_STREAMING = 1u,
    DSASM_MODE_BLOCK_BATCH = 2u
} dsasm_engine_mode;

typedef struct dsasm_mode_config {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t mode;
    uint32_t workers;
    uint32_t region_frames;
    uint32_t vocoder_bucket_frames;
    uint32_t overlap_frames;
    /* Incremented when a mode's promoted defaults change. */
    uint32_t profile_revision;
    /* Cache protocol revision, not a quality grade. Zero forbids canonical
       PCM commit; equal nonzero values permit logical cross-mode reuse. */
    uint32_t output_compatibility_revision;
} dsasm_mode_config;

enum {
    DSASM_OK = 0,
    DSASM_E_INVALID = -1,
    DSASM_E_IO = -2,
    DSASM_E_MODEL = -3,
    DSASM_E_NOMEM = -4,
    DSASM_E_UNSUPPORTED = -5,
    DSASM_E_CANCELLED = -6,
    DSASM_E_CALLBACK = -7,
    DSASM_E_INTERNAL = -8
};

enum {
    DSASM_REQUEST_USE_T_START = 1u << 0,
    DSASM_REQUEST_USE_TIME_SCALE = 1u << 1,
    DSASM_REQUEST_USE_STEPS = 1u << 2,
    DSASM_REQUEST_USE_SPEC_RANGE = 1u << 3,
    DSASM_REQUEST_USE_DEPTH = 1u << 4
};

/* Pointers only need to remain valid for the synchronous render call.
   mel2ph is 1-based. If it is NULL, durations[P] is expanded to mel2ph.
   speaker_embedding_frames is either 1 (broadcast) or mel_frames. */
typedef struct dsasm_request {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t flags;
    uint32_t reserved0;
    const int32_t *token_ids;
    size_t text_tokens;
    const int32_t *durations;
    const int32_t *mel2ph;
    const float *f0;
    size_t mel_frames;
    const int32_t *language_ids;
    const float *speaker_embedding;
    size_t speaker_embedding_frames;
    const float *breathiness;
    const float *voicing;
    const float *tension;
    const float *gender;
    const float *velocity;
    const float *noise;
    uint64_t noise_seed;
    float depth;
    float t_start;
    float time_scale_factor;
    uint32_t steps;
    const float *spec_min;
    const float *spec_max;
    size_t spec_range_dims;
    uint32_t overlap_frames;
    /* Exact fixed-shape vocoder bucket used for the whole request. Zero
       selects the smallest loaded bucket. */
    uint32_t vocoder_bucket_frames;
    /* Zero is reserved for the advanced benchmark engine. Product mode
       engines require one of dsasm_engine_mode. */
    uint32_t mode;
} dsasm_request;

static inline dsasm_request dsasm_request_init(void) {
    dsasm_request request = {0};
    request.struct_size = sizeof(request);
    request.abi_version = DSASM_ENGINE_ABI_VERSION;
    return request;
}
#define DSASM_REQUEST_INIT dsasm_request_init()
static inline dsasm_request dsasm_request_init_mode(dsasm_engine_mode mode) {
    dsasm_request request = dsasm_request_init();
    request.mode = (uint32_t)mode;
    if (mode == DSASM_MODE_REALTIME_STREAMING) {
        request.vocoder_bucket_frames = 32u;
        request.overlap_frames = 8u;
    } else if (mode == DSASM_MODE_BLOCK_BATCH) {
        request.vocoder_bucket_frames = 384u;
    }
    return request;
}

/* PCM is mono float32 and is valid only during the callback. Return zero to
   continue. A nonzero return stops rendering with DSASM_E_CALLBACK. */
typedef int (*dsasm_pcm_callback)(
    void *userdata,
    uint64_t sample_offset,
    const float *pcm,
    size_t sample_count,
    int is_final);

DSASM_API uint32_t dsasm_engine_abi_version(void);
/* Returns 1 when this build can run on the current OS/CPU. On failure, reason
   receives a short UTF-8 explanation when it is non-NULL and reason_size > 0. */
DSASM_API int dsasm_engine_is_supported(char *reason, size_t reason_size);
/* config must carry its struct size and ABI version before this call. */
DSASM_API int dsasm_engine_mode_config(dsasm_engine_mode mode, dsasm_mode_config *config);
DSASM_API dsasm_engine *dsasm_engine_create(
    const char *packed_acoustic_dir,
    const char *vocoder_bundle_dir,
    int workers);
DSASM_API dsasm_engine *dsasm_engine_create_mode(
    const char *packed_acoustic_dir,
    const char *vocoder_bundle_dir,
    dsasm_engine_mode mode);
DSASM_API int dsasm_engine_render(
    dsasm_engine *engine,
    const dsasm_request *request,
    dsasm_pcm_callback callback,
    void *userdata);
DSASM_API void dsasm_engine_cancel(dsasm_engine *engine);
DSASM_API void dsasm_engine_destroy(dsasm_engine *engine);
DSASM_API const char *dsasm_engine_last_error(const dsasm_engine *engine);
DSASM_API const char *dsasm_engine_last_create_error(void);
DSASM_API uint32_t dsasm_engine_sample_rate(const dsasm_engine *engine);
DSASM_API uint32_t dsasm_engine_hop_size(const dsasm_engine *engine);
DSASM_API uint32_t dsasm_engine_mel_bins(const dsasm_engine *engine);
DSASM_API size_t dsasm_engine_bucket_count(const dsasm_engine *engine);
DSASM_API uint32_t dsasm_engine_bucket_frames(const dsasm_engine *engine, size_t index);

#ifdef __cplusplus
}
#endif
#endif
