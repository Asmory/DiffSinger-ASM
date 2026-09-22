#include "dsasm_engine.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

enum { FRAMES = 400, SPEAKER_DIM = 384 };

typedef struct {
    uint64_t next_offset;
    size_t callbacks;
    size_t finals;
} CallbackState;

static int on_pcm(void *userdata, uint64_t offset, const float *pcm,
        size_t count, int is_final) {
    CallbackState *state = userdata;
    if (!pcm || !count || offset != state->next_offset) return 1;
    state->next_offset += count;
    state->callbacks++;
    state->finals += is_final != 0;
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s PACKED_ACOUSTIC VOCODER_BUCKET_DIR_OR_FILE\n", argv[0]);
        return 2;
    }
    dsasm_engine *engine = dsasm_engine_create_mode(
        argv[1], argv[2], DSASM_MODE_REALTIME_STREAMING);
    if (!engine) {
        fprintf(stderr, "create: %s\n", dsasm_engine_last_create_error());
        return 3;
    }
    int32_t token = 1, duration = FRAMES, language = 1;
    float *f0 = malloc(FRAMES * sizeof(*f0));
    float *speaker = calloc(SPEAKER_DIM, sizeof(*speaker));
    if (!f0 || !speaker) return 4;
    for (size_t i = 0; i < FRAMES; i++) f0[i] = 220.f;

    dsasm_request request = dsasm_request_init_mode(DSASM_MODE_REALTIME_STREAMING);
    request.flags = DSASM_REQUEST_USE_DEPTH | DSASM_REQUEST_USE_STEPS;
    request.token_ids = &token;
    request.text_tokens = 1;
    request.durations = &duration;
    request.f0 = f0;
    request.mel_frames = FRAMES;
    request.language_ids = &language;
    request.speaker_embedding = speaker;
    request.speaker_embedding_frames = 1;
    request.noise_seed = UINT64_C(123456789);
    request.depth = .6f;
    request.steps = 1;

    CallbackState state = {0};
    int rc = dsasm_engine_render(engine, &request, on_pcm, &state);
    size_t expected = FRAMES * dsasm_engine_hop_size(engine);
    const size_t expected_callbacks = 1 + (FRAMES - 1) / (32 - request.overlap_frames);
    if (rc || state.next_offset != expected || state.callbacks != expected_callbacks || state.finals != 1) {
        fprintf(stderr, "render rc=%d error=%s samples=%llu/%zu callbacks=%zu finals=%zu\n",
            rc, dsasm_engine_last_error(engine),
            (unsigned long long)state.next_offset, expected, state.callbacks, state.finals);
        return 5;
    }
    request.mode = DSASM_MODE_BLOCK_BATCH;
    rc = dsasm_engine_render(engine, &request, on_pcm, &state);
    if (rc != DSASM_E_INVALID) {
        fprintf(stderr, "mismatched mode rc=%d error=%s\n", rc,
            dsasm_engine_last_error(engine));
        return 6;
    }
    printf("DSASM stream: callbacks=%zu samples=%zu final=%zu OK\n",
        state.callbacks, expected, state.finals);
    free(speaker);free(f0);dsasm_engine_destroy(engine);return 0;
}
