#include "dsasm_engine.h"
#include <stddef.h>
#include <stdio.h>

int main(int argc, char **argv) {
    dsasm_request request = {0};
    request.struct_size = sizeof(request);
    request.abi_version = DSASM_ENGINE_ABI_VERSION;
    if (dsasm_engine_abi_version() != DSASM_ENGINE_ABI_VERSION) return 1;
    if (request.struct_size != sizeof(request)) return 2;
    if (request.abi_version != DSASM_ENGINE_ABI_VERSION) return 3;
    if (offsetof(dsasm_request, token_ids) % sizeof(void *) != 0) return 4;
    if (sizeof(dsasm_request) != 200) return 8;
    if (sizeof(dsasm_mode_config) != 36) return 9;
    dsasm_mode_config realtime = {
        sizeof(realtime), DSASM_ENGINE_ABI_VERSION, 0, 0, 0, 0, 0, 0, 0
    };
    dsasm_mode_config batch = {
        sizeof(batch), DSASM_ENGINE_ABI_VERSION, 0, 0, 0, 0, 0, 0, 0
    };
    if (dsasm_engine_mode_config(DSASM_MODE_REALTIME_STREAMING, &realtime) != DSASM_OK ||
            realtime.mode != DSASM_MODE_REALTIME_STREAMING ||
            realtime.workers != 4 || realtime.region_frames != 32 ||
            realtime.vocoder_bucket_frames != 32 || realtime.overlap_frames != 8 ||
            realtime.profile_revision != 2 ||
            realtime.output_compatibility_revision != 0) return 10;
    if (dsasm_engine_mode_config(DSASM_MODE_BLOCK_BATCH, &batch) != DSASM_OK ||
            batch.mode != DSASM_MODE_BLOCK_BATCH || batch.workers != 8 ||
            batch.region_frames != 384 || batch.vocoder_bucket_frames != 384 ||
            batch.overlap_frames != 0 || batch.profile_revision != 2 ||
            batch.output_compatibility_revision != 1) return 11;
    dsasm_request realtime_request = dsasm_request_init_mode(
        DSASM_MODE_REALTIME_STREAMING);
    dsasm_request batch_request = dsasm_request_init_mode(
        DSASM_MODE_BLOCK_BATCH);
    if (realtime_request.mode != realtime.mode ||
            realtime_request.vocoder_bucket_frames != realtime.vocoder_bucket_frames ||
            realtime_request.overlap_frames != realtime.overlap_frames ||
            batch_request.mode != batch.mode ||
            batch_request.vocoder_bucket_frames != batch.vocoder_bucket_frames ||
            batch_request.overlap_frames != batch.overlap_frames) return 12;
    batch.struct_size--;
    if (dsasm_engine_mode_config(DSASM_MODE_BLOCK_BATCH, &batch) != DSASM_E_INVALID) return 13;
    batch.struct_size++;
    batch.abi_version--;
    if (dsasm_engine_mode_config(DSASM_MODE_BLOCK_BATCH, &batch) != DSASM_E_INVALID ||
            dsasm_engine_mode_config((dsasm_engine_mode)0, &realtime) != DSASM_E_UNSUPPORTED) return 14;
    char support_reason[128];
    if (!dsasm_engine_is_supported(support_reason, sizeof(support_reason))) {
        fprintf(stderr, "unsupported runtime: %s\n", support_reason);
        return 5;
    }
    if (argc == 3) {
        dsasm_engine *engine = dsasm_engine_create(argv[1], argv[2], 2);
        if (!engine) {
            fprintf(stderr, "engine create failed: %s\n", dsasm_engine_last_create_error());
            return 6;
        }
        if (dsasm_engine_sample_rate(engine) != 44100 ||
                dsasm_engine_hop_size(engine) != 512 ||
                dsasm_engine_mel_bins(engine) != 128 ||
                dsasm_engine_bucket_count(engine) != 1 ||
                dsasm_engine_bucket_frames(engine, 0) != 384) {
            dsasm_engine_destroy(engine);
            return 7;
        }
        dsasm_engine_destroy(engine);
        puts("DSASM persistent engine load: OK");
    }
    puts("DSASM engine ABI v3 modes: OK");
    return 0;
}
