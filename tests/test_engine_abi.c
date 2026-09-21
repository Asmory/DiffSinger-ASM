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
    puts("DSASM engine ABI v1: OK");
    return 0;
}
