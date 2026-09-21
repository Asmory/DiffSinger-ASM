#ifndef DSASM_MODEL_H
#define DSASM_MODEL_H
#include <stddef.h>
#include <stdint.h>
#include "dsasm_fs2_encoder.h"
#include "dsasm_aux_decoder.h"
#include "dsasm_lynxnet2.h"
#include "dsasm_threadpool.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    DSAsmFS2AcousticWeights fs2;
    DSAsmFS2DeploymentExtras fs2_extras;
    DSAsmAuxConvNeXtWeights aux;
    DSAsmLynxNet2Weights rf;

    /* Loader-owned state. Treat as opaque. */
    void *fs2_map; size_t fs2_map_bytes;
    void *aux_map; size_t aux_map_bytes;
    void *rf_map;  size_t rf_map_bytes;
    DSAsmFS2EncoderLayer *fs2_layers_owned;
    DSAsmAuxConvNeXtBlock *aux_blocks_owned;
    DSAsmLynxNet2Block *rf_blocks_owned;
} DSAsmAcousticModel;

/* Load three packed runtime bundles produced by:
   pack_fs2_acoustic_checkpoint.py / ONNX M25 importer -> DSFS21 or DSFS25
   pack_aux_convnext_checkpoint.py  -> DSAUX20
   pack_lynxnet2_backbone.py        -> DSLYNX7
   Bundle payloads are mmap'd MAP_PRIVATE and remain valid until unload. */
int ds_acoustic_model_load(
    DSAsmAcousticModel *m,
    const char *fs2_path,
    const char *aux_path,
    const char *rf_path);

void ds_acoustic_model_unload(DSAsmAcousticModel *m);

/* Cross-bundle dimensional sanity check. */
int ds_acoustic_model_valid(const DSAsmAcousticModel *m);

size_t ds_acoustic_model_workspace_floats(
    const DSAsmAcousticModel *m,size_t text_tokens,size_t mel_frames);

/* Unified default-profile acoustic inference entry point. */
int ds_acoustic_model_infer_f32_avx2(
    const DSAsmAcousticModel *m,
    const int32_t *token_ids,size_t text_tokens,
    const int32_t *mel2ph,const float *f0,size_t mel_frames,
    const float *noise_tc,
    const float *spec_min,const float *spec_max,size_t range_dims,
    float t_start,float time_scale_factor,size_t steps,
    float *output_mel_tc,float *workspace,DSAsmThreadPool *pool);

int ds_acoustic_model_infer_deploy_f32_avx2(
    const DSAsmAcousticModel *m,const DSAsmFS2DeploymentInputs *inputs,
    const int32_t *token_ids,size_t text_tokens,
    const int32_t *mel2ph,const float *f0,size_t mel_frames,
    const float *noise_tc,
    const float *spec_min,const float *spec_max,size_t range_dims,
    float t_start,float time_scale_factor,size_t steps,
    float *output_mel_tc,float *workspace,DSAsmThreadPool *pool);

#ifdef __cplusplus
}
#endif
#endif
