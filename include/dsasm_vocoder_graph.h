#ifndef DSASM_VOCODER_GRAPH_H
#define DSASM_VOCODER_GRAPH_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct DSAsmVocoderGraph DSAsmVocoderGraph;

DSAsmVocoderGraph *ds_vocoder_graph_load(const char *path, size_t workers);
void ds_vocoder_graph_free(DSAsmVocoderGraph *g);
size_t ds_vocoder_graph_frames(const DSAsmVocoderGraph *g);
size_t ds_vocoder_graph_mel_bins(const DSAsmVocoderGraph *g);
size_t ds_vocoder_graph_samples(const DSAsmVocoderGraph *g);
size_t ds_vocoder_graph_workers(const DSAsmVocoderGraph *g);
int ds_vocoder_graph_infer(DSAsmVocoderGraph *g, const float *mel_tc, const float *f0_t, float *waveform, int profile);

#ifdef __cplusplus
}
#endif
#endif
