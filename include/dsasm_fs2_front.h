#ifndef DSASM_FS2_FRONT_H
#define DSASM_FS2_FRONT_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* M20-B exact non-neural FastSpeech2 acoustic front-end primitives, B=1. */
int ds_fs2_mel2ph_to_dur_i32(const int32_t*mel2ph,size_t T_mel,size_t T_txt,int32_t*dur);
int ds_fs2_stretch_f32(const int32_t*mel2ph,size_t T_mel,const int32_t*dur,size_t T_txt,float*out);
int ds_fs2_gather_encoder_f32(const float*encoder_txt,const int32_t*mel2ph,size_t T_mel,size_t T_txt,size_t H,float*condition);
void ds_fs2_pitch_input_f32(const float*f0,size_t T,float*out);
void ds_fs2_duration_input_f32(const int32_t*dur,size_t T_txt,float*out);
#ifdef __cplusplus
}
#endif
#endif
