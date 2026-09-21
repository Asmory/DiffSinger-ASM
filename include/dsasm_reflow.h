#ifndef DSASM_REFLOW_H
#define DSASM_REFLOW_H

#include <stddef.h>
#include "dsasm_lynxnet2.h"

#ifdef __cplusplus
extern "C" {
#endif

/* M18: OpenVPI RectifiedFlow Euler inference, normalized-domain core.
   Tensor layout is [T,D] (frames-major), equivalent to the acoustic model's
   [B=1,F=1,M,T] after the Python transposes.  Noise is supplied by the caller
   so RNG policy is independent from the sampler and parity tests are exact. */

/* Workspace contains one conditioner cache [T,C], one velocity buffer [T,D],
   and the existing LYNXNet2 forward workspace. */
size_t ds_reflow_euler_workspace_floats(
    const DSAsmLynxNet2Weights *w, size_t frames);

/* Normalize/denormalize the acoustic spec exactly as RectifiedFlow:
     norm   = (x-min)/(max-min)*2 - 1
     denorm = (x+1)/2*(max-min) + min
   range_dims must be 1 (broadcast scalar range) or dims (per-bin range). */
int ds_reflow_norm_spec_f32(
    const float *x_tc, const float *spec_min, const float *spec_max,
    size_t range_dims, float *y_tc, size_t frames, size_t dims);
int ds_reflow_denorm_spec_f32(
    const float *x_tc, const float *spec_min, const float *spec_max,
    size_t range_dims, float *y_tc, size_t frames, size_t dims);

/* Cached-condition Euler sampler. src_norm_tc may be NULL only when
   t_start==0.  For shallow diffusion (0<t_start<1):
      x = t_start*src + (1-t_start)*noise
   For t_start>=1, x=src and no denoiser step is run.
   For each Euler step:
      dt=(1-t_start)/max(1,steps)
      v=velocity_fn(x, time_scale_factor*(t_start+i*dt), cond)
      x += v*dt
   output_norm_tc may alias noise_tc or src_norm_tc. */
int ds_reflow_euler_sample_cached_condition_f32_avx2(
    const DSAsmLynxNet2Weights *w,
    const float *noise_tc,
    const float *src_norm_tc,
    const float *condition_projected_tc,
    float t_start,
    float time_scale_factor,
    size_t steps,
    float *output_norm_tc,
    float *workspace,
    size_t frames,
    DSAsmThreadPool *pool);

/* Convenience wrapper that projects condition once, caches it in workspace,
   then executes the same Euler loop. */
int ds_reflow_euler_sample_f32_avx2(
    const DSAsmLynxNet2Weights *w,
    const float *noise_tc,
    const float *src_norm_tc,
    const float *condition_tc,
    float t_start,
    float time_scale_factor,
    size_t steps,
    float *output_norm_tc,
    float *workspace,
    size_t frames,
    DSAsmThreadPool *pool);

#ifdef __cplusplus
}
#endif
#endif
