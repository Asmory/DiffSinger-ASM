#include "dsasm_reflow.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

static int reflow_valid(const DSAsmLynxNet2Weights *w, size_t frames) {
    return w && frames && w->input_dim && w->channels;
}

size_t ds_reflow_euler_workspace_floats(const DSAsmLynxNet2Weights *w, size_t frames) {
    if (!reflow_valid(w, frames)) return 0;
    const size_t den_ws = ds_lynxnet2_workspace_floats(w, frames);
    if (!den_ws) return 0;
    return frames * (size_t)w->channels + frames * (size_t)w->input_dim + den_ws;
}

static int range_valid(const float *lo, const float *hi, size_t range_dims, size_t dims) {
    if (!lo || !hi || (range_dims != 1 && range_dims != dims)) return 0;
    for (size_t d = 0; d < range_dims; ++d) {
        if (!(hi[d] > lo[d]) || !isfinite(lo[d]) || !isfinite(hi[d])) return 0;
    }
    return 1;
}

int ds_reflow_norm_spec_f32(
    const float *x, const float *lo, const float *hi,
    size_t range_dims, float *y, size_t frames, size_t dims) {
    if (!x || !y || !frames || !dims || !range_valid(lo, hi, range_dims, dims)) return -1;
    for (size_t t = 0; t < frames; ++t) {
        for (size_t d = 0; d < dims; ++d) {
            const size_t r = range_dims == 1 ? 0 : d;
            const float z = (x[t*dims+d] - lo[r]) / (hi[r] - lo[r]);
            y[t*dims+d] = z * 2.0f - 1.0f;
        }
    }
    return 0;
}

int ds_reflow_denorm_spec_f32(
    const float *x, const float *lo, const float *hi,
    size_t range_dims, float *y, size_t frames, size_t dims) {
    if (!x || !y || !frames || !dims || !range_valid(lo, hi, range_dims, dims)) return -1;
    for (size_t t = 0; t < frames; ++t) {
        for (size_t d = 0; d < dims; ++d) {
            const size_t r = range_dims == 1 ? 0 : d;
            const float z = (x[t*dims+d] + 1.0f) * 0.5f;
            y[t*dims+d] = z * (hi[r] - lo[r]) + lo[r];
        }
    }
    return 0;
}

static int sample_cached(
    const DSAsmLynxNet2Weights *w,
    const float *noise,
    const float *src,
    const float *cond_cache,
    float t_start,
    float time_scale_factor,
    size_t steps,
    float *x,
    float *workspace,
    size_t frames,
    DSAsmThreadPool *pool) {
    if (!reflow_valid(w, frames) || !cond_cache || !x || !workspace) return -1;
    if (!isfinite(t_start) || !isfinite(time_scale_factor) || t_start < 0.0f) return -1;
    if (t_start > 1.0f) t_start = 1.0f;
    if (t_start < 1.0f && !noise) return -1;
    if (t_start > 0.0f && !src) return -1;

    const size_t elems = frames * (size_t)w->input_dim;
    if (t_start <= 0.0f) {
        if (x != noise) memcpy(x, noise, elems * sizeof(float));
    } else if (t_start >= 1.0f) {
        if (x != src) memcpy(x, src, elems * sizeof(float));
        return 0;
    } else {
        const float a = t_start;
        const float b = 1.0f - t_start;
        for (size_t i = 0; i < elems; ++i) {
            /* Keep the two multiplies and add as separate IEEE-754 operations,
               matching PyTorch's scalar-tensor expression rather than FMA. */
            const float xs = src[i] * a;
            const float xn = noise[i] * b;
            x[i] = xs + xn;
        }
    }

    if (steps == 0 || t_start >= 1.0f) return 0;
    const float dt = (1.0f - t_start) / (float)(steps ? steps : 1u);

    /* Full M18 workspace begins with [T,C] conditioner cache.  The cached
       entry point intentionally skips that region so both APIs share a single
       workspace size/layout. */
    float *velocity = workspace + frames * (size_t)w->channels;
    float *den_ws = velocity + elems;

    for (size_t i = 0; i < steps; ++i) {
        const float t = t_start + (float)i * dt;
        const float timestep = time_scale_factor * t;
        int rc;
        if (pool) {
            rc = ds_lynxnet2_forward_cached_condition_parallel_f32_avx2(
                w, x, cond_cache, timestep, velocity, den_ws, frames, pool);
        } else {
            rc = ds_lynxnet2_forward_cached_condition_f32_avx2(
                w, x, cond_cache, timestep, velocity, den_ws, frames);
        }
        if (rc) return rc;
        for (size_t j = 0; j < elems; ++j) {
            const float delta = velocity[j] * dt;
            x[j] = x[j] + delta;
        }
    }
    return 0;
}

int ds_reflow_euler_sample_cached_condition_f32_avx2(
    const DSAsmLynxNet2Weights *w,
    const float *noise,
    const float *src_norm,
    const float *cond_cache,
    float t_start,
    float time_scale_factor,
    size_t steps,
    float *out,
    float *workspace,
    size_t frames,
    DSAsmThreadPool *pool) {
    return sample_cached(w, noise, src_norm, cond_cache, t_start, time_scale_factor,
                         steps, out, workspace, frames, pool);
}

int ds_reflow_euler_sample_f32_avx2(
    const DSAsmLynxNet2Weights *w,
    const float *noise,
    const float *src_norm,
    const float *condition,
    float t_start,
    float time_scale_factor,
    size_t steps,
    float *out,
    float *workspace,
    size_t frames,
    DSAsmThreadPool *pool) {
    if (!reflow_valid(w, frames) || !workspace) return -1;
    if (t_start >= 1.0f || steps == 0) {
        /* Upstream performs no velocity_fn call in these cases, so the
           conditioner projection is unnecessary as well. */
        return sample_cached(w, noise, src_norm, workspace, t_start, time_scale_factor,
                             steps, out, workspace, frames, pool);
    }
    if (!condition) return -1;
    float *cond_cache = workspace;
    int rc;
    if (pool) {
        rc = ds_lynxnet2_prepare_condition_parallel_f32_avx2(
            w, condition, cond_cache, frames, pool);
    } else {
        rc = ds_lynxnet2_prepare_condition_f32_avx2(w, condition, cond_cache, frames);
    }
    if (rc) return rc;
    return sample_cached(w, noise, src_norm, cond_cache, t_start, time_scale_factor,
                         steps, out, workspace, frames, pool);
}
