#include "dsasm_acoustic.h"

#include <math.h>
#include <stddef.h>

static int acoustic_valid(const DSAsmLynxNet2Weights *w, size_t frames) {
    return w && frames && w->input_dim && w->condition_dim && w->channels;
}

static int range_valid(const float *lo, const float *hi, size_t range_dims, size_t dims) {
    if (!lo || !hi || (range_dims != 1 && range_dims != dims)) return 0;
    for (size_t d = 0; d < range_dims; ++d) {
        if (!(hi[d] > lo[d]) || !isfinite(lo[d]) || !isfinite(hi[d])) return 0;
    }
    return 1;
}

size_t ds_acoustic_reflow_workspace_floats(
    const DSAsmLynxNet2Weights *w, size_t frames) {
    if (!acoustic_valid(w, frames)) return 0;
    const size_t rf = ds_reflow_euler_workspace_floats(w, frames);
    if (!rf) return 0;
    return frames * (size_t)w->input_dim + rf;
}

int ds_acoustic_reflow_decode_f32_avx2(
    const DSAsmLynxNet2Weights *w,
    const float *condition_tc,
    const float *aux_mel_tc,
    const float *noise_tc,
    const float *frame_mask_t,
    const float *spec_min,
    const float *spec_max,
    size_t range_dims,
    float t_start,
    float time_scale_factor,
    size_t steps,
    float *output_mel_tc,
    float *workspace,
    size_t frames,
    DSAsmThreadPool *pool) {
    if (!acoustic_valid(w, frames) || !output_mel_tc || !workspace) return -1;
    const size_t dims = (size_t)w->input_dim;
    const size_t elems = frames * dims;
    if (!range_valid(spec_min, spec_max, range_dims, dims)) return -1;
    if (!isfinite(t_start) || !isfinite(time_scale_factor) || t_start < 0.0f) return -1;
    if (t_start > 1.0f) t_start = 1.0f;
    if (t_start < 1.0f && !noise_tc) return -1;
    if (t_start > 0.0f && !aux_mel_tc) return -1;
    if (t_start < 1.0f && steps > 0 && !condition_tc) return -1;

    float *src_norm = workspace;
    float *rf_ws = src_norm + elems;
    const float *src_ptr = NULL;

    if (t_start > 0.0f) {
        /* Upstream masks aux_mel in raw mel space before RectifiedFlow.norm_spec. */
        for (size_t t = 0; t < frames; ++t) {
            const float m = frame_mask_t ? frame_mask_t[t] : 1.0f;
            if (!isfinite(m)) return -1;
            for (size_t d = 0; d < dims; ++d) {
                const size_t i = t * dims + d;
                const size_t r = range_dims == 1 ? 0 : d;
                const float masked = aux_mel_tc[i] * m;
                const float z = (masked - spec_min[r]) / (spec_max[r] - spec_min[r]);
                src_norm[i] = z * 2.0f - 1.0f;
            }
        }
        src_ptr = src_norm;
    }

    int rc = ds_reflow_euler_sample_f32_avx2(
        w, noise_tc, src_ptr, condition_tc,
        t_start, time_scale_factor, steps,
        output_mel_tc, rf_ws, frames, pool);
    if (rc) return rc;

    rc = ds_reflow_denorm_spec_f32(
        output_mel_tc, spec_min, spec_max, range_dims,
        output_mel_tc, frames, dims);
    if (rc) return rc;

    /* Upstream masks diffusion output again after denormalization. */
    if (frame_mask_t) {
        for (size_t t = 0; t < frames; ++t) {
            const float m = frame_mask_t[t];
            for (size_t d = 0; d < dims; ++d) {
                output_mel_tc[t*dims+d] = output_mel_tc[t*dims+d] * m;
            }
        }
    }
    return 0;
}

size_t ds_acoustic_reflow_normsrc_workspace_floats(
    const DSAsmLynxNet2Weights *w, size_t frames) {
    return ds_reflow_euler_workspace_floats(w, frames);
}

int ds_acoustic_reflow_decode_normsrc_f32_avx2(
    const DSAsmLynxNet2Weights *w,
    const float *condition_tc,const float *aux_norm_tc,const float *noise_tc,
    const float *frame_mask_t,const float *spec_min,const float *spec_max,size_t range_dims,
    float t_start,float time_scale_factor,size_t steps,float *output_mel_tc,
    float *workspace,size_t frames,DSAsmThreadPool *pool) {
    if (!acoustic_valid(w, frames) || !output_mel_tc || !workspace) return -1;
    const size_t dims=(size_t)w->input_dim, elems=frames*dims;
    if (!range_valid(spec_min,spec_max,range_dims,dims)) return -1;
    if (!isfinite(t_start)||!isfinite(time_scale_factor)||t_start<0.0f) return -1;
    if (t_start>1.0f) t_start=1.0f;
    if (t_start<1.0f && !noise_tc) return -1;
    if (t_start>0.0f && !aux_norm_tc) return -1;
    if (t_start<1.0f && steps>0 && !condition_tc) return -1;

    const float *src_ptr=NULL;
    if (t_start>0.0f) {
        for (size_t t=0;t<frames;++t) {
            const float m=frame_mask_t?frame_mask_t[t]:1.0f;
            if(!isfinite(m)) return -1;
            for(size_t d=0;d<dims;++d){
                const size_t i=t*dims+d, r=range_dims==1?0:d;
                const float k=(spec_max[r]-spec_min[r])*0.5f;
                const float b=(spec_max[r]+spec_min[r])*0.5f;
                const float raw=aux_norm_tc[i]*k+b;
                const float masked=raw*m;
                const float z=(masked-spec_min[r])/(spec_max[r]-spec_min[r]);
                output_mel_tc[i]=z*2.0f-1.0f;
            }
        }
        src_ptr=output_mel_tc;
    }
    int rc=ds_reflow_euler_sample_f32_avx2(
        w,noise_tc,src_ptr,condition_tc,t_start,time_scale_factor,steps,
        output_mel_tc,workspace,frames,pool);
    if(rc)return rc;
    rc=ds_reflow_denorm_spec_f32(output_mel_tc,spec_min,spec_max,range_dims,
        output_mel_tc,frames,dims);
    if(rc)return rc;
    if(frame_mask_t){
        for(size_t t=0;t<frames;++t){
            const float m=frame_mask_t[t];
            for(size_t d=0;d<dims;++d) output_mel_tc[t*dims+d]*=m;
        }
    }
    (void)elems;
    return 0;
}
