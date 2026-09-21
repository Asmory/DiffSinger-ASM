#ifndef DSASM_THREADPOOL_INTERNAL_H
#define DSASM_THREADPOOL_INTERNAL_H
#include "dsasm_threadpool.h"
#include <stddef.h>

typedef enum {
    DS_JOB_LINEAR=1,
    DS_JOB_LINEAR_RESIDUAL=2,
    DS_JOB_SOFTSIGN_GLU=3,
    /* M9: packed Linear(2N,K) followed immediately by ATanGLU into [M,N]. */
    DS_JOB_ATAN_GLU_LINEAR=4,
    /* M11: channel-parallel native [T,C] depthwise k31. */
    DS_JOB_DEPTHWISE_K31=5,
    /* M32: pure-ASM NSF-HiFiGAN group=1 stride=1 Conv1d. */
    DS_JOB_VOCODER_CONV1D=6,
    DS_JOB_VOCODER_CONVTRANSPOSE=7,
    DS_JOB_VOCODER_VNNI_PACK=8,
    DS_JOB_VOCODER_VNNI_CONV=9,
    /* M43: parallel same-shape Add for vocoder residual merges. */
    DS_JOB_ADD_F32=10,
    /* M55: channel-parallel leaky+pad copy for long vocoder Conv inputs. */
    DS_JOB_LEAKY_COPY_NCT=11
} DSJobKind;
typedef void (*DSVnniKernelFn)(
    const unsigned char*,const signed char*,const int*,const float*,const float*,float*,
    size_t,size_t,size_t,size_t);
typedef struct {
    DSJobKind kind;
    const float *x,*w,*b0,*b1,*residual;
    float *y;
    size_t M,N,K;
    /* Optional auxiliary integer fields for specialized jobs. */
    size_t P,Q,R,S,T;
    DSVnniKernelFn vnni_fn;
} DSAsmJob;
void ds_threadpool_run(DSAsmThreadPool *pool, const DSAsmJob *job);
#endif
