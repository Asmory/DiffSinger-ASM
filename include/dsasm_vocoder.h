#ifndef DSASM_VOCODER_H
#define DSASM_VOCODER_H
#include <stddef.h>
#include "dsasm_threadpool.h"
#ifdef __cplusplus
extern "C" {
#endif

/* Pure CPU/ASM Conv1d primitive used by the native NSF-HiFiGAN port.
   x: [Cin,Tin] NCT without batch; y: [Cout,Tout].
   w_packed4: [Cout/4,Cin,K,4], group=1, stride=1.
   workspace must contain at least Cin*(Tin+2*pad) floats. */
size_t ds_vocoder_conv1d_workspace_floats(size_t Cin, size_t Tin, size_t pad);

/* M36 graph-executor Conv1d. pack_width is 4 or 8. When fuse_leaky is non-zero,
   LeakyReLU(alpha) is applied while copying x into the padded workspace. */
int ds_vocoder_conv1d_ex_f32_avx2(
    const float *x, const float *w_packed, const float *bias, float *y,
    size_t Cin, size_t Cout, size_t K, size_t Tin, size_t pad, size_t dilation,
    size_t pack_width, int fuse_leaky, float alpha, float *workspace,
    DSAsmThreadPool *pool);


/* M38 graph-executor variant: optional residual [Cout,Tout] is accumulated
   directly into the oc8 ASM store path. residual must be NULL for pack4. */
int ds_vocoder_conv1d_ex_residual_f32_avx2(
    const float *x, const float *w_packed, const float *bias,
    const float *residual, float *y,
    size_t Cin, size_t Cout, size_t K, size_t Tin, size_t pad, size_t dilation,
    size_t pack_width, int fuse_leaky, float alpha, float *workspace,
    DSAsmThreadPool *pool);

int ds_vocoder_vnni_available(void);
int ds_vocoder_conv1d_vnni_u8s8(
    const float *x,const unsigned char *blob,const float *bias,float *y,
    size_t Cin,size_t Cout,size_t K,size_t Tin,size_t pad,size_t dilation,
    int fuse_leaky,float alpha,unsigned char *qx,unsigned char *xpack,float *scales,
    DSAsmThreadPool *pool,double *pack_ms,double *kernel_ms);

int ds_vocoder_conv1d_f32_avx2(
    const float *x, const float *w_packed4, const float *bias, float *y,
    size_t Cin, size_t Cout, size_t K, size_t Tin,
    size_t pad, size_t dilation, float *workspace,
    DSAsmThreadPool *pool);

/* Direct sparse ConvTranspose1d for NSF-HiFiGAN upsamplers. The offline
   packer transposes ONNX [Cin,Cout,K] to output-major [Cout,Cin,K] but does
   not insert zeros or flip taps. The ASM kernel visits only real input
   samples and the K outputs they actually affect. group=1, dilation=1,
   symmetric padding, output_padding=0 are supported. */
int ds_vocoder_convtranspose1d_f32_avx2(
    const float *x, const float *w_oc_major, const float *bias, float *y,
    size_t Cin, size_t Cout, size_t K, size_t Tin,
    size_t pad, size_t stride, DSAsmThreadPool *pool);

/* One HiFi-GAN residual unit:
     LeakyReLU -> Conv1 -> LeakyReLU -> Conv2 -> residual add.
   Both convolutions preserve [C,T]. workspace is returned by the helper. */
size_t ds_vocoder_resunit_workspace_floats(
    size_t C, size_t T, size_t pad1, size_t pad2);
int ds_vocoder_resunit_f32_avx2(
    const float *x,
    const float *w1_packed4, const float *b1, size_t K1, size_t pad1, size_t dil1,
    const float *w2_packed4, const float *b2, size_t K2, size_t pad2, size_t dil2,
    float alpha, float *y, size_t C, size_t T, float *workspace,
    DSAsmThreadPool *pool);

#ifdef __cplusplus
}
#endif
#endif
