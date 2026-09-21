#ifndef DSASM_KERNELS_H
#define DSASM_KERNELS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void ds_fused_linear_softsign_glu_f32_avx2(
    const float *x, const float *w_left, const float *w_gate,
    const float *b_left, const float *b_gate, float *y,
    size_t M, size_t N, size_t K);

void ds_fused_linear_softsign_glu_f32_avx2_packed4(
    const float *x, const float *w_packed,
    const float *b_left, const float *b_gate, float *y,
    size_t M, size_t N, size_t K);

void ds_linear_f32_avx2_packed4(
    const float *x, const float *w_packed, const float *bias, float *y,
    size_t M, size_t N, size_t K);

void ds_linear_residual_f32_avx2_packed4(
    const float *x, const float *w_packed, const float *bias,
    const float *residual, float *y,
    size_t M, size_t N, size_t K);

void ds_fused_linear_softsign_glu_f32_avx2_m4n8(
    const float *x, const float *w_packed,
    const float *b_left, const float *b_gate, float *y,
    size_t M, size_t N, size_t K);

void ds_linear_f32_avx2_m4n16(
    const float *x, const float *w_packed, const float *bias, float *y,
    size_t M, size_t N, size_t K);

/* M8 strided-N tile entry: N is the tile width, row_stride is the full output
   row width. w_packed/bias/y may point at an N-tile offset. */
void ds_linear_f32_avx2_m4n16_strided(
    const float *x, const float *w_packed, const float *bias, float *y,
    size_t M, size_t N, size_t K, size_t row_stride);

/* M13 front-end-reduced variant: same packed16 ABI and bit-exact arithmetic,
   but all four input rows share one K byte offset instead of incrementing four
   row pointers every inner-loop iteration. */
void ds_linear_f32_avx2_m4n16_idxstrided(
    const float *x, const float *w_packed, const float *bias, float *y,
    size_t M, size_t N, size_t K, size_t row_stride);

/* M15 K-block cache-local primitive: one packed16 output block, partial K
   accumulation. x may point at x[:,k0]; K_stride remains the full input row
   stride. init!=0 starts from bias, init==0 resumes from y. */
void ds_linear_f32_avx2_n16_kblock_accum(
    const float *x, const float *w_chunk, const float *bias, float *y,
    size_t M, size_t K_stride, size_t K_len, size_t y_stride, int init);

void ds_linear_residual_f32_avx2_m4n8(
    const float *x, const float *w_packed, const float *bias,
    const float *residual, float *y, size_t M, size_t N, size_t K);

void ds_linear_residual_f32_avx2_m4n16(
    const float *x, const float *w_packed, const float *bias,
    const float *residual, float *y, size_t M, size_t N, size_t K);

void ds_linear_residual_f32_avx2_m4n16_strided(
    const float *x, const float *w_packed, const float *bias,
    const float *residual, float *y, size_t M, size_t N, size_t K,
    size_t row_stride);

void ds_linear_residual_f32_avx2_m4n16_idxstrided(
    const float *x, const float *w_packed, const float *bias,
    const float *residual, float *y, size_t M, size_t N, size_t K,
    size_t row_stride);

void ds_layernorm_f32_avx2(
    const float *x, const float *gamma, const float *beta, float *y,
    size_t M, size_t K, float eps);

void ds_layernorm_f32_avx2_precise(
    const float *x, const float *gamma, const float *beta, float *y,
    size_t M, size_t K, float eps);

void ds_depthwise_conv1d_k31_f32_avx2(
    const float *x_padded, const float *weight, const float *bias, float *y,
    size_t C, size_t T);

void ds_depthwise_conv1d_k31_tc_f32_avx2(
    const float *x, const float *weight_tap_major, const float *bias, float *y,
    size_t T, size_t C);

/* M20 ConvNeXt1D support: native [T,C] depthwise Conv1d(k=7,pad=3). */
void ds_depthwise_conv1d_k7_tc_f32_avx2(
    const float *x, const float *weight_tap_major, const float *bias, float *y,
    size_t T, size_t C);

/* M11 channel-tiled depthwise entry. x/weight/bias/y may point at a channel
   offset; full_C preserves the original [T,C] row and tap stride. */
void ds_depthwise_conv1d_k31_tc_f32_avx2_cstrided(
    const float *x, const float *weight_tap_major, const float *bias, float *y,
    size_t T, size_t C_tile, size_t full_C);

void ds_depthwise_conv1d_k31_prelu_f32_avx2(
    const float *x_padded, const float *weight, const float *bias,
    const float *slope, float *y, size_t C, size_t T);

/* M7 official-LYNXNet2 support. */
void ds_atan_glu_f32_avx2(
    const float *x_left_gate, float *y, size_t M, size_t N);
void ds_atan_glu_f32_avx2_ystrided(
    const float *x_left_gate, float *y, size_t M, size_t N, size_t y_stride);
float ds_dot_f32_avx2_fma(const float *a, const float *b, size_t n);

void ds_convtranspose1d_oc_f32_avx2(
    const float *x, const float *w_oc, float bias, float *y,
    size_t Cin, size_t K, size_t Tin, size_t Tout, size_t pad, size_t stride);

/* M34 specialized late HiFi-GAN upsampler: stride=2,K=4,pad=1. */
void ds_convtranspose1d_s2k4_oc_f32_avx2(
    const float *x, const float *w_oc, float bias, float *y,
    size_t Cin, size_t Tin);

/* M42 specialized early HiFi-GAN upsampler: stride=8,K=16,pad=4. */
void ds_convtranspose1d_s8k16_oc_f32_avx2(
    const float *x, const float *w_oc, float bias, float *y,
    size_t Cin, size_t Tin);

void ds_leaky_relu_f32_avx2(const float *x, float *y, size_t n, float alpha);
void ds_add_f32_avx2(const float *a, const float *b, float *y, size_t n);

void ds_conv1d_nct_f32_avx2_oc4_t8(
    const float *x_padded, const float *w_packed4, const float *bias, float *y,
    size_t Cin, size_t Cout, size_t K, size_t Tout, size_t Tp, size_t dilation);

void ds_conv1d_nct_f32_avx2_oc8_t8(
    const float *x_padded, const float *w_packed8, const float *bias, float *y,
    size_t Cin, size_t Cout, size_t K, size_t Tout, size_t Tp, size_t dilation);

/* M37: one packed oc8 block over a time subrange; enables OC x time 2-D jobs. */
void ds_conv1d_nct_f32_avx2_oc8_t8_range(
    const float *x_padded, const float *w_block8, const float *bias8, float *y_block,
    size_t Cin, size_t K, size_t t0, size_t tcount, size_t Tp,
    size_t dilation, size_t y_stride);

/* M39 fixed-kernel-size HiFi-GAN Conv1d. These entry points are selected only
   for 8-aligned real vocoder shapes and preserve the generic accumulation order. */
void ds_conv1d_nct_f32_avx2_oc4_t16_k3(
    const float*,const float*,const float*,float*,size_t,size_t,size_t,size_t,size_t,size_t);
void ds_conv1d_nct_f32_avx2_oc4_t24_k3(
    const float*,const float*,const float*,float*,size_t,size_t,size_t,size_t,size_t,size_t);
void ds_conv1d_nct_f32_avx2_oc8_t8_k3(
    const float*,const float*,const float*,float*,size_t,size_t,size_t,size_t,size_t,size_t);
void ds_conv1d_nct_f32_avx2_oc8_t8_k7(
    const float*,const float*,const float*,float*,size_t,size_t,size_t,size_t,size_t,size_t);
void ds_conv1d_nct_f32_avx2_oc4_t24_k7(
    const float*,const float*,const float*,float*,size_t,size_t,size_t,size_t,size_t,size_t);
void ds_conv1d_nct_f32_avx2_oc4_t24_k11(
    const float*,const float*,const float*,float*,size_t,size_t,size_t,size_t,size_t,size_t);
/* M49: selective residual-store variants of the proven full t24 kernels. */
void ds_conv1d_nct_f32_avx2_oc4_t24_k3_residual(
    const float*,const float*,const float*,float*,size_t,size_t,size_t,size_t,size_t,size_t,const float*);
void ds_conv1d_nct_f32_avx2_oc4_t24_k7_residual(
    const float*,const float*,const float*,float*,size_t,size_t,size_t,size_t,size_t,size_t,const float*);
void ds_conv1d_nct_f32_avx2_oc4_t24_k11_residual(
    const float*,const float*,const float*,float*,size_t,size_t,size_t,size_t,size_t,size_t,const float*);
void ds_conv1d_nct_f32_avx2_oc4_t24_range_k7(
    const float*,const float*,const float*,float*,size_t,size_t,size_t,size_t,size_t,size_t,size_t);
void ds_conv1d_nct_f32_avx2_oc4_t24_range_k11(
    const float*,const float*,const float*,float*,size_t,size_t,size_t,size_t,size_t,size_t,size_t);
void ds_conv1d_nct_f32_avx2_oc8_t8_k11(
    const float*,const float*,const float*,float*,size_t,size_t,size_t,size_t,size_t,size_t);
void ds_conv1d_nct_f32_avx2_oc8_t8_range_k3(
    const float*,const float*,const float*,float*,size_t,size_t,size_t,size_t,size_t,size_t,size_t);
void ds_conv1d_nct_f32_avx2_oc8_t8_range_k7(
    const float*,const float*,const float*,float*,size_t,size_t,size_t,size_t,size_t,size_t,size_t);
void ds_conv1d_nct_f32_avx2_oc8_t8_range_k11(
    const float*,const float*,const float*,float*,size_t,size_t,size_t,size_t,size_t,size_t,size_t);


/* M38: oc8 Conv1d variants with residual Add fused into the store path. */
void ds_conv1d_nct_f32_avx2_oc8_t8_residual(
    const float *x_padded, const float *w_packed8, const float *bias, float *y,
    size_t Cin, size_t Cout, size_t K, size_t Tout, size_t Tp, size_t dilation,
    const float *residual);
void ds_conv1d_nct_f32_avx2_oc8_t8_range_residual(
    const float *x_padded, const float *w_block8, const float *bias8, float *y_block,
    size_t Cin, size_t K, size_t t0, size_t tcount, size_t Tp,
    size_t dilation, size_t y_stride, const float *residual_block);

/* M39 AVX-VNNI U8xS8 feasibility kernel for quantized Conv1d. */
void ds_vnni_pack_u8_4x8_avx2(
    const unsigned char *qpad, unsigned char *xpack, const int *offset4,
    size_t K4, size_t tb0, size_t tb1);

void ds_vnni_conv1d_u8s8_t8_oc8(
    const unsigned char *xpack, const signed char *wpack, const int *corr,
    const float *scale, const float *bias, float *y,
    size_t Tblocks, size_t K4, size_t Cout, size_t Tout);

void ds_leaky_copy_nct_f32_avx2(
    const float *x, float *dst, size_t C, size_t T, size_t Tp, size_t pad,
    float alpha);

void ds_add3_broadcast_f32_avx2(
    const float *a, const float *b, const float *c, float *y,
    size_t M, size_t N);

#ifdef __cplusplus
}
#endif
#endif
