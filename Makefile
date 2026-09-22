CC ?= gcc
CXX ?= c++
PYTHON ?= python
CFLAGS ?= -O3 -Wall -Wextra -std=c11 -march=native
CXXFLAGS ?= -O3 -Wall -Wextra -std=c++17 -march=native
CPPFLAGS ?= -Iinclude -Isrc/runtime
LDFLAGS ?= -lm
BUILD := build
DIST := release
KDIR := src/kernels/x86_64
RDIR := src/runtime
VERSION ?= $(shell git describe --tags --always --dirty)

# M31 realtime sprint defaults for the target i5-13420H P-core topology.
# Override on the make command line for another machine.
M31_STEPS ?= 20,16,12,10,8,6,4
M31_VOCODER_THREADS ?= 1,2,4,6,8
M31_CPUS ?= 0,2,4,6,1,3,5,7
M31_ACOUSTIC_ROUNDS ?= 2
M31_VOCODER_ROUNDS ?= 3
M31_PROFILE_RUNS ?= 3

.PHONY: all test bench engine engine-check engine-real-stream-check model-tool-check m6 m7 m7-bench m8 m8-bench m9 m9-bench m9-autotune m10 m10-bench m10-autotune m11 m11-bench m11-autotune m12 m12-bench m12-spin-bench m12-perf m13 m13-bench m13-kernel-bench m13-perf m14 m14-bench m14-perf m14-pytorch-check m15 m15-bench m15-perf m15-kernel-bench m15-pytorch-check m16 m16-bench m16-perf m16-pytorch-check m17 m17-bench m17-perf m17-pytorch-check m18 m18-bench m18-pytorch-check m19 m19-bench m19-pytorch-check m20 m20-bench m20-pytorch-check m20-fs2-front-check m20-postfs2-check m21 m21-bench m21-pytorch-check m21-condition-check m21-full-check m21-pack-check m22 m22-bench m22-check m23 m23-bench m23-check m23-model-check m24 m24-check m24-pack-check m25-deploy-check m29-check m30-check m31 m31-check m31-real-sprint pytorch-check bundle-check m7-pytorch-check m7-bundle-check m8-pytorch-check m9-pytorch-check m10-pytorch-check m11-pytorch-check m12-pytorch-check m13-pytorch-check verify clean m32 m32-check m32-real-kernel m33 m33-check m33-real-blocks m34 m34-check m34-real-ct
.PHONY: package

all: $(BUILD)/test_fused_glu $(BUILD)/test_fused_glu_m2 \
     $(BUILD)/test_depthwise_k31_prelu $(BUILD)/test_m4_lynxnet2_block \
     $(BUILD)/test_m5_lynxnet2_block $(BUILD)/test_glu_m6 \
     $(BUILD)/test_linear_m6 $(BUILD)/test_linear_residual_m6 \
     $(BUILD)/test_m6_lynxnet2_block $(BUILD)/test_m6_bundle \
     $(BUILD)/test_atan_glu_m7 $(BUILD)/test_atan_glu_m9_strided $(BUILD)/test_m8_strided $(BUILD)/test_m11_depthwise_cstrided $(BUILD)/test_m13_indexed $(BUILD)/test_m15_kblock $(BUILD)/test_m7_bundle $(BUILD)/libdsasm_m7.so $(BUILD)/libdsasm_m8.so $(BUILD)/libdsasm_m9.so $(BUILD)/libdsasm_m10.so $(BUILD)/libdsasm_m11.so $(BUILD)/libdsasm_m12.so $(BUILD)/libdsasm_m13.so $(BUILD)/libdsasm_m14.so $(BUILD)/libdsasm_m15.so $(BUILD)/libdsasm_m16.so $(BUILD)/libdsasm_m17.so $(BUILD)/libdsasm_m18.so $(BUILD)/libdsasm_m19.so $(BUILD)/libdsasm_m20.so $(BUILD)/libdsasm_m21.so $(BUILD)/libdsasm_m22.so $(BUILD)/libdsasm_m23.so $(BUILD)/libdsasm_m24.so $(BUILD)/dsasm-acoustic $(BUILD)/test_m23_model_loader

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/fused_glu_m1.o: $(KDIR)/fused_linear_softsign_glu_f32_avx2.S | $(BUILD)
	$(CC) -c $< -o $@
$(BUILD)/fused_glu_m2.o: $(KDIR)/fused_linear_softsign_glu_f32_avx2_packed4.S | $(BUILD)
	$(CC) -c $< -o $@
$(BUILD)/glu_m6.o: $(KDIR)/fused_linear_softsign_glu_f32_avx2_m4n8.S | $(BUILD)
	$(CC) -c $< -o $@
$(BUILD)/atan_glu_m7.o: $(KDIR)/atan_glu_f32_avx2.S | $(BUILD)
	$(CC) -c $< -o $@
$(BUILD)/atan_glu_m9_strided.o: $(KDIR)/atan_glu_f32_avx2_ystrided.S | $(BUILD)
	$(CC) -c $< -o $@
$(BUILD)/add3_m7.o: $(KDIR)/add3_broadcast_f32_avx2.S | $(BUILD)
	$(CC) -c $< -o $@
$(BUILD)/rope_m25.o: $(KDIR)/rope_f32_avx2.S | $(BUILD)
	$(CC) -c $< -o $@
$(BUILD)/silu_glu_m25.o: $(KDIR)/silu_glu_f32_avx2.S | $(BUILD)
	$(CC) -c $< -o $@
$(BUILD)/adaptive_affine_m25.o: $(KDIR)/adaptive_affine_f32_avx2.S | $(BUILD)
	$(CC) -c $< -o $@
$(BUILD)/phoneme_mean_m25.o: $(KDIR)/phoneme_mean_f32_avx2.S | $(BUILD)
	$(CC) -c $< -o $@
$(BUILD)/dwconv_k31_prelu.o: $(KDIR)/depthwise_conv1d_k31_prelu_f32_avx2.S | $(BUILD)
	$(CC) -c $< -o $@
$(BUILD)/linear_packed4.o: $(KDIR)/linear_f32_avx2_packed4.S | $(BUILD)
	$(CC) -c $< -o $@
$(BUILD)/linear_residual_packed4.o: $(KDIR)/linear_residual_f32_avx2_packed4.S | $(BUILD)
	$(CC) -c $< -o $@
$(BUILD)/linear_m4n16.o: $(KDIR)/linear_f32_avx2_m4n16.S | $(BUILD)
	$(CC) -c $< -o $@
$(BUILD)/linear_residual_m4n8.o: $(KDIR)/linear_residual_f32_avx2_m4n8.S | $(BUILD)
	$(CC) -c $< -o $@
$(BUILD)/linear_residual_m4n16.o: $(KDIR)/linear_residual_f32_avx2_m4n16.S | $(BUILD)
	$(CC) -c $< -o $@
$(BUILD)/linear_m4n16_strided.o: $(KDIR)/linear_f32_avx2_m4n16_strided.S | $(BUILD)
	$(CC) -c $< -o $@
$(BUILD)/linear_residual_m4n16_strided.o: $(KDIR)/linear_residual_f32_avx2_m4n16_strided.S | $(BUILD)
	$(CC) -c $< -o $@
$(BUILD)/linear_m4n16_idxstrided.o: $(KDIR)/linear_f32_avx2_m4n16_idxstrided.S | $(BUILD)
	$(CC) -c $< -o $@
$(BUILD)/linear_residual_m4n16_idxstrided.o: $(KDIR)/linear_residual_f32_avx2_m4n16_idxstrided.S | $(BUILD)
	$(CC) -c $< -o $@
$(BUILD)/linear_n16_kblock.o: $(KDIR)/linear_f32_avx2_n16_kblock_accum.S | $(BUILD)
	$(CC) -c $< -o $@
$(BUILD)/layernorm.o: $(KDIR)/layernorm_f32_avx2.S | $(BUILD)
	$(CC) -c $< -o $@
$(BUILD)/layernorm_precise.o: $(KDIR)/layernorm_f32_avx2_precise.S | $(BUILD)
	$(CC) -c $< -o $@
$(BUILD)/dwconv_k31.o: $(KDIR)/depthwise_conv1d_k31_f32_avx2.S | $(BUILD)
	$(CC) -c $< -o $@
$(BUILD)/dwconv_k31_tc.o: $(KDIR)/depthwise_conv1d_k31_tc_f32_avx2.S | $(BUILD)
	$(CC) -c $< -o $@

$(BUILD)/dwconv_k7_tc.o: $(KDIR)/depthwise_conv1d_k7_tc_f32_avx2.S | $(BUILD)
	$(CC) -c $< -o $@
$(BUILD)/dot_avx2.o: $(KDIR)/dot_f32_avx2_fma.S | $(BUILD)
	$(CC) -c $< -o $@
$(BUILD)/dwconv_k31_tc_cstrided.o: $(KDIR)/depthwise_conv1d_k31_tc_f32_avx2_cstrided.S | $(BUILD)
	$(CC) -c $< -o $@

$(BUILD)/lynxnet2_runtime.o: $(RDIR)/lynxnet2.c include/dsasm_lynxnet2.h include/dsasm_kernels.h $(RDIR)/threadpool_internal.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -fPIC -c $< -o $@
$(BUILD)/threadpool.o: $(RDIR)/threadpool.c include/dsasm_threadpool.h include/dsasm_kernels.h $(RDIR)/threadpool_internal.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -fPIC -pthread -c $< -o $@
$(BUILD)/reflow_runtime.o: $(RDIR)/reflow.c include/dsasm_reflow.h include/dsasm_lynxnet2.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -ffp-contract=off -fPIC -c $< -o $@
$(BUILD)/acoustic_runtime.o: $(RDIR)/acoustic.c include/dsasm_acoustic.h include/dsasm_reflow.h include/dsasm_lynxnet2.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -ffp-contract=off -fPIC -c $< -o $@

$(BUILD)/aux_decoder_runtime.o: $(RDIR)/aux_decoder.c include/dsasm_aux_decoder.h include/dsasm_kernels.h $(RDIR)/threadpool_internal.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -fPIC -c $< -o $@
$(BUILD)/post_fs2_runtime.o: $(RDIR)/post_fs2.c include/dsasm_acoustic.h include/dsasm_aux_decoder.h include/dsasm_reflow.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -ffp-contract=off -fPIC -c $< -o $@
$(BUILD)/fs2_front_runtime.o: $(RDIR)/fs2_front.c include/dsasm_fs2_front.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -ffp-contract=off -fPIC -c $< -o $@
$(BUILD)/fs2_encoder_runtime.o: $(RDIR)/fs2_encoder.c include/dsasm_fs2_encoder.h include/dsasm_kernels.h $(RDIR)/threadpool_internal.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -ffp-contract=off -fPIC -c $< -o $@
$(BUILD)/full_acoustic_runtime.o: $(RDIR)/full_acoustic.c include/dsasm_full_acoustic.h include/dsasm_fs2_encoder.h include/dsasm_acoustic.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -ffp-contract=off -fPIC -c $< -o $@
$(BUILD)/model_loader_runtime.o: $(RDIR)/model_loader.c include/dsasm_model.h include/dsasm_full_acoustic.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -ffp-contract=off -fPIC -c $< -o $@


$(BUILD)/test_fused_glu: $(BUILD)/fused_glu_m1.o tests/test_fused_glu.c
	$(CC) $(CFLAGS) tests/test_fused_glu.c $< -o $@ $(LDFLAGS)
$(BUILD)/test_fused_glu_m2: $(BUILD)/fused_glu_m1.o $(BUILD)/fused_glu_m2.o tests/test_fused_glu_m2.c
	$(CC) $(CFLAGS) tests/test_fused_glu_m2.c $(BUILD)/fused_glu_m1.o $(BUILD)/fused_glu_m2.o -o $@ $(LDFLAGS)
$(BUILD)/test_depthwise_k31_prelu: $(BUILD)/dwconv_k31_prelu.o tests/test_depthwise_k31_prelu.c
	$(CC) $(CFLAGS) tests/test_depthwise_k31_prelu.c $< -o $@ $(LDFLAGS)
$(BUILD)/test_m4_lynxnet2_block: $(BUILD)/linear_packed4.o $(BUILD)/layernorm.o $(BUILD)/dwconv_k31.o $(BUILD)/fused_glu_m2.o tests/test_m4_lynxnet2_block.c
	$(CC) $(CFLAGS) tests/test_m4_lynxnet2_block.c $(BUILD)/linear_packed4.o $(BUILD)/layernorm.o $(BUILD)/dwconv_k31.o $(BUILD)/fused_glu_m2.o -o $@ $(LDFLAGS)
$(BUILD)/test_m5_lynxnet2_block: $(BUILD)/linear_packed4.o $(BUILD)/linear_residual_packed4.o $(BUILD)/layernorm.o $(BUILD)/dwconv_k31.o $(BUILD)/dwconv_k31_tc.o $(BUILD)/fused_glu_m2.o tests/test_m5_lynxnet2_block.c
	$(CC) $(CFLAGS) tests/test_m5_lynxnet2_block.c $(BUILD)/linear_packed4.o $(BUILD)/linear_residual_packed4.o $(BUILD)/layernorm.o $(BUILD)/dwconv_k31.o $(BUILD)/dwconv_k31_tc.o $(BUILD)/fused_glu_m2.o -o $@ $(LDFLAGS)
$(BUILD)/test_glu_m6: $(BUILD)/glu_m6.o $(BUILD)/fused_glu_m2.o tests/test_glu_m6.c
	$(CC) $(CFLAGS) tests/test_glu_m6.c $(BUILD)/glu_m6.o $(BUILD)/fused_glu_m2.o -o $@ $(LDFLAGS)
$(BUILD)/test_linear_m6: $(BUILD)/linear_m4n16.o $(BUILD)/linear_packed4.o tests/test_linear_m6.c
	$(CC) $(CFLAGS) tests/test_linear_m6.c $(BUILD)/linear_m4n16.o $(BUILD)/linear_packed4.o -o $@ $(LDFLAGS)
$(BUILD)/test_linear_residual_m6: $(BUILD)/linear_residual_m4n16.o $(BUILD)/linear_residual_packed4.o tests/test_linear_residual_m6.c
	$(CC) $(CFLAGS) tests/test_linear_residual_m6.c $(BUILD)/linear_residual_m4n16.o $(BUILD)/linear_residual_packed4.o -o $@ $(LDFLAGS)
$(BUILD)/test_m6_lynxnet2_block: $(BUILD)/glu_m6.o $(BUILD)/linear_residual_m4n16.o $(BUILD)/linear_packed4.o $(BUILD)/linear_residual_packed4.o $(BUILD)/layernorm.o $(BUILD)/dwconv_k31.o $(BUILD)/dwconv_k31_tc.o $(BUILD)/fused_glu_m2.o tests/test_m6_lynxnet2_block.c
	$(CC) $(CFLAGS) tests/test_m6_lynxnet2_block.c $(BUILD)/linear_packed4.o $(BUILD)/linear_residual_packed4.o $(BUILD)/layernorm.o $(BUILD)/dwconv_k31.o $(BUILD)/dwconv_k31_tc.o $(BUILD)/fused_glu_m2.o $(BUILD)/glu_m6.o $(BUILD)/linear_residual_m4n16.o -o $@ $(LDFLAGS)
$(BUILD)/test_m6_bundle: $(BUILD)/glu_m6.o $(BUILD)/linear_residual_m4n16.o $(BUILD)/layernorm.o $(BUILD)/dwconv_k31_tc.o tests/test_m6_bundle.c
	$(CC) $(CFLAGS) tests/test_m6_bundle.c $(BUILD)/glu_m6.o $(BUILD)/linear_residual_m4n16.o $(BUILD)/layernorm.o $(BUILD)/dwconv_k31_tc.o -o $@ $(LDFLAGS)
$(BUILD)/test_atan_glu_m7: $(BUILD)/atan_glu_m7.o tests/test_atan_glu_m7.c
	$(CC) $(CFLAGS) tests/test_atan_glu_m7.c $(BUILD)/atan_glu_m7.o -o $@ $(LDFLAGS)
$(BUILD)/test_atan_glu_m9_strided: $(BUILD)/atan_glu_m7.o $(BUILD)/atan_glu_m9_strided.o tests/test_atan_glu_m9_strided.c
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_atan_glu_m9_strided.c $(BUILD)/atan_glu_m7.o $(BUILD)/atan_glu_m9_strided.o -o $@ $(LDFLAGS)
$(BUILD)/test_m8_strided: $(BUILD)/linear_m4n16.o $(BUILD)/linear_residual_m4n16.o $(BUILD)/linear_m4n16_strided.o $(BUILD)/linear_residual_m4n16_strided.o tests/test_m8_strided.c
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_m8_strided.c $(BUILD)/linear_m4n16.o $(BUILD)/linear_residual_m4n16.o $(BUILD)/linear_m4n16_strided.o $(BUILD)/linear_residual_m4n16_strided.o -o $@ $(LDFLAGS)
$(BUILD)/test_m11_depthwise_cstrided: $(BUILD)/dwconv_k31_tc.o $(BUILD)/dwconv_k31_tc_cstrided.o tests/test_m11_depthwise_cstrided.c
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_m11_depthwise_cstrided.c $(BUILD)/dwconv_k31_tc.o $(BUILD)/dwconv_k31_tc_cstrided.o -o $@ $(LDFLAGS)
$(BUILD)/test_m13_indexed: $(BUILD)/linear_m4n16_strided.o $(BUILD)/linear_residual_m4n16_strided.o $(BUILD)/linear_m4n16_idxstrided.o $(BUILD)/linear_residual_m4n16_idxstrided.o tests/test_m13_indexed.c
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_m13_indexed.c $(BUILD)/linear_m4n16_strided.o $(BUILD)/linear_residual_m4n16_strided.o $(BUILD)/linear_m4n16_idxstrided.o $(BUILD)/linear_residual_m4n16_idxstrided.o -o $@ $(LDFLAGS)
$(BUILD)/test_m15_kblock: $(BUILD)/linear_m4n16_strided.o $(BUILD)/linear_n16_kblock.o tests/test_m15_kblock.c
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_m15_kblock.c $(BUILD)/linear_m4n16_strided.o $(BUILD)/linear_n16_kblock.o -o $@ $(LDFLAGS)
$(BUILD)/test_m7_bundle: $(BUILD)/lynxnet2_runtime.o $(BUILD)/threadpool.o $(BUILD)/vnni_pack_m40_1.o $(BUILD)/conv1d_m32.o $(BUILD)/conv1d_m36.o $(BUILD)/conv1d_m37_range.o $(BUILD)/conv1d_m39_kspec.o $(BUILD)/conv1d_m38_residual.o $(BUILD)/conv1d_m38_range_residual.o $(BUILD)/convtranspose_m33.o $(BUILD)/glu_m6.o $(BUILD)/linear_m4n16.o $(BUILD)/linear_residual_m4n16.o $(BUILD)/linear_m4n16_strided.o $(BUILD)/linear_residual_m4n16_strided.o $(BUILD)/linear_m4n16_idxstrided.o $(BUILD)/linear_residual_m4n16_idxstrided.o $(BUILD)/linear_n16_kblock.o $(BUILD)/layernorm.o $(BUILD)/dwconv_k31_tc.o $(BUILD)/dwconv_k31_tc_cstrided.o $(BUILD)/atan_glu_m7.o $(BUILD)/atan_glu_m9_strided.o $(BUILD)/add3_m7.o tests/test_m7_bundle.c
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_m7_bundle.c $(BUILD)/lynxnet2_runtime.o $(BUILD)/threadpool.o $(BUILD)/conv1d_m32.o $(BUILD)/convtranspose_m33.o $(BUILD)/glu_m6.o $(BUILD)/linear_m4n16.o $(BUILD)/linear_residual_m4n16.o $(BUILD)/linear_m4n16_strided.o $(BUILD)/linear_residual_m4n16_strided.o $(BUILD)/linear_m4n16_idxstrided.o $(BUILD)/linear_residual_m4n16_idxstrided.o $(BUILD)/linear_n16_kblock.o $(BUILD)/layernorm.o $(BUILD)/dwconv_k31_tc.o $(BUILD)/dwconv_k31_tc_cstrided.o $(BUILD)/atan_glu_m7.o $(BUILD)/atan_glu_m9_strided.o $(BUILD)/add3_m7.o -o $@ $(LDFLAGS) -pthread

$(BUILD)/libdsasm_m6.so: $(BUILD)/glu_m6.o $(BUILD)/linear_m4n16.o $(BUILD)/linear_residual_m4n16.o $(BUILD)/layernorm.o $(BUILD)/dwconv_k31_tc.o
	$(CC) -shared -o $@ $^
$(BUILD)/libdsasm_m7.so: $(BUILD)/lynxnet2_runtime.o $(BUILD)/threadpool.o $(BUILD)/conv1d_m32.o $(BUILD)/convtranspose_m33.o $(BUILD)/glu_m6.o $(BUILD)/linear_m4n16.o $(BUILD)/linear_residual_m4n16.o $(BUILD)/linear_m4n16_strided.o $(BUILD)/linear_residual_m4n16_strided.o $(BUILD)/linear_m4n16_idxstrided.o $(BUILD)/linear_residual_m4n16_idxstrided.o $(BUILD)/linear_n16_kblock.o $(BUILD)/layernorm.o $(BUILD)/dwconv_k31_tc.o $(BUILD)/dwconv_k31_tc_cstrided.o $(BUILD)/atan_glu_m7.o $(BUILD)/atan_glu_m9_strided.o $(BUILD)/add3_m7.o
	$(CC) -shared -o $@ $^ -lm -pthread
$(BUILD)/libdsasm_m8.so: $(BUILD)/lynxnet2_runtime.o $(BUILD)/threadpool.o $(BUILD)/conv1d_m32.o $(BUILD)/convtranspose_m33.o $(BUILD)/glu_m6.o $(BUILD)/linear_m4n16.o $(BUILD)/linear_residual_m4n16.o $(BUILD)/linear_m4n16_strided.o $(BUILD)/linear_residual_m4n16_strided.o $(BUILD)/linear_m4n16_idxstrided.o $(BUILD)/linear_residual_m4n16_idxstrided.o $(BUILD)/linear_n16_kblock.o $(BUILD)/layernorm.o $(BUILD)/dwconv_k31_tc.o $(BUILD)/dwconv_k31_tc_cstrided.o $(BUILD)/atan_glu_m7.o $(BUILD)/atan_glu_m9_strided.o $(BUILD)/add3_m7.o
	$(CC) -shared -o $@ $^ -lm -pthread
$(BUILD)/libdsasm_m9.so: $(BUILD)/lynxnet2_runtime.o $(BUILD)/threadpool.o $(BUILD)/conv1d_m32.o $(BUILD)/convtranspose_m33.o $(BUILD)/glu_m6.o $(BUILD)/linear_m4n16.o $(BUILD)/linear_residual_m4n16.o $(BUILD)/linear_m4n16_strided.o $(BUILD)/linear_residual_m4n16_strided.o $(BUILD)/linear_m4n16_idxstrided.o $(BUILD)/linear_residual_m4n16_idxstrided.o $(BUILD)/linear_n16_kblock.o $(BUILD)/layernorm.o $(BUILD)/dwconv_k31_tc.o $(BUILD)/dwconv_k31_tc_cstrided.o $(BUILD)/atan_glu_m7.o $(BUILD)/atan_glu_m9_strided.o $(BUILD)/add3_m7.o
	$(CC) -shared -o $@ $^ -lm -pthread
$(BUILD)/libdsasm_m10.so: $(BUILD)/lynxnet2_runtime.o $(BUILD)/threadpool.o $(BUILD)/conv1d_m32.o $(BUILD)/convtranspose_m33.o $(BUILD)/glu_m6.o $(BUILD)/linear_m4n16.o $(BUILD)/linear_residual_m4n16.o $(BUILD)/linear_m4n16_strided.o $(BUILD)/linear_residual_m4n16_strided.o $(BUILD)/linear_m4n16_idxstrided.o $(BUILD)/linear_residual_m4n16_idxstrided.o $(BUILD)/linear_n16_kblock.o $(BUILD)/layernorm.o $(BUILD)/dwconv_k31_tc.o $(BUILD)/dwconv_k31_tc_cstrided.o $(BUILD)/atan_glu_m7.o $(BUILD)/atan_glu_m9_strided.o $(BUILD)/add3_m7.o
	$(CC) -shared -o $@ $^ -lm -pthread
$(BUILD)/libdsasm_m11.so: $(BUILD)/lynxnet2_runtime.o $(BUILD)/threadpool.o $(BUILD)/conv1d_m32.o $(BUILD)/convtranspose_m33.o $(BUILD)/glu_m6.o $(BUILD)/linear_m4n16.o $(BUILD)/linear_residual_m4n16.o $(BUILD)/linear_m4n16_strided.o $(BUILD)/linear_residual_m4n16_strided.o $(BUILD)/linear_m4n16_idxstrided.o $(BUILD)/linear_residual_m4n16_idxstrided.o $(BUILD)/linear_n16_kblock.o $(BUILD)/layernorm.o $(BUILD)/dwconv_k31_tc.o $(BUILD)/dwconv_k31_tc_cstrided.o $(BUILD)/atan_glu_m7.o $(BUILD)/atan_glu_m9_strided.o $(BUILD)/add3_m7.o
	$(CC) -shared -o $@ $^ -lm -pthread
$(BUILD)/libdsasm_m12.so: $(BUILD)/lynxnet2_runtime.o $(BUILD)/threadpool.o $(BUILD)/conv1d_m32.o $(BUILD)/convtranspose_m33.o $(BUILD)/glu_m6.o $(BUILD)/linear_m4n16.o $(BUILD)/linear_residual_m4n16.o $(BUILD)/linear_m4n16_strided.o $(BUILD)/linear_residual_m4n16_strided.o $(BUILD)/linear_m4n16_idxstrided.o $(BUILD)/linear_residual_m4n16_idxstrided.o $(BUILD)/linear_n16_kblock.o $(BUILD)/layernorm.o $(BUILD)/dwconv_k31_tc.o $(BUILD)/dwconv_k31_tc_cstrided.o $(BUILD)/atan_glu_m7.o $(BUILD)/atan_glu_m9_strided.o $(BUILD)/add3_m7.o
	$(CC) -shared -o $@ $^ -lm -pthread
$(BUILD)/libdsasm_m13.so: $(BUILD)/lynxnet2_runtime.o $(BUILD)/threadpool.o $(BUILD)/conv1d_m32.o $(BUILD)/convtranspose_m33.o $(BUILD)/glu_m6.o $(BUILD)/linear_m4n16.o $(BUILD)/linear_residual_m4n16.o $(BUILD)/linear_m4n16_strided.o $(BUILD)/linear_residual_m4n16_strided.o $(BUILD)/linear_m4n16_idxstrided.o $(BUILD)/linear_residual_m4n16_idxstrided.o $(BUILD)/linear_n16_kblock.o $(BUILD)/layernorm.o $(BUILD)/dwconv_k31_tc.o $(BUILD)/dwconv_k31_tc_cstrided.o $(BUILD)/atan_glu_m7.o $(BUILD)/atan_glu_m9_strided.o $(BUILD)/add3_m7.o
	$(CC) -shared -o $@ $^ -lm -pthread
$(BUILD)/libdsasm_m14.so: $(BUILD)/lynxnet2_runtime.o $(BUILD)/threadpool.o $(BUILD)/conv1d_m32.o $(BUILD)/convtranspose_m33.o $(BUILD)/glu_m6.o $(BUILD)/linear_m4n16.o $(BUILD)/linear_residual_m4n16.o $(BUILD)/linear_m4n16_strided.o $(BUILD)/linear_residual_m4n16_strided.o $(BUILD)/linear_m4n16_idxstrided.o $(BUILD)/linear_residual_m4n16_idxstrided.o $(BUILD)/linear_n16_kblock.o $(BUILD)/layernorm.o $(BUILD)/dwconv_k31_tc.o $(BUILD)/dwconv_k31_tc_cstrided.o $(BUILD)/atan_glu_m7.o $(BUILD)/atan_glu_m9_strided.o $(BUILD)/add3_m7.o
	$(CC) -shared -o $@ $^ -lm -pthread
$(BUILD)/libdsasm_m15.so: $(BUILD)/lynxnet2_runtime.o $(BUILD)/threadpool.o $(BUILD)/conv1d_m32.o $(BUILD)/convtranspose_m33.o $(BUILD)/glu_m6.o $(BUILD)/linear_m4n16.o $(BUILD)/linear_residual_m4n16.o $(BUILD)/linear_m4n16_strided.o $(BUILD)/linear_residual_m4n16_strided.o $(BUILD)/linear_m4n16_idxstrided.o $(BUILD)/linear_residual_m4n16_idxstrided.o $(BUILD)/linear_n16_kblock.o $(BUILD)/layernorm.o $(BUILD)/dwconv_k31_tc.o $(BUILD)/dwconv_k31_tc_cstrided.o $(BUILD)/atan_glu_m7.o $(BUILD)/atan_glu_m9_strided.o $(BUILD)/add3_m7.o
	$(CC) -shared -o $@ $^ -lm -pthread
$(BUILD)/libdsasm_m16.so: $(BUILD)/lynxnet2_runtime.o $(BUILD)/threadpool.o $(BUILD)/conv1d_m32.o $(BUILD)/convtranspose_m33.o $(BUILD)/glu_m6.o $(BUILD)/linear_m4n16.o $(BUILD)/linear_residual_m4n16.o $(BUILD)/linear_m4n16_strided.o $(BUILD)/linear_residual_m4n16_strided.o $(BUILD)/linear_m4n16_idxstrided.o $(BUILD)/linear_residual_m4n16_idxstrided.o $(BUILD)/linear_n16_kblock.o $(BUILD)/layernorm.o $(BUILD)/dwconv_k31_tc.o $(BUILD)/dwconv_k31_tc_cstrided.o $(BUILD)/atan_glu_m7.o $(BUILD)/atan_glu_m9_strided.o $(BUILD)/add3_m7.o
	$(CC) -shared -o $@ $^ -lm -pthread
$(BUILD)/libdsasm_m17.so: $(BUILD)/lynxnet2_runtime.o $(BUILD)/threadpool.o $(BUILD)/conv1d_m32.o $(BUILD)/convtranspose_m33.o $(BUILD)/glu_m6.o $(BUILD)/linear_m4n16.o $(BUILD)/linear_residual_m4n16.o $(BUILD)/linear_m4n16_strided.o $(BUILD)/linear_residual_m4n16_strided.o $(BUILD)/linear_m4n16_idxstrided.o $(BUILD)/linear_residual_m4n16_idxstrided.o $(BUILD)/linear_n16_kblock.o $(BUILD)/layernorm.o $(BUILD)/dwconv_k31_tc.o $(BUILD)/dwconv_k31_tc_cstrided.o $(BUILD)/atan_glu_m7.o $(BUILD)/atan_glu_m9_strided.o $(BUILD)/add3_m7.o
	$(CC) -shared -o $@ $^ -lm -pthread
$(BUILD)/libdsasm_m18.so: $(BUILD)/reflow_runtime.o $(BUILD)/lynxnet2_runtime.o $(BUILD)/threadpool.o $(BUILD)/conv1d_m32.o $(BUILD)/convtranspose_m33.o $(BUILD)/glu_m6.o $(BUILD)/linear_m4n16.o $(BUILD)/linear_residual_m4n16.o $(BUILD)/linear_m4n16_strided.o $(BUILD)/linear_residual_m4n16_strided.o $(BUILD)/linear_m4n16_idxstrided.o $(BUILD)/linear_residual_m4n16_idxstrided.o $(BUILD)/linear_n16_kblock.o $(BUILD)/layernorm.o $(BUILD)/dwconv_k31_tc.o $(BUILD)/dwconv_k31_tc_cstrided.o $(BUILD)/atan_glu_m7.o $(BUILD)/atan_glu_m9_strided.o $(BUILD)/add3_m7.o
	$(CC) -shared -o $@ $^ -lm -pthread
$(BUILD)/libdsasm_m19.so: $(BUILD)/acoustic_runtime.o $(BUILD)/reflow_runtime.o $(BUILD)/lynxnet2_runtime.o $(BUILD)/threadpool.o $(BUILD)/conv1d_m32.o $(BUILD)/convtranspose_m33.o $(BUILD)/glu_m6.o $(BUILD)/linear_m4n16.o $(BUILD)/linear_residual_m4n16.o $(BUILD)/linear_m4n16_strided.o $(BUILD)/linear_residual_m4n16_strided.o $(BUILD)/linear_m4n16_idxstrided.o $(BUILD)/linear_residual_m4n16_idxstrided.o $(BUILD)/linear_n16_kblock.o $(BUILD)/layernorm.o $(BUILD)/dwconv_k31_tc.o $(BUILD)/dwconv_k31_tc_cstrided.o $(BUILD)/atan_glu_m7.o $(BUILD)/atan_glu_m9_strided.o $(BUILD)/add3_m7.o
	$(CC) -shared -o $@ $^ -lm -pthread

$(BUILD)/libdsasm_m20.so: $(BUILD)/post_fs2_runtime.o $(BUILD)/aux_decoder_runtime.o $(BUILD)/fs2_front_runtime.o $(BUILD)/acoustic_runtime.o $(BUILD)/reflow_runtime.o $(BUILD)/lynxnet2_runtime.o $(BUILD)/threadpool.o $(BUILD)/conv1d_m32.o $(BUILD)/convtranspose_m33.o $(BUILD)/glu_m6.o $(BUILD)/linear_m4n16.o $(BUILD)/linear_residual_m4n16.o $(BUILD)/linear_m4n16_strided.o $(BUILD)/linear_residual_m4n16_strided.o $(BUILD)/linear_m4n16_idxstrided.o $(BUILD)/linear_residual_m4n16_idxstrided.o $(BUILD)/linear_n16_kblock.o $(BUILD)/layernorm.o $(BUILD)/dwconv_k31_tc.o $(BUILD)/dwconv_k31_tc_cstrided.o $(BUILD)/dwconv_k7_tc.o $(BUILD)/atan_glu_m7.o $(BUILD)/atan_glu_m9_strided.o $(BUILD)/add3_m7.o
	$(CC) -shared -o $@ $^ -lm -pthread

$(BUILD)/libdsasm_m21.so: $(BUILD)/full_acoustic_runtime.o $(BUILD)/post_fs2_runtime.o $(BUILD)/aux_decoder_runtime.o $(BUILD)/fs2_front_runtime.o $(BUILD)/acoustic_runtime.o $(BUILD)/reflow_runtime.o $(BUILD)/lynxnet2_runtime.o $(BUILD)/threadpool.o $(BUILD)/conv1d_m32.o $(BUILD)/convtranspose_m33.o $(BUILD)/glu_m6.o $(BUILD)/linear_m4n16.o $(BUILD)/linear_residual_m4n16.o $(BUILD)/linear_m4n16_strided.o $(BUILD)/linear_residual_m4n16_strided.o $(BUILD)/linear_m4n16_idxstrided.o $(BUILD)/linear_residual_m4n16_idxstrided.o $(BUILD)/linear_n16_kblock.o $(BUILD)/layernorm.o $(BUILD)/dwconv_k31_tc.o $(BUILD)/dwconv_k31_tc_cstrided.o $(BUILD)/dwconv_k7_tc.o $(BUILD)/atan_glu_m7.o $(BUILD)/atan_glu_m9_strided.o $(BUILD)/add3_m7.o $(BUILD)/fs2_encoder_runtime.o $(BUILD)/dot_avx2.o $(BUILD)/layernorm_precise.o
	$(CC) -shared -o $@ $^ -lm -pthread

$(BUILD)/libdsasm_m22.so: $(BUILD)/full_acoustic_runtime.o $(BUILD)/post_fs2_runtime.o $(BUILD)/aux_decoder_runtime.o $(BUILD)/fs2_front_runtime.o $(BUILD)/acoustic_runtime.o $(BUILD)/reflow_runtime.o $(BUILD)/lynxnet2_runtime.o $(BUILD)/threadpool.o $(BUILD)/conv1d_m32.o $(BUILD)/convtranspose_m33.o $(BUILD)/glu_m6.o $(BUILD)/linear_m4n16.o $(BUILD)/linear_residual_m4n16.o $(BUILD)/linear_m4n16_strided.o $(BUILD)/linear_residual_m4n16_strided.o $(BUILD)/linear_m4n16_idxstrided.o $(BUILD)/linear_residual_m4n16_idxstrided.o $(BUILD)/linear_n16_kblock.o $(BUILD)/layernorm.o $(BUILD)/dwconv_k31_tc.o $(BUILD)/dwconv_k31_tc_cstrided.o $(BUILD)/dwconv_k7_tc.o $(BUILD)/atan_glu_m7.o $(BUILD)/atan_glu_m9_strided.o $(BUILD)/add3_m7.o $(BUILD)/fs2_encoder_runtime.o $(BUILD)/dot_avx2.o $(BUILD)/layernorm_precise.o
	$(CC) -shared -o $@ $^ -lm -pthread

$(BUILD)/libdsasm_m23.so: $(BUILD)/model_loader_runtime.o $(BUILD)/full_acoustic_runtime.o $(BUILD)/post_fs2_runtime.o $(BUILD)/aux_decoder_runtime.o $(BUILD)/fs2_front_runtime.o $(BUILD)/acoustic_runtime.o $(BUILD)/reflow_runtime.o $(BUILD)/lynxnet2_runtime.o $(BUILD)/threadpool.o $(BUILD)/conv1d_m32.o $(BUILD)/convtranspose_m33.o $(BUILD)/glu_m6.o $(BUILD)/linear_m4n16.o $(BUILD)/linear_residual_m4n16.o $(BUILD)/linear_m4n16_strided.o $(BUILD)/linear_residual_m4n16_strided.o $(BUILD)/linear_m4n16_idxstrided.o $(BUILD)/linear_residual_m4n16_idxstrided.o $(BUILD)/linear_n16_kblock.o $(BUILD)/layernorm.o $(BUILD)/dwconv_k31_tc.o $(BUILD)/dwconv_k31_tc_cstrided.o $(BUILD)/dwconv_k7_tc.o $(BUILD)/atan_glu_m7.o $(BUILD)/atan_glu_m9_strided.o $(BUILD)/add3_m7.o $(BUILD)/fs2_encoder_runtime.o $(BUILD)/dot_avx2.o $(BUILD)/layernorm_precise.o
	$(CC) -shared -o $@ $^ -lm -pthread

$(BUILD)/test_m23_model_loader: tests/test_m23_model_loader.c $(BUILD)/model_loader_runtime.o $(BUILD)/full_acoustic_runtime.o $(BUILD)/post_fs2_runtime.o $(BUILD)/aux_decoder_runtime.o $(BUILD)/fs2_front_runtime.o $(BUILD)/acoustic_runtime.o $(BUILD)/reflow_runtime.o $(BUILD)/lynxnet2_runtime.o $(BUILD)/threadpool.o $(BUILD)/conv1d_m32.o $(BUILD)/convtranspose_m33.o $(BUILD)/glu_m6.o $(BUILD)/linear_m4n16.o $(BUILD)/linear_residual_m4n16.o $(BUILD)/linear_m4n16_strided.o $(BUILD)/linear_residual_m4n16_strided.o $(BUILD)/linear_m4n16_idxstrided.o $(BUILD)/linear_residual_m4n16_idxstrided.o $(BUILD)/linear_n16_kblock.o $(BUILD)/layernorm.o $(BUILD)/dwconv_k31_tc.o $(BUILD)/dwconv_k31_tc_cstrided.o $(BUILD)/dwconv_k7_tc.o $(BUILD)/atan_glu_m7.o $(BUILD)/atan_glu_m9_strided.o $(BUILD)/add3_m7.o $(BUILD)/fs2_encoder_runtime.o $(BUILD)/dot_avx2.o $(BUILD)/layernorm_precise.o
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_m23_model_loader.c $(filter-out tests/test_m23_model_loader.c,$^) -o $@ -lm -pthread

CORE_TEST_BINS := $(BUILD)/test_fused_glu $(BUILD)/test_fused_glu_m2 \
                  $(BUILD)/test_depthwise_k31_prelu $(BUILD)/test_m4_lynxnet2_block \
                  $(BUILD)/test_m5_lynxnet2_block $(BUILD)/test_glu_m6 \
                  $(BUILD)/test_linear_m6 $(BUILD)/test_linear_residual_m6 \
                  $(BUILD)/test_m6_lynxnet2_block $(BUILD)/test_atan_glu_m7 \
                  $(BUILD)/test_atan_glu_m9_strided $(BUILD)/test_m8_strided \
                  $(BUILD)/test_m11_depthwise_cstrided $(BUILD)/test_m13_indexed \
                  $(BUILD)/test_m15_kblock

test: $(CORE_TEST_BINS)
	./$(BUILD)/test_fused_glu
	./$(BUILD)/test_fused_glu_m2
	./$(BUILD)/test_depthwise_k31_prelu
	./$(BUILD)/test_m4_lynxnet2_block
	./$(BUILD)/test_m5_lynxnet2_block
	./$(BUILD)/test_glu_m6
	./$(BUILD)/test_linear_m6
	./$(BUILD)/test_linear_residual_m6
	./$(BUILD)/test_m6_lynxnet2_block
	./$(BUILD)/test_atan_glu_m7
	./$(BUILD)/test_atan_glu_m9_strided
	./$(BUILD)/test_m8_strided
	./$(BUILD)/test_m11_depthwise_cstrided
	./$(BUILD)/test_m13_indexed
	./$(BUILD)/test_m15_kblock

bench: all
	./$(BUILD)/test_glu_m6
	./$(BUILD)/test_linear_m6
	./$(BUILD)/test_linear_residual_m6
	./$(BUILD)/test_m6_lynxnet2_block
	./$(BUILD)/test_atan_glu_m7

m6: $(BUILD)/test_m6_lynxnet2_block
	./$(BUILD)/test_m6_lynxnet2_block
m7: m7-pytorch-check m7-bundle-check

pytorch-check: $(BUILD)/libdsasm_m6.so
	$(PYTHON) tools/validate_pytorch_block.py --lib $(BUILD)/libdsasm_m6.so --large
bundle-check: $(BUILD)/test_m6_bundle
	$(PYTHON) tools/make_fake_lynx_checkpoint.py $(BUILD)/fake_lynx.ckpt
	rm -rf $(BUILD)/fake_bundle
	$(PYTHON) tools/pack_lynxnet2_checkpoint.py $(BUILD)/fake_lynx.ckpt --out $(BUILD)/fake_bundle --test-frames 37
	./$(BUILD)/test_m6_bundle $(BUILD)/fake_bundle/lynxnet2_block0.dsb $(BUILD)/fake_bundle/test_input.f32 $(BUILD)/fake_bundle/test_output_pytorch.f32

m7-pytorch-check: $(BUILD)/libdsasm_m7.so
	$(PYTHON) tools/validate_pytorch_lynxnet2.py --lib $(BUILD)/libdsasm_m7.so --threads 4
m7-bundle-check: $(BUILD)/test_m7_bundle
	$(PYTHON) tools/make_fake_lynxnet2_backbone.py $(BUILD)/fake_full.ckpt
	rm -rf $(BUILD)/fake_full_bundle
	$(PYTHON) tools/pack_lynxnet2_backbone.py $(BUILD)/fake_full.ckpt --prefix model.diffusion.denoise_fn --out $(BUILD)/fake_full_bundle --test-frames 37
	./$(BUILD)/test_m7_bundle $(BUILD)/fake_full_bundle/lynxnet2.dsn $(BUILD)/fake_full_bundle/test_spec.f32 $(BUILD)/fake_full_bundle/test_condition.f32 $(BUILD)/fake_full_bundle/test_timestep.f32 $(BUILD)/fake_full_bundle/test_output_pytorch.f32 4
m7-bench: $(BUILD)/libdsasm_m7.so
	$(PYTHON) tools/validate_pytorch_lynxnet2.py --lib $(BUILD)/libdsasm_m7.so --threads 4 --official-shape
m8-pytorch-check: $(BUILD)/libdsasm_m8.so
	$(PYTHON) tools/validate_pytorch_lynxnet2.py --lib $(BUILD)/libdsasm_m8.so --threads 0
m8-bench: $(BUILD)/libdsasm_m8.so
	$(PYTHON) tools/validate_pytorch_lynxnet2.py --lib $(BUILD)/libdsasm_m8.so --threads 0 --official-shape
m8: m8-pytorch-check m7-bundle-check
m9-pytorch-check: $(BUILD)/libdsasm_m9.so
	$(PYTHON) tools/validate_pytorch_lynxnet2.py --lib $(BUILD)/libdsasm_m9.so --threads 0
m9-bench: $(BUILD)/libdsasm_m9.so
	$(PYTHON) tools/validate_pytorch_lynxnet2.py --lib $(BUILD)/libdsasm_m9.so --threads 0 --official-shape
m9-autotune: $(BUILD)/libdsasm_m9.so
	$(PYTHON) tools/validate_pytorch_lynxnet2.py --lib $(BUILD)/libdsasm_m9.so --threads 0 --official-shape --sweep-tiles
m9: m9-pytorch-check m7-bundle-check
m10-pytorch-check: $(BUILD)/libdsasm_m10.so
	$(PYTHON) tools/validate_pytorch_lynxnet2.py --lib $(BUILD)/libdsasm_m10.so --threads 0
m10-bench: $(BUILD)/libdsasm_m10.so
	$(PYTHON) tools/validate_pytorch_lynxnet2.py --lib $(BUILD)/libdsasm_m10.so --threads 0 --official-shape
m10-autotune: $(BUILD)/libdsasm_m10.so
	$(PYTHON) tools/validate_pytorch_lynxnet2.py --lib $(BUILD)/libdsasm_m10.so --threads 0 --official-shape --sweep-tiles
m10: m10-pytorch-check m7-bundle-check
m11-pytorch-check: $(BUILD)/libdsasm_m11.so
	$(PYTHON) tools/validate_pytorch_lynxnet2.py --lib $(BUILD)/libdsasm_m11.so --threads 0
m11-bench: $(BUILD)/libdsasm_m11.so
	$(PYTHON) tools/validate_pytorch_lynxnet2.py --lib $(BUILD)/libdsasm_m11.so --threads 0 --official-shape
m11-autotune: $(BUILD)/libdsasm_m11.so
	$(PYTHON) tools/validate_pytorch_lynxnet2.py --lib $(BUILD)/libdsasm_m11.so --threads 0 --official-shape --sweep-tiles
m11: m11-pytorch-check m7-bundle-check
m12-pytorch-check: $(BUILD)/libdsasm_m12.so
	DSASM_PARALLEL_DW=0 $(PYTHON) tools/validate_pytorch_lynxnet2.py --lib $(BUILD)/libdsasm_m12.so --threads 0
m12-bench: $(BUILD)/libdsasm_m12.so
	DSASM_PARALLEL_DW=0 $(PYTHON) tools/validate_pytorch_lynxnet2.py --lib $(BUILD)/libdsasm_m12.so --threads 0 --official-shape
m12-spin-bench: $(BUILD)/libdsasm_m12.so
	$(PYTHON) tools/bench_m12_spin.py --lib $(BUILD)/libdsasm_m12.so
m12-perf: $(BUILD)/libdsasm_m12.so
	@if command -v perf >/dev/null 2>&1; then \
	  perf stat -e cycles,instructions,cache-references,cache-misses,branches,branch-misses,context-switches,cpu-migrations \
	    $(PYTHON) tools/bench_m12_perf.py --lib $(BUILD)/libdsasm_m12.so --iters 80; \
	else \
	  echo 'perf not found; on Arch install the perf package, then rerun make m12-perf'; exit 2; \
	fi
m12: m12-pytorch-check m7-bundle-check
m13-pytorch-check: $(BUILD)/libdsasm_m13.so
	DSASM_PARALLEL_DW=0 DSASM_SPIN_POOL=0 $(PYTHON) tools/validate_pytorch_lynxnet2.py --lib $(BUILD)/libdsasm_m13.so --threads 0
m13-kernel-bench: $(BUILD)/test_m13_indexed
	./$(BUILD)/test_m13_indexed
m13-bench: $(BUILD)/libdsasm_m13.so
	DSASM_PARALLEL_DW=0 DSASM_SPIN_POOL=0 $(PYTHON) tools/bench_m13_matrix.py --lib $(BUILD)/libdsasm_m13.so
m13-perf: $(BUILD)/libdsasm_m13.so
	@if command -v perf >/dev/null 2>&1; then \
	  echo '--- M12 legacy inner loop ---'; \
	  DSASM_INDEXED_LINEAR=0 perf stat -e cycles,instructions,cache-references,cache-misses,branches,branch-misses,context-switches,cpu-migrations \
	    $(PYTHON) tools/bench_m12_perf.py --lib $(BUILD)/libdsasm_m13.so --iters 80; \
	  echo '--- M13 indexed inner loop ---'; \
	  DSASM_INDEXED_LINEAR=1 perf stat -e cycles,instructions,cache-references,cache-misses,branches,branch-misses,context-switches,cpu-migrations \
	    $(PYTHON) tools/bench_m12_perf.py --lib $(BUILD)/libdsasm_m13.so --iters 80; \
	else \
	  echo 'perf not found; on Arch install the perf package, then rerun make m13-perf'; exit 2; \
	fi
m13: m13-pytorch-check m7-bundle-check

m14-pytorch-check: $(BUILD)/libdsasm_m14.so
	DSASM_PARALLEL_DW=0 DSASM_SPIN_POOL=0 DSASM_N_OWNER=0 $(PYTHON) tools/validate_pytorch_lynxnet2.py --lib $(BUILD)/libdsasm_m14.so --threads 0
m14-bench: $(BUILD)/libdsasm_m14.so
	DSASM_PARALLEL_DW=0 DSASM_SPIN_POOL=0 $(PYTHON) tools/bench_m14_owner.py --lib $(BUILD)/libdsasm_m14.so
m14-perf: $(BUILD)/libdsasm_m14.so
	@if command -v perf >/dev/null 2>&1; then \
	  echo '--- M13 dynamic 32x64 old ---'; \
	  DSASM_N_OWNER=0 DSASM_INDEXED_LINEAR=0 DSASM_AUTO_TILES=0 DSASM_M_TILE=32 DSASM_N_TILE=64 perf stat -e cycles,instructions,cache-references,cache-misses,branches,branch-misses,context-switches,cpu-migrations \
	    $(PYTHON) tools/bench_m14_perf.py --lib $(BUILD)/libdsasm_m14.so --iters 80 --owner 0 --indexed 0; \
	  echo '--- M14 N-owner 64-wide old ---'; \
	  DSASM_N_OWNER=1 DSASM_INDEXED_LINEAR=0 perf stat -e cycles,instructions,cache-references,cache-misses,branches,branch-misses,context-switches,cpu-migrations \
	    $(PYTHON) tools/bench_m14_perf.py --lib $(BUILD)/libdsasm_m14.so --iters 80 --owner 1 --indexed 0; \
	  echo '--- M14 N-owner 64-wide indexed ---'; \
	  DSASM_N_OWNER=1 DSASM_INDEXED_LINEAR=1 perf stat -e cycles,instructions,cache-references,cache-misses,branches,branch-misses,context-switches,cpu-migrations \
	    $(PYTHON) tools/bench_m14_perf.py --lib $(BUILD)/libdsasm_m14.so --iters 80 --owner 1 --indexed 1; \
	else \
	  echo 'perf not found; install perf then rerun make m14-perf'; exit 2; \
	fi
m14: m14-pytorch-check m7-bundle-check


m15-pytorch-check: $(BUILD)/libdsasm_m15.so
	DSASM_PARALLEL_DW=0 DSASM_SPIN_POOL=0 DSASM_N_OWNER=0 DSASM_INDEXED_LINEAR=0 DSASM_KBLOCK_ATAN=1 DSASM_K_BLOCK=512 $(PYTHON) tools/validate_pytorch_lynxnet2.py --lib $(BUILD)/libdsasm_m15.so --threads 0
m15-kernel-bench: $(BUILD)/test_m15_kblock
	./$(BUILD)/test_m15_kblock
m15-bench: $(BUILD)/libdsasm_m15.so
	DSASM_PARALLEL_DW=0 DSASM_SPIN_POOL=0 DSASM_N_OWNER=0 DSASM_INDEXED_LINEAR=0 $(PYTHON) tools/bench_m15_kblock.py --lib $(BUILD)/libdsasm_m15.so
m15-perf: $(BUILD)/libdsasm_m15.so
	@if command -v perf >/dev/null 2>&1; then \
	  echo '--- M15 full-K baseline ---'; \
	  perf stat -e cycles,instructions,cache-references,cache-misses,branches,branch-misses,context-switches,cpu-migrations \
	    $(PYTHON) tools/bench_m15_perf.py --lib $(BUILD)/libdsasm_m15.so --iters 80 --kblock 0; \
	  echo '--- M15 Kblock=512 ---'; \
	  perf stat -e cycles,instructions,cache-references,cache-misses,branches,branch-misses,context-switches,cpu-migrations \
	    $(PYTHON) tools/bench_m15_perf.py --lib $(BUILD)/libdsasm_m15.so --iters 80 --kblock 512; \
	else \
	  echo 'perf not found; install perf then rerun make m15-perf'; exit 2; \
	fi
m15: m15-pytorch-check m7-bundle-check

m16-pytorch-check: $(BUILD)/libdsasm_m16.so
	DSASM_PARALLEL_DW=0 DSASM_SPIN_POOL=0 DSASM_INDEXED_LINEAR=0 $(PYTHON) tools/validate_pytorch_lynxnet2.py --lib $(BUILD)/libdsasm_m16.so --threads 0
m16-bench: $(BUILD)/libdsasm_m16.so
	DSASM_PARALLEL_DW=0 DSASM_SPIN_POOL=0 DSASM_INDEXED_LINEAR=0 $(PYTHON) tools/bench_m16_policy.py --lib $(BUILD)/libdsasm_m16.so
m16-perf: $(BUILD)/libdsasm_m16.so
	@if command -v perf >/dev/null 2>&1; then \
	  echo '--- M15 dynamic full-K baseline ---'; \
	  perf stat -e cycles,instructions,cache-references,cache-misses,branches,branch-misses,context-switches,cpu-migrations \
	    $(PYTHON) tools/bench_m16_perf.py --lib $(BUILD)/libdsasm_m16.so --iters 80 --owner 0 --kblock 0; \
	  echo '--- dynamic Kblock=512 ---'; \
	  perf stat -e cycles,instructions,cache-references,cache-misses,branches,branch-misses,context-switches,cpu-migrations \
	    $(PYTHON) tools/bench_m16_perf.py --lib $(BUILD)/libdsasm_m16.so --iters 80 --owner 0 --kblock 512; \
	  echo '--- N-owner full-K ---'; \
	  perf stat -e cycles,instructions,cache-references,cache-misses,branches,branch-misses,context-switches,cpu-migrations \
	    $(PYTHON) tools/bench_m16_perf.py --lib $(BUILD)/libdsasm_m16.so --iters 80 --owner 1 --kblock 0; \
	  echo '--- M16 N-owner + Kblock=512 ---'; \
	  perf stat -e cycles,instructions,cache-references,cache-misses,branches,branch-misses,context-switches,cpu-migrations \
	    $(PYTHON) tools/bench_m16_perf.py --lib $(BUILD)/libdsasm_m16.so --iters 80 --owner 1 --kblock 512; \
	else \
	  echo 'perf not found; install perf then rerun make m16-perf'; exit 2; \
	fi
m16: m16-pytorch-check m7-bundle-check

m17-pytorch-check: $(BUILD)/libdsasm_m17.so
	DSASM_PARALLEL_DW=0 DSASM_SPIN_POOL=0 DSASM_INDEXED_LINEAR=0 $(PYTHON) tools/validate_pytorch_lynxnet2.py --lib $(BUILD)/libdsasm_m17.so --threads 0
m17-bench: $(BUILD)/libdsasm_m17.so
	DSASM_PARALLEL_DW=0 DSASM_SPIN_POOL=0 DSASM_INDEXED_LINEAR=0 $(PYTHON) tools/bench_m17_policy.py --lib $(BUILD)/libdsasm_m17.so
m17-perf: $(BUILD)/libdsasm_m17.so
	@if command -v perf >/dev/null 2>&1; then \
	  echo '--- M17 dynamic full-K baseline ---'; \
	  perf stat -e cycles,instructions,cache-references,cache-misses,branches,branch-misses,context-switches,cpu-migrations \
	    $(PYTHON) tools/bench_m16_perf.py --lib $(BUILD)/libdsasm_m17.so --iters 80 --owner 0 --kblock 0; \
	  echo '--- M17 promoted dynamic Kblock=512 ---'; \
	  perf stat -e cycles,instructions,cache-references,cache-misses,branches,branch-misses,context-switches,cpu-migrations \
	    $(PYTHON) tools/bench_m16_perf.py --lib $(BUILD)/libdsasm_m17.so --iters 80 --owner 0 --kblock 512; \
	  echo '--- owner + Kblock=512 control ---'; \
	  perf stat -e cycles,instructions,cache-references,cache-misses,branches,branch-misses,context-switches,cpu-migrations \
	    $(PYTHON) tools/bench_m16_perf.py --lib $(BUILD)/libdsasm_m17.so --iters 80 --owner 1 --kblock 512; \
	else \
	  echo 'perf not found; install perf then rerun make m17-perf'; exit 2; \
	fi
m17: m17-pytorch-check m7-bundle-check

m18-pytorch-check: $(BUILD)/libdsasm_m18.so
	DSASM_PARALLEL_DW=0 DSASM_SPIN_POOL=0 DSASM_N_OWNER=0 DSASM_INDEXED_LINEAR=0 DSASM_KBLOCK_ATAN=0 $(PYTHON) tools/validate_pytorch_reflow.py --lib $(BUILD)/libdsasm_m18.so
m18-bench: $(BUILD)/libdsasm_m18.so
	DSASM_PARALLEL_DW=0 DSASM_SPIN_POOL=0 DSASM_N_OWNER=0 DSASM_INDEXED_LINEAR=0 DSASM_KBLOCK_ATAN=0 $(PYTHON) tools/validate_pytorch_reflow.py --lib $(BUILD)/libdsasm_m18.so --official-shape --steps 20
m18: m18-pytorch-check m7-bundle-check

m19-pytorch-check: $(BUILD)/libdsasm_m19.so
	DSASM_PARALLEL_DW=0 DSASM_SPIN_POOL=0 DSASM_N_OWNER=0 DSASM_INDEXED_LINEAR=0 DSASM_KBLOCK_ATAN=0 $(PYTHON) tools/validate_pytorch_acoustic.py --lib $(BUILD)/libdsasm_m19.so
m19-bench: $(BUILD)/libdsasm_m19.so
	DSASM_PARALLEL_DW=0 DSASM_SPIN_POOL=0 DSASM_N_OWNER=0 DSASM_INDEXED_LINEAR=0 DSASM_KBLOCK_ATAN=0 $(PYTHON) tools/validate_pytorch_acoustic.py --lib $(BUILD)/libdsasm_m19.so --official-shape --steps 20
m19: m19-pytorch-check m7-bundle-check

m20-fs2-front-check: $(BUILD)/libdsasm_m20.so
	$(PYTHON) tools/validate_fs2_front_m20.py --lib $(BUILD)/libdsasm_m20.so
m20-postfs2-check: $(BUILD)/libdsasm_m20.so
	DSASM_PARALLEL_DW=0 DSASM_SPIN_POOL=0 DSASM_N_OWNER=0 DSASM_INDEXED_LINEAR=0 DSASM_KBLOCK_ATAN=0 $(PYTHON) tools/validate_pytorch_postfs2_m20.py --lib $(BUILD)/libdsasm_m20.so
m20-pytorch-check: $(BUILD)/libdsasm_m20.so
	DSASM_PARALLEL_DW=0 DSASM_SPIN_POOL=0 DSASM_N_OWNER=0 DSASM_INDEXED_LINEAR=0 DSASM_KBLOCK_ATAN=0 $(PYTHON) tools/validate_pytorch_aux_m20.py --lib $(BUILD)/libdsasm_m20.so
m20-bench: $(BUILD)/libdsasm_m20.so
	DSASM_PARALLEL_DW=0 DSASM_SPIN_POOL=0 DSASM_N_OWNER=0 DSASM_INDEXED_LINEAR=0 DSASM_KBLOCK_ATAN=0 $(PYTHON) tools/validate_pytorch_aux_m20.py --lib $(BUILD)/libdsasm_m20.so --official-shape
	DSASM_PARALLEL_DW=0 DSASM_SPIN_POOL=0 DSASM_N_OWNER=0 DSASM_INDEXED_LINEAR=0 DSASM_KBLOCK_ATAN=0 $(PYTHON) tools/validate_pytorch_postfs2_m20.py --lib $(BUILD)/libdsasm_m20.so --official-shape
m20: m20-fs2-front-check m20-pytorch-check m20-postfs2-check m19-pytorch-check m7-bundle-check

m21-pytorch-check: $(BUILD)/libdsasm_m21.so
	DSASM_PARALLEL_DW=0 DSASM_SPIN_POOL=0 DSASM_N_OWNER=0 DSASM_INDEXED_LINEAR=0 DSASM_KBLOCK_ATAN=0 $(PYTHON) tools/validate_pytorch_fs2_encoder_m21.py --lib $(BUILD)/libdsasm_m21.so
m21-condition-check: $(BUILD)/libdsasm_m21.so
	DSASM_PARALLEL_DW=0 DSASM_SPIN_POOL=0 DSASM_N_OWNER=0 DSASM_INDEXED_LINEAR=0 DSASM_KBLOCK_ATAN=0 $(PYTHON) tools/validate_pytorch_fs2_condition_m21.py --lib $(BUILD)/libdsasm_m21.so
m21-full-check: $(BUILD)/libdsasm_m21.so
	DSASM_PARALLEL_DW=0 DSASM_SPIN_POOL=0 DSASM_N_OWNER=0 DSASM_INDEXED_LINEAR=0 DSASM_KBLOCK_ATAN=0 $(PYTHON) tools/validate_pytorch_full_acoustic_m21.py --lib $(BUILD)/libdsasm_m21.so
m21-pack-check:
	$(PYTHON) tools/make_fake_fs2_m21_checkpoint.py $(BUILD)/fake_fs2_m21.ckpt
	rm -rf $(BUILD)/fake_fs2_m21_bundle
	$(PYTHON) tools/pack_fs2_acoustic_checkpoint.py $(BUILD)/fake_fs2_m21.ckpt --out $(BUILD)/fake_fs2_m21_bundle
	test -s $(BUILD)/fake_fs2_m21_bundle/fs2_acoustic.dsfs

m21-bench: $(BUILD)/libdsasm_m21.so
	DSASM_PARALLEL_DW=0 DSASM_SPIN_POOL=0 DSASM_N_OWNER=0 DSASM_INDEXED_LINEAR=0 DSASM_KBLOCK_ATAN=0 $(PYTHON) tools/validate_pytorch_fs2_encoder_m21.py --lib $(BUILD)/libdsasm_m21.so --official-shape
	DSASM_PARALLEL_DW=0 DSASM_SPIN_POOL=0 DSASM_N_OWNER=0 DSASM_INDEXED_LINEAR=0 DSASM_KBLOCK_ATAN=0 $(PYTHON) tools/validate_pytorch_fs2_condition_m21.py --lib $(BUILD)/libdsasm_m21.so --official-shape
	DSASM_PARALLEL_DW=0 DSASM_SPIN_POOL=0 DSASM_N_OWNER=0 DSASM_INDEXED_LINEAR=0 DSASM_KBLOCK_ATAN=0 $(PYTHON) tools/validate_pytorch_full_acoustic_m21.py --lib $(BUILD)/libdsasm_m21.so --official-shape
m21: m21-pytorch-check m21-condition-check m21-full-check m21-pack-check m20-pytorch-check m20-postfs2-check m19-pytorch-check m7-bundle-check


m22-check: $(BUILD)/libdsasm_m22.so
	DSASM_PARALLEL_DW=0 DSASM_SPIN_POOL=0 DSASM_N_OWNER=0 DSASM_INDEXED_LINEAR=0 DSASM_KBLOCK_ATAN=0 $(PYTHON) tools/validate_m22_normfast.py --lib $(BUILD)/libdsasm_m22.so
m22-bench: $(BUILD)/libdsasm_m22.so
	DSASM_PARALLEL_DW=0 DSASM_SPIN_POOL=0 DSASM_N_OWNER=0 DSASM_INDEXED_LINEAR=0 DSASM_KBLOCK_ATAN=0 $(PYTHON) tools/bench_m22_stages.py --lib $(BUILD)/libdsasm_m22.so --official-shape
m22: m22-check m21-pytorch-check m21-condition-check m20-pytorch-check m19-pytorch-check m7-bundle-check

# M23: benchmark reliability + native three-bundle model loader.
m23-model-check: $(BUILD)/test_m23_model_loader
	$(PYTHON) tools/make_fake_fs2_m21_checkpoint.py $(BUILD)/m23_fs.ckpt --c 64 --layers 2
	rm -rf $(BUILD)/m23_fs_bundle && $(PYTHON) tools/pack_fs2_acoustic_checkpoint.py $(BUILD)/m23_fs.ckpt --out $(BUILD)/m23_fs_bundle
	$(PYTHON) tools/make_fake_aux_checkpoint.py $(BUILD)/m23_aux.ckpt --I 64 --C 64 --D 64 --L 2
	rm -rf $(BUILD)/m23_aux_bundle && $(PYTHON) tools/pack_aux_convnext_checkpoint.py $(BUILD)/m23_aux.ckpt --out $(BUILD)/m23_aux_bundle
	$(PYTHON) tools/make_fake_lynxnet2_backbone.py $(BUILD)/m23_rf.ckpt --input 64 --condition 64 --channels 64 --hidden 64 --layers 2
	rm -rf $(BUILD)/m23_rf_bundle && $(PYTHON) tools/pack_lynxnet2_backbone.py $(BUILD)/m23_rf.ckpt --prefix model.diffusion.denoise_fn --out $(BUILD)/m23_rf_bundle --test-frames 16
	./$(BUILD)/test_m23_model_loader $(BUILD)/m23_fs_bundle/fs2_acoustic.dsfs $(BUILD)/m23_aux_bundle/aux_convnext.dsa $(BUILD)/m23_rf_bundle/lynxnet2.dsn
m23-check: $(BUILD)/libdsasm_m23.so m23-model-check
	DSASM_PARALLEL_DW=0 DSASM_SPIN_POOL=0 DSASM_N_OWNER=0 DSASM_INDEXED_LINEAR=0 DSASM_KBLOCK_ATAN=0 $(PYTHON) tools/validate_m22_normfast.py --lib $(BUILD)/libdsasm_m23.so
m23-bench: $(BUILD)/libdsasm_m23.so
	DSASM_PARALLEL_DW=0 DSASM_SPIN_POOL=0 DSASM_N_OWNER=0 DSASM_INDEXED_LINEAR=0 DSASM_KBLOCK_ATAN=0 $(PYTHON) tools/bench_m23_paired.py --lib $(BUILD)/libdsasm_m23.so --blocks 4
m23: m23-check m21-pytorch-check m21-condition-check m20-pytorch-check m19-pytorch-check m7-bundle-check

verify: test pytorch-check bundle-check m7-pytorch-check m7-bundle-check m8-pytorch-check m9-pytorch-check m10-pytorch-check m11-pytorch-check m12-pytorch-check m13-pytorch-check m14-pytorch-check m15-pytorch-check m16-pytorch-check m17-pytorch-check m18-pytorch-check m19-pytorch-check m20-fs2-front-check m20-pytorch-check m20-postfs2-check m21-pytorch-check m21-condition-check m21-full-check m21-pack-check

clean:
	rm -rf $(BUILD)

# M24: real-checkpoint pack-all + native CLI
M24_OBJS := $(BUILD)/model_loader_runtime.o $(BUILD)/full_acoustic_runtime.o $(BUILD)/post_fs2_runtime.o $(BUILD)/aux_decoder_runtime.o $(BUILD)/fs2_front_runtime.o $(BUILD)/acoustic_runtime.o $(BUILD)/reflow_runtime.o $(BUILD)/lynxnet2_runtime.o $(BUILD)/threadpool.o $(BUILD)/vnni_pack_m40_1.o $(BUILD)/conv1d_m32.o $(BUILD)/conv1d_m36.o $(BUILD)/conv1d_m37_range.o $(BUILD)/conv1d_m39_kspec.o $(BUILD)/conv1d_m38_residual.o $(BUILD)/conv1d_m38_range_residual.o $(BUILD)/convtranspose_m33.o $(BUILD)/glu_m6.o $(BUILD)/linear_m4n16.o $(BUILD)/linear_residual_m4n16.o $(BUILD)/linear_m4n16_strided.o $(BUILD)/linear_residual_m4n16_strided.o $(BUILD)/linear_m4n16_idxstrided.o $(BUILD)/linear_residual_m4n16_idxstrided.o $(BUILD)/linear_n16_kblock.o $(BUILD)/layernorm.o $(BUILD)/dwconv_k31_tc.o $(BUILD)/dwconv_k31_tc_cstrided.o $(BUILD)/dwconv_k7_tc.o $(BUILD)/atan_glu_m7.o $(BUILD)/atan_glu_m9_strided.o $(BUILD)/add3_m7.o $(BUILD)/add_m33.o $(BUILD)/rope_m25.o $(BUILD)/silu_glu_m25.o $(BUILD)/adaptive_affine_m25.o $(BUILD)/phoneme_mean_m25.o $(BUILD)/fs2_encoder_runtime.o $(BUILD)/dot_avx2.o $(BUILD)/layernorm_precise.o

$(BUILD)/libdsasm_m24.so: $(M24_OBJS)
	$(CC) -shared -o $@ $^ -lm -pthread

$(BUILD)/dsasm-acoustic: src/cli/dsasm_acoustic.c $(M24_OBJS)
	$(CC) $(CPPFLAGS) $(CFLAGS) src/cli/dsasm_acoustic.c $(M24_OBJS) -o $@ -lm -pthread

m24-pack-check: $(BUILD)/dsasm-acoustic
	$(PYTHON) tools/make_fake_acoustic_m24_checkpoint.py $(BUILD)/m24_fake.ckpt --c 64 --layers-fs 2 --layers-aux 2 --layers-rf 2
	rm -rf $(BUILD)/m24_model
	$(PYTHON) tools/pack_acoustic_model_m24.py $(BUILD)/m24_fake.ckpt --config tests/m24_profile.yaml --out $(BUILD)/m24_model
	./$(BUILD)/dsasm-acoustic inspect $(BUILD)/m24_model
	printf '1 2 3 4\n' > $(BUILD)/m24_tokens.txt
	printf '3 4 5 4\n' > $(BUILD)/m24_durations.txt
	$(PYTHON) -c "print(' '.join(str(220.0 + i*2.5) for i in range(16)))" > $(BUILD)/m24_f0.txt
	./$(BUILD)/dsasm-acoustic infer $(BUILD)/m24_model --tokens $(BUILD)/m24_tokens.txt --durations $(BUILD)/m24_durations.txt --f0 $(BUILD)/m24_f0.txt --out $(BUILD)/m24_mel_a.f32 --seed 24 --steps 2
	./$(BUILD)/dsasm-acoustic infer $(BUILD)/m24_model --tokens $(BUILD)/m24_tokens.txt --durations $(BUILD)/m24_durations.txt --f0 $(BUILD)/m24_f0.txt --out $(BUILD)/m24_mel_b.f32 --seed 24 --steps 2
	cmp $(BUILD)/m24_mel_a.f32 $(BUILD)/m24_mel_b.f32
	test "$$(stat -c%s $(BUILD)/m24_mel_a.f32)" -eq $$((16*64*4))
	@echo 'M24 pack-all + native CLI deterministic smoke OK'

m24-check: $(BUILD)/libdsasm_m24.so m24-pack-check m23-model-check
	DSASM_PARALLEL_DW=0 DSASM_SPIN_POOL=0 DSASM_N_OWNER=0 DSASM_INDEXED_LINEAR=0 DSASM_KBLOCK_ATAN=0 $(PYTHON) tools/validate_m22_normfast.py --lib $(BUILD)/libdsasm_m24.so

m24: m24-check

# M25: real-model acceptance harness (checkpoint tensor PyTorch vs native CLI).
m25-real-check: $(BUILD)/dsasm-acoustic
	$(PYTHON) tools/make_fake_acoustic_m24_checkpoint.py $(BUILD)/m25_fake.ckpt --c 64 --layers-fs 2 --layers-aux 2 --layers-rf 2
	rm -rf $(BUILD)/m25_accept
	DSASM_PARALLEL_DW=0 DSASM_SPIN_POOL=0 DSASM_N_OWNER=0 DSASM_INDEXED_LINEAR=0 DSASM_KBLOCK_ATAN=0 $(PYTHON) tools/validate_real_model_m25.py $(BUILD)/m25_fake.ckpt --config tests/m24_profile.yaml --cli $(BUILD)/dsasm-acoustic --out $(BUILD)/m25_accept --frames 16 --tokens 4 --steps 2 --max-abs 1e-3
m25-ckpt-legacy: m25-real-check m24-check

m25-real: $(BUILD)/dsasm-acoustic
	@test -n "$(CKPT)" || (echo 'usage: make m25-real CKPT=/path/model.ckpt CONFIG=/path/acoustic.yaml [FRAMES=64 TOKENS=32 STEPS=20 THREADS=0]'; exit 2)
	@test -n "$(CONFIG)" || (echo 'usage: make m25-real CKPT=/path/model.ckpt CONFIG=/path/acoustic.yaml [FRAMES=64 TOKENS=32 STEPS=20 THREADS=0]'; exit 2)
	rm -rf $(BUILD)/m25_real_accept
	DSASM_PARALLEL_DW=0 DSASM_SPIN_POOL=0 DSASM_N_OWNER=0 DSASM_INDEXED_LINEAR=0 DSASM_KBLOCK_ATAN=0 $(PYTHON) tools/validate_real_model_m25.py "$(CKPT)" --config "$(CONFIG)" --cli $(BUILD)/dsasm-acoustic --out $(BUILD)/m25_real_accept --frames $(or $(FRAMES),64) --tokens $(or $(TOKENS),32) $(if $(STEPS),--steps $(STEPS),) --threads $(or $(THREADS),0) --max-abs 1e-3

# M25 deployment ONNX: extended conditioner + DSFS25 + real deployment importer.
$(BUILD)/libdsasm_m25.so: $(M24_OBJS)
	$(CC) -shared -o $@ $^ -lm -pthread

m25-deploy-check: $(BUILD)/libdsasm_m25.so $(BUILD)/dsasm-acoustic
	$(PYTHON) tools/validate_deploy_features_m25.py --lib $(BUILD)/libdsasm_m25.so
	$(PYTHON) -m py_compile tools/pack_acoustic_onnx_m25.py tools/make_fake_dsfs25_bundle.py
	$(PYTHON) tools/make_fake_acoustic_m24_checkpoint.py $(BUILD)/m25_deploy_fake.ckpt --c 64 --layers-fs 2 --layers-aux 2 --layers-rf 2
	rm -rf $(BUILD)/m25_deploy_base $(BUILD)/m25_deploy_model
	$(PYTHON) tools/pack_acoustic_model_m24.py $(BUILD)/m25_deploy_fake.ckpt --config tests/m24_profile.yaml --out $(BUILD)/m25_deploy_base
	mkdir -p $(BUILD)/m25_deploy_model
	$(PYTHON) tools/make_fake_dsfs25_bundle.py $(BUILD)/m25_deploy_base/fs2_acoustic.dsfs $(BUILD)/m25_deploy_model/fs2_acoustic.dsfs
	cp $(BUILD)/m25_deploy_base/aux_convnext.dsa $(BUILD)/m25_deploy_model/aux_convnext.dsa
	cp $(BUILD)/m25_deploy_base/lynxnet2.dsn $(BUILD)/m25_deploy_model/lynxnet2.dsn
	cp $(BUILD)/m25_deploy_base/model.conf $(BUILD)/m25_deploy_model/model.conf
	./$(BUILD)/dsasm-acoustic inspect $(BUILD)/m25_deploy_model
	printf '1 2 3 4\n' > $(BUILD)/m25_tokens.txt
	printf '3 4 5 4\n' > $(BUILD)/m25_durations.txt
	$(PYTHON) -c "print(' '.join(str(220.0 + i*2.5) for i in range(16)))" > $(BUILD)/m25_f0.txt
	$(PYTHON) -c "import numpy as np; np.zeros(64,np.float32).tofile('$(BUILD)/m25_speaker.emb')"
	./$(BUILD)/dsasm-acoustic infer $(BUILD)/m25_deploy_model --tokens $(BUILD)/m25_tokens.txt --durations $(BUILD)/m25_durations.txt --f0 $(BUILD)/m25_f0.txt --language-id 4 --speaker-emb $(BUILD)/m25_speaker.emb --depth 0.6 --steps 2 --seed 25 --out $(BUILD)/m25_mel_a.f32
	./$(BUILD)/dsasm-acoustic infer $(BUILD)/m25_deploy_model --tokens $(BUILD)/m25_tokens.txt --durations $(BUILD)/m25_durations.txt --f0 $(BUILD)/m25_f0.txt --language-id 4 --speaker-emb $(BUILD)/m25_speaker.emb --depth 0.6 --steps 2 --seed 25 --out $(BUILD)/m25_mel_b.f32
	cmp $(BUILD)/m25_mel_a.f32 $(BUILD)/m25_mel_b.f32
	@echo 'M25 DSFS25 loader + extended CLI deterministic smoke OK'

m25-onnx-real: $(BUILD)/libdsasm_m25.so $(BUILD)/dsasm-acoustic
	@test -n "$(MODEL_DIR)" || (echo 'usage: make m25-onnx-real MODEL_DIR=/path/to/DiffSinger-voicebank'; exit 2)
	test -f "$(MODEL_DIR)/acoustic.onnx"
	rm -rf $(BUILD)/m25_real_onnx
	$(PYTHON) tools/pack_acoustic_onnx_m25.py "$(MODEL_DIR)/acoustic.onnx" --model-dir "$(MODEL_DIR)" --out $(BUILD)/m25_real_onnx
	./$(BUILD)/dsasm-acoustic inspect $(BUILD)/m25_real_onnx

# Redefinition below intentionally makes M25 mean the deployment-ONNX milestone.
m25: m25-deploy-check m24-check

# M27: real deployment FS2 stage locator. No model math changes; exposes
# matched ONNX/native intermediate tensors to identify the first divergence.
m27-debug-check: m25-deploy-check
	$(PYTHON) -m py_compile tools/validate_real_onnx_m27.py
	rm -rf $(BUILD)/m27_fake_stages && mkdir -p $(BUILD)/m27_fake_stages
	./$(BUILD)/dsasm-acoustic infer $(BUILD)/m25_deploy_model --tokens $(BUILD)/m25_tokens.txt --durations $(BUILD)/m25_durations.txt --f0 $(BUILD)/m25_f0.txt --language-id 4 --speaker-emb $(BUILD)/m25_speaker.emb --depth 0.6 --steps 2 --seed 27 --dump-fs2-stages $(BUILD)/m27_fake_stages --out $(BUILD)/m27_fake_mel.f32
	test "$$(stat -c%s $(BUILD)/m27_fake_stages/encoder_txt.f32)" -eq $$((4*64*4))
	@for f in gathered stretch gru pitch variance key_shift speed speaker; do test "$$(stat -c%s $(BUILD)/m27_fake_stages/$$f.f32)" -eq $$((16*64*4)) || exit 1; done
	@echo 'M27 FS2 stage dump smoke OK'

m27-real-parity: $(BUILD)/dsasm-acoustic $(BUILD)/libdsasm_m25.so
	@test -n "$(MODEL_DIR)" || (echo 'usage: make m27-real-parity MODEL_DIR=/path/voicebank SPEAKER_EMB=/path/singer.emb'; exit 2)
	@test -n "$(SPEAKER_EMB)" || (echo 'usage: make m27-real-parity MODEL_DIR=/path/voicebank SPEAKER_EMB=/path/singer.emb'; exit 2)
	rm -rf $(BUILD)/m27_real_model $(BUILD)/m27_real
	$(PYTHON) tools/pack_acoustic_onnx_m25.py "$(MODEL_DIR)/acoustic.onnx" --model-dir "$(MODEL_DIR)" --out $(BUILD)/m27_real_model
	./$(BUILD)/dsasm-acoustic inspect $(BUILD)/m27_real_model
	$(PYTHON) tools/validate_real_onnx_m27.py --fs2-stages --onnx "$(MODEL_DIR)/acoustic.onnx" --packed $(BUILD)/m27_real_model --cli ./$(BUILD)/dsasm-acoustic --speaker-emb "$(SPEAKER_EMB)" --language-id $(or $(LANGUAGE_ID),4) --depth $(or $(DEPTH),0.6) --steps $(or $(STEPS),20) --work $(BUILD)/m27_real

m27: m27-debug-check

# M28: real-model encoder diagnosis + accuracy-oriented FS2 LayerNorm A/B.
# Also validates the ONNX importer's real speed shared-bias topology.
m28-check: $(BUILD)/dsasm-acoustic $(BUILD)/libdsasm_m25.so
	$(PYTHON) -m py_compile tools/validate_real_onnx_m28.py tools/pack_acoustic_onnx_m25.py
	DSASM_FS2_PRECISE_LN=0 $(PYTHON) tools/validate_pytorch_fs2_encoder_m21.py --lib $(BUILD)/libdsasm_m25.so
	DSASM_FS2_PRECISE_LN=1 $(PYTHON) tools/validate_pytorch_fs2_encoder_m21.py --lib $(BUILD)/libdsasm_m25.so
	@echo 'M28 fast/precise FS2 LayerNorm synthetic parity OK'

m28-real-parity: $(BUILD)/dsasm-acoustic $(BUILD)/libdsasm_m25.so
	@test -n "$(MODEL_DIR)" || (echo 'usage: make m28-real-parity MODEL_DIR=/path/voicebank SPEAKER_EMB=/path/singer.emb'; exit 2)
	@test -n "$(SPEAKER_EMB)" || (echo 'usage: make m28-real-parity MODEL_DIR=/path/voicebank SPEAKER_EMB=/path/singer.emb'; exit 2)
	rm -rf $(BUILD)/m28_real_model $(BUILD)/m28_real
	$(PYTHON) tools/pack_acoustic_onnx_m25.py "$(MODEL_DIR)/acoustic.onnx" --model-dir "$(MODEL_DIR)" --out $(BUILD)/m28_real_model
	./$(BUILD)/dsasm-acoustic inspect $(BUILD)/m28_real_model
	$(PYTHON) tools/validate_real_onnx_m28.py --onnx "$(MODEL_DIR)/acoustic.onnx" --packed $(BUILD)/m28_real_model --cli ./$(BUILD)/dsasm-acoustic --speaker-emb "$(SPEAKER_EMB)" --language-id $(or $(LANGUAGE_ID),4) --depth $(or $(DEPTH),0.6) --steps $(or $(STEPS),20) --work $(BUILD)/m28_real

m28: m28-check

# M29: real deployment cross-lingual language mask.  The upstream ONNX
# applies language IDs only to tokens listed in cross_lingual_token_idx.
m29-check: $(BUILD)/dsasm-acoustic $(BUILD)/libdsasm_m25.so
	$(PYTHON) -m py_compile tools/pack_acoustic_onnx_m25.py tools/validate_real_onnx_m29.py
	$(PYTHON) tools/validate_deploy_features_m25.py --lib $(BUILD)/libdsasm_m25.so
	@echo 'M29 cross-lingual language-mask synthetic parity OK'

m29-real-parity: $(BUILD)/dsasm-acoustic $(BUILD)/libdsasm_m25.so
	@test -n "$(MODEL_DIR)" || (echo 'usage: make m29-real-parity MODEL_DIR=/path/voicebank SPEAKER_EMB=/path/singer.emb'; exit 2)
	@test -n "$(SPEAKER_EMB)" || (echo 'usage: make m29-real-parity MODEL_DIR=/path/voicebank SPEAKER_EMB=/path/singer.emb'; exit 2)
	rm -rf $(BUILD)/m29_real_model $(BUILD)/m29_real
	$(PYTHON) tools/pack_acoustic_onnx_m25.py "$(MODEL_DIR)/acoustic.onnx" --model-dir "$(MODEL_DIR)" --out $(BUILD)/m29_real_model
	./$(BUILD)/dsasm-acoustic inspect $(BUILD)/m29_real_model
	$(PYTHON) tools/validate_real_onnx_m29.py --onnx "$(MODEL_DIR)/acoustic.onnx" --packed $(BUILD)/m29_real_model --cli ./$(BUILD)/dsasm-acoustic --speaker-emb "$(SPEAKER_EMB)" --language-id $(or $(LANGUAGE_ID),4) --depth $(or $(DEPTH),0.6) --steps $(or $(STEPS),20) --work $(BUILD)/m29_real

m29: m29-check

# M30: first real voicebank waveform path. Native packed acoustic remains the
# production acoustic engine; the bundled NSF-HiFiGAN ONNX is intentionally run
# through ORT to establish an end-to-end waveform golden reference before a
# native vocoder port.
m30-check: $(BUILD)/dsasm-acoustic $(BUILD)/libdsasm_m25.so
	$(PYTHON) -m py_compile tools/run_real_voicebank_m30.py tools/validate_real_onnx_m29.py
	$(PYTHON) tools/validate_deploy_features_m25.py --lib $(BUILD)/libdsasm_m25.so
	@echo 'M30 acoustic regression + waveform harness syntax OK'

m30-real-wave: $(BUILD)/dsasm-acoustic $(BUILD)/libdsasm_m25.so
	@test -n "$(MODEL_DIR)" || (echo 'usage: make m30-real-wave MODEL_DIR=/path/voicebank SPEAKER_EMB=/path/singer.emb'; exit 2)
	@test -n "$(SPEAKER_EMB)" || (echo 'usage: make m30-real-wave MODEL_DIR=/path/voicebank SPEAKER_EMB=/path/singer.emb'; exit 2)
	test -f "$(MODEL_DIR)/acoustic.onnx"
	test -f "$(MODEL_DIR)/dsvocoder/nsf_hifigan.onnx"
	rm -rf $(BUILD)/m30_real_model $(BUILD)/m30_real
	$(PYTHON) tools/pack_acoustic_onnx_m25.py "$(MODEL_DIR)/acoustic.onnx" --model-dir "$(MODEL_DIR)" --out $(BUILD)/m30_real_model
	./$(BUILD)/dsasm-acoustic inspect $(BUILD)/m30_real_model
	$(PYTHON) tools/run_real_voicebank_m30.py \
	  --acoustic-onnx "$(MODEL_DIR)/acoustic.onnx" \
	  --vocoder-onnx "$(MODEL_DIR)/dsvocoder/nsf_hifigan.onnx" \
	  --packed $(BUILD)/m30_real_model \
	  --cli ./$(BUILD)/dsasm-acoustic \
	  --speaker-emb "$(SPEAKER_EMB)" \
	  --language-id $(or $(LANGUAGE_ID),4) \
	  --depth $(or $(DEPTH),0.6) \
	  --steps $(or $(STEPS),20) \
	  --threads $(or $(THREADS),0) \
	  --work $(BUILD)/m30_real

m30: m30-check

# M31: CPU realtime performance sprint.  The hard product requirement is
# end-to-end CPU RTF < 1.0 on the target i5-13420H; engineering target <=0.8.
# No acoustic math changes here: sweep Rectified-Flow steps, split native
# FS2/Aux/RF wall time, sweep NSF-HiFiGAN ORT CPU threading, and profile the
# vocoder graph/nodes to choose the next native/ASM optimization.
m31-check: m25-deploy-check $(BUILD)/dsasm-acoustic
	$(PYTHON) -m py_compile tools/realtime_sprint_m31.py tools/run_real_voicebank_m30.py
	./$(BUILD)/dsasm-acoustic infer $(BUILD)/m25_deploy_model --tokens $(BUILD)/m25_tokens.txt --durations $(BUILD)/m25_durations.txt --f0 $(BUILD)/m25_f0.txt --language-id 4 --speaker-emb $(BUILD)/m25_speaker.emb --depth 0.6 --steps 2 --seed 25 --profile-stages --out $(BUILD)/m31_profile_mel.f32 | tee $(BUILD)/m31_profile_smoke.txt
	cmp $(BUILD)/m25_mel_a.f32 $(BUILD)/m31_profile_mel.f32
	grep -q 'stages: fs2=' $(BUILD)/m31_profile_smoke.txt
	@echo 'M31 stage profiler is bit-exact with full wrapper; realtime harness syntax OK'

m31-real-sprint: $(BUILD)/dsasm-acoustic $(BUILD)/libdsasm_m25.so
	@test -n "$(MODEL_DIR)" || (echo 'usage: make m31-real-sprint MODEL_DIR=/path/voicebank SPEAKER_EMB=/path/singer.emb'; exit 2)
	@test -n "$(SPEAKER_EMB)" || (echo 'usage: make m31-real-sprint MODEL_DIR=/path/voicebank SPEAKER_EMB=/path/singer.emb'; exit 2)
	test -f "$(MODEL_DIR)/acoustic.onnx"
	test -f "$(MODEL_DIR)/dsvocoder/nsf_hifigan.onnx"
	rm -rf $(BUILD)/m31_real_model $(BUILD)/m31_realtime
	$(PYTHON) tools/pack_acoustic_onnx_m25.py "$(MODEL_DIR)/acoustic.onnx" --model-dir "$(MODEL_DIR)" --out $(BUILD)/m31_real_model
	./$(BUILD)/dsasm-acoustic inspect $(BUILD)/m31_real_model
	$(PYTHON) tools/realtime_sprint_m31.py \
	  --vocoder-onnx "$(MODEL_DIR)/dsvocoder/nsf_hifigan.onnx" \
	  --packed $(BUILD)/m31_real_model \
	  --cli ./$(BUILD)/dsasm-acoustic \
	  --speaker-emb "$(SPEAKER_EMB)" \
	  --language-id $(or $(LANGUAGE_ID),4) \
	  --depth $(or $(DEPTH),0.6) \
	  --steps "$(M31_STEPS)" \
	  --native-threads $(or $(NATIVE_THREADS),8) \
	  --acoustic-rounds $(M31_ACOUSTIC_ROUNDS) \
	  --vocoder-threads "$(M31_VOCODER_THREADS)" \
	  --vocoder-rounds $(M31_VOCODER_ROUNDS) \
	  --profile-runs $(M31_PROFILE_RUNS) \
	  --cpus "$(M31_CPUS)" \
	  --work $(BUILD)/m31_realtime

m31: m31-check

# ---------------------------------------------------------------------------
# M32: PURE CPU/ASM NSF-HiFiGAN Conv1d kernel lab.
# ORT is used only OFFLINE to expose a real node input/output as golden data.
# The timed execution path is our own C runtime + x86-64 AVX2/FMA assembly.
# No oneDNN/OpenVINO/BLAS/ORT execution is used by test_m32_vocoder_conv.
# ---------------------------------------------------------------------------
M32_ROUNDS ?= 5
M32_NODE ?= /generator/resblocks.5/convs1.0/Conv

$(BUILD)/conv1d_m32.o: $(KDIR)/conv1d_nct_f32_avx2_oc4_t8.S | $(BUILD)
	$(CC) -c $< -o $@

$(BUILD)/vocoder_conv_runtime.o: $(RDIR)/vocoder_conv.c include/dsasm_vocoder.h include/dsasm_kernels.h $(RDIR)/threadpool_internal.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -ffp-contract=off -fPIC -c $< -o $@

M32_POOL_KERNELS := $(BUILD)/linear_m4n16.o $(BUILD)/linear_residual_m4n16.o \
                   $(BUILD)/linear_m4n16_strided.o $(BUILD)/linear_residual_m4n16_strided.o \
                   $(BUILD)/linear_m4n16_idxstrided.o $(BUILD)/linear_residual_m4n16_idxstrided.o \
                   $(BUILD)/linear_n16_kblock.o $(BUILD)/dwconv_k31_tc_cstrided.o \
                   $(BUILD)/glu_m6.o $(BUILD)/atan_glu_m9_strided.o $(BUILD)/conv1d_m39_vnni.o $(BUILD)/vnni_pack_m40_1.o $(BUILD)/conv1d_m36.o $(BUILD)/conv1d_m37_range.o $(BUILD)/conv1d_m39_kspec.o $(BUILD)/conv1d_m38_residual.o $(BUILD)/conv1d_m38_range_residual.o $(BUILD)/leaky_copy_m36.o

$(BUILD)/test_m32_vocoder_conv: tests/test_m32_vocoder_conv.c $(BUILD)/vocoder_conv_runtime.o $(BUILD)/threadpool.o $(BUILD)/conv1d_m32.o $(BUILD)/convtranspose_m33.o $(BUILD)/leaky_relu_m33.o $(BUILD)/add_m33.o $(M32_POOL_KERNELS)
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_m32_vocoder_conv.c $(BUILD)/vocoder_conv_runtime.o $(BUILD)/threadpool.o $(BUILD)/conv1d_m32.o $(BUILD)/convtranspose_m33.o $(BUILD)/leaky_relu_m33.o $(BUILD)/add_m33.o $(M32_POOL_KERNELS) -o $@ -lm -pthread

m32-check: $(BUILD)/test_m32_vocoder_conv
	$(PYTHON) -m py_compile tools/make_fake_vocoder_conv_m32.py tools/pack_vocoder_hot_conv_m32.py
	$(PYTHON) tools/make_fake_vocoder_conv_m32.py $(BUILD)/m32_fake.dsv32 --cin 32 --cout 32 --k 11 --tin 257 --dilation 1
	./$(BUILD)/test_m32_vocoder_conv $(BUILD)/m32_fake.dsv32 3
	$(PYTHON) tools/make_fake_vocoder_conv_m32.py $(BUILD)/m32_fake_d3.dsv32 --cin 16 --cout 32 --k 7 --tin 259 --dilation 3
	./$(BUILD)/test_m32_vocoder_conv $(BUILD)/m32_fake_d3.dsv32 3
	@echo 'M32 pure-ASM Conv1d synthetic parity OK'

m32-real-kernel: $(BUILD)/test_m32_vocoder_conv
	@test -n "$(MODEL_DIR)" || (echo 'usage: make m32-real-kernel MODEL_DIR=/path/voicebank'; exit 2)
	test -f "$(MODEL_DIR)/dsvocoder/nsf_hifigan.onnx"
	rm -rf $(BUILD)/m32_real && mkdir -p $(BUILD)/m32_real
	$(PYTHON) tools/pack_vocoder_hot_conv_m32.py "$(MODEL_DIR)/dsvocoder/nsf_hifigan.onnx" --node "$(M32_NODE)" --frames 48 --out $(BUILD)/m32_real/hot0.dsv32
	cat $(BUILD)/m32_real/hot0.json
	./$(BUILD)/test_m32_vocoder_conv $(BUILD)/m32_real/hot0.dsv32 $(M32_ROUNDS)
	$(PYTHON) tools/pack_vocoder_hot_conv_m32.py "$(MODEL_DIR)/dsvocoder/nsf_hifigan.onnx" --node "/generator/resblocks.5/convs2.0/Conv" --frames 48 --out $(BUILD)/m32_real/hot1.dsv32
	./$(BUILD)/test_m32_vocoder_conv $(BUILD)/m32_real/hot1.dsv32 $(M32_ROUNDS)
	$(PYTHON) tools/pack_vocoder_hot_conv_m32.py "$(MODEL_DIR)/dsvocoder/nsf_hifigan.onnx" --node "/generator/resblocks.4/convs1.0/Conv" --frames 48 --out $(BUILD)/m32_real/hot2.dsv32
	./$(BUILD)/test_m32_vocoder_conv $(BUILD)/m32_real/hot2.dsv32 $(M32_ROUNDS)
	@echo 'M32 real hot-Conv pure-ASM benchmark complete'

m32: m32-check


# ---------------------------------------------------------------------------
# M33: PURE CPU/ASM NSF-HiFiGAN structural units.
# - direct sparse ConvTranspose1d (no zero-insertion waste)
# - AVX2 LeakyReLU + residual Add
# - complete HiFi-GAN residual unit on the persistent P-core pool
# ORT remains OFFLINE-only for extracting real tensors/golden outputs.
# ---------------------------------------------------------------------------
M33_ROUNDS ?= 7
M33_RES_PREFIX ?= /generator/resblocks.5
M33_RES_UNIT ?= 0
M33_UP_NODE ?= /generator/ups.1/ConvTranspose

$(BUILD)/convtranspose_m33.o: $(KDIR)/convtranspose1d_nct_f32_avx2.S | $(BUILD)
	$(CC) -c $< -o $@
$(BUILD)/leaky_relu_m33.o: $(KDIR)/leaky_relu_f32_avx2.S | $(BUILD)
	$(CC) -c $< -o $@
$(BUILD)/add_m33.o: $(KDIR)/add_f32_avx2.S | $(BUILD)
	$(CC) -c $< -o $@

M33_VOCODER_OBJS := $(BUILD)/vocoder_conv_runtime.o $(BUILD)/threadpool.o \
                    $(BUILD)/conv1d_m32.o $(BUILD)/convtranspose_m33.o \
                    $(BUILD)/leaky_relu_m33.o $(BUILD)/add_m33.o $(M32_POOL_KERNELS)

$(BUILD)/test_m33_vocoder_resunit: tests/test_m33_vocoder_resunit.c $(M33_VOCODER_OBJS)
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_m33_vocoder_resunit.c $(M33_VOCODER_OBJS) -o $@ -lm -pthread
$(BUILD)/test_m33_vocoder_convtranspose: tests/test_m33_vocoder_convtranspose.c $(M33_VOCODER_OBJS)
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_m33_vocoder_convtranspose.c $(M33_VOCODER_OBJS) -o $@ -lm -pthread

m33-check: $(BUILD)/test_m33_vocoder_resunit $(BUILD)/test_m33_vocoder_convtranspose
	$(PYTHON) -m py_compile tools/make_fake_vocoder_resunit_m33.py tools/make_fake_vocoder_convtranspose_m33.py tools/pack_vocoder_resunit_m33.py tools/pack_vocoder_convtranspose_m33.py
	$(PYTHON) tools/make_fake_vocoder_resunit_m33.py $(BUILD)/m33_fake_resunit.dsvru33 --c 16 --t 97 --k1 7 --d1 3 --k2 7 --d2 1
	./$(BUILD)/test_m33_vocoder_resunit $(BUILD)/m33_fake_resunit.dsvru33 3
	$(PYTHON) tools/make_fake_vocoder_convtranspose_m33.py $(BUILD)/m33_fake_up.dsvct33 --cin 16 --cout 8 --k 8 --tin 33 --stride 4 --pad 2
	./$(BUILD)/test_m33_vocoder_convtranspose $(BUILD)/m33_fake_up.dsvct33 3
	@if ldd $(BUILD)/test_m33_vocoder_resunit $(BUILD)/test_m33_vocoder_convtranspose | grep -Eqi 'onnx|dnnl|onednn|openvino|mkl|blas'; then echo 'ERROR: forbidden runtime backend dependency detected'; exit 1; else echo 'M33 runtime purity: OK (no ORT/oneDNN/OpenVINO/MKL/BLAS)'; fi
	@echo 'M33 pure-ASM residual-unit + ConvTranspose synthetic parity OK'

m33-real-blocks: $(BUILD)/test_m33_vocoder_resunit $(BUILD)/test_m33_vocoder_convtranspose
	@test -n "$(MODEL_DIR)" || (echo 'usage: make m33-real-blocks MODEL_DIR=/path/voicebank'; exit 2)
	test -f "$(MODEL_DIR)/dsvocoder/nsf_hifigan.onnx"
	rm -rf $(BUILD)/m33_real && mkdir -p $(BUILD)/m33_real
	@for u in 0 1 2; do \
	  echo "===== REAL RESBLOCK.5 UNIT $$u ====="; \
	  $(PYTHON) tools/pack_vocoder_resunit_m33.py "$(MODEL_DIR)/dsvocoder/nsf_hifigan.onnx" --prefix "/generator/resblocks.5" --unit $$u --frames 48 --out $(BUILD)/m33_real/res5_u$$u.dsvru33 || exit $$?; \
	  cat $(BUILD)/m33_real/res5_u$$u.json; \
	  ./$(BUILD)/test_m33_vocoder_resunit $(BUILD)/m33_real/res5_u$$u.dsvru33 $(M33_ROUNDS) || exit $$?; \
	done
	@for u in 0 1 2 3 4; do \
	  echo "===== REAL UPSAMPLER $$u ====="; \
	  $(PYTHON) tools/pack_vocoder_convtranspose_m33.py "$(MODEL_DIR)/dsvocoder/nsf_hifigan.onnx" --node "/generator/ups.$$u/ConvTranspose" --frames 48 --out $(BUILD)/m33_real/up$$u.dsvct33 || exit $$?; \
	  cat $(BUILD)/m33_real/up$$u.json; \
	  ./$(BUILD)/test_m33_vocoder_convtranspose $(BUILD)/m33_real/up$$u.dsvct33 $(M33_ROUNDS) || exit $$?; \
	done
	@echo 'M33 real pure-ASM vocoder structural-unit benchmark complete'

m33: m33-check


# M34: phase-vectorized pure-ASM ConvTranspose for stride=2,K=4,pad=1.
$(BUILD)/test_m34_vocoder_convtranspose: tests/test_m34_vocoder_convtranspose.c $(M33_VOCODER_OBJS)
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_m34_vocoder_convtranspose.c $(M33_VOCODER_OBJS) -o $@ -lm -pthread

m34-check: $(BUILD)/test_m34_vocoder_convtranspose
	$(PYTHON) -m py_compile tools/make_fake_vocoder_convtranspose_m33.py tools/pack_vocoder_convtranspose_m33.py
	$(PYTHON) tools/make_fake_vocoder_convtranspose_m33.py $(BUILD)/m34_fake_s2k4.dsvct33 --cin 16 --cout 8 --k 4 --tin 259 --stride 2 --pad 1
	./$(BUILD)/test_m34_vocoder_convtranspose $(BUILD)/m34_fake_s2k4.dsvct33 5
	$(PYTHON) tools/make_fake_vocoder_convtranspose_m33.py $(BUILD)/m34_fake_generic.dsvct33 --cin 16 --cout 8 --k 8 --tin 33 --stride 4 --pad 2
	./$(BUILD)/test_m34_vocoder_convtranspose $(BUILD)/m34_fake_generic.dsvct33 5
	@if ldd $(BUILD)/test_m34_vocoder_convtranspose | grep -Eqi 'onnx|dnnl|onednn|openvino|mkl|blas'; then echo 'ERROR: forbidden runtime backend dependency detected'; exit 1; else echo 'M34 runtime purity: OK (no ORT/oneDNN/OpenVINO/MKL/BLAS)'; fi
	@echo 'M34 specialized s2/k4 + generic ConvTranspose parity OK'

M34_ROUNDS ?= 9
m34-real-ct: $(BUILD)/test_m34_vocoder_convtranspose
	@test -n "$(MODEL_DIR)" || (echo 'usage: make m34-real-ct MODEL_DIR=/path/voicebank'; exit 2)
	@test -f "$(MODEL_DIR)/dsvocoder/nsf_hifigan.onnx"
	rm -rf $(BUILD)/m34_real && mkdir -p $(BUILD)/m34_real
	@for u in 0 1 2 3 4; do \
	  echo "===== M34 REAL UPSAMPLER $$u ====="; \
	  $(PYTHON) tools/pack_vocoder_convtranspose_m33.py "$(MODEL_DIR)/dsvocoder/nsf_hifigan.onnx" --node "/generator/ups.$$u/ConvTranspose" --frames 48 --out $(BUILD)/m34_real/up$$u.dsvct33 || exit $$?; \
	  cat $(BUILD)/m34_real/up$$u.json; \
	  ./$(BUILD)/test_m34_vocoder_convtranspose $(BUILD)/m34_real/up$$u.dsvct33 $(M34_ROUNDS) || exit $$?; \
	done
	@echo 'M34 real ConvTranspose benchmark complete (ups.0/1 generic, ups.2/3/4 s2k4-time8)'

m34: m34-check

# -----------------------------------------------------------------------------
# M35: full fixed-shape pure-native NSF-HiFiGAN graph executor.
# ONNX/ORT are pack-time/golden-only; build/dsasm-vocoder links no inference lib.
# -----------------------------------------------------------------------------
.PHONY: m35 m35-check m35-real-vocoder

$(BUILD)/vocoder_graph_runtime.o: $(RDIR)/vocoder_graph.c include/dsasm_vocoder_graph.h include/dsasm_vocoder.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -fPIC -c $< -o $@
$(BUILD)/vocoder_vnni_runtime.o: $(RDIR)/vocoder_vnni.c include/dsasm_vocoder.h $(RDIR)/threadpool_internal.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -fPIC -c $< -o $@

$(BUILD)/engine_runtime.o: $(RDIR)/engine.c include/dsasm_engine.h include/dsasm_model.h include/dsasm_vocoder_graph.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -fPIC -pthread -c $< -o $@

M35_VOCODER_OBJS := $(BUILD)/vocoder_graph_runtime.o $(BUILD)/vocoder_vnni_runtime.o $(M33_VOCODER_OBJS)

PRODUCT_OBJS = $(sort $(M24_OBJS) $(M35_VOCODER_OBJS) $(BUILD)/engine_runtime.o)

$(BUILD)/libdsasm.so: $(PRODUCT_OBJS) src/runtime/libdsasm.map
	$(CC) -shared -Wl,--version-script=src/runtime/libdsasm.map -o $@ $(PRODUCT_OBJS) -lm -pthread

$(BUILD)/test_engine_abi: tests/test_engine_abi.c $(BUILD)/libdsasm.so
	$(CC) $(CPPFLAGS) $(CFLAGS) $< -L$(BUILD) -Wl,-rpath,'$$ORIGIN' -ldsasm -o $@

$(BUILD)/test_engine_stream: tests/test_engine_stream.c $(BUILD)/libdsasm.so
	$(CC) $(CPPFLAGS) $(CFLAGS) $< -L$(BUILD) -Wl,-rpath,'$$ORIGIN' -ldsasm -o $@

$(BUILD)/bench_engine_stream: tools/bench_engine_stream.c $(BUILD)/libdsasm.so
	$(CC) $(CPPFLAGS) $(CFLAGS) $< -L$(BUILD) -Wl,-rpath,'$$ORIGIN' -ldsasm -o $@ -lm

$(BUILD)/bench_engine_cancel: tools/bench_engine_cancel.c $(BUILD)/libdsasm.so
	$(CC) $(CPPFLAGS) $(CFLAGS) $< -L$(BUILD) -Wl,-rpath,'$$ORIGIN' -ldsasm -o $@ -lm -pthread

$(BUILD)/probe_stage_pipeline: tools/probe_stage_pipeline.c $(M24_OBJS) $(M35_VOCODER_OBJS)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(sort $(M24_OBJS) $(M35_VOCODER_OBJS)) -o $@ -lm -pthread

$(BUILD)/bench_onednn_conv: tools/bench_onednn_conv.cpp $(M35_VOCODER_OBJS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(M35_VOCODER_OBJS) -o $@ -ldnnl -lm -pthread

engine-check: $(BUILD)/test_engine_abi
	./$(BUILD)/test_engine_abi
	@if ldd $(BUILD)/libdsasm.so | grep -Eqi 'onnx|onnxruntime|dnnl|onednn|openvino|mkl|blas'; then echo 'ERROR: forbidden runtime backend dependency detected'; exit 1; else echo 'engine runtime purity: OK'; fi
	@bad="$$(nm -D --defined-only $(BUILD)/libdsasm.so | awk '$$2 ~ /^[TW]$$/ { sub(/@@.*/, "", $$3); print $$3 }' | grep -Ev '^dsasm_engine_' || true)"; test -z "$$bad" || { echo "ERROR: unexpected public symbols:"; echo "$$bad"; exit 1; }

engine: engine-check

model-tool-check:
	$(PYTHON) -m py_compile tools/dsasm_model_tool.py
	$(PYTHON) -m unittest -v tests.test_model_tool

package: engine-check model-tool-check $(BUILD)/dsasm-acoustic $(BUILD)/dsasm-vocoder-m40
	./tools/package-engine.sh "$(VERSION)" "$(DIST)"

engine-real-stream-check: $(BUILD)/test_engine_stream
	@test -n "$(ENGINE_REAL_ACOUSTIC)" -a -n "$(ENGINE_REAL_VOCODER)" || (echo 'usage: make engine-real-stream-check ENGINE_REAL_ACOUSTIC=/packed/acoustic ENGINE_REAL_VOCODER=/bucket/dir-or-file'; exit 2)
	./$(BUILD)/test_engine_stream "$(ENGINE_REAL_ACOUSTIC)" "$(ENGINE_REAL_VOCODER)"

$(BUILD)/dsasm-vocoder: src/cli/dsasm_vocoder.c $(M35_VOCODER_OBJS)
	$(CC) $(CPPFLAGS) $(CFLAGS) src/cli/dsasm_vocoder.c $(M35_VOCODER_OBJS) -o $@ -lm -pthread

m35-check: $(BUILD)/dsasm-vocoder
	$(PYTHON) -m py_compile tools/dsv35_common.py tools/make_fake_vocoder_graph_m35.py tools/pack_vocoder_graph_m35.py
	rm -rf $(BUILD)/m35_fake && mkdir -p $(BUILD)/m35_fake
	$(PYTHON) tools/make_fake_vocoder_graph_m35.py $(BUILD)/m35_fake/model.dsv35 --work $(BUILD)/m35_fake
	./$(BUILD)/dsasm-vocoder inspect $(BUILD)/m35_fake/model.dsv35
	./$(BUILD)/dsasm-vocoder infer $(BUILD)/m35_fake/model.dsv35 --mel $(BUILD)/m35_fake/mel.f32 --f0 $(BUILD)/m35_fake/f0.f32 --out $(BUILD)/m35_fake/native.f32 --golden $(BUILD)/m35_fake/golden_wave.f32 --workers 4 --rounds 5 --profile
	@if ldd $(BUILD)/dsasm-vocoder | grep -Eqi 'onnx|onnxruntime|dnnl|onednn|openvino|mkl|blas'; then echo 'ERROR: forbidden runtime backend dependency detected'; exit 1; else echo 'M35 runtime purity: OK (pure C/ASM + libc/libm/pthread)'; fi
	@echo 'M35 fixed-shape pure-native graph executor synthetic parity OK'

M35_FRAMES ?= 48
M35_WORKERS ?= 8
M35_ROUNDS ?= 5
m35-real-vocoder: $(BUILD)/dsasm-vocoder
	@test -n "$(MODEL_DIR)" || (echo 'usage: make m35-real-vocoder MODEL_DIR=/path/voicebank'; exit 2)
	@test -f "$(MODEL_DIR)/dsvocoder/nsf_hifigan.onnx"
	rm -rf $(BUILD)/m35_real && mkdir -p $(BUILD)/m35_real
	$(PYTHON) tools/pack_vocoder_graph_m35.py "$(MODEL_DIR)/dsvocoder/nsf_hifigan.onnx" --frames $(M35_FRAMES) --out $(BUILD)/m35_real/nsf_hifigan.dsv35 --work $(BUILD)/m35_real
	./$(BUILD)/dsasm-vocoder inspect $(BUILD)/m35_real/nsf_hifigan.dsv35
	./$(BUILD)/dsasm-vocoder infer $(BUILD)/m35_real/nsf_hifigan.dsv35 --mel $(BUILD)/m35_real/mel.f32 --f0 $(BUILD)/m35_real/f0.f32 --out $(BUILD)/m35_real/native_wave.f32 --wav $(BUILD)/m35_real/native_wave.wav --golden $(BUILD)/m35_real/golden_wave.f32 --workers $(M35_WORKERS) --rounds $(M35_ROUNDS) --profile
	@if ldd $(BUILD)/dsasm-vocoder | grep -Eqi 'onnx|onnxruntime|dnnl|onednn|openvino|mkl|blas'; then echo 'ERROR: forbidden runtime backend dependency detected'; exit 1; else echo 'M35 REAL runtime purity: OK'; fi
	@echo 'M35 real fixed-shape pure-ASM NSF-HiFiGAN run complete'

m35: m35-check

M35_STEPS ?= 4
M35_E2E_WORKERS ?= 8
M35_E2E_ROUNDS ?= 5
m35-real-e2e: $(BUILD)/dsasm-vocoder $(BUILD)/dsasm-acoustic
	@test -n "$(MODEL_DIR)" || (echo 'usage: make m35-real-e2e MODEL_DIR=/path/voicebank SPEAKER_EMB=/path/speaker.emb'; exit 2)
	@test -n "$(SPEAKER_EMB)" || (echo 'SPEAKER_EMB required'; exit 2)
	rm -rf $(BUILD)/m35_e2e_model $(BUILD)/m35_e2e && mkdir -p $(BUILD)/m35_e2e
	$(PYTHON) tools/pack_acoustic_onnx_m25.py "$(MODEL_DIR)/acoustic.onnx" --model-dir "$(MODEL_DIR)" --out $(BUILD)/m35_e2e_model
	$(PYTHON) tools/pack_vocoder_graph_m35.py "$(MODEL_DIR)/dsvocoder/nsf_hifigan.onnx" --frames 48 --out $(BUILD)/m35_e2e/nsf_hifigan.dsv35 --work $(BUILD)/m35_e2e/packer
	$(PYTHON) tools/run_native_e2e_m35.py --packed-acoustic $(BUILD)/m35_e2e_model --acoustic-cli ./$(BUILD)/dsasm-acoustic --vocoder-bundle $(BUILD)/m35_e2e/nsf_hifigan.dsv35 --vocoder-cli ./$(BUILD)/dsasm-vocoder --vocoder-onnx "$(MODEL_DIR)/dsvocoder/nsf_hifigan.onnx" --speaker-emb "$(SPEAKER_EMB)" --language-id 4 --depth 0.6 --steps $(M35_STEPS) --workers $(M35_E2E_WORKERS) --rounds $(M35_E2E_ROUNDS) --work $(BUILD)/m35_e2e/run
	@if ldd $(BUILD)/dsasm-acoustic $(BUILD)/dsasm-vocoder | grep -Eqi 'onnx|onnxruntime|dnnl|onednn|openvino|mkl|blas'; then echo 'ERROR: forbidden runtime backend dependency detected'; exit 1; else echo 'M35 end-to-end runtime purity: OK'; fi
.PHONY: m35-real-e2e

# -----------------------------------------------------------------------------
# M36: oc8xT8 Conv, fused Leaky->Conv padding transform, lifetime arena reuse.
# -----------------------------------------------------------------------------
.PHONY: m36 m36-check m36-real-vocoder m36-real-e2e

$(BUILD)/conv1d_m36.o: $(KDIR)/conv1d_nct_f32_avx2_oc8_t8.S | $(BUILD)
	$(CC) -c $< -o $@

$(BUILD)/conv1d_m37_range.o: $(KDIR)/conv1d_nct_f32_avx2_oc8_t8_range.S | $(BUILD)
	$(CC) -c $< -o $@
$(BUILD)/conv1d_m39_kspec.o: $(KDIR)/conv1d_nct_f32_avx2_oc8_t8_kspec.S | $(BUILD)
	$(CC) -c $< -o $@
$(BUILD)/conv1d_m38_residual.o: $(KDIR)/conv1d_nct_f32_avx2_oc8_t8_residual.S | $(BUILD)
	$(CC) -c $< -o $@
$(BUILD)/conv1d_m38_range_residual.o: $(KDIR)/conv1d_nct_f32_avx2_oc8_t8_range_residual.S | $(BUILD)
	$(CC) -c $< -o $@
$(BUILD)/leaky_copy_m36.o: $(KDIR)/leaky_copy_nct_f32_avx2.S | $(BUILD)
	$(CC) -c $< -o $@

M36_EXTRA_OBJS :=

# Override the vocoder executable target with M36 kernels available to the shared pool.
$(BUILD)/dsasm-vocoder-m36: src/cli/dsasm_vocoder.c $(M35_VOCODER_OBJS) $(M36_EXTRA_OBJS)
	$(CC) $(CPPFLAGS) $(CFLAGS) src/cli/dsasm_vocoder.c $(M35_VOCODER_OBJS) $(M36_EXTRA_OBJS) -o $@ -lm -pthread

m36-check: $(BUILD)/dsasm-vocoder-m36
	$(PYTHON) -m py_compile tools/dsv35_common.py tools/make_fake_vocoder_graph_m35.py tools/pack_vocoder_graph_m35.py
	rm -rf $(BUILD)/m36_fake && mkdir -p $(BUILD)/m36_fake
	$(PYTHON) tools/make_fake_vocoder_graph_m35.py $(BUILD)/m36_fake/model.dsv35 --work $(BUILD)/m36_fake
	./$(BUILD)/dsasm-vocoder-m36 infer $(BUILD)/m36_fake/model.dsv35 --mel $(BUILD)/m36_fake/mel.f32 --f0 $(BUILD)/m36_fake/f0.f32 --out $(BUILD)/m36_fake/native.f32 --golden $(BUILD)/m36_fake/golden_wave.f32 --workers 4 --rounds 5 --profile
	@if ldd $(BUILD)/dsasm-vocoder-m36 | grep -Eqi 'onnx|onnxruntime|dnnl|onednn|openvino|mkl|blas'; then echo 'ERROR: forbidden runtime backend dependency detected'; exit 1; else echo 'M36 runtime purity: OK'; fi
	@echo 'M36 oc8 + fused Leaky->Conv + lifetime arena synthetic parity OK'

M36_FRAMES ?= 48
M36_ROUNDS ?= 7
M36_WORKERS ?= 8
m36-real-vocoder: $(BUILD)/dsasm-vocoder-m36
	@test -n "$(MODEL_DIR)" || (echo 'MODEL_DIR required'; exit 2)
	rm -rf $(BUILD)/m36_real && mkdir -p $(BUILD)/m36_real
	$(PYTHON) tools/pack_vocoder_graph_m35.py "$(MODEL_DIR)/dsvocoder/nsf_hifigan.onnx" --frames $(M36_FRAMES) --out $(BUILD)/m36_real/nsf_hifigan.dsv35 --work $(BUILD)/m36_real
	cat $(BUILD)/m36_real/nsf_hifigan.json
	./$(BUILD)/dsasm-vocoder-m36 inspect $(BUILD)/m36_real/nsf_hifigan.dsv35
	@for w in 4 6 8; do \
	  echo "===== M36 REAL VOCODER workers=$$w ====="; \
	  ./$(BUILD)/dsasm-vocoder-m36 infer $(BUILD)/m36_real/nsf_hifigan.dsv35 --mel $(BUILD)/m36_real/mel.f32 --f0 $(BUILD)/m36_real/f0.f32 --out $(BUILD)/m36_real/native_w$$w.f32 --golden $(BUILD)/m36_real/golden_wave.f32 --workers $$w --rounds $(M36_ROUNDS) --profile || exit $$?; \
	done

M36_STEPS ?= 4
M36_E2E_WORKERS ?= 8
M36_E2E_ROUNDS ?= 7
m36-real-e2e: $(BUILD)/dsasm-vocoder-m36 $(BUILD)/dsasm-acoustic
	@test -n "$(MODEL_DIR)" || (echo 'MODEL_DIR required'; exit 2)
	@test -n "$(SPEAKER_EMB)" || (echo 'SPEAKER_EMB required'; exit 2)
	rm -rf $(BUILD)/m36_e2e_model $(BUILD)/m36_e2e && mkdir -p $(BUILD)/m36_e2e
	$(PYTHON) tools/pack_acoustic_onnx_m25.py "$(MODEL_DIR)/acoustic.onnx" --model-dir "$(MODEL_DIR)" --out $(BUILD)/m36_e2e_model
	$(PYTHON) tools/pack_vocoder_graph_m35.py "$(MODEL_DIR)/dsvocoder/nsf_hifigan.onnx" --frames 48 --out $(BUILD)/m36_e2e/nsf_hifigan.dsv35 --work $(BUILD)/m36_e2e/packer
	$(PYTHON) tools/run_native_e2e_m35.py --packed-acoustic $(BUILD)/m36_e2e_model --acoustic-cli ./$(BUILD)/dsasm-acoustic --vocoder-bundle $(BUILD)/m36_e2e/nsf_hifigan.dsv35 --vocoder-cli ./$(BUILD)/dsasm-vocoder-m36 --vocoder-onnx "$(MODEL_DIR)/dsvocoder/nsf_hifigan.onnx" --speaker-emb "$(SPEAKER_EMB)" --language-id 4 --depth 0.6 --steps $(M36_STEPS) --workers $(M36_E2E_WORKERS) --rounds $(M36_E2E_ROUNDS) --work $(BUILD)/m36_e2e/run

m36: m36-check


# ---------------------------------------------------------------------------
# M37: pure-ASM vocoder Conv 2-D OC-block x time-tile scheduling.
# All Cout divisible by 8 now use packed8; long low-channel layers dynamically
# spread time tiles across the full persistent P-core worker pool.
# ---------------------------------------------------------------------------
.PHONY: m37 m37-check m37-real-vocoder m37-real-e2e
M37_FRAMES ?= 48
M37_ROUNDS ?= 7
M37_BASELINE_ROUNDS ?= 5
M37_STEPS ?= 4
M37_E2E_WORKERS ?= 8
M37_E2E_ROUNDS ?= 7

$(BUILD)/dsasm-vocoder-m37: src/cli/dsasm_vocoder.c $(M35_VOCODER_OBJS)
	$(CC) $(CPPFLAGS) $(CFLAGS) src/cli/dsasm_vocoder.c $(M35_VOCODER_OBJS) -o $@ -lm -pthread

m37-check: $(BUILD)/dsasm-vocoder-m37
	$(PYTHON) -m py_compile tools/dsv35_common.py tools/make_fake_vocoder_graph_m35.py tools/pack_vocoder_graph_m35.py
	rm -rf $(BUILD)/m37_fake && mkdir -p $(BUILD)/m37_fake
	$(PYTHON) tools/make_fake_vocoder_graph_m35.py $(BUILD)/m37_fake/model.dsv35 --work $(BUILD)/m37_fake --frames 2048 --channels 8
	@echo '--- M37 2-D enabled ---'
	./$(BUILD)/dsasm-vocoder-m37 infer $(BUILD)/m37_fake/model.dsv35 --mel $(BUILD)/m37_fake/mel.f32 --f0 $(BUILD)/m37_fake/f0.f32 --out $(BUILD)/m37_fake/native_2d.f32 --golden $(BUILD)/m37_fake/golden_wave.f32 --workers 4 --rounds 5 --profile
	@echo '--- M37 2-D disabled baseline ---'
	DSASM_2D=0 ./$(BUILD)/dsasm-vocoder-m37 infer $(BUILD)/m37_fake/model.dsv35 --mel $(BUILD)/m37_fake/mel.f32 --f0 $(BUILD)/m37_fake/f0.f32 --out $(BUILD)/m37_fake/native_static.f32 --golden $(BUILD)/m37_fake/golden_wave.f32 --workers 4 --rounds 5 --profile
	cmp $(BUILD)/m37_fake/native_2d.f32 $(BUILD)/m37_fake/native_static.f32
	@if ldd $(BUILD)/dsasm-vocoder-m37 | grep -Eqi 'onnx|onnxruntime|dnnl|onednn|openvino|mkl|blas'; then echo 'ERROR: forbidden runtime backend dependency detected'; exit 1; else echo 'M37 runtime purity: OK'; fi
	@echo 'M37 long-sequence packed8 2-D scheduler parity OK'

m37-real-vocoder: $(BUILD)/dsasm-vocoder-m37
	@test -n "$(MODEL_DIR)" || (echo 'usage: make m37-real-vocoder MODEL_DIR=/path/voicebank'; exit 2)
	test -f "$(MODEL_DIR)/dsvocoder/nsf_hifigan.onnx"
	rm -rf $(BUILD)/m37_real && mkdir -p $(BUILD)/m37_real
	$(PYTHON) tools/pack_vocoder_graph_m35.py "$(MODEL_DIR)/dsvocoder/nsf_hifigan.onnx" --frames $(M37_FRAMES) --out $(BUILD)/m37_real/nsf_hifigan.dsv35 --work $(BUILD)/m37_real
	cat $(BUILD)/m37_real/nsf_hifigan.json
	./$(BUILD)/dsasm-vocoder-m37 inspect $(BUILD)/m37_real/nsf_hifigan.dsv35
	@echo '===== M37 8-worker STATIC-OC BASELINE (2-D disabled) ====='
	DSASM_2D=0 ./$(BUILD)/dsasm-vocoder-m37 infer $(BUILD)/m37_real/nsf_hifigan.dsv35 --mel $(BUILD)/m37_real/mel.f32 --f0 $(BUILD)/m37_real/f0.f32 --out $(BUILD)/m37_real/native_static8.f32 --golden $(BUILD)/m37_real/golden_wave.f32 --workers 8 --rounds $(M37_BASELINE_ROUNDS) --profile
	@for w in 4 6 8; do \
	  echo "===== M37 REAL VOCODER workers=$$w 2-D AUTO ====="; \
	  ./$(BUILD)/dsasm-vocoder-m37 infer $(BUILD)/m37_real/nsf_hifigan.dsv35 --mel $(BUILD)/m37_real/mel.f32 --f0 $(BUILD)/m37_real/f0.f32 --out $(BUILD)/m37_real/native_w$$w.f32 --golden $(BUILD)/m37_real/golden_wave.f32 --workers $$w --rounds $(M37_ROUNDS) --profile || exit $$?; \
	done
	@echo 'M37 real pure-ASM vocoder benchmark complete'

m37-real-e2e: $(BUILD)/dsasm-vocoder-m37 $(BUILD)/dsasm-acoustic
	@test -n "$(MODEL_DIR)" || (echo 'usage: make m37-real-e2e MODEL_DIR=/path/voicebank SPEAKER_EMB=/path/speaker.emb'; exit 2)
	@test -n "$(SPEAKER_EMB)" || (echo 'SPEAKER_EMB required'; exit 2)
	rm -rf $(BUILD)/m37_e2e_model $(BUILD)/m37_e2e && mkdir -p $(BUILD)/m37_e2e
	$(PYTHON) tools/pack_acoustic_onnx_m25.py "$(MODEL_DIR)/acoustic.onnx" --model-dir "$(MODEL_DIR)" --out $(BUILD)/m37_e2e_model
	$(PYTHON) tools/pack_vocoder_graph_m35.py "$(MODEL_DIR)/dsvocoder/nsf_hifigan.onnx" --frames 48 --out $(BUILD)/m37_e2e/nsf_hifigan.dsv35 --work $(BUILD)/m37_e2e/packer
	$(PYTHON) tools/run_native_e2e_m35.py --packed-acoustic $(BUILD)/m37_e2e_model --acoustic-cli ./$(BUILD)/dsasm-acoustic --vocoder-bundle $(BUILD)/m37_e2e/nsf_hifigan.dsv35 --vocoder-cli ./$(BUILD)/dsasm-vocoder-m37 --vocoder-onnx "$(MODEL_DIR)/dsvocoder/nsf_hifigan.onnx" --speaker-emb "$(SPEAKER_EMB)" --language-id 4 --depth 0.6 --steps $(M37_STEPS) --workers $(M37_E2E_WORKERS) --rounds $(M37_E2E_ROUNDS) --work $(BUILD)/m37_e2e/run

m37: m37-check


# ---------------------------------------------------------------------------
# M38: fuse HiFi-GAN residual Add into Conv2 AVX2 store path.
# ---------------------------------------------------------------------------
.PHONY: m38 m38-check m38-real-vocoder m38-real-e2e
M38_FRAMES ?= 48
M38_ROUNDS ?= 7
M38_STEPS ?= 4
M38_E2E_WORKERS ?= 4
M38_E2E_ROUNDS ?= 7

$(BUILD)/dsasm-vocoder-m38: src/cli/dsasm_vocoder.c $(M35_VOCODER_OBJS)
	$(CC) $(CPPFLAGS) $(CFLAGS) src/cli/dsasm_vocoder.c $(M35_VOCODER_OBJS) -o $@ -lm -pthread

m38-check: $(BUILD)/dsasm-vocoder-m38
	$(PYTHON) -m py_compile tools/dsv35_common.py tools/make_fake_vocoder_graph_m35.py tools/pack_vocoder_graph_m35.py
	rm -rf $(BUILD)/m38_fake && mkdir -p $(BUILD)/m38_fake
	$(PYTHON) tools/make_fake_vocoder_graph_m35.py $(BUILD)/m38_fake/model.dsv35 --work $(BUILD)/m38_fake --frames 2048 --channels 8
	./$(BUILD)/dsasm-vocoder-m38 infer $(BUILD)/m38_fake/model.dsv35 --mel $(BUILD)/m38_fake/mel.f32 --f0 $(BUILD)/m38_fake/f0.f32 --out $(BUILD)/m38_fake/native_2d.f32 --golden $(BUILD)/m38_fake/golden_wave.f32 --workers 4 --rounds 5 --profile
	DSASM_2D=0 ./$(BUILD)/dsasm-vocoder-m38 infer $(BUILD)/m38_fake/model.dsv35 --mel $(BUILD)/m38_fake/mel.f32 --f0 $(BUILD)/m38_fake/f0.f32 --out $(BUILD)/m38_fake/native_static.f32 --golden $(BUILD)/m38_fake/golden_wave.f32 --workers 4 --rounds 5 --profile
	cmp $(BUILD)/m38_fake/native_2d.f32 $(BUILD)/m38_fake/native_static.f32
	@if ldd $(BUILD)/dsasm-vocoder-m38 | grep -Eqi 'onnx|onnxruntime|dnnl|onednn|openvino|mkl|blas'; then echo 'ERROR: forbidden runtime backend dependency detected'; exit 1; else echo 'M38 runtime purity: OK'; fi
	@echo 'M38 fused residual-store synthetic parity OK'

m38-real-vocoder: $(BUILD)/dsasm-vocoder-m38
	@test -n "$(MODEL_DIR)" || (echo 'MODEL_DIR required'; exit 2)
	rm -rf $(BUILD)/m38_real && mkdir -p $(BUILD)/m38_real
	$(PYTHON) tools/pack_vocoder_graph_m35.py "$(MODEL_DIR)/dsvocoder/nsf_hifigan.onnx" --frames $(M38_FRAMES) --out $(BUILD)/m38_real/nsf_hifigan.dsv35 --work $(BUILD)/m38_real
	cat $(BUILD)/m38_real/nsf_hifigan.json
	@for w in 4 8; do \
	  echo "===== M38 REAL VOCODER workers=$$w ====="; \
	  ./$(BUILD)/dsasm-vocoder-m38 infer $(BUILD)/m38_real/nsf_hifigan.dsv35 --mel $(BUILD)/m38_real/mel.f32 --f0 $(BUILD)/m38_real/f0.f32 --out $(BUILD)/m38_real/native_w$$w.f32 --golden $(BUILD)/m38_real/golden_wave.f32 --workers $$w --rounds $(M38_ROUNDS) --profile || exit $$?; \
	done

m38-real-e2e: $(BUILD)/dsasm-vocoder-m38 $(BUILD)/dsasm-acoustic
	@test -n "$(MODEL_DIR)" || (echo 'MODEL_DIR required'; exit 2)
	@test -n "$(SPEAKER_EMB)" || (echo 'SPEAKER_EMB required'; exit 2)
	rm -rf $(BUILD)/m38_e2e_model $(BUILD)/m38_e2e && mkdir -p $(BUILD)/m38_e2e
	$(PYTHON) tools/pack_acoustic_onnx_m25.py "$(MODEL_DIR)/acoustic.onnx" --model-dir "$(MODEL_DIR)" --out $(BUILD)/m38_e2e_model
	$(PYTHON) tools/pack_vocoder_graph_m35.py "$(MODEL_DIR)/dsvocoder/nsf_hifigan.onnx" --frames 48 --out $(BUILD)/m38_e2e/nsf_hifigan.dsv35 --work $(BUILD)/m38_e2e/packer
	$(PYTHON) tools/run_native_e2e_m35.py --packed-acoustic $(BUILD)/m38_e2e_model --acoustic-cli ./$(BUILD)/dsasm-acoustic --vocoder-bundle $(BUILD)/m38_e2e/nsf_hifigan.dsv35 --vocoder-cli ./$(BUILD)/dsasm-vocoder-m38 --vocoder-onnx "$(MODEL_DIR)/dsvocoder/nsf_hifigan.onnx" --speaker-emb "$(SPEAKER_EMB)" --language-id 4 --depth 0.6 --steps $(M38_STEPS) --workers $(M38_E2E_WORKERS) --rounds $(M38_E2E_ROUNDS) --work $(BUILD)/m38_e2e/run

m38: m38-check

# ---------------------------------------------------------------------------
# M39: restore M37 winning graph semantics; reduce Conv workspace traffic.
# Default: residual-store fusion OFF, halo-zero integrated in Leaky-copy ASM.
# Fixed K3/K7/K11 unrolled kernels are experimental and A/B controlled by
# DSASM_KSPEC=1; DSASM_FULL_MEMSET=1 restores the pre-M39 workspace path.
# ---------------------------------------------------------------------------
.PHONY: m39 m39-check m39-real-vocoder m39-real-e2e
M39_FRAMES ?= 48
M39_ROUNDS ?= 9
M39_STEPS ?= 4
M39_E2E_WORKERS ?= 4
M39_E2E_ROUNDS ?= 9

$(BUILD)/test_m39_kspec: tests/test_m39_kspec.c $(BUILD)/conv1d_m36.o $(BUILD)/conv1d_m37_range.o $(BUILD)/conv1d_m39_kspec.o
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_m39_kspec.c $(BUILD)/conv1d_m36.o $(BUILD)/conv1d_m37_range.o $(BUILD)/conv1d_m39_kspec.o -o $@ -lm

$(BUILD)/dsasm-vocoder-m39: src/cli/dsasm_vocoder.c $(M35_VOCODER_OBJS)
	$(CC) $(CPPFLAGS) $(CFLAGS) src/cli/dsasm_vocoder.c $(M35_VOCODER_OBJS) -o $@ -lm -pthread

m39-check: $(BUILD)/dsasm-vocoder-m39 $(BUILD)/test_m39_kspec
	./$(BUILD)/test_m39_kspec
	$(PYTHON) -m py_compile tools/dsv35_common.py tools/make_fake_vocoder_graph_m35.py tools/pack_vocoder_graph_m35.py
	rm -rf $(BUILD)/m39_fake && mkdir -p $(BUILD)/m39_fake
	$(PYTHON) tools/make_fake_vocoder_graph_m35.py $(BUILD)/m39_fake/model.dsv35 --work $(BUILD)/m39_fake --frames 4096 --channels 8
	./$(BUILD)/dsasm-vocoder-m39 infer $(BUILD)/m39_fake/model.dsv35 --mel $(BUILD)/m39_fake/mel.f32 --f0 $(BUILD)/m39_fake/f0.f32 --out $(BUILD)/m39_fake/halo.f32 --golden $(BUILD)/m39_fake/golden_wave.f32 --workers 4 --rounds 5
	DSASM_FULL_MEMSET=1 ./$(BUILD)/dsasm-vocoder-m39 infer $(BUILD)/m39_fake/model.dsv35 --mel $(BUILD)/m39_fake/mel.f32 --f0 $(BUILD)/m39_fake/f0.f32 --out $(BUILD)/m39_fake/full.f32 --golden $(BUILD)/m39_fake/golden_wave.f32 --workers 4 --rounds 5
	cmp $(BUILD)/m39_fake/halo.f32 $(BUILD)/m39_fake/full.f32
	@if ldd $(BUILD)/dsasm-vocoder-m39 | grep -Eqi 'onnx|onnxruntime|dnnl|onednn|openvino|mkl|blas'; then echo 'ERROR: forbidden runtime backend dependency detected'; exit 1; else echo 'M39 runtime purity: OK'; fi
	@echo 'M39 workspace-halo + experimental fixed-K parity OK'

m39-real-vocoder: $(BUILD)/dsasm-vocoder-m39
	@test -n "$(MODEL_DIR)" || (echo 'MODEL_DIR required'; exit 2)
	rm -rf $(BUILD)/m39_real && mkdir -p $(BUILD)/m39_real
	$(PYTHON) tools/pack_vocoder_graph_m35.py "$(MODEL_DIR)/dsvocoder/nsf_hifigan.onnx" --frames $(M39_FRAMES) --out $(BUILD)/m39_real/nsf_hifigan.dsv35 --work $(BUILD)/m39_real
	cat $(BUILD)/m39_real/nsf_hifigan.json
	@echo '===== M39 A/B A: M37-style generic K + old FULL workspace memset ====='
	DSASM_KSPEC=0 DSASM_FULL_MEMSET=1 ./$(BUILD)/dsasm-vocoder-m39 infer $(BUILD)/m39_real/nsf_hifigan.dsv35 --mel $(BUILD)/m39_real/mel.f32 --f0 $(BUILD)/m39_real/f0.f32 --out $(BUILD)/m39_real/a_old.f32 --golden $(BUILD)/m39_real/golden_wave.f32 --workers 4 --rounds $(M39_ROUNDS) --profile
	@echo '===== M39 A/B B: generic K + integrated halo zero (DEFAULT) ====='
	DSASM_KSPEC=0 ./$(BUILD)/dsasm-vocoder-m39 infer $(BUILD)/m39_real/nsf_hifigan.dsv35 --mel $(BUILD)/m39_real/mel.f32 --f0 $(BUILD)/m39_real/f0.f32 --out $(BUILD)/m39_real/b_halo.f32 --golden $(BUILD)/m39_real/golden_wave.f32 --workers 4 --rounds $(M39_ROUNDS) --profile
	@echo '===== M39 A/B C: experimental fixed K3/K7/K11 + integrated halo ====='
	DSASM_KSPEC=1 ./$(BUILD)/dsasm-vocoder-m39 infer $(BUILD)/m39_real/nsf_hifigan.dsv35 --mel $(BUILD)/m39_real/mel.f32 --f0 $(BUILD)/m39_real/f0.f32 --out $(BUILD)/m39_real/c_kspec.f32 --golden $(BUILD)/m39_real/golden_wave.f32 --workers 4 --rounds $(M39_ROUNDS) --profile
	cmp $(BUILD)/m39_real/a_old.f32 $(BUILD)/m39_real/b_halo.f32
	cmp $(BUILD)/m39_real/a_old.f32 $(BUILD)/m39_real/c_kspec.f32
	@echo 'M39 real A/B complete; all three paths are bit-identical.'

m39-real-e2e: $(BUILD)/dsasm-vocoder-m39 $(BUILD)/dsasm-acoustic
	@test -n "$(MODEL_DIR)" || (echo 'MODEL_DIR required'; exit 2)
	@test -n "$(SPEAKER_EMB)" || (echo 'SPEAKER_EMB required'; exit 2)
	rm -rf $(BUILD)/m39_e2e_model $(BUILD)/m39_e2e && mkdir -p $(BUILD)/m39_e2e
	$(PYTHON) tools/pack_acoustic_onnx_m25.py "$(MODEL_DIR)/acoustic.onnx" --model-dir "$(MODEL_DIR)" --out $(BUILD)/m39_e2e_model
	$(PYTHON) tools/pack_vocoder_graph_m35.py "$(MODEL_DIR)/dsvocoder/nsf_hifigan.onnx" --frames 48 --out $(BUILD)/m39_e2e/nsf_hifigan.dsv35 --work $(BUILD)/m39_e2e/packer
	DSASM_KSPEC=0 $(PYTHON) tools/run_native_e2e_m35.py --packed-acoustic $(BUILD)/m39_e2e_model --acoustic-cli ./$(BUILD)/dsasm-acoustic --vocoder-bundle $(BUILD)/m39_e2e/nsf_hifigan.dsv35 --vocoder-cli ./$(BUILD)/dsasm-vocoder-m39 --vocoder-onnx "$(MODEL_DIR)/dsvocoder/nsf_hifigan.onnx" --speaker-emb "$(SPEAKER_EMB)" --language-id 4 --depth 0.6 --steps $(M39_STEPS) --workers $(M39_E2E_WORKERS) --rounds $(M39_E2E_ROUNDS) --work $(BUILD)/m39_e2e/run

m39: m39-check

$(BUILD)/conv1d_m39_vnni.o: $(KDIR)/conv1d_vnni_u8s8_t8_oc8.S | $(BUILD)
	$(CC) -c $< -o $@


$(BUILD)/vnni_pack_m40_1.o: $(KDIR)/vnni_pack_u8_4x8_avx2.S | $(BUILD)
	$(CC) -c $< -o $@
$(BUILD)/test_m39_vnni_conv: tests/test_m39_vnni_conv.c $(BUILD)/conv1d_m36.o $(BUILD)/conv1d_m39_vnni.o
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_m39_vnni_conv.c $(BUILD)/conv1d_m36.o $(BUILD)/conv1d_m39_vnni.o -o $@ -lm

.PHONY: m39-vnni-check m39-vnni-real
m39-vnni-check: $(BUILD)/test_m39_vnni_conv
	@echo 'M39.1 verifying AVX-VNNI VEX encoding (must be c4..., never EVEX 62...)'
	@objdump -d -M intel $(BUILD)/conv1d_m39_vnni.o | grep -q '{vex} vpdpbusd' || (echo 'ERROR: vpdpbusd is not VEX encoded'; exit 1)
	@if objdump -d -M intel $(BUILD)/conv1d_m39_vnni.o | grep 'vpdpbusd' | grep -q '^[[:space:]]*[0-9a-f]*:[[:space:]]*62 '; then echo 'ERROR: EVEX/AVX-512 VNNI encoding detected'; exit 1; fi
	$(PYTHON) tools/make_fake_vnni_m39.py $(BUILD)/m39_fake_vnni.dsvn39 --cin 16 --cout 16 --k 7 --tin 256 --dilation 3
	./$(BUILD)/test_m39_vnni_conv $(BUILD)/m39_fake_vnni.dsvn39 7
	@echo 'M39.1 AVX-VNNI synthetic feasibility check complete'

M39_VNNI_ROUNDS ?= 7
m39-vnni-real: $(BUILD)/test_m39_vnni_conv
	@test -n "$(MODEL_DIR)" || (echo 'MODEL_DIR required'; exit 2)
	rm -rf $(BUILD)/m39_vnni && mkdir -p $(BUILD)/m39_vnni
	@for spec in 'k11:/generator/resblocks.5/convs1.0/Conv' 'k7:/generator/resblocks.4/convs1.0/Conv' 'k3:/generator/resblocks.3/convs1.0/Conv'; do \
	  tag=$${spec%%:*}; node=$${spec#*:}; \
	  echo "===== M39 VNNI REAL $$tag $$node ====="; \
	  $(PYTHON) tools/pack_vocoder_vnni_hot_m39.py "$(MODEL_DIR)/dsvocoder/nsf_hifigan.onnx" --node "$$node" --frames 48 --out $(BUILD)/m39_vnni/$$tag.dsvn39 || exit $$?; \
	  cat $(BUILD)/m39_vnni/$$tag.json; \
	  ./$(BUILD)/test_m39_vnni_conv $(BUILD)/m39_vnni/$$tag.dsvn39 $(M39_VNNI_ROUNDS) || exit $$?; \
	done

# ---------------------------------------------------------------------------
# M40: selective full-graph AVX-VNNI for the C128/T3072 K7/K11 hot stage.
# FP32 remains embedded as exact fallback; DSASM_VNNI=0|k11|k117 selects runtime.
# ---------------------------------------------------------------------------
.PHONY: m40 m40-check m40-real-vocoder m40-real-e2e
M40_FRAMES ?= 48
M40_ROUNDS ?= 9
M40_STEPS ?= 4
M40_E2E_WORKERS ?= 4
M40_E2E_ROUNDS ?= 9

$(BUILD)/dsasm-vocoder-m40: src/cli/dsasm_vocoder.c $(M35_VOCODER_OBJS)
	$(CC) $(CPPFLAGS) $(CFLAGS) src/cli/dsasm_vocoder.c $(M35_VOCODER_OBJS) -o $@ -lm -pthread

m40-check: $(BUILD)/dsasm-vocoder-m40
	$(PYTHON) -m py_compile tools/dsv35_common.py tools/make_fake_vnni_graph_m40.py tools/pack_vocoder_graph_m35.py
	rm -rf $(BUILD)/m40_fake && mkdir -p $(BUILD)/m40_fake
	$(PYTHON) tools/make_fake_vnni_graph_m40.py $(BUILD)/m40_fake/model.dsv35 --work $(BUILD)/m40_fake
	DSASM_VNNI=0 ./$(BUILD)/dsasm-vocoder-m40 infer $(BUILD)/m40_fake/model.dsv35 --mel $(BUILD)/m40_fake/mel.f32 --f0 $(BUILD)/m40_fake/f0.f32 --out $(BUILD)/m40_fake/fp32.f32 --golden $(BUILD)/m40_fake/golden.f32 --workers 4 --rounds 5 --profile
	DSASM_VNNI=k117 ./$(BUILD)/dsasm-vocoder-m40 infer $(BUILD)/m40_fake/model.dsv35 --mel $(BUILD)/m40_fake/mel.f32 --f0 $(BUILD)/m40_fake/f0.f32 --out $(BUILD)/m40_fake/vnni.f32 --golden $(BUILD)/m40_fake/golden.f32 --workers 4 --rounds 5 --profile
	@if ldd $(BUILD)/dsasm-vocoder-m40 | grep -Eqi 'onnx|onnxruntime|dnnl|onednn|openvino|mkl|blas'; then echo 'ERROR: forbidden runtime backend'; exit 1; else echo 'M40 runtime purity: OK'; fi
	@echo 'M40 selective VNNI full-executor synthetic check OK'

m40-real-vocoder: $(BUILD)/dsasm-vocoder-m40
	@test -n "$(MODEL_DIR)" || (echo 'MODEL_DIR required'; exit 2)
	rm -rf $(BUILD)/m40_real && mkdir -p $(BUILD)/m40_real
	$(PYTHON) tools/pack_vocoder_graph_m35.py "$(MODEL_DIR)/dsvocoder/nsf_hifigan.onnx" --frames $(M40_FRAMES) --vnni-scope stage128 --out $(BUILD)/m40_real/nsf_hifigan.dsv35 --work $(BUILD)/m40_real
	cat $(BUILD)/m40_real/nsf_hifigan.json
	@for mode in 0 k11 k117; do \
	  echo "===== M40 REAL VOCODER DSASM_VNNI=$$mode ====="; \
	  if [ "$$mode" = 0 ]; then tol=0.005; else tol=0.05; fi; \
	  DSASM_GOLDEN_TOL=$$tol DSASM_VNNI=$$mode DSASM_KSPEC=1 ./$(BUILD)/dsasm-vocoder-m40 infer $(BUILD)/m40_real/nsf_hifigan.dsv35 --mel $(BUILD)/m40_real/mel.f32 --f0 $(BUILD)/m40_real/f0.f32 --out $(BUILD)/m40_real/$$mode.f32 --golden $(BUILD)/m40_real/golden_wave.f32 --workers 4 --rounds $(M40_ROUNDS) --profile || exit $$?; \
	done
	@echo 'M40 real selective-VNNI sweep complete'

m40-real-e2e: $(BUILD)/dsasm-vocoder-m40 $(BUILD)/dsasm-acoustic
	@test -n "$(MODEL_DIR)" || (echo 'MODEL_DIR required'; exit 2)
	@test -n "$(SPEAKER_EMB)" || (echo 'SPEAKER_EMB required'; exit 2)
	rm -rf $(BUILD)/m40_e2e_model $(BUILD)/m40_e2e && mkdir -p $(BUILD)/m40_e2e
	$(PYTHON) tools/pack_acoustic_onnx_m25.py "$(MODEL_DIR)/acoustic.onnx" --model-dir "$(MODEL_DIR)" --out $(BUILD)/m40_e2e_model
	$(PYTHON) tools/pack_vocoder_graph_m35.py "$(MODEL_DIR)/dsvocoder/nsf_hifigan.onnx" --frames 48 --vnni-scope stage128 --out $(BUILD)/m40_e2e/nsf_hifigan.dsv35 --work $(BUILD)/m40_e2e/packer
	DSASM_GOLDEN_TOL=0.05 DSASM_VNNI=k117 DSASM_KSPEC=1 $(PYTHON) tools/run_native_e2e_m35.py --packed-acoustic $(BUILD)/m40_e2e_model --acoustic-cli ./$(BUILD)/dsasm-acoustic --vocoder-bundle $(BUILD)/m40_e2e/nsf_hifigan.dsv35 --vocoder-cli ./$(BUILD)/dsasm-vocoder-m40 --vocoder-onnx "$(MODEL_DIR)/dsvocoder/nsf_hifigan.onnx" --speaker-emb "$(SPEAKER_EMB)" --language-id 4 --depth 0.6 --steps $(M40_STEPS) --workers $(M40_E2E_WORKERS) --rounds $(M40_E2E_ROUNDS) --work $(BUILD)/m40_e2e/run

m40: m40-check


# ---------------------------------------------------------------------------
# M40.1: faster branch-free ASM VNNI activation pack + relative quality gate.
# ---------------------------------------------------------------------------
.PHONY: m40-1-check m40-1-real-vocoder m40-1-real-e2e m40-1
M40_1_FRAMES ?= 48
M40_1_ROUNDS ?= 9
M40_1_STEPS ?= 4
M40_1_E2E_WORKERS ?= 4
M40_1_E2E_ROUNDS ?= 9

m40-1-check: $(BUILD)/dsasm-vocoder-m40
	$(PYTHON) -m py_compile tools/run_native_e2e_m35.py tools/pack_vocoder_graph_m35.py
	rm -rf $(BUILD)/m40_1_fake && mkdir -p $(BUILD)/m40_1_fake
	$(PYTHON) tools/make_fake_vnni_graph_m40.py $(BUILD)/m40_1_fake/model.dsv35 --work $(BUILD)/m40_1_fake
	DSASM_GOLDEN_COS=0.999 DSASM_GOLDEN_SNR=25 DSASM_VNNI=k117 ./$(BUILD)/dsasm-vocoder-m40 infer $(BUILD)/m40_1_fake/model.dsv35 --mel $(BUILD)/m40_1_fake/mel.f32 --f0 $(BUILD)/m40_1_fake/f0.f32 --out $(BUILD)/m40_1_fake/vnni.f32 --golden $(BUILD)/m40_1_fake/golden.f32 --workers 4 --rounds 7 --profile
	@echo 'M40.1 fast VNNI pack + relative quality gate OK'

m40-1-real-vocoder: $(BUILD)/dsasm-vocoder-m40
	@test -n "$(MODEL_DIR)" || (echo 'MODEL_DIR required'; exit 2)
	rm -rf $(BUILD)/m40_1_real && mkdir -p $(BUILD)/m40_1_real
	$(PYTHON) tools/pack_vocoder_graph_m35.py "$(MODEL_DIR)/dsvocoder/nsf_hifigan.onnx" --frames $(M40_1_FRAMES) --vnni-scope stage128 --out $(BUILD)/m40_1_real/nsf_hifigan.dsv35 --work $(BUILD)/m40_1_real
	@for mode in 0 k11 k117; do \
	  echo "===== M40.1 REAL VOCODER DSASM_VNNI=$$mode ====="; \
	  if [ "$$mode" = 0 ]; then qenv="DSASM_GOLDEN_TOL=0.005"; else qenv="DSASM_GOLDEN_COS=0.999 DSASM_GOLDEN_SNR=25"; fi; \
	  eval $$qenv DSASM_VNNI=$$mode DSASM_KSPEC=1 ./$(BUILD)/dsasm-vocoder-m40 infer $(BUILD)/m40_1_real/nsf_hifigan.dsv35 --mel $(BUILD)/m40_1_real/mel.f32 --f0 $(BUILD)/m40_1_real/f0.f32 --out $(BUILD)/m40_1_real/$$mode.f32 --golden $(BUILD)/m40_1_real/golden_wave.f32 --workers 4 --rounds $(M40_1_ROUNDS) --profile || exit $$?; \
	done

m40-1-real-e2e: $(BUILD)/dsasm-vocoder-m40 $(BUILD)/dsasm-acoustic
	@test -n "$(MODEL_DIR)" || (echo 'MODEL_DIR required'; exit 2)
	@test -n "$(SPEAKER_EMB)" || (echo 'SPEAKER_EMB required'; exit 2)
	rm -rf $(BUILD)/m40_1_e2e_model $(BUILD)/m40_1_e2e && mkdir -p $(BUILD)/m40_1_e2e
	$(PYTHON) tools/pack_acoustic_onnx_m25.py "$(MODEL_DIR)/acoustic.onnx" --model-dir "$(MODEL_DIR)" --out $(BUILD)/m40_1_e2e_model
	$(PYTHON) tools/pack_vocoder_graph_m35.py "$(MODEL_DIR)/dsvocoder/nsf_hifigan.onnx" --frames 48 --vnni-scope stage128 --out $(BUILD)/m40_1_e2e/nsf_hifigan.dsv35 --work $(BUILD)/m40_1_e2e/packer
	$(PYTHON) tools/run_native_e2e_m40_1.py --packed-acoustic $(BUILD)/m40_1_e2e_model --acoustic-cli ./$(BUILD)/dsasm-acoustic --vocoder-bundle $(BUILD)/m40_1_e2e/nsf_hifigan.dsv35 --vocoder-cli ./$(BUILD)/dsasm-vocoder-m40 --vocoder-onnx "$(MODEL_DIR)/dsvocoder/nsf_hifigan.onnx" --speaker-emb "$(SPEAKER_EMB)" --language-id 4 --depth 0.6 --steps $(M40_1_STEPS) --workers $(M40_1_E2E_WORKERS) --rounds $(M40_1_E2E_ROUNDS) --modes k11,k117 --work $(BUILD)/m40_1_e2e/run

m40-1: m40-1-check

# M58: bit-exact late-stage range + residual-store kernels. Performance is
# measured separately by scripts/run_m58_micro.sh so correctness tests do not
# become flaky under system load.
.PHONY: m58-check m58-bench k3-range-bench persistent-e2e

PERSISTENT_E2E_OBJS := $(sort $(M24_OBJS) $(M35_VOCODER_OBJS))

$(BUILD)/persistent-e2e: tools/persistent_e2e.c $(PERSISTENT_E2E_OBJS)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(PERSISTENT_E2E_OBJS) -o $@ -lm -pthread

persistent-e2e: $(BUILD)/persistent-e2e

$(BUILD)/test_m58_range_residual: tools/test_m58_range_residual.c $(BUILD)/conv1d_m39_kspec.o $(BUILD)/conv1d_m38_range_residual.o
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(BUILD)/conv1d_m39_kspec.o $(BUILD)/conv1d_m38_range_residual.o -o $@ $(LDFLAGS)

m58-check: $(BUILD)/test_m58_range_residual
	./$(BUILD)/test_m58_range_residual

m58-bench: $(BUILD)/test_m58_range_residual
	bash scripts/run_m58_micro.sh

k3-range-bench: $(BUILD)/test_m58_range_residual
	bash scripts/run_k3_range_micro.sh
