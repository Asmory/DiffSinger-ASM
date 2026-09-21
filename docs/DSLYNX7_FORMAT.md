# DSLYNX7 complete LYNXNet2 bundle

`DSLYNX7` is the M7 runtime format for one complete DiffSinger LYNXNet2
backbone. It removes ONNX/protobuf/PyTorch from runtime and stores matrices in
the exact layouts consumed by the AVX2/FMA kernels.

Current OpenVPI `configs/acoustic.yaml` defaults (2026-09-20) are compatible:
mel input 128, conditioner 384, channels 1024, 6 residual blocks, kernel 31,
expansion factor 1, and ATanGLU. SoftSignGLU is also supported.

All integers are little-endian `u32`, tensors are float32, and every tensor
section begins at a 64-byte aligned offset.

## Header (64 bytes)

`char magic[8] = "DSLYNX7\0"`, followed by 12 `u32` values:

1. version = 1
2. input dimension `I`
3. conditioner dimension `Q`
4. channels `C`
5. block hidden dimension `H`
6. residual-layer count `L`
7. kernel size = 31
8. GLU type: 1=ATanGLU, 2=SoftSignGLU
9. Linear M tile = 4
10. Linear N tile = 16
11. GLU M tile = 4
12. GLU N tile = 8

The last eight header bytes are reserved.

## Global sections

1. input projection packed16 `[C,I]`, bias `[C]`
2. conditioner projection packed16 `[C,Q]`, bias `[C]`
3. timestep MLP Linear1 packed16 `[4C,C]`, bias `[4C]`
4. timestep MLP Linear2 packed16 `[C,4C]`, bias `[C]`

## Per residual block

For each block 0..L-1:

1. LayerNorm gamma `[C]`, beta `[C]`
2. depthwise weight tap-major `[31,C]`, bias `[C]`
3. GLU Linear1 weight + bias `[2H]`
4. GLU Linear2 weight + bias `[2H]`
5. output Linear packed16 `[C,H]`, bias `[C]`

ATanGLU stores its two GLU matrices in normal M4N16 packed-Linear layout and
uses a separate AVX2 range-reduced atan activation. SoftSignGLU stores the two
matrices in M4N8 split-left/gate layout for the fused kernel. Both occupy the
same number of float elements.

## Final sections

1. post LayerNorm gamma `[C]`, beta `[C]`
2. output projection packed16 `[I,C]`, bias `[I]`

`tools/pack_lynxnet2_backbone.py` converts a trusted DiffSinger checkpoint and
writes deterministic PyTorch test vectors. `tests/test_m7_bundle.c` parses the
binary directly and validates serial and persistent-thread-pool execution.
