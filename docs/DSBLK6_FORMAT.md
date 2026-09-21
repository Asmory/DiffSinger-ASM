# DSBLK6 bundle format

`DSBLK6` is the Milestone-6 binary representation of one DiffSinger
`LYNXNet2Block`. It stores weights directly in the layouts consumed by the M6
AVX2/FMA microkernels, so runtime execution does not repack tensors and does not
need ONNX, protobuf, PyTorch, or JSON parsing.

All integers are little-endian. Tensor payloads are IEEE-754 `float32`. Every
payload section begins at a 64-byte aligned file offset.

## 64-byte header

| Offset | Type | Meaning |
|---:|---|---|
| 0 | `char[8]` | magic `DSBLK6\0` |
| 8 | `u32` | version = `1` |
| 12 | `u32` | `D`, block channel dimension |
| 16 | `u32` | `H`, GLU inner dimension |
| 20 | `u32` | depthwise kernel size = `31` |
| 24 | `u32` | GLU M tile = `4` |
| 28 | `u32` | GLU N tile = `8` |
| 32 | `u32` | output Linear M tile = `4` |
| 36 | `u32` | output Linear N tile = `16` |
| 40 | 24 bytes | reserved, zero |

The current fast path requires `D % 16 == 0` and `H % 8 == 0`.

## Section order

Starting at byte 64, align to 64 bytes before every section:

1. `ln_gamma[D]`
2. `ln_beta[D]`
3. `dw_weight_tap_major[31,D]`
4. `dw_bias[D]`
5. `glu1_weight_m4n8[2*H*D]`
6. `glu1_bias_left[H]`
7. `glu1_bias_gate[H]`
8. `glu2_weight_m4n8[2*H*H]`
9. `glu2_bias_left[H]`
10. `glu2_bias_gate[H]`
11. `out_weight_m4n16[D*H]`
12. `out_bias[D]`

## GLU M4N8 packing

For a PyTorch Linear weight `[2N,K]`, split the output dimension into left and
gate halves. For every eight-output block and scalar input channel `k`, write:

```text
WL[n0:n0+8, k]
WG[n0:n0+8, k]
```

or 16 floats / 64 bytes per `k`. This lets the 4x8 ASM microkernel load the two
weight vectors once and reuse them across four input rows. Results are already
vectorized across output channels, so no horizontal dot-product reduction is
needed.

## Linear M4N16 packing

For a normal Linear weight `[N,K]`, for every sixteen-output block and scalar
input channel `k`, write:

```text
W[n0:n0+8, k]
W[n0+8:n0+16, k]
```

or 16 floats / 64 bytes per `k`. The same layout is used by the generic M6
Linear and the final `Linear + residual` fused kernel.

## Converter and runtime check

`tools/pack_lynxnet2_checkpoint.py` locates the current OpenVPI DiffSinger
sequential block parameters by state-dict suffix, writes `DSBLK6`, and generates
a deterministic PyTorch input/output pair using the exact checkpoint weights.

`tests/test_m6_bundle.c` parses the 64-byte header and aligned sections in plain
C, calls the handwritten ASM kernels directly, and compares its output to the
exported PyTorch reference.
