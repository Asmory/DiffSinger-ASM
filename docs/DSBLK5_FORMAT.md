# DSBLK5 bundle format

`DSBLK5` is the Milestone-5 binary representation of one DiffSinger
`LYNXNet2Block`. It is intentionally simple: the runtime can memory-map the
file, validate a 64-byte header, and point ASM kernels directly at packed FP32
sections without parsing ONNX, protobuf, JSON, or a PyTorch checkpoint.

All integers are little-endian. All tensor payloads are IEEE-754 `float32`.
Every tensor section starts at a 64-byte aligned file offset.

## Header — 64 bytes

| Offset | Type | Meaning |
|---:|---|---|
| 0 | `char[8]` | magic `DSBLK5\0\0` |
| 8 | `u32` | format version, currently `1` |
| 12 | `u32` | `D`, block/model channel dimension |
| 16 | `u32` | `H`, GLU inner dimension |
| 20 | `u32` | depthwise kernel size, currently `31` |
| 24 | 40 bytes | reserved, zero |

## Section order

Starting at offset 64, align the current offset to 64 bytes before every
section.

1. `ln_gamma[D]`
2. `ln_beta[D]`
3. `dw_weight_tap_major[31,D]`
4. `dw_bias[D]`
5. `glu1_weight_packed4`
6. `glu1_bias_left[H]`
7. `glu1_bias_gate[H]`
8. `glu2_weight_packed4`
9. `glu2_bias_left[H]`
10. `glu2_bias_gate[H]`
11. `out_weight_packed4`
12. `out_bias[D]`

`glu*_weight_packed4` uses four output channels per block and eight input
channels per K chunk. For each `(n0,k0)` chunk the order is:

```
L0[8] L1[8] L2[8] L3[8]
G0[8] G1[8] G2[8] G3[8]
```

The first/left half of the original PyTorch Linear is `L`; the second half is
`G`. Missing lanes in the final four-output block are zero-filled.

`out_weight_packed4` uses:

```
W0[8] W1[8] W2[8] W3[8]
```

The current M5 fast path requires `D % 8 == 0` and `H % 8 == 0`, which matches
the common DiffSinger LYNXNet2 dimensions targeted by this prototype.

## Converter

`tools/pack_lynxnet2_checkpoint.py` locates the current DiffSinger sequential
block parameters by state-dict suffix:

```
residual_layers.<i>.net.0  LayerNorm
residual_layers.<i>.net.2  depthwise Conv1d
residual_layers.<i>.net.4  Linear -> GLU
residual_layers.<i>.net.6  Linear -> GLU
residual_layers.<i>.net.8  output Linear
```

It also writes a JSON manifest plus deterministic `test_input.f32` and
`test_output_pytorch.f32` files. `tests/test_m5_bundle.c` loads the `.dsb`
without Python and checks the ASM output against that PyTorch reference.
