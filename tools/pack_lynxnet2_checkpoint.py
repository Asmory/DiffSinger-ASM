#!/usr/bin/env python3
"""Pack one real DiffSinger LYNXNet2Block from a trusted PyTorch checkpoint.

The script does not import DiffSinger. It discovers the current LYNXNet2Block
state-dict keys by suffix, packs weights into M6 runtime layouts, and emits a
small deterministic PyTorch reference vector using the *real checkpoint
weights*. This is the bridge from synthetic regression tests to real models.
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ALIGN = 64


def align_up(x: int, a: int = ALIGN) -> int:
    return (x + a - 1) // a * a


def as_f32(t: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(t.detach().cpu().float().numpy())


def unwrap_state_dict(obj):
    if not isinstance(obj, dict):
        raise TypeError(f"checkpoint root must be a dict, got {type(obj)!r}")
    # Prefer standard Lightning / PyTorch wrappers.
    for k in ("state_dict", "model_state_dict", "model"):
        v = obj.get(k)
        if isinstance(v, dict) and any(torch.is_tensor(x) for x in v.values()):
            return v
    if any(torch.is_tensor(x) for x in obj.values()):
        return obj
    raise ValueError("could not locate a tensor state_dict in checkpoint")


def find_key(sd, suffix: str, prefix: str | None):
    hits = [k for k in sd if k.endswith(suffix) and (prefix is None or prefix in k)]
    if len(hits) != 1:
        lines = "\n  ".join(hits[:20]) if hits else "<none>"
        raise KeyError(f"expected exactly one key ending with {suffix!r}; found {len(hits)}:\n  {lines}\nUse --prefix to disambiguate.")
    return hits[0]


def pack_linear16(w: np.ndarray) -> np.ndarray:
    n, k = w.shape
    if n % 16:
        raise ValueError(f"M6 linear fast path requires N % 16 == 0, got N={n}")
    out = np.empty(n * k, dtype=np.float32)
    p = 0
    for n0 in range(0, n, 16):
        for kk in range(k):
            out[p:p+8] = w[n0:n0+8, kk]
            p += 8
            out[p:p+8] = w[n0+8:n0+16, kk]
            p += 8
    return out


def pack_glu8(w: np.ndarray) -> np.ndarray:
    two_n, k = w.shape
    if two_n % 2:
        raise ValueError("GLU Linear output dimension must be even")
    n = two_n // 2
    if n % 8:
        raise ValueError(f"M6 GLU fast path requires N % 8 == 0, got N={n}")
    wl, wg = w[:n], w[n:]
    out = np.empty(2 * n * k, dtype=np.float32)
    p = 0
    for n0 in range(0, n, 8):
        for kk in range(k):
            out[p:p+8] = wl[n0:n0+8, kk]
            p += 8
            out[p:p+8] = wg[n0:n0+8, kk]
            p += 8
    return out


def torch_block(x, gamma, beta, dw_w, dw_b, w1, b1, w2, b2, w3, b3):
    d = x.shape[-1]
    z = F.layer_norm(x, (d,), gamma, beta, 1e-5)
    z = F.conv1d(z.T[None], dw_w[:, None, :], dw_b, padding=15, groups=d)[0].T.contiguous()
    z = F.linear(z, w1, b1)
    l, g = torch.chunk(z, 2, -1)
    z = l * F.softsign(g)
    z = F.linear(z, w2, b2)
    l, g = torch.chunk(z, 2, -1)
    z = l * F.softsign(g)
    z = F.linear(z, w3, b3)
    return x + z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path, help="trusted DiffSinger .ckpt/.pt file")
    ap.add_argument("--block-index", type=int, default=0)
    ap.add_argument("--prefix", help="substring used to disambiguate multiple LYNXNet2 backbones")
    ap.add_argument("--out", type=Path, default=Path("packed_block"))
    ap.add_argument("--test-frames", type=int, default=128)
    ap.add_argument("--seed", type=int, default=20260919)
    args = ap.parse_args()

    # weights_only avoids executing arbitrary pickled Python objects. The input
    # should still be a checkpoint you trust.
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    sd = unwrap_state_dict(ckpt)
    base = f"residual_layers.{args.block_index}.net"
    suffixes = {
        "gamma": f"{base}.0.weight",
        "beta": f"{base}.0.bias",
        "dw_w": f"{base}.2.weight",
        "dw_b": f"{base}.2.bias",
        "w1": f"{base}.4.weight",
        "b1": f"{base}.4.bias",
        "w2": f"{base}.6.weight",
        "b2": f"{base}.6.bias",
        "w3": f"{base}.8.weight",
        "b3": f"{base}.8.bias",
    }
    keys = {name: find_key(sd, suf, args.prefix) for name, suf in suffixes.items()}
    ts = {name: sd[key].detach().cpu().float().contiguous() for name, key in keys.items()}

    gamma, beta = ts["gamma"], ts["beta"]
    dw_w = ts["dw_w"]
    if dw_w.ndim == 3 and dw_w.shape[1] == 1:
        dw_w = dw_w[:, 0, :]
    d = gamma.numel()
    if tuple(beta.shape) != (d,) or tuple(dw_w.shape) != (d, 31) or tuple(ts["dw_b"].shape) != (d,):
        raise ValueError("unexpected LayerNorm/depthwise shapes; this does not look like current LYNXNet2Block")
    two_h, d1 = ts["w1"].shape
    if d1 != d or two_h % 2:
        raise ValueError(f"bad first GLU Linear shape {tuple(ts['w1'].shape)}")
    h = two_h // 2
    if tuple(ts["w2"].shape) != (2*h, h) or tuple(ts["w3"].shape) != (d, h):
        raise ValueError("unexpected second GLU/final Linear shapes")
    if d % 16 or h % 8:
        raise ValueError(f"M6 fast path requires D % 16 == 0 and H % 8 == 0; got D={d}, H={h}")

    arrays = {
        "ln_gamma": as_f32(gamma),
        "ln_beta": as_f32(beta),
        "dw_weight_tap_major": np.ascontiguousarray(as_f32(dw_w).T),
        "dw_bias": as_f32(ts["dw_b"]),
        "glu1_weight_m4n8": pack_glu8(as_f32(ts["w1"])),
        "glu1_bias_left": as_f32(ts["b1"][:h]),
        "glu1_bias_gate": as_f32(ts["b1"][h:]),
        "glu2_weight_m4n8": pack_glu8(as_f32(ts["w2"])),
        "glu2_bias_left": as_f32(ts["b2"][:h]),
        "glu2_bias_gate": as_f32(ts["b2"][h:]),
        "out_weight_m4n16": pack_linear16(as_f32(ts["w3"])),
        "out_bias": as_f32(ts["b3"]),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    blob_path = args.out / f"lynxnet2_block{args.block_index}.dsb"
    manifest_path = args.out / f"lynxnet2_block{args.block_index}.json"
    sections = {}
    with blob_path.open("wb") as f:
        # 64-byte fixed header. Data sections begin at offset 64.
        header = struct.pack("<8s8I24x", b"DSBLK6\0", 1, d, h, 31, 4, 8, 4, 16)
        assert len(header) == 64
        f.write(header)
        offset = 64
        for name, a in arrays.items():
            aligned = align_up(offset)
            if aligned > offset:
                f.write(b"\0" * (aligned - offset))
                offset = aligned
            raw = np.ascontiguousarray(a, dtype=np.float32).tobytes()
            f.write(raw)
            sections[name] = {"offset": offset, "bytes": len(raw), "shape": list(a.shape), "dtype": "float32"}
            offset += len(raw)

    # Deterministic real-weight reference pair for the ASM runtime.
    gen = torch.Generator(device="cpu")
    gen.manual_seed(args.seed)
    x = torch.randn((args.test_frames, d), generator=gen, dtype=torch.float32) * 0.6
    with torch.inference_mode():
        y = torch_block(x, gamma, beta, dw_w, ts["dw_b"], ts["w1"], ts["b1"], ts["w2"], ts["b2"], ts["w3"], ts["b3"])
    x_path = args.out / "test_input.f32"
    y_path = args.out / "test_output_pytorch.f32"
    x.numpy().tofile(x_path)
    y.numpy().tofile(y_path)

    manifest = {
        "format": "DiffSinger-ASM LYNXNet2 block bundle",
        "version": 1,
        "magic": "DSBLK6",
        "glu_tile": [4, 8],
        "output_linear_tile": [4, 16],
        "D": d,
        "H": h,
        "kernel_size": 31,
        "block_index": args.block_index,
        "checkpoint": str(args.checkpoint),
        "matched_state_dict_keys": keys,
        "sections": sections,
        "test_vector": {
            "frames": args.test_frames,
            "seed": args.seed,
            "input": x_path.name,
            "pytorch_output": y_path.name,
            "shape": [args.test_frames, d],
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"packed real LYNXNet2 block: D={d} H={h} -> {blob_path}")
    print(f"manifest: {manifest_path}")
    print(f"reference: {x_path.name} -> {y_path.name}")


if __name__ == "__main__":
    main()
