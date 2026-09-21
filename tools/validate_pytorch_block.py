#!/usr/bin/env python3
"""Validate the M6 handwritten ASM LYNXNet2 block against PyTorch semantics."""
from __future__ import annotations

import argparse
import ctypes
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def ptr(a: np.ndarray):
    assert a.dtype == np.float32 and a.flags.c_contiguous
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def pack_glu8(w: np.ndarray) -> np.ndarray:
    two_n, k = w.shape
    assert two_n % 2 == 0
    n = two_n // 2
    assert n % 8 == 0
    wl, wg = w[:n], w[n:]
    out = np.empty(2 * n * k, np.float32)
    p = 0
    for n0 in range(0, n, 8):
        for kk in range(k):
            out[p:p+8] = wl[n0:n0+8, kk]
            p += 8
            out[p:p+8] = wg[n0:n0+8, kk]
            p += 8
    return out


def pack_linear16(w: np.ndarray) -> np.ndarray:
    n, k = w.shape
    assert n % 16 == 0
    out = np.empty(n * k, np.float32)
    p = 0
    for n0 in range(0, n, 16):
        for kk in range(k):
            out[p:p+8] = w[n0:n0+8, kk]
            p += 8
            out[p:p+8] = w[n0+8:n0+16, kk]
            p += 8
    return out


def make_case(seed: int, t: int, d: int, h: int):
    rng = np.random.default_rng(seed)
    f = lambda shape, scale=1.0: rng.standard_normal(shape).astype(np.float32) * scale
    return {
        "x": f((t, d), 0.6),
        "gamma": 1.0 + f((d,), 0.08),
        "beta": f((d,), 0.05),
        "dw_w": f((d, 31), 0.04),
        "dw_b": f((d,), 0.03),
        "w1": f((2*h, d), 0.04),
        "b1": f((2*h,), 0.03),
        "w2": f((2*h, h), 0.04),
        "b2": f((2*h,), 0.03),
        "w3": f((d, h), 0.04),
        "b3": f((d,), 0.03),
    }


def torch_ref(q):
    x = torch.from_numpy(q["x"])
    z = F.layer_norm(x, (x.shape[-1],), torch.from_numpy(q["gamma"]), torch.from_numpy(q["beta"]), 1e-5)
    z = F.conv1d(z.T[None], torch.from_numpy(q["dw_w"][:, None, :]), torch.from_numpy(q["dw_b"]),
                 padding=15, groups=x.shape[-1])[0].T.contiguous()
    z = F.linear(z, torch.from_numpy(q["w1"]), torch.from_numpy(q["b1"]))
    left, gate = torch.chunk(z, 2, dim=-1)
    z = left * F.softsign(gate)
    z = F.linear(z, torch.from_numpy(q["w2"]), torch.from_numpy(q["b2"]))
    left, gate = torch.chunk(z, 2, dim=-1)
    z = left * F.softsign(gate)
    z = F.linear(z, torch.from_numpy(q["w3"]), torch.from_numpy(q["b3"]))
    return (x + z).numpy()


def configure(lib):
    fp = ctypes.POINTER(ctypes.c_float)
    lib.ds_layernorm_f32_avx2.argtypes = [fp, fp, fp, fp, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_float]
    lib.ds_depthwise_conv1d_k31_tc_f32_avx2.argtypes = [fp, fp, fp, fp, ctypes.c_size_t, ctypes.c_size_t]
    lib.ds_fused_linear_softsign_glu_f32_avx2_m4n8.argtypes = [fp, fp, fp, fp, fp, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]
    lib.ds_linear_residual_f32_avx2_m4n16.argtypes = [fp, fp, fp, fp, fp, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]


def asm_run(lib, q):
    x = np.ascontiguousarray(q["x"])
    t, d = x.shape
    h = q["w3"].shape[1]
    assert d % 16 == 0 and h % 8 == 0
    dw_wp = np.ascontiguousarray(q["dw_w"].T)
    p1 = pack_glu8(q["w1"])
    p2 = pack_glu8(q["w2"])
    p3 = pack_linear16(q["w3"])
    b1l, b1g = np.ascontiguousarray(q["b1"][:h]), np.ascontiguousarray(q["b1"][h:])
    b2l, b2g = np.ascontiguousarray(q["b2"][:h]), np.ascontiguousarray(q["b2"][h:])
    ln = np.empty((t, d), np.float32)
    dw = np.empty((t, d), np.float32)
    h1 = np.empty((t, h), np.float32)
    h2 = np.empty((t, h), np.float32)
    y = np.empty((t, d), np.float32)
    lib.ds_layernorm_f32_avx2(ptr(x), ptr(q["gamma"]), ptr(q["beta"]), ptr(ln), t, d, ctypes.c_float(1e-5))
    lib.ds_depthwise_conv1d_k31_tc_f32_avx2(ptr(ln), ptr(dw_wp), ptr(q["dw_b"]), ptr(dw), t, d)
    lib.ds_fused_linear_softsign_glu_f32_avx2_m4n8(ptr(dw), ptr(p1), ptr(b1l), ptr(b1g), ptr(h1), t, h, d)
    lib.ds_fused_linear_softsign_glu_f32_avx2_m4n8(ptr(h1), ptr(p2), ptr(b2l), ptr(b2g), ptr(h2), t, h, h)
    lib.ds_linear_residual_f32_avx2_m4n16(ptr(h2), ptr(p3), ptr(q["b3"]), ptr(x), ptr(y), t, d, h)
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", type=Path, default=Path("build/libdsasm_m6.so"))
    ap.add_argument("--large", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(1)
    lib = ctypes.CDLL(str(args.lib.resolve()))
    configure(lib)
    cases = [(37, 64, 64)]
    if args.large:
        cases.append((128, 512, 512))
    failed = False
    for i, (t, d, h) in enumerate(cases):
        q = make_case(0x5A17 + i, t, d, h)
        ref = torch_ref(q)
        got = asm_run(lib, q)
        abs_err = float(np.max(np.abs(ref - got)))
        rel = float(np.max(np.abs(ref - got) / np.maximum(np.abs(ref), 1e-5)))
        ok = abs_err < 1e-4
        print(f"PyTorch vs M6 ASM T={t} D={d} H={h}: max_abs={abs_err:.8g} max_rel={rel:.8g} {'OK' if ok else 'FAIL'}")
        failed |= not ok
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
