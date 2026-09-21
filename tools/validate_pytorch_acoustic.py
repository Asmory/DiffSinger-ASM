#!/usr/bin/env python3
"""M19 acoustic-decoder envelope parity.

Mirrors current DiffSingerAcoustic.forward() after FS2+aux decoder:
  aux_mel *= (mel2ph > 0)
  mel = RectifiedFlow(condition, src_spec=aux_mel, infer=True)
  mel *= (mel2ph > 0)

The native boundary accepts condition [T,Q], raw aux mel [T,D], explicit noise,
and a 0/1 frame mask, then returns raw/denormalized mel [T,D].
"""
from __future__ import annotations
import argparse, ctypes, time
from pathlib import Path
import numpy as np
import torch

from validate_pytorch_lynxnet2 import FP, Net, ptr, configure, make_case, build_ffi
from validate_pytorch_reflow import configure_reflow, torch_reflow, torch_denorm


def configure_acoustic(lib):
    configure_reflow(lib)
    lib.ds_acoustic_reflow_workspace_floats.argtypes=[ctypes.POINTER(Net),ctypes.c_size_t]
    lib.ds_acoustic_reflow_workspace_floats.restype=ctypes.c_size_t
    lib.ds_acoustic_reflow_decode_f32_avx2.argtypes=[
        ctypes.POINTER(Net), FP, FP, FP, FP, FP, FP, ctypes.c_size_t,
        ctypes.c_float, ctypes.c_float, ctypes.c_size_t,
        FP, FP, ctypes.c_size_t, ctypes.c_void_p
    ]
    lib.ds_acoustic_reflow_decode_f32_avx2.restype=ctypes.c_int


def make_outer_inputs(q, seed, vector_range=False):
    rng=np.random.default_rng(seed)
    T,D=q['spec'].shape
    noise=np.ascontiguousarray(rng.standard_normal((T,D)).astype(np.float32))
    if vector_range:
        lo=np.ascontiguousarray(np.linspace(-13.0,-11.0,D,dtype=np.float32))
        hi=np.ascontiguousarray(np.linspace(-1.0,1.0,D,dtype=np.float32))
        span=hi-lo
        aux=np.ascontiguousarray(lo[None,:]+span[None,:]*(0.1+0.75*rng.random((T,D))).astype(np.float32))
    else:
        lo=np.array([-12.0],np.float32); hi=np.array([0.0],np.float32)
        aux=np.ascontiguousarray((-10.5+8.5*rng.random((T,D))).astype(np.float32))
    mask=np.ones(T,np.float32)
    # Exercise both internal and tail padding without changing tensor shape.
    if T >= 8:
        mask[3]=0.0
        mask[-3:]=0.0
    return noise,aux,mask,lo,hi


def torch_acoustic(q, noise, aux, mask, lo, hi, t_start, steps):
    mask_t=torch.from_numpy(mask)[:,None]
    aux_t=torch.from_numpy(aux)*mask_t
    src=aux_t.numpy() if t_start>0 else None
    x=torch_reflow(q,noise,src,lo,hi,t_start,steps,1000.0)
    mel=torch_denorm(x,lo,hi)
    mel=mel*mask_t
    return mel.numpy()


def run_case(lib,seed,T,D,Q,C,H,L,t_start,steps,vector_range=False):
    q=make_case(seed,T,D,Q,C,H,L,'atan')
    noise,aux,mask,lo,hi=make_outer_inputs(q,seed+1000,vector_range)
    ref=torch_acoustic(q,noise,aux,mask,lo,hi,t_start,steps)
    net,keep=build_ffi(q)
    ws=np.empty(lib.ds_acoustic_reflow_workspace_floats(ctypes.byref(net),T),np.float32)
    out=np.empty_like(aux)
    cond=np.ascontiguousarray(q['cond'])
    pool=lib.ds_threadpool_create_auto(); assert pool
    rc=lib.ds_acoustic_reflow_decode_f32_avx2(
        ctypes.byref(net),ptr(cond),ptr(aux) if t_start>0 else None,ptr(noise) if t_start<1 else None,
        ptr(mask),ptr(lo),ptr(hi),len(lo),ctypes.c_float(t_start),ctypes.c_float(1000.0),steps,
        ptr(out),ptr(ws),T,pool)
    assert rc==0
    ae=float(np.max(np.abs(out-ref)))
    re=float(np.max(np.abs(out-ref)/np.maximum(np.abs(ref),1e-5)))
    pad=float(np.max(np.abs(out[mask==0]))) if np.any(mask==0) else 0.0
    n=int(lib.ds_threadpool_threads(pool));cpus=[int(lib.ds_threadpool_cpu_at(pool,i)) for i in range(n)]
    print(f"M19 acoustic T={T} D={D} Q={Q} C={C} L={L} t0={t_start:g} steps={steps} "
          f"range={'vector' if vector_range else 'scalar'} workers={n}: max_abs={ae:.8g} max_rel={re:.8g} pad={pad:.3g} "
          f"{'OK' if ae<5e-4 and pad==0 else 'FAIL'}")
    print(f"  pool cpus={cpus}")
    lib.ds_threadpool_destroy(pool)
    assert ae<5e-4 and pad==0


def bench_official(lib,steps):
    T,D,Q,C,H,L=64,128,384,1024,1024,6
    q=make_case(190,T,D,Q,C,H,L,'atan')
    noise,aux,mask,lo,hi=make_outer_inputs(q,1190,False)
    net,keep=build_ffi(q);cond=np.ascontiguousarray(q['cond'])
    ws=np.empty(lib.ds_acoustic_reflow_workspace_floats(ctypes.byref(net),T),np.float32)
    out=np.empty_like(aux)
    pool=lib.ds_threadpool_create_auto();assert pool
    n=int(lib.ds_threadpool_threads(pool));cpus=[int(lib.ds_threadpool_cpu_at(pool,i)) for i in range(n)]
    def one():
        return lib.ds_acoustic_reflow_decode_f32_avx2(
            ctypes.byref(net),ptr(cond),ptr(aux),ptr(noise),ptr(mask),ptr(lo),ptr(hi),1,
            ctypes.c_float(.4),ctypes.c_float(1000.0),steps,ptr(out),ptr(ws),T,pool)
    assert one()==0
    vals=[]
    for _ in range(3):
        a=time.perf_counter();assert one()==0;b=time.perf_counter();vals.append(b-a)
    med=float(np.median(vals)); audio_s=T*512/44100.0
    print(f"M19 official acoustic decoder: T=64 mel=128 cond=384 C=1024 L=6 steps={steps} T_start=0.4 spec=[-12,0]")
    print(f"  workers={n} cpus={cpus}")
    print(f"  total median={med*1e3:.3f} ms; per-step={med*1e3/max(1,steps):.3f} ms")
    print(f"  chunk audio={audio_s*1e3:.3f} ms; RTF={med/audio_s:.3f}; realtime-speed={audio_s/med:.3f}x")
    print(f"  padded-output max_abs={float(np.max(np.abs(out[mask==0]))):.3g}")
    lib.ds_threadpool_destroy(pool)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--lib',type=Path,default=Path('build/libdsasm_m19.so'))
    ap.add_argument('--official-shape',action='store_true')
    ap.add_argument('--steps',type=int,default=20)
    args=ap.parse_args()
    torch.set_num_threads(1)
    lib=ctypes.CDLL(str(args.lib.resolve()));configure_acoustic(lib)
    run_case(lib,191,19,64,48,64,64,2,0.0,4)
    run_case(lib,192,31,64,48,64,64,2,0.4,5)
    run_case(lib,193,17,64,48,64,64,2,1.0,20)
    run_case(lib,194,29,64,48,64,64,2,0.4,5,vector_range=True)
    run_case(lib,195,64,128,96,256,256,6,0.4,6)
    if args.official_shape: bench_official(lib,args.steps)

if __name__=='__main__': main()
