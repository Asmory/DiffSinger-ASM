#!/usr/bin/env python3
"""M18 Rectified Flow Euler parity against current OpenVPI semantics.

Reference equations mirror modules/core/reflow.py:
  x0 = t_start * x_end + (1 - t_start) * noise   (shallow)
  dt = (1 - t_start) / max(1, steps)
  x <- x + velocity_fn(x, time_scale_factor * t) * dt
  t = t_start + i * float32(dt)

Noise is explicitly shared between PyTorch and the native runtime; M18 does not
choose an RNG policy yet.
"""
from __future__ import annotations
import argparse, ctypes, time
from pathlib import Path
import numpy as np
import torch

from validate_pytorch_lynxnet2 import FP, Net, ptr, configure, make_case, build_ffi, torch_ref


def configure_reflow(lib):
    configure(lib)
    lib.ds_reflow_euler_workspace_floats.argtypes=[ctypes.POINTER(Net),ctypes.c_size_t]
    lib.ds_reflow_euler_workspace_floats.restype=ctypes.c_size_t
    lib.ds_reflow_norm_spec_f32.argtypes=[FP,FP,FP,ctypes.c_size_t,FP,ctypes.c_size_t,ctypes.c_size_t]
    lib.ds_reflow_norm_spec_f32.restype=ctypes.c_int
    lib.ds_reflow_denorm_spec_f32.argtypes=[FP,FP,FP,ctypes.c_size_t,FP,ctypes.c_size_t,ctypes.c_size_t]
    lib.ds_reflow_denorm_spec_f32.restype=ctypes.c_int
    args=[ctypes.POINTER(Net),FP,FP,FP,ctypes.c_float,ctypes.c_float,ctypes.c_size_t,FP,FP,ctypes.c_size_t,ctypes.c_void_p]
    lib.ds_reflow_euler_sample_cached_condition_f32_avx2.argtypes=args
    lib.ds_reflow_euler_sample_cached_condition_f32_avx2.restype=ctypes.c_int
    lib.ds_reflow_euler_sample_f32_avx2.argtypes=args
    lib.ds_reflow_euler_sample_f32_avx2.restype=ctypes.c_int
    if hasattr(lib,'ds_threadpool_get_n_owner'):
        lib.ds_threadpool_get_n_owner.argtypes=[ctypes.c_void_p];lib.ds_threadpool_get_n_owner.restype=ctypes.c_int
    if hasattr(lib,'ds_threadpool_get_kblocked_atan'):
        lib.ds_threadpool_get_kblocked_atan.argtypes=[ctypes.c_void_p];lib.ds_threadpool_get_kblocked_atan.restype=ctypes.c_int
    if hasattr(lib,'ds_threadpool_k_block'):
        lib.ds_threadpool_k_block.argtypes=[ctypes.c_void_p];lib.ds_threadpool_k_block.restype=ctypes.c_size_t


def torch_norm(x, lo, hi):
    lo_t=torch.tensor(lo,dtype=torch.float32)
    hi_t=torch.tensor(hi,dtype=torch.float32)
    return (x-lo_t)/(hi_t-lo_t)*2-1


def torch_denorm(x, lo, hi):
    lo_t=torch.tensor(lo,dtype=torch.float32)
    hi_t=torch.tensor(hi,dtype=torch.float32)
    return (x+1)/2*(hi_t-lo_t)+lo_t


def ref_velocity(q, x, timestep):
    qq=dict(q)
    qq['spec']=np.ascontiguousarray(x.detach().cpu().numpy(),dtype=np.float32)
    # Upstream Euler keeps t as a one-element float32 tensor for B=1.
    # validate_pytorch_lynxnet2.torch_ref() models a single sample and expects
    # a scalar timestep, so unwrap exactly that one element without changing
    # float32 precision/order.
    ts=np.asarray(timestep.detach().cpu().numpy() if torch.is_tensor(timestep) else timestep,dtype=np.float32)
    if ts.size != 1:
        raise ValueError(f'expected one timestep for B=1 reference, got shape={ts.shape}')
    qq['timestep']=np.float32(ts.reshape(-1)[0])
    return torch.from_numpy(torch_ref(qq))


def torch_reflow(q, noise, src_raw, lo, hi, t_start, steps, scale):
    noise=torch.from_numpy(np.ascontiguousarray(noise,dtype=np.float32))
    if t_start>0:
        assert src_raw is not None
        src=torch_norm(torch.from_numpy(np.ascontiguousarray(src_raw,dtype=np.float32)),lo,hi)
        if t_start>=1:
            return src.clone()
        # Keep the same scalar-tensor operation ordering as upstream.
        x=t_start*src+(1-t_start)*noise
    else:
        x=noise.clone()
    if t_start>=1:
        return x
    dt=(1.0-t_start)/max(1,steps)
    dts=torch.tensor([dt],dtype=x.dtype)
    for i in range(steps):
        t=t_start+i*dts
        v=ref_velocity(q,x,scale*t)
        x += v*dt
    return x.float()


def make_inputs(q, seed):
    rng=np.random.default_rng(seed)
    T,I=q['spec'].shape
    noise=np.ascontiguousarray(rng.standard_normal((T,I)).astype(np.float32))
    # Current acoustic default range is [-12,0]; keep source comfortably inside.
    src=np.ascontiguousarray((-10.5+8.5*rng.random((T,I))).astype(np.float32))
    return noise,src


def run_case(lib, seed, T,I,Q,C,H,L, t_start, steps, parallel=True):
    q=make_case(seed,T,I,Q,C,H,L,'atan')
    noise,src_raw=make_inputs(q,seed+1000)
    lo=np.array([-12.0],np.float32); hi=np.array([0.0],np.float32)
    src_norm=np.empty_like(src_raw)
    assert lib.ds_reflow_norm_spec_f32(ptr(src_raw),ptr(lo),ptr(hi),1,ptr(src_norm),T,I)==0
    src_ref=torch_norm(torch.from_numpy(src_raw),lo,hi).numpy()
    norm_err=float(np.max(np.abs(src_norm-src_ref)))

    ref=torch_reflow(q,noise,src_raw if t_start>0 else None,lo,hi,t_start,steps,1000.0).numpy()
    net,keep=build_ffi(q)
    ws=np.empty(lib.ds_reflow_euler_workspace_floats(ctypes.byref(net),T),np.float32)
    out=np.empty_like(noise)
    cond=np.ascontiguousarray(q['cond'])
    pool=lib.ds_threadpool_create_auto() if parallel else None
    if pool:
        n=int(lib.ds_threadpool_threads(pool)); cpus=[int(lib.ds_threadpool_cpu_at(pool,i)) for i in range(n)]
    else:
        n=1; cpus=[]
    srcp=ptr(src_norm) if t_start>0 else None
    rc=lib.ds_reflow_euler_sample_f32_avx2(ctypes.byref(net),ptr(noise),srcp,ptr(cond),
            ctypes.c_float(t_start),ctypes.c_float(1000.0),steps,ptr(out),ptr(ws),T,pool)
    assert rc==0
    ae=float(np.max(np.abs(out-ref))); re=float(np.max(np.abs(out-ref)/np.maximum(np.abs(ref),1e-5)))

    # Conditioner-cache path must be identical to the convenience wrapper.
    cache=np.empty((T,C),np.float32)
    if pool:
        assert lib.ds_lynxnet2_prepare_condition_parallel_f32_avx2(ctypes.byref(net),ptr(cond),ptr(cache),T,pool)==0
    else:
        assert lib.ds_lynxnet2_prepare_condition_f32_avx2(ctypes.byref(net),ptr(cond),ptr(cache),T)==0
    out2=np.empty_like(out)
    assert lib.ds_reflow_euler_sample_cached_condition_f32_avx2(ctypes.byref(net),ptr(noise),srcp,ptr(cache),
            ctypes.c_float(t_start),ctypes.c_float(1000.0),steps,ptr(out2),ptr(ws),T,pool)==0
    cache_err=float(np.max(np.abs(out2-out)))

    raw=np.empty_like(out)
    assert lib.ds_reflow_denorm_spec_f32(ptr(out),ptr(lo),ptr(hi),1,ptr(raw),T,I)==0
    raw_ref=torch_denorm(torch.from_numpy(ref),lo,hi).numpy()
    den_err=float(np.max(np.abs(raw-raw_ref)))
    print(f"M18 Euler T={T} I={I} C={C} L={L} t0={t_start:g} steps={steps} workers={n}: "
          f"norm={norm_err:.3g} max_abs={ae:.8g} max_rel={re:.8g} cache={cache_err:.3g} denorm={den_err:.3g} "
          f"{'OK' if ae<3e-4 and den_err<3e-4 else 'FAIL'}")
    if pool:
        owner=int(lib.ds_threadpool_get_n_owner(pool)) if hasattr(lib,'ds_threadpool_get_n_owner') else -1
        kb=int(lib.ds_threadpool_get_kblocked_atan(pool)) if hasattr(lib,'ds_threadpool_get_kblocked_atan') else -1
        kbs=int(lib.ds_threadpool_k_block(pool)) if hasattr(lib,'ds_threadpool_k_block') else 0
        print(f"  pool cpus={cpus} defaults: owner={owner} kblocked={kb} kblock={kbs}")
        lib.ds_threadpool_destroy(pool)
    assert ae<3e-4 and cache_err<1e-6 and den_err<3e-4


def bench_official(lib, steps):
    T,I,Q,C,H,L=64,128,384,1024,1024,6
    q=make_case(180,T,I,Q,C,H,L,'atan')
    noise,src_raw=make_inputs(q,1180)
    lo=np.array([-12.0],np.float32); hi=np.array([0.0],np.float32)
    src_norm=np.empty_like(src_raw); assert lib.ds_reflow_norm_spec_f32(ptr(src_raw),ptr(lo),ptr(hi),1,ptr(src_norm),T,I)==0
    net,keep=build_ffi(q); cond=np.ascontiguousarray(q['cond'])
    ws=np.empty(lib.ds_reflow_euler_workspace_floats(ctypes.byref(net),T),np.float32); out=np.empty_like(noise)
    pool=lib.ds_threadpool_create_auto(); assert pool
    n=int(lib.ds_threadpool_threads(pool)); cpus=[int(lib.ds_threadpool_cpu_at(pool,i)) for i in range(n)]
    cache=np.empty((T,C),np.float32)
    t0=time.perf_counter();
    assert lib.ds_lynxnet2_prepare_condition_parallel_f32_avx2(ctypes.byref(net),ptr(cond),ptr(cache),T,pool)==0
    t1=time.perf_counter()
    # one warm-up sampler, then three timed runs using identical inputs
    for _ in range(1):
        assert lib.ds_reflow_euler_sample_cached_condition_f32_avx2(ctypes.byref(net),ptr(noise),ptr(src_norm),ptr(cache),ctypes.c_float(.4),ctypes.c_float(1000.0),steps,ptr(out),ptr(ws),T,pool)==0
    vals=[]
    for _ in range(3):
        a=time.perf_counter();
        assert lib.ds_reflow_euler_sample_cached_condition_f32_avx2(ctypes.byref(net),ptr(noise),ptr(src_norm),ptr(cache),ctypes.c_float(.4),ctypes.c_float(1000.0),steps,ptr(out),ptr(ws),T,pool)==0
        b=time.perf_counter(); vals.append(b-a)
    med=float(np.median(vals))
    print(f"M18 official Euler shallow: T=64 I=128 Q=384 C=1024 L=6 steps={steps} T_start=0.4")
    print(f"  workers={n} cpus={cpus}")
    print(f"  conditioner projection once={(t1-t0)*1e3:.3f} ms")
    print(f"  sampler median={med*1e3:.3f} ms; per-step={med*1e3/max(1,steps):.3f} ms")
    print(f"  20-step realtime estimate @44.1k/hop512/T64 chunk: model-only {med*1e3:.3f} ms")
    lib.ds_threadpool_destroy(pool)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--lib',type=Path,default=Path('build/libdsasm_m18.so'))
    ap.add_argument('--official-shape',action='store_true')
    ap.add_argument('--steps',type=int,default=20)
    args=ap.parse_args()
    torch.set_num_threads(1)
    lib=ctypes.CDLL(str(args.lib.resolve())); configure_reflow(lib)
    # Full diffusion, shallow diffusion, and t_start=1 edge semantics.
    run_case(lib,181,19,64,48,64,64,2,0.0,4)
    run_case(lib,182,31,64,48,64,64,2,0.4,5)
    run_case(lib,183,17,64,48,64,64,2,1.0,20)
    # A medium six-layer case exercises the persistent parallel path.
    run_case(lib,184,64,128,96,256,256,6,0.4,6)
    if args.official_shape:
        bench_official(lib,args.steps)

if __name__=='__main__': main()
