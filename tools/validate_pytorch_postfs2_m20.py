#!/usr/bin/env python3
from __future__ import annotations
import argparse,ctypes,time
from pathlib import Path
import numpy as np
import torch
from validate_pytorch_lynxnet2 import FP,Net,ptr,make_case,build_ffi
from validate_pytorch_acoustic import configure_acoustic,torch_acoustic,make_outer_inputs
from validate_pytorch_aux_m20 import Block as AuxBlock,Net as AuxNet,make as make_aux,build as build_aux,ref as ref_aux,configure as configure_aux

def configure(lib):
    configure_acoustic(lib);configure_aux(lib)
    lib.ds_acoustic_post_fs2_workspace_floats.argtypes=[ctypes.POINTER(AuxNet),ctypes.POINTER(Net),ctypes.c_size_t]
    lib.ds_acoustic_post_fs2_workspace_floats.restype=ctypes.c_size_t
    lib.ds_acoustic_post_fs2_f32_avx2.argtypes=[ctypes.POINTER(AuxNet),ctypes.POINTER(Net),FP,FP,FP,FP,FP,ctypes.c_size_t,ctypes.c_float,ctypes.c_float,ctypes.c_size_t,FP,FP,ctypes.c_size_t,ctypes.c_void_p]
    lib.ds_acoustic_post_fs2_f32_avx2.restype=ctypes.c_int

def one(lib,seed,T,D,Q,C,H,L,A,AL,steps=5,t0=.4,bench=False):
    q=make_case(seed,T,D,Q,C,H,L,'atan')
    aq=make_aux(seed+77,T,Q,A,D,AL); aq['x']=np.ascontiguousarray(q['cond'].copy())
    aux_raw=ref_aux(aq)*6.0-6.0
    rng=np.random.default_rng(seed+900);noise=np.ascontiguousarray(rng.standard_normal((T,D)).astype(np.float32));mask=np.ones(T,np.float32)
    if T>=8: mask[3]=0;mask[-2:]=0
    lo=np.array([-12.],np.float32);hi=np.array([0.],np.float32)
    ref=torch_acoustic(q,noise,np.ascontiguousarray(aux_raw),mask,lo,hi,t0,steps)
    rf,rkeep=build_ffi(q);aux,akeep=build_aux(aq);cond=np.ascontiguousarray(q['cond']);out=np.empty((T,D),np.float32)
    ws=np.empty(lib.ds_acoustic_post_fs2_workspace_floats(ctypes.byref(aux),ctypes.byref(rf),T),np.float32)
    pool=lib.ds_threadpool_create_auto();assert pool
    rc=lib.ds_acoustic_post_fs2_f32_avx2(ctypes.byref(aux),ctypes.byref(rf),ptr(cond),ptr(noise),ptr(mask),ptr(lo),ptr(hi),1,ctypes.c_float(t0),ctypes.c_float(1000),steps,ptr(out),ptr(ws),T,pool);assert rc==0
    ae=float(np.max(np.abs(out-ref)));re=float(np.max(np.abs(out-ref)/np.maximum(np.abs(ref),1e-5)));pad=float(np.max(np.abs(out[mask==0])))
    n=int(lib.ds_threadpool_threads(pool));cpus=[int(lib.ds_threadpool_cpu_at(pool,i)) for i in range(n)]
    print(f'M20 post-FS2 T={T} Q={Q} auxC={A} mel={D} rfC={C} steps={steps} workers={n}: max_abs={ae:.8g} max_rel={re:.8g} pad={pad:.3g} cpus={cpus} {"OK" if ae<1e-3 and pad==0 else "FAIL"}')
    assert ae<1e-3 and pad==0
    if bench:
        for _ in range(1):rc=lib.ds_acoustic_post_fs2_f32_avx2(ctypes.byref(aux),ctypes.byref(rf),ptr(cond),ptr(noise),ptr(mask),ptr(lo),ptr(hi),1,ctypes.c_float(t0),ctypes.c_float(1000),steps,ptr(out),ptr(ws),T,pool);assert rc==0
        vals=[]
        for _ in range(3):
            a=time.perf_counter();rc=lib.ds_acoustic_post_fs2_f32_avx2(ctypes.byref(aux),ctypes.byref(rf),ptr(cond),ptr(noise),ptr(mask),ptr(lo),ptr(hi),1,ctypes.c_float(t0),ctypes.c_float(1000),steps,ptr(out),ptr(ws),T,pool);b=time.perf_counter();assert rc==0;vals.append(b-a)
        med=float(np.median(vals));audio=T*512/44100
        print(f'  full post-FS2 median={med*1e3:.3f} ms; audio={audio*1e3:.3f} ms; RTF={med/audio:.3f}; realtime-speed={audio/med:.3f}x')
    lib.ds_threadpool_destroy(pool)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--lib',type=Path,default=Path('build/libdsasm_m20.so'));ap.add_argument('--official-shape',action='store_true');a=ap.parse_args();torch.set_num_threads(1)
    lib=ctypes.CDLL(str(a.lib.resolve()));configure(lib)
    one(lib,2001,19,64,48,64,64,2,64,2,4,0.4)
    if a.official_shape: one(lib,2002,64,128,384,1024,1024,6,512,6,20,0.4,True)
if __name__=='__main__':main()
