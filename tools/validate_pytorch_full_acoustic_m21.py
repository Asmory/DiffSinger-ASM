#!/usr/bin/env python3
from __future__ import annotations
import argparse,ctypes as C,time
from pathlib import Path
import numpy as np
import torch
import validate_pytorch_fs2_condition_m21 as F2
import validate_pytorch_aux_m20 as AUX
from validate_pytorch_lynxnet2 import Net as RFNet,FP,ptr,make_case as make_rf,build_ffi as build_rf,configure as configure_rf
from validate_pytorch_acoustic import configure_acoustic,torch_acoustic

torch.set_num_threads(1)

def configure(lib):
    F2.setup(Path(lib._name)) # only for type setup side effect on another handle is not useful
    AUX.configure(lib);configure_acoustic(lib)
    lib.ds_fs2_acoustic_condition_workspace_floats.argtypes=[C.POINTER(F2.Acoustic),C.c_size_t,C.c_size_t];lib.ds_fs2_acoustic_condition_workspace_floats.restype=C.c_size_t
    lib.ds_fs2_acoustic_condition_f32_avx2.argtypes=[C.POINTER(F2.Acoustic),F2.PI,C.c_size_t,F2.PI,F2.PF,C.c_size_t,F2.PF,F2.PF,C.c_void_p];lib.ds_fs2_acoustic_condition_f32_avx2.restype=C.c_int
    lib.ds_full_acoustic_workspace_floats.argtypes=[C.POINTER(F2.Acoustic),C.POINTER(AUX.Net),C.POINTER(RFNet),C.c_size_t,C.c_size_t];lib.ds_full_acoustic_workspace_floats.restype=C.c_size_t
    lib.ds_full_acoustic_infer_f32_avx2.argtypes=[C.POINTER(F2.Acoustic),C.POINTER(AUX.Net),C.POINTER(RFNet),F2.PI,C.c_size_t,F2.PI,F2.PF,C.c_size_t,FP,FP,FP,C.c_size_t,C.c_float,C.c_float,C.c_size_t,FP,FP,C.c_void_p]
    lib.ds_full_acoustic_infer_f32_avx2.restype=C.c_int

def one(lib,seed,P,T,Q,fsL,auxC,D,auxL,rfC,rfH,rfL,steps=.4,t0=.4,bench=False):
    if isinstance(steps,float): steps_i=4
    else: steps_i=int(steps)
    aw,akeep,refs,emb,dw,db,fg,fb,tok,dur,mel2ph,f0,extra=F2.make_case(seed,P,T,Q,fsL,2)
    cond=F2.ref_full(refs,emb,dw,db,fg,fb,tok,dur,mel2ph,f0,2,extra)
    aq=AUX.make(seed+77,T,Q,auxC,D,auxL);aq['x']=np.ascontiguousarray(cond.copy());aux_norm=AUX.ref(aq);aux_raw=np.ascontiguousarray(aux_norm*6.0-6.0)
    rq=make_rf(seed+155,T,D,Q,rfC,rfH,rfL,'atan');rq['cond']=np.ascontiguousarray(cond.copy())
    rng=np.random.default_rng(seed+333);noise=np.ascontiguousarray(rng.standard_normal((T,D)).astype(np.float32));mask=np.ascontiguousarray((mel2ph>0).astype(np.float32));lo=np.array([-12.],np.float32);hi=np.array([0.],np.float32)
    ref=torch_acoustic(rq,noise,aux_raw,mask,lo,hi,t0,steps_i)
    aux,auxkeep=AUX.build(aq);rf,rfkeep=build_rf(rq)
    ws=np.empty(lib.ds_full_acoustic_workspace_floats(C.byref(aw),C.byref(aux),C.byref(rf),P,T),np.float32);out=np.empty((T,D),np.float32)
    pool=lib.ds_threadpool_create_auto();assert pool
    def call():return lib.ds_full_acoustic_infer_f32_avx2(C.byref(aw),C.byref(aux),C.byref(rf),F2.pint(tok),P,F2.pint(mel2ph),F2.pfloat(f0),T,ptr(noise),ptr(lo),ptr(hi),1,C.c_float(t0),C.c_float(1000.0),steps_i,ptr(out),ptr(ws),pool)
    rc=call();assert rc==0,rc
    e=np.abs(out-ref);ma=float(e.max());mr=float((e/np.maximum(np.abs(ref),1e-5)).max());pad=float(np.abs(out[mask==0]).max(initial=0))
    n=int(lib.ds_threadpool_threads(pool));cp=[int(lib.ds_threadpool_cpu_at(pool,i)) for i in range(n)]
    ok=ma<2e-3 and pad==0
    print(f'M21 full acoustic P={P} T={T} Q={Q} auxC={auxC} D={D} rfC={rfC} steps={steps_i} workers={n}: max_abs={ma:.8g} max_rel={mr:.8g} pad={pad:.3g} cpus={cp} {"OK" if ok else "FAIL"}')
    if not ok:raise SystemExit(2)
    if bench:
        call();vals=[]
        for _ in range(3):a=time.perf_counter();call();vals.append(time.perf_counter()-a)
        med=float(np.median(vals));audio=T*512/44100
        print(f'  M21 tokens-to-mel median={med*1e3:.3f} ms; audio={audio*1e3:.3f} ms; RTF={med/audio:.3f}; realtime-speed={audio/med:.3f}x')
    lib.ds_threadpool_destroy(pool)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--lib',type=Path,required=True);ap.add_argument('--official-shape',action='store_true');a=ap.parse_args()
    lib=C.CDLL(str(a.lib.resolve()));configure(lib)
    one(lib,2301,9,24,64,1,64,64,2,64,64,2,4,.4)
    if a.official_shape:one(lib,2321,32,64,384,4,512,128,6,1024,1024,6,20,.4,True)
if __name__=='__main__':main()
