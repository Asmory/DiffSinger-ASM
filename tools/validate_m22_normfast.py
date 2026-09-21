#!/usr/bin/env python3
from __future__ import annotations
import argparse,ctypes as C
from pathlib import Path
import numpy as np
import validate_pytorch_full_acoustic_m21 as M21
import validate_pytorch_fs2_condition_m21 as F2
import validate_pytorch_aux_m20 as AUX
from validate_pytorch_lynxnet2 import Net as RFNet,ptr,make_case as make_rf,build_ffi as build_rf
from validate_pytorch_acoustic import torch_acoustic

def cfg(lib):
    M21.configure(lib)
    lib.ds_full_acoustic_normfast_workspace_floats.argtypes=[C.POINTER(F2.Acoustic),C.POINTER(AUX.Net),C.POINTER(RFNet),C.c_size_t,C.c_size_t]
    lib.ds_full_acoustic_normfast_workspace_floats.restype=C.c_size_t
    lib.ds_full_acoustic_infer_normfast_f32_avx2.argtypes=[C.POINTER(F2.Acoustic),C.POINTER(AUX.Net),C.POINTER(RFNet),F2.PI,C.c_size_t,F2.PI,F2.PF,C.c_size_t,AUX.FP,AUX.FP,AUX.FP,C.c_size_t,C.c_float,C.c_float,C.c_size_t,AUX.FP,AUX.FP,C.c_void_p]
    lib.ds_full_acoustic_infer_normfast_f32_avx2.restype=C.c_int

def one(lib,seed,P,T,Q,fsL,auxC,D,auxL,rfC,rfH,rfL,steps,t0):
    aw,akeep,refs,emb,dw,db,fg,fb,tok,dur,mel2ph,f0,extra=F2.make_case(seed,P,T,Q,fsL,2)
    cond=F2.ref_full(refs,emb,dw,db,fg,fb,tok,dur,mel2ph,f0,2,extra)
    aq=AUX.make(seed+77,T,Q,auxC,D,auxL);aq['x']=np.ascontiguousarray(cond.copy())
    aux_norm=AUX.ref(aq);aux_raw=np.ascontiguousarray(aux_norm*6.0-6.0)
    rq=make_rf(seed+155,T,D,Q,rfC,rfH,rfL,'atan');rq['cond']=np.ascontiguousarray(cond.copy())
    rng=np.random.default_rng(seed+333);noise=np.ascontiguousarray(rng.standard_normal((T,D)).astype(np.float32))
    mask=np.ascontiguousarray((mel2ph>0).astype(np.float32));lo=np.array([-12.],np.float32);hi=np.array([0.],np.float32)
    ref=torch_acoustic(rq,noise,aux_raw,mask,lo,hi,t0,steps)
    aux,auxkeep=AUX.build(aq);rf,rfkeep=build_rf(rq)
    pool=lib.ds_threadpool_create_auto();assert pool
    nold=lib.ds_full_acoustic_workspace_floats(C.byref(aw),C.byref(aux),C.byref(rf),P,T)
    nnew=lib.ds_full_acoustic_normfast_workspace_floats(C.byref(aw),C.byref(aux),C.byref(rf),P,T)
    wold=np.empty(nold,np.float32);wnew=np.empty(nnew,np.float32);o1=np.empty((T,D),np.float32);o2=np.empty_like(o1)
    a=(C.byref(aw),C.byref(aux),C.byref(rf),F2.pint(tok),P,F2.pint(mel2ph),F2.pfloat(f0),T,ptr(noise),ptr(lo),ptr(hi),1,C.c_float(t0),C.c_float(1000.0),steps)
    assert lib.ds_full_acoustic_infer_f32_avx2(*a,ptr(o1),ptr(wold),pool)==0
    assert lib.ds_full_acoustic_infer_normfast_f32_avx2(*a,ptr(o2),ptr(wnew),pool)==0
    e=np.abs(o2-ref);ma=float(e.max());ab=float(np.abs(o2-o1).max());pad=float(np.abs(o2[mask==0]).max(initial=0))
    print(f'M22 normfast P={P} T={T} Q={Q} steps={steps}: ref_max_abs={ma:.8g} old_vs_new={ab:.8g} pad={pad:.3g} workspace={nold}->{nnew} floats {"OK" if ma<2e-3 and pad==0 else "FAIL"}')
    lib.ds_threadpool_destroy(pool)
    if ma>=2e-3 or pad!=0: raise SystemExit(2)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--lib',type=Path,required=True);a=ap.parse_args()
    lib=C.CDLL(str(a.lib.resolve()));cfg(lib)
    one(lib,2401,9,24,64,1,64,64,2,64,64,2,4,.4)
    one(lib,2402,17,48,128,2,128,64,3,256,256,3,6,.4)
if __name__=='__main__':main()
