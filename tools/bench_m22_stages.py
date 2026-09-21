#!/usr/bin/env python3
from __future__ import annotations
import argparse,ctypes as C,time
from pathlib import Path
import numpy as np
import validate_pytorch_full_acoustic_m21 as M21
import validate_pytorch_fs2_condition_m21 as F2
import validate_pytorch_aux_m20 as AUX
from validate_pytorch_lynxnet2 import Net as RFNet,ptr,make_case as make_rf,build_ffi as build_rf

def cfg(lib):
    M21.configure(lib);AUX.configure(lib)
    lib.ds_acoustic_reflow_normsrc_workspace_floats.argtypes=[C.POINTER(RFNet),C.c_size_t];lib.ds_acoustic_reflow_normsrc_workspace_floats.restype=C.c_size_t
    lib.ds_acoustic_reflow_decode_normsrc_f32_avx2.argtypes=[C.POINTER(RFNet),AUX.FP,AUX.FP,AUX.FP,AUX.FP,AUX.FP,AUX.FP,C.c_size_t,C.c_float,C.c_float,C.c_size_t,AUX.FP,AUX.FP,C.c_size_t,C.c_void_p];lib.ds_acoustic_reflow_decode_normsrc_f32_avx2.restype=C.c_int
    lib.ds_full_acoustic_normfast_workspace_floats.argtypes=[C.POINTER(F2.Acoustic),C.POINTER(AUX.Net),C.POINTER(RFNet),C.c_size_t,C.c_size_t];lib.ds_full_acoustic_normfast_workspace_floats.restype=C.c_size_t
    lib.ds_full_acoustic_infer_normfast_f32_avx2.argtypes=[C.POINTER(F2.Acoustic),C.POINTER(AUX.Net),C.POINTER(RFNet),F2.PI,C.c_size_t,F2.PI,F2.PF,C.c_size_t,AUX.FP,AUX.FP,AUX.FP,C.c_size_t,C.c_float,C.c_float,C.c_size_t,AUX.FP,AUX.FP,C.c_void_p];lib.ds_full_acoustic_infer_normfast_f32_avx2.restype=C.c_int

def med(fn,n):
    xs=[]
    for _ in range(n):
        a=time.perf_counter();rc=fn();b=time.perf_counter();assert rc==0;xs.append(b-a)
    return float(np.median(xs))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--lib',type=Path,required=True);ap.add_argument('--official-shape',action='store_true');a=ap.parse_args()
    lib=C.CDLL(str(a.lib.resolve()));cfg(lib)
    P,T,Q,fsL,auxC,D,auxL,rfC,rfH,rfL,steps=32,64,384,4,512,128,6,1024,1024,6,20
    seed=2422
    aw,akeep,refs,emb,dw,db,fg,fb,tok,dur,mel2ph,f0,extra=F2.make_case(seed,P,T,Q,fsL,2)
    cond_ref=F2.ref_full(refs,emb,dw,db,fg,fb,tok,dur,mel2ph,f0,2,extra)
    aq=AUX.make(seed+77,T,Q,auxC,D,auxL);aq['x']=np.ascontiguousarray(cond_ref.copy());aux,auxkeep=AUX.build(aq)
    rq=make_rf(seed+155,T,D,Q,rfC,rfH,rfL,'atan');rq['cond']=np.ascontiguousarray(cond_ref.copy());rf,rfkeep=build_rf(rq)
    rng=np.random.default_rng(seed+333);noise=np.ascontiguousarray(rng.standard_normal((T,D)).astype(np.float32));mask=np.ascontiguousarray((mel2ph>0).astype(np.float32));lo=np.array([-12.],np.float32);hi=np.array([0.],np.float32)
    pool=lib.ds_threadpool_create_auto();assert pool
    n=int(lib.ds_threadpool_threads(pool));cp=[int(lib.ds_threadpool_cpu_at(pool,i)) for i in range(n)]
    fws=np.empty(lib.ds_fs2_acoustic_condition_workspace_floats(C.byref(aw),P,T),np.float32);cond=np.empty((T,Q),np.float32)
    aws=np.empty(lib.ds_aux_convnext_workspace_floats(C.byref(aux),T),np.float32);an=np.empty((T,D),np.float32)
    rws=np.empty(lib.ds_acoustic_reflow_normsrc_workspace_floats(C.byref(rf),T),np.float32);rout=np.empty((T,D),np.float32)
    oldws=np.empty(lib.ds_full_acoustic_workspace_floats(C.byref(aw),C.byref(aux),C.byref(rf),P,T),np.float32);oldout=np.empty((T,D),np.float32)
    newws=np.empty(lib.ds_full_acoustic_normfast_workspace_floats(C.byref(aw),C.byref(aux),C.byref(rf),P,T),np.float32);newout=np.empty((T,D),np.float32)
    def fs(): return lib.ds_fs2_acoustic_condition_f32_avx2(C.byref(aw),F2.pint(tok),P,F2.pint(mel2ph),F2.pfloat(f0),T,F2.pfloat(cond),F2.pfloat(fws),pool)
    def auxf(): return lib.ds_aux_convnext_forward_norm_f32_avx2(C.byref(aux),AUX.ptr(cond),AUX.ptr(an),AUX.ptr(aws),T,pool)
    def rff(): return lib.ds_acoustic_reflow_decode_normsrc_f32_avx2(C.byref(rf),AUX.ptr(cond),AUX.ptr(an),ptr(noise),ptr(mask),ptr(lo),ptr(hi),1,C.c_float(.4),C.c_float(1000.),steps,ptr(rout),ptr(rws),T,pool)
    base=(C.byref(aw),C.byref(aux),C.byref(rf),F2.pint(tok),P,F2.pint(mel2ph),F2.pfloat(f0),T,ptr(noise),ptr(lo),ptr(hi),1,C.c_float(.4),C.c_float(1000.),steps)
    def old(): return lib.ds_full_acoustic_infer_f32_avx2(*base,ptr(oldout),ptr(oldws),pool)
    def new(): return lib.ds_full_acoustic_infer_normfast_f32_avx2(*base,ptr(newout),ptr(newws),pool)
    fs();auxf();rff();old();new()
    tfs=med(fs,7);fs();taux=med(auxf,7);fs();auxf();trf=med(rff,3)
    olds=[];news=[]
    for i in range(5):
        order=(old,new) if i%2==0 else (new,old)
        for fn in order:
            z=time.perf_counter();rc=fn();q=time.perf_counter();assert rc==0
            (olds if fn is old else news).append(q-z)
    mo=float(np.median(olds));mn=float(np.median(news));diff=float(np.abs(oldout-newout).max())
    audio=T*512/44100
    print(f'M22 stage profile official P={P} T={T} Q={Q} workers={n} cpus={cp}')
    print(f'  FS2 condition median={tfs*1e3:.3f} ms')
    print(f'  Aux normalized median={taux*1e3:.3f} ms')
    print(f'  RF20 normsrc median={trf*1e3:.3f} ms')
    print(f'  stage sum={((tfs+taux+trf)*1e3):.3f} ms')
    print(f'  M21 legacy full median={mo*1e3:.3f} ms')
    print(f'  M22 normfast full median={mn*1e3:.3f} ms; speedup={mo/mn:.3f}x; old_vs_new max_abs={diff:.8g}')
    print(f'  M22 RTF={mn/audio:.3f}; realtime-speed={audio/mn:.3f}x')
    print(f'  workspace legacy={oldws.size*4/1048576:.3f} MiB normfast={newws.size*4/1048576:.3f} MiB')
    lib.ds_threadpool_destroy(pool)
if __name__=='__main__':main()
