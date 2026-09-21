#!/usr/bin/env python3
from __future__ import annotations
import argparse,ctypes as C,time
from pathlib import Path
import numpy as np
import validate_pytorch_fs2_condition_m21 as F2
import validate_pytorch_aux_m20 as AUX
import validate_pytorch_full_acoustic_m21 as M21
from validate_pytorch_lynxnet2 import make_case as make_rf,build_ffi as build_rf,ptr

def cfg(lib):
    M21.configure(lib);AUX.configure(lib)
    lib.ds_acoustic_reflow_normsrc_workspace_floats.argtypes=[C.POINTER(M21.RFNet),C.c_size_t];lib.ds_acoustic_reflow_normsrc_workspace_floats.restype=C.c_size_t
    lib.ds_acoustic_reflow_decode_normsrc_f32_avx2.argtypes=[C.POINTER(M21.RFNet),AUX.FP,AUX.FP,AUX.FP,AUX.FP,AUX.FP,AUX.FP,C.c_size_t,C.c_float,C.c_float,C.c_size_t,AUX.FP,AUX.FP,C.c_size_t,C.c_void_p];lib.ds_acoustic_reflow_decode_normsrc_f32_avx2.restype=C.c_int
    lib.ds_full_acoustic_normfast_workspace_floats.argtypes=[C.POINTER(F2.Acoustic),C.POINTER(AUX.Net),C.POINTER(M21.RFNet),C.c_size_t,C.c_size_t];lib.ds_full_acoustic_normfast_workspace_floats.restype=C.c_size_t
    lib.ds_full_acoustic_infer_normfast_f32_avx2.argtypes=[C.POINTER(F2.Acoustic),C.POINTER(AUX.Net),C.POINTER(M21.RFNet),F2.PI,C.c_size_t,F2.PI,F2.PF,C.c_size_t,AUX.FP,AUX.FP,AUX.FP,C.c_size_t,C.c_float,C.c_float,C.c_size_t,AUX.FP,AUX.FP,C.c_void_p];lib.ds_full_acoustic_infer_normfast_f32_avx2.restype=C.c_int

def freq_mhz(cpus):
    vals=[]
    for c in cpus:
        p=Path(f'/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_cur_freq')
        try: vals.append(float(p.read_text().strip())/1000.0)
        except Exception: pass
    return float(np.mean(vals)) if vals else float('nan')

def stats(xs):
    a=np.asarray(xs)*1e3; med=float(np.median(a));p90=float(np.percentile(a,90));mx=float(a.max());
    return med,p90,mx,int(np.sum(a>1.5*med))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--lib',type=Path,required=True);ap.add_argument('--blocks',type=int,default=4);a=ap.parse_args()
    lib=C.CDLL(str(a.lib.resolve()));cfg(lib)
    P,T,Q,fsL,auxC,D,auxL,rfC,rfH,rfL,steps=32,64,384,4,512,128,6,1024,1024,6,20; seed=2323
    aw,akeep,refs,emb,dw,db,fg,fb,tok,dur,mel2ph,f0,extra=F2.make_case(seed,P,T,Q,fsL,2)
    aq=AUX.make(seed+77,T,Q,auxC,D,auxL);aux,auxkeep=AUX.build(aq)
    rq=make_rf(seed+155,T,D,Q,rfC,rfH,rfL,'atan');rf,rfkeep=build_rf(rq)
    rng=np.random.default_rng(seed+333);noise=np.ascontiguousarray(rng.standard_normal((T,D)).astype(np.float32));mask=np.ascontiguousarray((mel2ph>0).astype(np.float32));lo=np.array([-12.],np.float32);hi=np.array([0.],np.float32)
    pool=lib.ds_threadpool_create_auto();assert pool;n=int(lib.ds_threadpool_threads(pool));cp=[int(lib.ds_threadpool_cpu_at(pool,i)) for i in range(n)]
    fws=np.empty(lib.ds_fs2_acoustic_condition_workspace_floats(C.byref(aw),P,T),np.float32);cond=np.empty((T,Q),np.float32)
    aws=np.empty(lib.ds_aux_convnext_workspace_floats(C.byref(aux),T),np.float32);an=np.empty((T,D),np.float32)
    rws=np.empty(lib.ds_acoustic_reflow_normsrc_workspace_floats(C.byref(rf),T),np.float32);chainout=np.empty((T,D),np.float32)
    fullws=np.empty(lib.ds_full_acoustic_normfast_workspace_floats(C.byref(aw),C.byref(aux),C.byref(rf),P,T),np.float32);fullout=np.empty((T,D),np.float32)
    base=(C.byref(aw),C.byref(aux),C.byref(rf),F2.pint(tok),P,F2.pint(mel2ph),F2.pfloat(f0),T,ptr(noise),ptr(lo),ptr(hi),1,C.c_float(.4),C.c_float(1000.),steps)
    def fs():return lib.ds_fs2_acoustic_condition_f32_avx2(C.byref(aw),F2.pint(tok),P,F2.pint(mel2ph),F2.pfloat(f0),T,F2.pfloat(cond),F2.pfloat(fws),pool)
    def auxf():return lib.ds_aux_convnext_forward_norm_f32_avx2(C.byref(aux),AUX.ptr(cond),AUX.ptr(an),AUX.ptr(aws),T,pool)
    def rff():return lib.ds_acoustic_reflow_decode_normsrc_f32_avx2(C.byref(rf),AUX.ptr(cond),AUX.ptr(an),ptr(noise),ptr(mask),ptr(lo),ptr(hi),1,C.c_float(.4),C.c_float(1000.),steps,ptr(chainout),ptr(rws),T,pool)
    stage_samples=[]
    def chain():
        z=time.perf_counter();rc=fs();a1=time.perf_counter();
        if rc:return rc
        rc=auxf();a2=time.perf_counter();
        if rc:return rc
        rc=rff();a3=time.perf_counter();stage_samples.append((a1-z,a2-a1,a3-a2));return rc
    def full():return lib.ds_full_acoustic_infer_normfast_f32_avx2(*base,ptr(fullout),ptr(fullws),pool)
    # matched warmup
    for fn in (chain,full,full,chain): assert fn()==0
    chain_times=[];full_times=[];freqs=[]
    seq=(chain,full,full,chain)
    for bi in range(a.blocks):
        for fn in seq:
            f0mhz=freq_mhz(cp);z=time.perf_counter();rc=fn();dt=time.perf_counter()-z;f1mhz=freq_mhz(cp);assert rc==0
            (chain_times if fn is chain else full_times).append(dt);freqs.append((fn.__name__,f0mhz,f1mhz,dt))
            print(f'  block={bi} {fn.__name__:5s} {dt*1e3:9.3f} ms  freq={f0mhz:.0f}->{f1mhz:.0f} MHz')
    cm,cp90,cmax,cout=stats(chain_times);fm,fp90,fmax,fout=stats(full_times)
    # stage samples include warmup; keep only last 2*blocks chain calls, which are measured chain calls.
    ss=np.asarray(stage_samples[-2*a.blocks:])*1e3
    smed=np.median(ss,axis=0)
    diff=float(np.max(np.abs(chainout-fullout)));audio=T*512/44100
    print(f'M23 paired ABBA official P={P} T={T} Q={Q} workers={n} cpus={cp}')
    print(f'  manual-chain median={cm:.3f} ms p90={cp90:.3f} max={cmax:.3f} outliers>1.5x={cout}')
    print(f'    in-chain stage medians: FS2={smed[0]:.3f} Aux={smed[1]:.3f} RF20={smed[2]:.3f} sum={float(smed.sum()):.3f} ms')
    print(f'  full-wrapper median={fm:.3f} ms p90={fp90:.3f} max={fmax:.3f} outliers>1.5x={fout}')
    print(f'  chain/full={cm/fm:.3f}x old_vs_new max_abs={diff:.8g}')
    print(f'  full RTF={fm/1e3/audio:.3f}; realtime-speed={audio/(fm/1e3):.3f}x')
    print(f'  workspace full={fullws.size*4/1048576:.3f} MiB')
    lib.ds_threadpool_destroy(pool)
if __name__=='__main__':main()
