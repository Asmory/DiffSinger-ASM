#!/usr/bin/env python3
"""M12 dispatch experiment: pthread-condvar pool vs low-latency spin-dispatch pool.

The spin pool is experimental and intentionally NOT the default runtime mode.
It burns CPU while the pool exists, so this benchmark creates it only for a short
measurement window. ABBA ordering reduces thermal/frequency-order bias.
"""
from __future__ import annotations
import argparse, ctypes, os, time
from pathlib import Path
import numpy as np
import validate_pytorch_lynxnet2 as v


def read_freqs(cpus=(0,2,4,6)):
    out=[]
    for cpu in cpus:
        p=Path(f'/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq')
        try: out.append((cpu,int(p.read_text().strip())/1000.0))
        except Exception: pass
    return out


def fmt_freqs(xs):
    return ', '.join(f'cpu{c}={mhz:.0f}MHz' for c,mhz in xs) if xs else 'unavailable'


def prepare_case(lib,C,Q,seed):
    T=64; I=128; H=C; L=6
    q=v.make_case(seed,T,I,Q,C,H,L,'atan')
    net,keep=v.build_ffi(q)
    spec=np.ascontiguousarray(q['spec']);cond=np.ascontiguousarray(q['cond'])
    cache=np.empty((T,C),np.float32);ref=np.empty((T,I),np.float32);out=np.empty_like(ref)
    ws=np.empty(lib.ds_lynxnet2_workspace_floats(ctypes.byref(net),T),np.float32)
    assert lib.ds_lynxnet2_prepare_condition_f32_avx2(ctypes.byref(net),v.ptr(cond),v.ptr(cache),T)==0
    # The fully serial runtime is already PyTorch-validated by make verify. Use it
    # here as a cheap same-binary reference so dispatch A/B does not spend time in Torch.
    assert lib.ds_lynxnet2_forward_cached_condition_f32_avx2(ctypes.byref(net),v.ptr(spec),v.ptr(cache),ctypes.c_float(q['timestep']),v.ptr(ref),v.ptr(ws),T)==0
    return q,net,keep,spec,cache,ref,out,ws


def one_window(lib,data,spin,reps=7):
    q,net,keep,spec,cache,ref,out,ws=data; T=spec.shape[0]
    os.environ['DSASM_SPIN_POOL']='1' if spin else '0'
    os.environ['DSASM_PARALLEL_DW']='0'
    pool=lib.ds_threadpool_create_auto(); assert pool
    actual=int(lib.ds_threadpool_threads(pool)); cpus=[int(lib.ds_threadpool_cpu_at(pool,i)) for i in range(actual)]
    lib.ds_threadpool_set_2d(pool,1); lib.ds_threadpool_set_atan_pipeline(pool,1)
    lib.ds_threadpool_set_auto_tiles(pool,1); lib.ds_threadpool_set_parallel_depthwise(pool,0)
    for _ in range(2):
        assert lib.ds_lynxnet2_forward_cached_condition_parallel_f32_avx2(ctypes.byref(net),v.ptr(spec),v.ptr(cache),ctypes.c_float(q['timestep']),v.ptr(out),v.ptr(ws),T,pool)==0
    ae=float(np.max(np.abs(out-ref))); assert ae<2e-4,ae
    ts=[]
    for _ in range(reps):
        a=time.perf_counter_ns()
        lib.ds_lynxnet2_forward_cached_condition_parallel_f32_avx2(ctypes.byref(net),v.ptr(spec),v.ptr(cache),ctypes.c_float(q['timestep']),v.ptr(out),v.ptr(ws),T,pool)
        b=time.perf_counter_ns(); ts.append((b-a)*1e-6)
    lib.ds_threadpool_destroy(pool)
    return float(np.median(ts)), actual, cpus, ae


def compare(lib,C,Q,seed):
    data=prepare_case(lib,C,Q,seed)
    print(f'M12 dispatch A/B: T=64 C={C} L=6')
    print('  freq before:',fmt_freqs(read_freqs()))
    rows=[]
    # ABBA. Separate pools keep always-spin workers from contaminating the
    # condvar measurement while the other mode is running.
    for mode in (0,1,1,0):
        ms,n,cpus,ae=one_window(lib,data,bool(mode))
        rows.append((mode,ms))
        print(f"  {'spin' if mode else 'cond':4s}: {ms:8.3f} ms  workers={n} cpus={cpus} max_abs={ae:.3g}")
    cond=[x for m,x in rows if m==0]; spin=[x for m,x in rows if m==1]
    cm=float(np.median(cond)); sm=float(np.median(spin))
    print('  freq after :',fmt_freqs(read_freqs()))
    print(f'  RESULT cond={cm:.3f} ms spin={sm:.3f} ms spin_speedup={cm/sm:.3f}x')
    return cm,sm


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--lib',type=Path,default=Path('build/libdsasm_m12.so')); args=ap.parse_args()
    lib=ctypes.CDLL(str(args.lib.resolve())); v.configure(lib)
    compare(lib,256,96,119)
    compare(lib,1024,384,120)

if __name__=='__main__': main()
