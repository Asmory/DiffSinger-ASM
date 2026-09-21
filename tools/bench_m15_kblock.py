#!/usr/bin/env python3
"""M15 same-process A/B: K-blocked ATan packed16 vs legacy full-K inner loop."""
from __future__ import annotations
import argparse, ctypes, os, time
from pathlib import Path
import numpy as np
import validate_pytorch_lynxnet2 as v


def freqs(cpus=(0,2,4,6,8,9,10,11)):
    out=[]
    for c in cpus:
        try:
            mhz=int(Path(f'/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_cur_freq').read_text())/1000
            out.append((c,mhz))
        except Exception: pass
    return ', '.join(f'cpu{c}={m:.0f}MHz' for c,m in out) if out else 'unavailable'


def prepare(lib):
    q=v.make_case(715,64,128,384,1024,1024,6,'atan')
    net,keep=v.build_ffi(q); T=64; C=1024
    spec=np.ascontiguousarray(q['spec']); cond=np.ascontiguousarray(q['cond'])
    cache=np.empty((T,C),np.float32); out=np.empty((T,128),np.float32)
    ws=np.empty(lib.ds_lynxnet2_workspace_floats(ctypes.byref(net),T),np.float32)
    assert lib.ds_lynxnet2_prepare_condition_f32_avx2(ctypes.byref(net),v.ptr(cond),v.ptr(cache),T)==0
    return q,net,keep,spec,cache,out,ws


def setup_api(lib):
    lib.ds_threadpool_set_n_owner.argtypes=[ctypes.c_void_p,ctypes.c_int]
    lib.ds_threadpool_set_kblocked_atan.argtypes=[ctypes.c_void_p,ctypes.c_int]
    lib.ds_threadpool_set_k_block.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
    lib.ds_threadpool_set_k_block.restype=ctypes.c_int


def config(lib,pool,owner,kblock,kb):
    lib.ds_threadpool_set_2d(pool,1); lib.ds_threadpool_set_atan_pipeline(pool,1)
    lib.ds_threadpool_set_parallel_depthwise(pool,0); lib.ds_threadpool_set_auto_tiles(pool,0)
    assert lib.ds_threadpool_set_tiles(pool,32,64)==0
    lib.ds_threadpool_set_indexed_linear(pool,0)
    lib.ds_threadpool_set_n_owner(pool,owner)
    lib.ds_threadpool_set_kblocked_atan(pool,kblock)
    assert lib.ds_threadpool_set_k_block(pool,kb)==0


def call(lib,d,pool):
    q,net,keep,spec,cache,out,ws=d
    rc=lib.ds_lynxnet2_forward_cached_condition_parallel_f32_avx2(
        ctypes.byref(net),v.ptr(spec),v.ptr(cache),ctypes.c_float(q['timestep']),v.ptr(out),v.ptr(ws),64,pool)
    assert rc==0
    return out.copy()


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--lib',type=Path,default=Path('build/libdsasm_m15.so')); ap.add_argument('--reps',type=int,default=21); args=ap.parse_args()
    os.environ['DSASM_SPIN_POOL']='0'; os.environ['DSASM_PARALLEL_DW']='0'; os.environ['DSASM_INDEXED_LINEAR']='0'
    lib=ctypes.CDLL(str(args.lib.resolve())); v.configure(lib); setup_api(lib)
    data=prepare(lib); pool=lib.ds_threadpool_create(8); assert pool
    n=int(lib.ds_threadpool_threads(pool)); cpus=[int(lib.ds_threadpool_cpu_at(pool,i)) for i in range(n)]
    configs=[
        ('dynamic old',0,0,256),
        ('dynamic kb128',0,1,128),('dynamic kb256',0,1,256),('dynamic kb512',0,1,512),
        ('owner kb256',1,1,256),('owner kb512',1,1,512),
    ]
    print(f'M15 K-block A/B: official T=64 C=1024 L=6 workers={n} cpus={cpus}')
    print('freq before:',freqs())
    config(lib,pool,*configs[0][1:]); ref=call(lib,data,pool)
    for _,o,k,kb in configs:
        config(lib,pool,o,k,kb); out=call(lib,data,pool); d=float(np.max(np.abs(out-ref))); print(f'  parity {_:13s}: max_abs={d:g}'); assert d==0.0
    buckets={name:[] for name,*_ in configs}
    # Round-robin/reversed rotation reduces first/last thermal bias.
    for r in range(args.reps):
        order=configs[r%len(configs):]+configs[:r%len(configs)]
        if r&1: order=list(reversed(order))
        for name,o,k,kb in order:
            config(lib,pool,o,k,kb)
            a=time.perf_counter_ns(); call(lib,data,pool); b=time.perf_counter_ns()
            buckets[name].append((b-a)*1e-6)
    for name,*_ in configs:
        xs=np.asarray(buckets[name]); print(f'  {name:13s}: median={np.median(xs):7.3f} ms p90={np.percentile(xs,90):7.3f}')
    best=min(buckets,key=lambda k:float(np.median(buckets[k])))
    base=float(np.median(buckets['dynamic old'])); bv=float(np.median(buckets[best]))
    print(f'  BEST {best} -> {bv:.3f} ms; vs dynamic-old={base/bv:.3f}x')
    print('freq after :',freqs())
    lib.ds_threadpool_destroy(pool)
if __name__=='__main__': main()
