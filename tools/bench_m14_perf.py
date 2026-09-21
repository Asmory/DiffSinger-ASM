#!/usr/bin/env python3
"""Perf-friendly M14 official-shape cached-condition workload."""
from __future__ import annotations
import argparse, ctypes, os, time
from pathlib import Path
import numpy as np
import validate_pytorch_lynxnet2 as v

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--lib',type=Path,default=Path('build/libdsasm_m14.so')); ap.add_argument('--iters',type=int,default=80); ap.add_argument('--owner',type=int,default=0); ap.add_argument('--indexed',type=int,default=0); args=ap.parse_args()
    os.environ['DSASM_SPIN_POOL']='0'; os.environ['DSASM_PARALLEL_DW']='0'
    lib=ctypes.CDLL(str(args.lib.resolve())); v.configure(lib); lib.ds_threadpool_set_n_owner.argtypes=[ctypes.c_void_p,ctypes.c_int]
    q=v.make_case(615,64,128,384,1024,1024,6,'atan'); net,keep=v.build_ffi(q); T=64; C=1024
    spec=np.ascontiguousarray(q['spec']); cond=np.ascontiguousarray(q['cond']); cache=np.empty((T,C),np.float32); out=np.empty((T,128),np.float32)
    ws=np.empty(lib.ds_lynxnet2_workspace_floats(ctypes.byref(net),T),np.float32)
    assert lib.ds_lynxnet2_prepare_condition_f32_avx2(ctypes.byref(net),v.ptr(cond),v.ptr(cache),T)==0
    pool=lib.ds_threadpool_create(8); assert pool
    lib.ds_threadpool_set_2d(pool,1); lib.ds_threadpool_set_atan_pipeline(pool,1); lib.ds_threadpool_set_parallel_depthwise(pool,0); lib.ds_threadpool_set_auto_tiles(pool,0); assert lib.ds_threadpool_set_tiles(pool,32,64)==0
    lib.ds_threadpool_set_n_owner(pool,args.owner); lib.ds_threadpool_set_indexed_linear(pool,args.indexed)
    n=int(lib.ds_threadpool_threads(pool)); cpus=[int(lib.ds_threadpool_cpu_at(pool,i)) for i in range(n)]
    def f():
        assert lib.ds_lynxnet2_forward_cached_condition_parallel_f32_avx2(ctypes.byref(net),v.ptr(spec),v.ptr(cache),ctypes.c_float(q['timestep']),v.ptr(out),v.ptr(ws),T,pool)==0
    for _ in range(5): f()
    ts=[]
    for _ in range(args.iters):
        a=time.perf_counter_ns(); f(); b=time.perf_counter_ns(); ts.append((b-a)*1e-6)
    print(f'M14 PERF owner={args.owner} indexed={args.indexed} workers={n} cpus={cpus} median={np.median(ts):.3f} ms mean={np.mean(ts):.3f} p90={np.percentile(ts,90):.3f} ms')
    lib.ds_threadpool_destroy(pool)
if __name__=='__main__': main()
