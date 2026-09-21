#!/usr/bin/env python3
"""M13 target-machine matrix: P-only vs hybrid workers, tile ownership, indexed K addressing.

The benchmark keeps one pool alive per worker topology and interleaves candidate
forwards so laptop frequency/temperature drift is shared as much as possible.
"""
from __future__ import annotations
import argparse, ctypes, os, time
from pathlib import Path
import numpy as np
import validate_pytorch_lynxnet2 as v


def read_freqs(cpus=(0,2,4,6,8,9,10,11)):
    out=[]
    for cpu in cpus:
        p=Path(f'/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq')
        try: out.append((cpu,int(p.read_text().strip())/1000.0))
        except Exception: pass
    return out

def fmt_freqs(xs):
    return ', '.join(f'cpu{c}={mhz:.0f}MHz' for c,mhz in xs) if xs else 'unavailable'

def prepare(lib):
    q=v.make_case(413,64,128,384,1024,1024,6,'atan')
    net,keep=v.build_ffi(q); T=64; C=1024
    spec=np.ascontiguousarray(q['spec']); cond=np.ascontiguousarray(q['cond'])
    cache=np.empty((T,C),np.float32); out=np.empty((T,128),np.float32)
    ws=np.empty(lib.ds_lynxnet2_workspace_floats(ctypes.byref(net),T),np.float32)
    assert lib.ds_lynxnet2_prepare_condition_f32_avx2(ctypes.byref(net),v.ptr(cond),v.ptr(cache),T)==0
    return q,net,keep,spec,cache,out,ws

def call(lib,data,pool,mt,nt,indexed):
    q,net,keep,spec,cache,out,ws=data; T=64
    lib.ds_threadpool_set_2d(pool,1); lib.ds_threadpool_set_atan_pipeline(pool,1)
    lib.ds_threadpool_set_parallel_depthwise(pool,0); lib.ds_threadpool_set_auto_tiles(pool,0)
    assert lib.ds_threadpool_set_tiles(pool,mt,nt)==0
    lib.ds_threadpool_set_indexed_linear(pool,1 if indexed else 0)
    rc=lib.ds_lynxnet2_forward_cached_condition_parallel_f32_avx2(
        ctypes.byref(net),v.ptr(spec),v.ptr(cache),ctypes.c_float(q['timestep']),v.ptr(out),v.ptr(ws),T,pool)
    assert rc==0
    return out.copy()

def bench_pool(lib,data,requested,reps=13):
    pool=lib.ds_threadpool_create(requested); assert pool
    n=int(lib.ds_threadpool_threads(pool)); cpus=[int(lib.ds_threadpool_cpu_at(pool,i)) for i in range(n)]
    print(f'  pool requested={requested}: workers={n} cpus={cpus}')
    configs=[(16,64,0),(32,64,0),(64,64,0),(32,64,1),(64,64,1)]
    names={
        (16,64,0):'16x64 old',
        (32,64,0):'32x64 old',
        (64,64,0):'64x64 old',
        (32,64,1):'32x64 indexed',
        (64,64,1):'64x64 indexed',
    }
    # Warm every path and establish exact parity.
    ref=call(lib,data,pool,*configs[1])
    for c in configs:
        out=call(lib,data,pool,*c)
        ae=float(np.max(np.abs(out-ref))); assert ae==0.0,(c,ae)
    buckets={c:[] for c in configs}
    # Rotate order every repetition to avoid always giving one mode the coolest slot.
    for r in range(reps):
        order=configs[r%len(configs):]+configs[:r%len(configs)]
        if r&1: order=list(reversed(order))
        for c in order:
            a=time.perf_counter_ns(); call(lib,data,pool,*c); b=time.perf_counter_ns()
            buckets[c].append((b-a)*1e-6)
    for c in configs:
        xs=np.asarray(buckets[c]); print(f"    {names[c]:15s}: median={np.median(xs):7.3f} ms  p90={np.percentile(xs,90):7.3f}")
    best=min(configs,key=lambda c:float(np.median(buckets[c])))
    print(f"    BEST {names[best]} -> {np.median(buckets[best]):.3f} ms")
    lib.ds_threadpool_destroy(pool)
    return n,cpus,{names[c]:float(np.median(buckets[c])) for c in configs}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--lib',type=Path,default=Path('build/libdsasm_m13.so')); args=ap.parse_args()
    os.environ['DSASM_SPIN_POOL']='0'; os.environ['DSASM_PARALLEL_DW']='0'
    lib=ctypes.CDLL(str(args.lib.resolve())); v.configure(lib); data=prepare(lib)
    print('M13 target matrix: official T=64 C=1024 L=6')
    print('freq before:',fmt_freqs(read_freqs()))
    n8,c8,r8=bench_pool(lib,data,8)
    n12,c12,r12=bench_pool(lib,data,12)
    print('freq after :',fmt_freqs(read_freqs()))
    if n12>n8:
        b8=min(r8.values()); b12=min(r12.values())
        print(f'  HYBRID verdict: best-P={b8:.3f} ms best-all={b12:.3f} ms all/P={b8/b12:.3f}x')
    else:
        print('  HYBRID verdict: host exposes no extra workers beyond the P/preferred pool; target i5-13420H will decide.')

if __name__=='__main__': main()
