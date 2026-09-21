#!/usr/bin/env python3
"""End-to-end M16 LYNXNet2 forward validation against PyTorch semantics.

Covers the current OpenVPI default ATanGLU path and the existing SoftSignGLU
path.  All GEMMs, depthwise conv, layer norms, residual projection, ATanGLU and
broadcast adds in the runtime are handwritten x86-64 AVX2/FMA kernels; the tiny
per-timestep sinusoidal/GELU embedding is orchestrated in C.
"""
from __future__ import annotations
import argparse, ctypes, math, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

FP=ctypes.POINTER(ctypes.c_float)

def ptr(a):
    assert a.dtype==np.float32 and a.flags.c_contiguous
    return a.ctypes.data_as(FP)

def pack_linear16(w):
    w=np.ascontiguousarray(w,dtype=np.float32); n,k=w.shape
    assert n%16==0
    return np.ascontiguousarray(w.reshape(n//16,16,k).transpose(0,2,1)).ravel()

def pack_glu8(w):
    w=np.ascontiguousarray(w,dtype=np.float32); two_n,k=w.shape; n=two_n//2
    assert two_n%2==0 and n%8==0
    a=w[:n].reshape(n//8,8,k).transpose(0,2,1)
    b=w[n:].reshape(n//8,8,k).transpose(0,2,1)
    return np.ascontiguousarray(np.concatenate([a,b],axis=2)).ravel()

class Block(ctypes.Structure):
    _fields_=[('ln_gamma',FP),('ln_beta',FP),('dw_weight_tap_major',FP),('dw_bias',FP),
              ('glu1_weight',FP),('glu1_bias',FP),('glu2_weight',FP),('glu2_bias',FP),
              ('out_weight_m4n16',FP),('out_bias',FP)]
class Net(ctypes.Structure):
    _fields_=[('input_dim',ctypes.c_uint32),('condition_dim',ctypes.c_uint32),
              ('channels',ctypes.c_uint32),('hidden_dim',ctypes.c_uint32),
              ('num_layers',ctypes.c_uint32),('kernel_size',ctypes.c_uint32),('glu_type',ctypes.c_uint32),
              ('input_weight_m4n16',FP),('input_bias',FP),('condition_weight_m4n16',FP),('condition_bias',FP),
              ('time1_weight_m4n16',FP),('time1_bias',FP),('time2_weight_m4n16',FP),('time2_bias',FP),
              ('blocks',ctypes.POINTER(Block)),('post_norm_gamma',FP),('post_norm_beta',FP),
              ('output_weight_m4n16',FP),('output_bias',FP)]

def configure(lib):
    lib.ds_lynxnet2_workspace_floats.argtypes=[ctypes.POINTER(Net),ctypes.c_size_t];lib.ds_lynxnet2_workspace_floats.restype=ctypes.c_size_t
    lib.ds_lynxnet2_prepare_condition_f32_avx2.argtypes=[ctypes.POINTER(Net),FP,FP,ctypes.c_size_t];lib.ds_lynxnet2_prepare_condition_f32_avx2.restype=ctypes.c_int
    lib.ds_lynxnet2_forward_f32_avx2.argtypes=[ctypes.POINTER(Net),FP,FP,ctypes.c_float,FP,FP,ctypes.c_size_t];lib.ds_lynxnet2_forward_f32_avx2.restype=ctypes.c_int
    lib.ds_lynxnet2_forward_cached_condition_f32_avx2.argtypes=[ctypes.POINTER(Net),FP,FP,ctypes.c_float,FP,FP,ctypes.c_size_t];lib.ds_lynxnet2_forward_cached_condition_f32_avx2.restype=ctypes.c_int
    lib.ds_threadpool_create.argtypes=[ctypes.c_size_t];lib.ds_threadpool_create.restype=ctypes.c_void_p
    lib.ds_threadpool_create_auto.argtypes=[];lib.ds_threadpool_create_auto.restype=ctypes.c_void_p
    lib.ds_threadpool_destroy.argtypes=[ctypes.c_void_p]
    lib.ds_threadpool_threads.argtypes=[ctypes.c_void_p];lib.ds_threadpool_threads.restype=ctypes.c_size_t
    lib.ds_threadpool_cpu_at.argtypes=[ctypes.c_void_p,ctypes.c_size_t];lib.ds_threadpool_cpu_at.restype=ctypes.c_int
    lib.ds_threadpool_affinity_enabled.argtypes=[ctypes.c_void_p];lib.ds_threadpool_affinity_enabled.restype=ctypes.c_int
    lib.ds_threadpool_set_2d.argtypes=[ctypes.c_void_p,ctypes.c_int]
    lib.ds_threadpool_get_2d.argtypes=[ctypes.c_void_p];lib.ds_threadpool_get_2d.restype=ctypes.c_int
    lib.ds_threadpool_set_atan_pipeline.argtypes=[ctypes.c_void_p,ctypes.c_int]
    lib.ds_threadpool_get_atan_pipeline.argtypes=[ctypes.c_void_p];lib.ds_threadpool_get_atan_pipeline.restype=ctypes.c_int
    lib.ds_threadpool_set_tiles.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_size_t];lib.ds_threadpool_set_tiles.restype=ctypes.c_int
    lib.ds_threadpool_m_tile.argtypes=[ctypes.c_void_p];lib.ds_threadpool_m_tile.restype=ctypes.c_size_t
    lib.ds_threadpool_n_tile.argtypes=[ctypes.c_void_p];lib.ds_threadpool_n_tile.restype=ctypes.c_size_t
    lib.ds_threadpool_set_auto_tiles.argtypes=[ctypes.c_void_p,ctypes.c_int]
    lib.ds_threadpool_get_auto_tiles.argtypes=[ctypes.c_void_p];lib.ds_threadpool_get_auto_tiles.restype=ctypes.c_int
    lib.ds_threadpool_set_parallel_depthwise.argtypes=[ctypes.c_void_p,ctypes.c_int]
    lib.ds_threadpool_get_parallel_depthwise.argtypes=[ctypes.c_void_p];lib.ds_threadpool_get_parallel_depthwise.restype=ctypes.c_int
    lib.ds_threadpool_set_indexed_linear.argtypes=[ctypes.c_void_p,ctypes.c_int]
    if hasattr(lib,"ds_threadpool_set_n_owner"):
        lib.ds_threadpool_set_n_owner.argtypes=[ctypes.c_void_p,ctypes.c_int]
    lib.ds_threadpool_get_indexed_linear.argtypes=[ctypes.c_void_p];lib.ds_threadpool_get_indexed_linear.restype=ctypes.c_int
    lib.ds_lynxnet2_prepare_condition_parallel_f32_avx2.argtypes=[ctypes.POINTER(Net),FP,FP,ctypes.c_size_t,ctypes.c_void_p];lib.ds_lynxnet2_prepare_condition_parallel_f32_avx2.restype=ctypes.c_int
    lib.ds_lynxnet2_forward_cached_condition_parallel_f32_avx2.argtypes=[ctypes.POINTER(Net),FP,FP,ctypes.c_float,FP,FP,ctypes.c_size_t,ctypes.c_void_p];lib.ds_lynxnet2_forward_cached_condition_parallel_f32_avx2.restype=ctypes.c_int

def make_case(seed,T,I,Q,C,H,L,glu):
    rng=np.random.default_rng(seed)
    f=lambda sh,s=.02: (rng.standard_normal(sh).astype(np.float32)*s)
    q={'spec':f((T,I),.4),'cond':f((T,Q),.25),'timestep':np.float32(327.5),
       'iw':f((C,I),.025),'ib':f((C,),.015),'cw':f((C,Q),.025),'cb':f((C,),.015),
       'tw1':f((4*C,C),.018),'tb1':f((4*C,),.01),'tw2':f((C,4*C),.018),'tb2':f((C,),.01),
       'post_g':1+f((C,),.04),'post_b':f((C,),.02),'ow':f((I,C),.025),'ob':f((I,),.015),'blocks':[],'glu':glu}
    for _ in range(L):
        q['blocks'].append({'g':1+f((C,),.04),'b':f((C,),.02),'dw':f((C,31),.025),'dwb':f((C,),.01),
            'w1':f((2*H,C),.022),'b1':f((2*H,),.01),'w2':f((2*H,H),.022),'b2':f((2*H,),.01),
            'w3':f((C,H),.022),'b3':f((C,),.01)})
    return q

def sinusoidal(t,c):
    half=c//2
    scale=-math.log(10000)/(half-1)
    e=torch.exp(torch.arange(half,dtype=torch.float32)*scale)*float(t)
    return torch.cat((torch.sin(e),torch.cos(e)))

def torch_ref(q):
    x=F.linear(torch.from_numpy(q['spec']),torch.from_numpy(q['iw']),torch.from_numpy(q['ib']))
    x=x+F.linear(torch.from_numpy(q['cond']),torch.from_numpy(q['cw']),torch.from_numpy(q['cb']))
    st=sinusoidal(q['timestep'],x.shape[-1])
    st=F.linear(st,torch.from_numpy(q['tw1']),torch.from_numpy(q['tb1']))
    st=F.gelu(st)
    st=F.linear(st,torch.from_numpy(q['tw2']),torch.from_numpy(q['tb2']))
    x=x+st
    for b in q['blocks']:
        z=F.layer_norm(x,(x.shape[-1],),torch.from_numpy(b['g']),torch.from_numpy(b['b']),1e-5)
        z=F.conv1d(z.T[None],torch.from_numpy(b['dw'][:,None,:]),torch.from_numpy(b['dwb']),padding=15,groups=x.shape[-1])[0].T
        z=F.linear(z,torch.from_numpy(b['w1']),torch.from_numpy(b['b1']));l,g=torch.chunk(z,2,-1)
        z=l*(torch.atan(g) if q['glu']=='atan' else F.softsign(g))
        z=F.linear(z,torch.from_numpy(b['w2']),torch.from_numpy(b['b2']));l,g=torch.chunk(z,2,-1)
        z=l*(torch.atan(g) if q['glu']=='atan' else F.softsign(g))
        z=F.linear(z,torch.from_numpy(b['w3']),torch.from_numpy(b['b3']))
        x=x+z
    x=F.layer_norm(x,(x.shape[-1],),torch.from_numpy(q['post_g']),torch.from_numpy(q['post_b']),1e-5)
    return F.linear(x,torch.from_numpy(q['ow']),torch.from_numpy(q['ob'])).numpy()

def build_ffi(q):
    keep=[]
    def a(x): x=np.ascontiguousarray(x,dtype=np.float32);keep.append(x);return x
    C=q['iw'].shape[0];I=q['iw'].shape[1];Q=q['cw'].shape[1];H=q['blocks'][0]['w1'].shape[0]//2;L=len(q['blocks'])
    ba=(Block*L)(); keep.append(ba)
    for i,b in enumerate(q['blocks']):
        g=a(b['g']);bb=a(b['b']);dw=a(b['dw'].T);dwb=a(b['dwb']);b1=a(b['b1']);b2=a(b['b2']);b3=a(b['b3'])
        w1=a(pack_linear16(b['w1']) if q['glu']=='atan' else pack_glu8(b['w1']))
        w2=a(pack_linear16(b['w2']) if q['glu']=='atan' else pack_glu8(b['w2']))
        w3=a(pack_linear16(b['w3']))
        ba[i]=Block(ptr(g),ptr(bb),ptr(dw),ptr(dwb),ptr(w1),ptr(b1),ptr(w2),ptr(b2),ptr(w3),ptr(b3))
    iw=a(pack_linear16(q['iw'])); ib=a(q['ib']); cw=a(pack_linear16(q['cw'])); cb=a(q['cb'])
    tw1=a(pack_linear16(q['tw1']));tb1=a(q['tb1']);tw2=a(pack_linear16(q['tw2']));tb2=a(q['tb2'])
    pg=a(q['post_g']);pb=a(q['post_b']);ow=a(pack_linear16(q['ow']));ob=a(q['ob'])
    n=Net(I,Q,C,H,L,31,1 if q['glu']=='atan' else 2,ptr(iw),ptr(ib),ptr(cw),ptr(cb),ptr(tw1),ptr(tb1),ptr(tw2),ptr(tb2),ba,ptr(pg),ptr(pb),ptr(ow),ptr(ob))
    keep.append(n)
    return n,keep

def run(lib,q,bench=False,threads=0,sweep_tiles=False):
    ref=torch_ref(q)
    net,keep=build_ffi(q);T=q['spec'].shape[0];C=net.channels
    spec=np.ascontiguousarray(q['spec']);cond=np.ascontiguousarray(q['cond']);out=np.empty_like(ref);cache=np.empty((T,C),np.float32)
    ws=np.empty(lib.ds_lynxnet2_workspace_floats(ctypes.byref(net),T),np.float32)
    rc=lib.ds_lynxnet2_forward_f32_avx2(ctypes.byref(net),ptr(spec),ptr(cond),ctypes.c_float(q['timestep']),ptr(out),ptr(ws),T);assert rc==0
    ae=float(np.max(np.abs(out-ref)));re=float(np.max(np.abs(out-ref)/np.maximum(np.abs(ref),1e-5)))
    print(f"PyTorch vs M16 full {q['glu']} T={T} I={net.input_dim} Q={net.condition_dim} C={net.channels} L={net.num_layers}: max_abs={ae:.8g} max_rel={re:.8g} {'OK' if ae<2e-4 else 'FAIL'}")
    assert ae<2e-4
    assert lib.ds_lynxnet2_prepare_condition_f32_avx2(ctypes.byref(net),ptr(cond),ptr(cache),T)==0
    out2=np.empty_like(out)
    assert lib.ds_lynxnet2_forward_cached_condition_f32_avx2(ctypes.byref(net),ptr(spec),ptr(cache),ctypes.c_float(q['timestep']),ptr(out2),ptr(ws),T)==0
    ce=float(np.max(np.abs(out-out2)));print(f"  cached conditioner parity: {ce:.3g}");assert ce<1e-6
    if bench:
        # Benchmark the repeated-sampling path. Conditioner projection is timed
        # separately because official deployments cache it across solver steps.
        for _ in range(3):
            lib.ds_lynxnet2_forward_cached_condition_f32_avx2(ctypes.byref(net),ptr(spec),ptr(cache),ctypes.c_float(q['timestep']),ptr(out2),ptr(ws),T)
        proj_times=[]
        for _ in range(9):
            z0=time.perf_counter();lib.ds_lynxnet2_prepare_condition_f32_avx2(ctypes.byref(net),ptr(cond),ptr(cache),T);z1=time.perf_counter();proj_times.append(z1-z0)
        print(f"  conditioner projection standalone median={np.median(proj_times)*1e3:.3f} ms (cache once per phrase/segment)")
        serial=[]
        if threads!=1:
            pool=lib.ds_threadpool_create(threads);assert pool
            actual=int(lib.ds_threadpool_threads(pool))
            cpus=[int(lib.ds_threadpool_cpu_at(pool,i)) for i in range(actual)]
            aff=bool(lib.ds_threadpool_affinity_enabled(pool))
            print(f"  M16 pool: requested={'auto' if threads==0 else threads}, workers={actual}, affinity={aff}, cpus={cpus}")
            if actual>=8:
                policy=(32,64) if C>=768 else ((8,256) if C<=256 else (16,128))
            else:
                policy=(16,64)
            print(f"  M10 adaptive 2-D policy for this shape: {policy[0]}x{policy[1]}")
            pcache=np.empty_like(cache);pout=np.empty_like(out)
            assert lib.ds_lynxnet2_prepare_condition_parallel_f32_avx2(ctypes.byref(net),ptr(cond),ptr(pcache),T,pool)==0
            assert np.max(np.abs(pcache-cache))<1e-6
            assert lib.ds_lynxnet2_forward_cached_condition_parallel_f32_avx2(ctypes.byref(net),ptr(spec),ptr(pcache),ctypes.c_float(q['timestep']),ptr(pout),ptr(ws),T,pool)==0
            pe=float(np.max(np.abs(pout-ref)));print(f"  parallel {actual}T parity max_abs={pe:.3g}");assert pe<2e-4

            def bench_pool(use2d,atan_pipeline,m_tile=None,n_tile=None,reps=9,auto_tiles=False,parallel_dw=False):
                lib.ds_threadpool_set_2d(pool,1 if use2d else 0)
                lib.ds_threadpool_set_atan_pipeline(pool,1 if atan_pipeline else 0)
                lib.ds_threadpool_set_parallel_depthwise(pool,1 if parallel_dw else 0)
                lib.ds_threadpool_set_auto_tiles(pool,1 if auto_tiles else 0)
                if m_tile is not None:
                    assert lib.ds_threadpool_set_tiles(pool,m_tile,n_tile)==0
                if auto_tiles:
                    lib.ds_threadpool_set_auto_tiles(pool,1)
                for _ in range(3):
                    lib.ds_lynxnet2_forward_cached_condition_parallel_f32_avx2(ctypes.byref(net),ptr(spec),ptr(pcache),ctypes.c_float(q['timestep']),ptr(pout),ptr(ws),T,pool)
                ts=[]
                for _ in range(reps):
                    a=time.perf_counter();lib.ds_lynxnet2_forward_cached_condition_parallel_f32_avx2(ctypes.byref(net),ptr(spec),ptr(pcache),ctypes.c_float(q['timestep']),ptr(pout),ptr(ws),T,pool);b=time.perf_counter();ts.append(b-a)
                return float(np.median(ts))
            # Same-process A/B: M7 M-only, M8 2-D, M9 fixed ATan tile, M10 adaptive.
            m_only=bench_pool(False,False)
            two_d=bench_pool(True,False,16,128)
            m9=bench_pool(True,True,16,128)
            m10=bench_pool(True,True,reps=9,auto_tiles=True,parallel_dw=False)

            # M12 fair A/B: alternate serial-DW and parallel-DW forward calls
            # in the same process/pool so thermal/frequency drift affects both.
            def bench_dw_pair(reps=15):
                lib.ds_threadpool_set_2d(pool,1); lib.ds_threadpool_set_atan_pipeline(pool,1)
                lib.ds_threadpool_set_auto_tiles(pool,1)
                a10=[]; a11=[]
                for pd in (0,1,0,1):
                    lib.ds_threadpool_set_parallel_depthwise(pool,pd)
                    lib.ds_lynxnet2_forward_cached_condition_parallel_f32_avx2(ctypes.byref(net),ptr(spec),ptr(pcache),ctypes.c_float(q['timestep']),ptr(pout),ptr(ws),T,pool)
                for i in range(reps):
                    order=(0,1) if (i&1)==0 else (1,0)
                    for pd in order:
                        lib.ds_threadpool_set_parallel_depthwise(pool,pd)
                        a=time.perf_counter(); lib.ds_lynxnet2_forward_cached_condition_parallel_f32_avx2(ctypes.byref(net),ptr(spec),ptr(pcache),ctypes.c_float(q['timestep']),ptr(pout),ptr(ws),T,pool); b=time.perf_counter()
                        (a11 if pd else a10).append(b-a)
                return float(np.median(a10)),float(np.median(a11))
            m10_ab,m11=bench_dw_pair()
            best=(m9,16,128)
            if sweep_tiles and q['glu']=='atan':
                print("  fixed tile sweep:")
                for mt in (8,16,32,64):
                    for nt in (64,128,256):
                        v=bench_pool(True,True,mt,nt,reps=7,parallel_dw=False)
                        print(f"    tile {mt:2d}x{nt:3d}: {v*1e3:.3f} ms")
                        if v<best[0]: best=(v,mt,nt)
                best_recheck=bench_pool(True,True,best[1],best[2],parallel_dw=False)
                print(f"    best fixed={best[1]}x{best[2]} -> {best_recheck*1e3:.3f} ms")
                m10=bench_pool(True,True,reps=9,auto_tiles=True,parallel_dw=False)
                print(f"    M10 adaptive recheck -> {m10*1e3:.3f} ms")
                m10_ab,m11=bench_dw_pair()
                print(f"    interleaved M10/M12 -> {m10_ab*1e3:.3f} / {m11*1e3:.3f} ms")
            for _ in range(3):lib.ds_lynxnet2_forward_cached_condition_f32_avx2(ctypes.byref(net),ptr(spec),ptr(cache),ctypes.c_float(q['timestep']),ptr(out2),ptr(ws),T)
            for _ in range(9):
                a=time.perf_counter();lib.ds_lynxnet2_forward_cached_condition_f32_avx2(ctypes.byref(net),ptr(spec),ptr(cache),ctypes.c_float(q['timestep']),ptr(out2),ptr(ws),T);b=time.perf_counter();serial.append(b-a)
            sm=float(np.median(serial))
            print(f"  repeated-step serial median={sm*1e3:.3f} ms")
            print(f"  M7-style M-only {actual}T median={m_only*1e3:.3f} ms; speedup={sm/m_only:.2f}x")
            print(f"  M8 2-D      {actual}T median={two_d*1e3:.3f} ms; speedup={sm/two_d:.2f}x; vs M-only={m_only/two_d:.2f}x")
            print(f"  M9 fixed16x128 {actual}T median={m9*1e3:.3f} ms; speedup={sm/m9:.2f}x; vs M8={two_d/m9:.2f}x")
            print(f"  M10 adaptive    {actual}T median={m10*1e3:.3f} ms; speedup={sm/m10:.2f}x; vs M9={m9/m10:.2f}x")
            print(f"  M10 interleaved serial-DW median={m10_ab*1e3:.3f} ms")
            print(f"  M12 parallel-DW          median={m11*1e3:.3f} ms; speedup={sm/m11:.2f}x; vs M10-interleaved={m10_ab/m11:.2f}x")
            lib.ds_threadpool_destroy(pool)
        else:
            for _ in range(9):
                a=time.perf_counter();lib.ds_lynxnet2_forward_cached_condition_f32_avx2(ctypes.byref(net),ptr(spec),ptr(cache),ctypes.c_float(q['timestep']),ptr(out2),ptr(ws),T);b=time.perf_counter();serial.append(b-a)
            print(f"  repeated-step serial median={np.median(serial)*1e3:.3f} ms")

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--lib',type=Path,default=Path('build/libdsasm_m13.so'));ap.add_argument('--official-shape',action='store_true');ap.add_argument('--threads',type=int,default=0,help='0=auto performance-core pool');ap.add_argument('--sweep-tiles',action='store_true');args=ap.parse_args()
    torch.set_num_threads(1);lib=ctypes.CDLL(str(args.lib.resolve()));configure(lib)
    run(lib,make_case(17,19,64,48,64,64,2,'atan'))
    run(lib,make_case(18,37,64,48,64,64,2,'softsign'))
    run(lib,make_case(19,64,128,96,256,256,6,'atan'),bench=True,threads=args.threads,sweep_tiles=args.sweep_tiles)
    if args.official_shape:
        # Current official acoustic.yaml dimensions: mel=128, hidden_size=384,
        # LYNXNet2 channels=1024, layers=6, expansion=1, ATanGLU.
        run(lib,make_case(20,64,128,384,1024,1024,6,'atan'),bench=True,threads=args.threads,sweep_tiles=args.sweep_tiles)
if __name__=='__main__':main()
