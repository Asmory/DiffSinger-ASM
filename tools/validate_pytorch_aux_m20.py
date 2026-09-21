#!/usr/bin/env python3
from __future__ import annotations
import argparse, ctypes, time
import numpy as np
import torch
import torch.nn.functional as F

FP=ctypes.POINTER(ctypes.c_float)
def ptr(a):
    assert a.dtype==np.float32 and a.flags.c_contiguous
    return a.ctypes.data_as(FP)
def pack16(w):
    w=np.ascontiguousarray(w,np.float32);n,k=w.shape
    assert n%16==0
    return np.ascontiguousarray(w.reshape(n//16,16,k).transpose(0,2,1)).ravel()

class Block(ctypes.Structure):
    _fields_=[('dw_weight_tap_major',FP),('dw_bias',FP),('ln_gamma',FP),('ln_beta',FP),
              ('pw1_weight_m4n16',FP),('pw1_bias',FP),('pw2_weight_m4n16',FP),('pw2_bias',FP),('gamma',FP)]
class Net(ctypes.Structure):
    _fields_=[('input_dim',ctypes.c_uint32),('channels',ctypes.c_uint32),('output_dim',ctypes.c_uint32),
              ('num_layers',ctypes.c_uint32),('kernel_size',ctypes.c_uint32),('in_weight_m4n16',FP),
              ('in_bias',FP),('blocks',ctypes.POINTER(Block)),('out_weight_m4n16',FP),('out_bias',FP)]

def configure(lib):
    lib.ds_aux_convnext_workspace_floats.argtypes=[ctypes.POINTER(Net),ctypes.c_size_t];lib.ds_aux_convnext_workspace_floats.restype=ctypes.c_size_t
    lib.ds_aux_convnext_forward_norm_f32_avx2.argtypes=[ctypes.POINTER(Net),FP,FP,FP,ctypes.c_size_t,ctypes.c_void_p];lib.ds_aux_convnext_forward_norm_f32_avx2.restype=ctypes.c_int
    lib.ds_aux_convnext_infer_f32_avx2.argtypes=[ctypes.POINTER(Net),FP,FP,FP,ctypes.c_size_t,FP,FP,ctypes.c_size_t,ctypes.c_void_p];lib.ds_aux_convnext_infer_f32_avx2.restype=ctypes.c_int
    lib.ds_threadpool_create_auto.argtypes=[];lib.ds_threadpool_create_auto.restype=ctypes.c_void_p
    lib.ds_threadpool_destroy.argtypes=[ctypes.c_void_p]
    lib.ds_threadpool_threads.argtypes=[ctypes.c_void_p];lib.ds_threadpool_threads.restype=ctypes.c_size_t
    lib.ds_threadpool_cpu_at.argtypes=[ctypes.c_void_p,ctypes.c_size_t];lib.ds_threadpool_cpu_at.restype=ctypes.c_int

def make(seed,T,I,C,D,L):
    rng=np.random.default_rng(seed); f=lambda sh,s=.02:(rng.standard_normal(sh).astype(np.float32)*s)
    q={'x':f((T,I),.3),'inw':f((C,I,7),.025),'inb':f((C,),.01),'outw':f((D,C,7),.025),'outb':f((D,),.01),'blocks':[]}
    for _ in range(L):
        q['blocks'].append({'dw':f((C,7),.025),'dwb':f((C,),.01),'g':1+f((C,),.03),'b':f((C,),.015),
            'w1':f((4*C,C),.02),'b1':f((4*C,),.01),'w2':f((C,4*C),.02),'b2':f((C,),.01),
            'gamma':(1e-6*(1+f((C,),.05))).astype(np.float32)})
    return q

def ref(q):
    x=torch.from_numpy(q['x']).T[None]
    x=F.conv1d(x,torch.from_numpy(q['inw']),torch.from_numpy(q['inb']),padding=3)
    for b in q['blocks']:
        r=x
        z=F.conv1d(x,torch.from_numpy(b['dw'][:,None,:]),torch.from_numpy(b['dwb']),padding=3,groups=x.shape[1])
        z=z.transpose(1,2)
        z=F.layer_norm(z,(z.shape[-1],),torch.from_numpy(b['g']),torch.from_numpy(b['b']),1e-6)
        z=F.linear(z,torch.from_numpy(b['w1']),torch.from_numpy(b['b1']))
        z=F.gelu(z,approximate='none')
        z=F.linear(z,torch.from_numpy(b['w2']),torch.from_numpy(b['b2']))
        z=z*torch.from_numpy(b['gamma'])
        x=r+z.transpose(1,2)
    y=F.conv1d(x,torch.from_numpy(q['outw']),torch.from_numpy(q['outb']),padding=3)
    return y[0].T.contiguous().numpy()

def build(q):
    keep=[]
    def a(x):x=np.ascontiguousarray(x,np.float32);keep.append(x);return x
    C=q['inw'].shape[0];I=q['inw'].shape[1];D=q['outw'].shape[0];L=len(q['blocks'])
    # Flatten Conv1d weights to [N, tap*Cin] in im2col tap-major order.
    inraw=np.ascontiguousarray(q['inw'].transpose(0,2,1).reshape(C,7*I))
    outraw=np.ascontiguousarray(q['outw'].transpose(0,2,1).reshape(D,7*C))
    iw=a(pack16(inraw));ib=a(q['inb']);ow=a(pack16(outraw));ob=a(q['outb'])
    ba=(Block*L)();keep.append(ba)
    for i,b in enumerate(q['blocks']):
        dw=a(b['dw'].T);dwb=a(b['dwb']);g=a(b['g']);bb=a(b['b']);w1=a(pack16(b['w1']));b1=a(b['b1']);w2=a(pack16(b['w2']));b2=a(b['b2']);ga=a(b['gamma'])
        ba[i]=Block(ptr(dw),ptr(dwb),ptr(g),ptr(bb),ptr(w1),ptr(b1),ptr(w2),ptr(b2),ptr(ga))
    net=Net(I,C,D,L,7,ptr(iw),ptr(ib),ba,ptr(ow),ptr(ob));keep.append(net)
    return net,keep

def one(lib,seed,T,I,C,D,L,bench=False):
    q=make(seed,T,I,C,D,L);r=ref(q);net,keep=build(q)
    x=np.ascontiguousarray(q['x']);out=np.empty((T,D),np.float32)
    ws=np.empty(lib.ds_aux_convnext_workspace_floats(ctypes.byref(net),T),np.float32)
    pool=lib.ds_threadpool_create_auto();assert pool
    rc=lib.ds_aux_convnext_forward_norm_f32_avx2(ctypes.byref(net),ptr(x),ptr(out),ptr(ws),T,pool);assert rc==0
    ae=float(np.max(np.abs(out-r)));re=float(np.max(np.abs(out-r)/np.maximum(np.abs(r),1e-5)))
    n=int(lib.ds_threadpool_threads(pool));cpus=[int(lib.ds_threadpool_cpu_at(pool,i)) for i in range(n)]
    print(f'M20 aux T={T} I={I} C={C} D={D} L={L} workers={n}: max_abs={ae:.8g} max_rel={re:.8g} cpus={cpus} {"OK" if ae<2e-4 else "FAIL"}')
    assert ae<2e-4
    lo=np.array([-12.],np.float32);hi=np.array([0.],np.float32);raw=np.empty_like(out)
    assert lib.ds_aux_convnext_infer_f32_avx2(ctypes.byref(net),ptr(x),ptr(lo),ptr(hi),1,ptr(raw),ptr(ws),T,pool)==0
    raw_ref=r*6.0-6.0
    de=float(np.max(np.abs(raw-raw_ref)));print(f'  infer denorm max_abs={de:.3g}');assert de<2e-3
    if bench:
        for _ in range(2):lib.ds_aux_convnext_infer_f32_avx2(ctypes.byref(net),ptr(x),ptr(lo),ptr(hi),1,ptr(raw),ptr(ws),T,pool)
        ts=[]
        for _ in range(7):
            z=time.perf_counter();lib.ds_aux_convnext_infer_f32_avx2(ctypes.byref(net),ptr(x),ptr(lo),ptr(hi),1,ptr(raw),ptr(ws),T,pool);ts.append(time.perf_counter()-z)
        print(f'  official aux median={np.median(ts)*1e3:.3f} ms')
    lib.ds_threadpool_destroy(pool)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--lib',default='build/libdsasm_m20.so');ap.add_argument('--official-shape',action='store_true');a=ap.parse_args()
    lib=ctypes.CDLL(str(a.lib));configure(lib)
    one(lib,20260919,5,64,64,32,1)
    one(lib,20260920,17,64,64,32,2)
    one(lib,20260921,37,64,128,64,3)
    if a.official_shape:one(lib,20260922,64,384,512,128,6,True)
if __name__=='__main__':main()
