#!/usr/bin/env python3
from __future__ import annotations
import argparse,ctypes as C,math,statistics,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(1)
PF=C.POINTER(C.c_float); PI=C.POINTER(C.c_int32)
def pfloat(a): return a.ctypes.data_as(PF)
def pint(a): return a.ctypes.data_as(PI)

def pack16(w):
    w=np.ascontiguousarray(w,np.float32); n,k=w.shape
    assert n%16==0
    return np.ascontiguousarray(w.reshape(n//16,16,k).transpose(0,2,1)).reshape(-1)
def conv3_flat(w):
    # torch [N,C,3] -> [N,3*C], tap-major im2col order
    return np.ascontiguousarray(w.transpose(0,2,1).reshape(w.shape[0],-1),np.float32)

class Layer(C.Structure):
    _fields_=[('ln1_gamma',PF),('ln1_beta',PF),('qkv_weight_m4n16',PF),('qkv_bias',PF),
              ('out_weight_m4n16',PF),('out_bias',PF),('ln2_gamma',PF),('ln2_beta',PF),
              ('ffn1_weight_m4n16',PF),('ffn1_bias',PF),('ffn2_weight_m4n16',PF),('ffn2_bias',PF),
              ('ln1_affine_weight_m4n16',PF),('ln1_affine_bias',PF),
              ('ln2_affine_weight_m4n16',PF),('ln2_affine_bias',PF)]
class Weights(C.Structure):
    _fields_=[('vocab_size',C.c_uint32),('hidden_size',C.c_uint32),('num_layers',C.c_uint32),
              ('num_heads',C.c_uint32),('ffn_kernel_size',C.c_uint32),('rope_interleaved',C.c_uint32),
              ('rope_theta',C.c_float),('token_embedding',PF),('dur_weight',PF),('dur_bias',PF),
              ('layers',C.POINTER(Layer)),('final_ln_gamma',PF),('final_ln_beta',PF)]

def randn(rng,shape,scale): return np.ascontiguousarray(rng.standard_normal(shape,dtype=np.float32)*np.float32(scale))
def make_case(seed,T,Cc,L,H,vocab=80,pad=3):
    rng=np.random.default_rng(seed); hd=Cc//H
    keep=[]
    emb=randn(rng,(vocab,Cc),1/math.sqrt(Cc));emb[0]=0;keep.append(emb)
    dw=randn(rng,(Cc,),0.035); db=randn(rng,(Cc,),0.01); keep += [dw,db]
    refs=[]; ls=(Layer*L)()
    for i in range(L):
        g1=np.ascontiguousarray(1+randn(rng,(Cc,),0.03)); b1=randn(rng,(Cc,),0.01)
        qkv=randn(rng,(3*Cc,Cc),0.035); qkvp=pack16(qkv); qkvb=np.zeros(3*Cc,np.float32)
        ow=randn(rng,(Cc,Cc),0.035); owp=pack16(ow); ob=np.zeros(Cc,np.float32)
        g2=np.ascontiguousarray(1+randn(rng,(Cc,),0.03)); b2=randn(rng,(Cc,),0.01)
        fw1=randn(rng,(4*Cc,Cc,3),0.025); fw1p=pack16(conv3_flat(fw1)); fb1=randn(rng,(4*Cc,),0.01)
        fw2=randn(rng,(Cc,4*Cc),0.025); fw2p=pack16(fw2); fb2=randn(rng,(Cc,),0.01)
        keep += [g1,b1,qkv,qkvp,qkvb,ow,owp,ob,g2,b2,fw1,fw1p,fb1,fw2,fw2p,fb2]
        refs.append(dict(g1=g1,b1=b1,qkv=qkv,ow=ow,g2=g2,b2=b2,fw1=fw1,fb1=fb1,fw2=fw2,fb2=fb2))
        ls[i]=Layer(pfloat(g1),pfloat(b1),pfloat(qkvp),pfloat(qkvb),pfloat(owp),pfloat(ob),pfloat(g2),pfloat(b2),pfloat(fw1p),pfloat(fb1),pfloat(fw2p),pfloat(fb2))
    fg=np.ascontiguousarray(1+randn(rng,(Cc,),0.03)); fb=randn(rng,(Cc,),0.01); keep += [fg,fb,ls]
    w=Weights(vocab,Cc,L,H,3,0,np.float32(10000.0),pfloat(emb),pfloat(dw),pfloat(db),ls,pfloat(fg),pfloat(fb))
    tok=rng.integers(1,vocab,size=T,dtype=np.int32)
    if pad:
        tok[-pad:]=0
    dur=rng.integers(1,7,size=T,dtype=np.int32);dur[tok==0]=0
    return w,keep,refs,emb,dw,db,fg,fb,tok,dur

def rope(x,theta=10000.0,interleaved=False):
    # x [T,H,D]
    T,H,D=x.shape
    inv=1.0/(theta**(torch.arange(0,D,2,dtype=torch.float32)/D))
    freq=torch.einsum('i,j->ij',torch.arange(T,dtype=torch.float32),inv)
    if interleaved: freq=torch.repeat_interleave(freq,2,-1)
    else: freq=torch.cat((freq,freq),-1)
    co,si=freq.cos(),freq.sin()
    if interleaved:
        z=x.reshape(T,H,D//2,2);x1,x2=z.unbind(-1);rh=torch.stack((-x2,x1),-1).reshape_as(x)
    else:
        x1,x2=x.split(D//2,-1);rh=torch.cat((-x2,x1),-1)
    return x*co[:,None,:]+rh*si[:,None,:]

def ref_forward(refs,emb,dw,db,fg,fb,tok,dur,H):
    et=torch.from_numpy(emb); tok_t=torch.from_numpy(tok.astype(np.int64));dur_t=torch.from_numpy(dur.astype(np.float32))
    Cc=emb.shape[1]; mask=tok_t.eq(0); non=(~mask).float()[:,None]
    x=math.sqrt(Cc)*et[tok_t]
    x=x + torch.from_numpy(dw)[None,:]*torch.log1p(dur_t)[:,None] + torch.from_numpy(db)[None,:]
    x=x*non
    hd=Cc//H
    for q in refs:
        r=x
        z=F.layer_norm(x,(Cc,),torch.from_numpy(q['g1']),torch.from_numpy(q['b1']),1e-5)
        qq=F.linear(z,torch.from_numpy(q['qkv'])).reshape(len(tok),3,H,hd)
        Q=rope(qq[:,0],interleaved=False).permute(1,0,2)
        K=rope(qq[:,1],interleaved=False).permute(1,0,2)
        V=qq[:,2].permute(1,0,2)
        s=torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(hd)
        s=s.masked_fill(mask[None,None,:],float('-inf'))
        a=torch.softmax(s,dim=-1)
        a=torch.matmul(a,V).permute(1,0,2).contiguous().reshape(len(tok),Cc)
        a=F.linear(a,torch.from_numpy(q['ow']))
        x=(r+a)*non
        r=x
        z=F.layer_norm(x,(Cc,),torch.from_numpy(q['g2']),torch.from_numpy(q['b2']),1e-5)
        z=F.conv1d(z.T[None],torch.from_numpy(q['fw1']),torch.from_numpy(q['fb1']),padding=1)[0].T
        z=z*(3.0**-0.5)
        z=F.gelu(z,approximate='none')
        z=F.linear(z,torch.from_numpy(q['fw2']),torch.from_numpy(q['fb2']))
        x=(r+z)*non
    x=F.layer_norm(x,(Cc,),torch.from_numpy(fg),torch.from_numpy(fb),1e-5)*non
    return x.numpy()

def setup(path):
    lib=C.CDLL(str(path));
    lib.ds_threadpool_create.argtypes=[C.c_size_t];lib.ds_threadpool_create.restype=C.c_void_p
    lib.ds_threadpool_destroy.argtypes=[C.c_void_p]
    lib.ds_threadpool_threads.argtypes=[C.c_void_p];lib.ds_threadpool_threads.restype=C.c_size_t
    lib.ds_threadpool_cpu_at.argtypes=[C.c_void_p,C.c_size_t];lib.ds_threadpool_cpu_at.restype=C.c_int
    lib.ds_fs2_encoder_workspace_floats.argtypes=[C.POINTER(Weights),C.c_size_t];lib.ds_fs2_encoder_workspace_floats.restype=C.c_size_t
    lib.ds_fs2_encoder_forward_f32_avx2.argtypes=[C.POINTER(Weights),PI,PI,C.c_size_t,PF,PF,C.c_void_p];lib.ds_fs2_encoder_forward_f32_avx2.restype=C.c_int
    return lib

def run(lib,seed,T,Cc,L,H,bench=False):
    w,keep,refs,emb,dw,db,fg,fb,tok,dur=make_case(seed,T,Cc,L,H,pad=max(1,T//10))
    ref=ref_forward(refs,emb,dw,db,fg,fb,tok,dur,H)
    out=np.empty((T,Cc),np.float32); wn=lib.ds_fs2_encoder_workspace_floats(C.byref(w),T);ws=np.empty(wn,np.float32)
    pool=lib.ds_threadpool_create(0)
    try:
        rc=lib.ds_fs2_encoder_forward_f32_avx2(C.byref(w),pint(tok),pint(dur),T,pfloat(out),pfloat(ws),pool)
        if rc: raise RuntimeError(f'native rc={rc}')
        e=np.abs(out-ref);ma=float(e.max());den=np.maximum(np.abs(ref),1e-7);mr=float((e/den).max());pad=float(np.abs(out[tok==0]).max(initial=0))
        n=lib.ds_threadpool_threads(pool);cp=[lib.ds_threadpool_cpu_at(pool,i) for i in range(n)]
        ok=ma < (2e-4 if Cc>=384 else 8e-5) and pad==0.0
        print(f'M21 FS2 encoder Ttxt={T} C={Cc} L={L} H={H} workers={n}: max_abs={ma:.8g} max_rel={mr:.8g} pad={pad:.3g} cpus={cp} {"OK" if ok else "FAIL"}')
        if not ok: raise SystemExit(2)
        if bench:
            for _ in range(4):lib.ds_fs2_encoder_forward_f32_avx2(C.byref(w),pint(tok),pint(dur),T,pfloat(out),pfloat(ws),pool)
            ts=[]
            for _ in range(21):
                t0=time.perf_counter();lib.ds_fs2_encoder_forward_f32_avx2(C.byref(w),pint(tok),pint(dur),T,pfloat(out),pfloat(ws),pool);ts.append((time.perf_counter()-t0)*1000)
            print(f'  M21 encoder median={statistics.median(ts):.3f} ms p90={sorted(ts)[int(.9*(len(ts)-1))]:.3f} ms')
    finally:lib.ds_threadpool_destroy(pool)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--lib',type=Path,required=True);ap.add_argument('--official-shape',action='store_true');a=ap.parse_args();lib=setup(a.lib)
    run(lib,2101,7,64,1,2)
    run(lib,2102,19,64,2,2)
    run(lib,2103,31,128,3,2)
    if a.official_shape:
        run(lib,2121,32,384,4,2,True)
        run(lib,2122,64,384,4,2,True)
if __name__=='__main__':main()
