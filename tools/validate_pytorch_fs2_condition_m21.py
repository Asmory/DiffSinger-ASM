#!/usr/bin/env python3
from __future__ import annotations
import argparse,ctypes as C,math,statistics,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import validate_pytorch_fs2_encoder_m21 as E

torch.set_num_threads(1)
PF=E.PF; PI=E.PI

def pfloat(a): return E.pfloat(a)
def pint(a): return E.pint(a)

class Acoustic(C.Structure):
    _fields_=[('encoder',E.Weights),('stretch_w1_m4n16',PF),('stretch_b1',PF),('stretch_w2_m4n16',PF),('stretch_b2',PF),
              ('gru_w_ih_m4n16',PF),('gru_b_ih',PF),('gru_w_hh_m4n16',PF),('gru_b_hh',PF),('pitch_weight',PF),('pitch_bias',PF)]

def make_mel2ph(rng,Pvalid,Tmel):
    core=Tmel-2
    cuts=np.ones(Pvalid,dtype=np.int32)
    rem=core-Pvalid
    for _ in range(rem): cuts[int(rng.integers(0,Pvalid))]+=1
    vals=[]
    for i,d in enumerate(cuts,1): vals += [i]*int(d)
    vals += [0,0]
    return np.asarray(vals,np.int32)

def make_case(seed,P,T,Cc,L,H,vocab=96,pad_text=2):
    ew,keep,refs,emb,dw,db,fg,fb,tok,_=E.make_case(seed,P,Cc,L,H,vocab=vocab,pad=pad_text)
    rng=np.random.default_rng(seed+99); valid=P-pad_text
    mel2ph=make_mel2ph(rng,valid,T)
    dur=np.bincount(mel2ph,minlength=P+1)[1:].astype(np.int32)
    f0=np.ascontiguousarray(rng.uniform(70,900,size=T).astype(np.float32)); f0[mel2ph==0]=0
    sw1=E.randn(rng,(4*Cc,Cc),0.03); sw1p=E.pack16(sw1); sb1=E.randn(rng,(4*Cc,),0.01)
    sw2=E.randn(rng,(Cc,4*Cc),0.03); sw2p=E.pack16(sw2); sb2=E.randn(rng,(Cc,),0.01)
    gwih=E.randn(rng,(3*Cc,Cc),0.025); gwihp=E.pack16(gwih); gbih=E.randn(rng,(3*Cc,),0.01)
    gwhh=E.randn(rng,(3*Cc,Cc),0.025); gwhhp=E.pack16(gwhh); gbhh=E.randn(rng,(3*Cc,),0.01)
    pw=E.randn(rng,(Cc,),0.035); pb=E.randn(rng,(Cc,),0.01)
    keep += [mel2ph,dur,f0,sw1,sw1p,sb1,sw2,sw2p,sb2,gwih,gwihp,gbih,gwhh,gwhhp,gbhh,pw,pb]
    aw=Acoustic(ew,pfloat(sw1p),pfloat(sb1),pfloat(sw2p),pfloat(sb2),pfloat(gwihp),pfloat(gbih),pfloat(gwhhp),pfloat(gbhh),pfloat(pw),pfloat(pb))
    extra=dict(sw1=sw1,sb1=sb1,sw2=sw2,sb2=sb2,gwih=gwih,gbih=gbih,gwhh=gwhh,gbhh=gbhh,pw=pw,pb=pb)
    return aw,keep,refs,emb,dw,db,fg,fb,tok,dur,mel2ph,f0,extra

def stretch_ref(mel2ph,dur):
    m=torch.from_numpy(mel2ph.astype(np.int64));d=torch.from_numpy(dur.astype(np.int64))[None]
    dp=torch.cat([torch.ones_like(d[:,:1]),d],1)[0]
    md=dp[m]
    bound=m[1:]>m[:-1]
    delta=1-bound.to(torch.int64)*md[:-1]
    delta=F.pad(delta,[1,0])
    den=torch.cumsum(delta,0)
    return den.float()/md * (m>0)

def sinemb(x,Cc):
    half=Cc//2; a=math.log(10000)/(half-1)
    freq=torch.exp(torch.arange(half,dtype=torch.float32)*-a)
    z=x[:,None]*freq[None]
    return torch.cat([z.sin(),z.cos()],-1)

def ref_full(refs,emb,dw,db,fg,fb,tok,dur,mel2ph,f0,H,q):
    enc=E.ref_forward(refs,emb,dw,db,fg,fb,tok,dur,H)
    enc_t=torch.from_numpy(enc)
    m=torch.from_numpy(mel2ph.astype(np.int64));
    padded=torch.cat([torch.zeros(1,enc_t.shape[1]),enc_t],0)
    cond=padded[m]
    st=stretch_ref(mel2ph,dur)
    se=sinemb(torch.round(1000*st),enc_t.shape[1])
    z=F.gelu(F.linear(se,torch.from_numpy(q['sw1']),torch.from_numpy(q['sb1'])),approximate='none')
    z=F.linear(z,torch.from_numpy(q['sw2']),torch.from_numpy(q['sb2']))
    cond=cond+z
    Cc=cond.shape[1]
    gru=torch.nn.GRU(Cc,Cc,1,batch_first=True)
    with torch.no_grad():
        gru.weight_ih_l0.copy_(torch.from_numpy(q['gwih']));gru.bias_ih_l0.copy_(torch.from_numpy(q['gbih']))
        gru.weight_hh_l0.copy_(torch.from_numpy(q['gwhh']));gru.bias_hh_l0.copy_(torch.from_numpy(q['gbhh']))
    gout,_=gru(cond[None]);cond=cond+gout[0]
    pi=torch.log1p(torch.from_numpy(f0)/700.0)
    cond=cond + pi[:,None]*torch.from_numpy(q['pw'])[None] + torch.from_numpy(q['pb'])[None]
    return cond.detach().numpy()

def setup(path):
    lib=E.setup(path)
    lib.ds_fs2_acoustic_condition_workspace_floats.argtypes=[C.POINTER(Acoustic),C.c_size_t,C.c_size_t];lib.ds_fs2_acoustic_condition_workspace_floats.restype=C.c_size_t
    lib.ds_fs2_acoustic_condition_f32_avx2.argtypes=[C.POINTER(Acoustic),PI,C.c_size_t,PI,PF,C.c_size_t,PF,PF,C.c_void_p];lib.ds_fs2_acoustic_condition_f32_avx2.restype=C.c_int
    return lib

def run(lib,seed,P,T,Cc,L,H,bench=False):
    aw,keep,refs,emb,dw,db,fg,fb,tok,dur,m,f0,q=make_case(seed,P,T,Cc,L,H)
    ref=ref_full(refs,emb,dw,db,fg,fb,tok,dur,m,f0,H,q)
    out=np.empty((T,Cc),np.float32);wn=lib.ds_fs2_acoustic_condition_workspace_floats(C.byref(aw),P,T);ws=np.empty(wn,np.float32)
    pool=lib.ds_threadpool_create(0)
    try:
        rc=lib.ds_fs2_acoustic_condition_f32_avx2(C.byref(aw),pint(tok),P,pint(m),pfloat(f0),T,pfloat(out),pfloat(ws),pool)
        if rc:raise RuntimeError(rc)
        e=np.abs(out-ref);ma=float(e.max());mr=float((e/np.maximum(np.abs(ref),1e-7)).max())
        n=lib.ds_threadpool_threads(pool);cp=[lib.ds_threadpool_cpu_at(pool,i) for i in range(n)]
        ok=ma < (4e-4 if Cc>=384 else 1.2e-4)
        print(f'M21 full condition Ttxt={P} Tmel={T} C={Cc} L={L}: max_abs={ma:.8g} max_rel={mr:.8g} workers={n} cpus={cp} {"OK" if ok else "FAIL"}')
        if not ok:raise SystemExit(2)
        if bench:
            for _ in range(3):lib.ds_fs2_acoustic_condition_f32_avx2(C.byref(aw),pint(tok),P,pint(m),pfloat(f0),T,pfloat(out),pfloat(ws),pool)
            ts=[]
            for _ in range(15):
                t0=time.perf_counter();lib.ds_fs2_acoustic_condition_f32_avx2(C.byref(aw),pint(tok),P,pint(m),pfloat(f0),T,pfloat(out),pfloat(ws),pool);ts.append((time.perf_counter()-t0)*1000)
            print(f'  M21 full-condition median={statistics.median(ts):.3f} ms p90={sorted(ts)[int(.9*(len(ts)-1))]:.3f} ms')
    finally:lib.ds_threadpool_destroy(pool)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--lib',type=Path,required=True);ap.add_argument('--official-shape',action='store_true');a=ap.parse_args();lib=setup(a.lib)
    run(lib,2201,9,24,64,1,2)
    run(lib,2202,17,48,128,2,2)
    if a.official_shape:run(lib,2221,32,64,384,4,2,True)
if __name__=='__main__':main()
