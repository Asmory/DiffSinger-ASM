#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
from dsv35_common import Builder

def conv_ref(x,w,b,pad,dil):
    C,T=x.shape;O,_,K=w.shape;Tp=T+2*pad;xp=np.zeros((C,Tp),np.float32);xp[:,pad:pad+T]=x
    y=np.empty((O,T),np.float32)
    for o in range(O):
        z=np.full(T,b[o],np.float32)
        for c in range(C):
            for k in range(K): z += xp[c,k*dil:k*dil+T]*w[o,c,k]
        y[o]=z
    return y

def main():
    ap=argparse.ArgumentParser();ap.add_argument('out');ap.add_argument('--work',default='build/m40_fake');ap.add_argument('--c',type=int,default=16);ap.add_argument('--t',type=int,default=256);ap.add_argument('--k',type=int,default=7);ap.add_argument('--dilation',type=int,default=3);a=ap.parse_args()
    C,T,K,D=a.c,a.t,a.k,a.dilation;assert C%8==0 and T%8==0
    pad=D*(K-1)//2;rng=np.random.default_rng(40)
    mel=rng.normal(0,.3,(1,T,C)).astype(np.float32);f0=np.full((1,T),220,np.float32)
    W=rng.normal(0,.03,(C,C,K)).astype(np.float32);B=rng.normal(0,.01,C).astype(np.float32)
    packed=np.zeros((C//8,C,K,8),np.float32)
    for oc in range(C):packed[oc//8,:,:,oc%8]=W[oc]
    ws=np.empty(C,np.float32);qw=np.zeros_like(W,dtype=np.int8);corr=np.empty(C,np.int32);kred=C*K;k4=(kred+3)//4
    for oc in range(C):
        mx=float(np.max(np.abs(W[oc])));sc=mx/127. if mx>0 else 1.;ws[oc]=sc
        q=np.rint(W[oc]/sc).clip(-127,127).astype(np.int8);qw[oc]=q;corr[oc]=-128*int(q.astype(np.int32).sum())
    wp=np.zeros((C//8,k4,8,4),np.int8)
    for oc in range(C):
        flat=qw[oc].reshape(-1)
        for r,v in enumerate(flat):wp[oc//8,r//4,oc%8,r%4]=v
    blob=ws.tobytes()+corr.tobytes()+wp.tobytes()
    b=Builder(T,C,C*T);tm=b.add_work(mel.shape);tf=b.add_work(f0.shape);tx=b.add_work((1,C,T));ty=b.add_work((1,C,T))
    wid=b.add_const(packed);bid=b.add_const(B);vbid=b.add_blob(blob)
    b.add_op('Transpose',[tm],tx,p=[0,2,1]);b.add_op('Conv',[tx,wid,bid],ty,p=[C,C,C,K,T,pad,D,8],f=[0.1],flags=1|4,reserved=vbid)
    x=np.transpose(mel,(0,2,1))[0];x=np.where(x>=0,x,.1*x);gold=conv_ref(x,W,B,pad,D).reshape(-1)
    w=Path(a.work);w.mkdir(parents=True,exist_ok=True);mel.tofile(w/'mel.f32');f0.tofile(w/'f0.f32');gold.tofile(w/'golden.f32')
    b.plan_arena_lifetimes(tm,tf,ty);b.write(a.out,tm,tf,ty)
    print('M40 fake VNNI graph ->',a.out,'golden',w/'golden.f32')
if __name__=='__main__':main()
