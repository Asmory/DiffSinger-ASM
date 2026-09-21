#!/usr/bin/env python3
import argparse, pathlib, struct
import numpy as np
MAGIC=b'DSVRU33\0'

def pack4(w):
    co,ci,k=w.shape
    assert co%4==0
    return w.reshape(co//4,4,ci,k).transpose(0,2,3,1).copy()

def conv(x,w,b,pad,dil):
    co,ci,k=w.shape; T=x.shape[1]; y=np.empty((co,T),np.float32)
    for o in range(co):
        for t in range(T):
            s=np.float32(b[o])
            for c in range(ci):
                for q in range(k):
                    u=t+q*dil-pad
                    if 0<=u<T:s=np.float32(s+np.float32(x[c,u]*w[o,c,q]))
            y[o,t]=s
    return y

def lrelu(x,a): return np.where(x>=0,x,x*np.float32(a)).astype(np.float32)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('out');ap.add_argument('--c',type=int,default=16);ap.add_argument('--t',type=int,default=97);ap.add_argument('--k1',type=int,default=7);ap.add_argument('--d1',type=int,default=3);ap.add_argument('--k2',type=int,default=7);ap.add_argument('--d2',type=int,default=1);ap.add_argument('--alpha',type=float,default=.1);a=ap.parse_args()
    rng=np.random.default_rng(33); C=a.c;T=a.t
    assert C%4==0
    p1=a.d1*(a.k1-1)//2;p2=a.d2*(a.k2-1)//2
    x=(rng.standard_normal((C,T))*0.1).astype(np.float32)
    w1=(rng.standard_normal((C,C,a.k1))*0.02).astype(np.float32);b1=(rng.standard_normal(C)*0.01).astype(np.float32)
    w2=(rng.standard_normal((C,C,a.k2))*0.02).astype(np.float32);b2=(rng.standard_normal(C)*0.01).astype(np.float32)
    y=conv(lrelu(x,a.alpha),w1,b1,p1,a.d1); y=conv(lrelu(y,a.alpha),w2,b2,p2,a.d2); y=(y+x).astype(np.float32)
    hdr=struct.pack('<8s9If16x',MAGIC,1,C,a.k1,a.k2,T,p1,a.d1,p2,a.d2,float(a.alpha))
    out=pathlib.Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('wb') as f:
        f.write(hdr);f.write(b1.tobytes());f.write(pack4(w1).tobytes());f.write(b2.tobytes());f.write(pack4(w2).tobytes());f.write(x.tobytes());f.write(y.tobytes())
    print(f'M33 fake resunit: C={C} T={T} k1={a.k1}/d{a.d1} k2={a.k2}/d{a.d2} -> {out}')
if __name__=='__main__':main()
