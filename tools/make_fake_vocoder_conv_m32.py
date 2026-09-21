#!/usr/bin/env python3
import argparse, struct
import numpy as np

MAGIC=b'DSVOC32\0'

def pack4(w):
    # w [Cout,Cin,K] -> [Cout/4,Cin,K,4]
    co,ci,k=w.shape
    assert co%4==0
    return w.reshape(co//4,4,ci,k).transpose(0,2,3,1).copy()

def ref_conv(x,w,b,pad,dil):
    ci,tin=x.shape; co,ci2,k=w.shape; assert ci==ci2
    tout=tin+2*pad-dil*(k-1)
    xp=np.pad(x,((0,0),(pad,pad)))
    y=np.empty((co,tout),np.float32)
    for o in range(co):
        acc=np.full(tout,b[o],np.float32)
        for c in range(ci):
            for q in range(k):
                acc += xp[c,q*dil:q*dil+tout]*w[o,c,q]
        y[o]=acc
    return y

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--cin',type=int,default=32);ap.add_argument('--cout',type=int,default=32)
    ap.add_argument('--k',type=int,default=11);ap.add_argument('--tin',type=int,default=257)
    ap.add_argument('--dilation',type=int,default=1);ap.add_argument('--pad',type=int,default=None)
    ap.add_argument('--seed',type=int,default=32)
    a=ap.parse_args(); pad=a.pad if a.pad is not None else a.dilation*(a.k-1)//2
    if a.cout%4: raise SystemExit('cout must be multiple of 4')
    rng=np.random.default_rng(a.seed)
    x=(rng.standard_normal((a.cin,a.tin))*0.2).astype(np.float32)
    w=(rng.standard_normal((a.cout,a.cin,a.k))*0.03).astype(np.float32)
    b=(rng.standard_normal(a.cout)*0.02).astype(np.float32)
    y=ref_conv(x,w,b,pad,a.dilation)
    wp=pack4(w)
    hdr=struct.pack('<8s10I16x',MAGIC,1,a.cin,a.cout,a.k,a.tin,y.shape[1],pad,a.dilation,1,1)
    assert len(hdr)==64
    with open(a.out,'wb') as f:
        f.write(hdr); f.write(b.tobytes()); f.write(wp.tobytes()); f.write(x.tobytes()); f.write(y.tobytes())
    print(f'M32 fake Conv1d: Cin={a.cin} Cout={a.cout} K={a.k} Tin={a.tin} Tout={y.shape[1]} pad={pad} dil={a.dilation} -> {a.out}')
if __name__=='__main__':main()
