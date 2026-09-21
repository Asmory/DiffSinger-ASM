#!/usr/bin/env python3
import argparse,pathlib,struct
import numpy as np
MAGIC=b'DSVCT33\0'

def pack_output_major(w):
    # ONNX ConvTranspose [Cin,Cout,K] -> output-major [Cout,Cin,K].
    return w.transpose(1,0,2).copy()

def deconv(x,w,b,pad,dil,stride):
    ci,T=x.shape;ci2,co,k=w.shape;assert ci==ci2
    Tout=(T-1)*stride-2*pad+dil*(k-1)+1
    y=np.repeat(b[:,None].astype(np.float32),Tout,axis=1)
    for c in range(ci):
        for t in range(T):
            xv=np.float32(x[c,t])
            for o in range(co):
                for q in range(k):
                    u=t*stride+q*dil-pad
                    if 0<=u<Tout:y[o,u]=np.float32(y[o,u]+np.float32(xv*w[c,o,q]))
    return y

def main():
    ap=argparse.ArgumentParser();ap.add_argument('out');ap.add_argument('--cin',type=int,default=16);ap.add_argument('--cout',type=int,default=8);ap.add_argument('--k',type=int,default=8);ap.add_argument('--tin',type=int,default=33);ap.add_argument('--stride',type=int,default=4);ap.add_argument('--pad',type=int,default=2);ap.add_argument('--dilation',type=int,default=1);a=ap.parse_args()
    rng=np.random.default_rng(330);assert a.cout%4==0
    x=(rng.standard_normal((a.cin,a.tin))*0.1).astype(np.float32);w=(rng.standard_normal((a.cin,a.cout,a.k))*0.02).astype(np.float32);b=(rng.standard_normal(a.cout)*0.01).astype(np.float32)
    y=deconv(x,w,b,a.pad,a.dilation,a.stride);Tout=y.shape[1]
    hdr=struct.pack('<8s10I16x',MAGIC,1,a.cin,a.cout,a.k,a.tin,Tout,a.pad,a.dilation,a.stride,1)
    out=pathlib.Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('wb') as f:f.write(hdr);f.write(b.tobytes());f.write(pack_output_major(w).tobytes());f.write(x.tobytes());f.write(y.tobytes())
    print(f'M33 fake ConvTranspose: Cin={a.cin} Cout={a.cout} K={a.k} Tin={a.tin} Tout={Tout} stride={a.stride} pad={a.pad} -> {out}')
if __name__=='__main__':main()
