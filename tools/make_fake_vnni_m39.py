#!/usr/bin/env python3
import argparse,struct
import numpy as np
MAGIC=b'DSVN39\0\0'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('out');ap.add_argument('--cin',type=int,default=16);ap.add_argument('--cout',type=int,default=16);ap.add_argument('--k',type=int,default=7);ap.add_argument('--tin',type=int,default=256);ap.add_argument('--dilation',type=int,default=3);a=ap.parse_args()
 ci,co,k,tin,d=a.cin,a.cout,a.k,a.tin,a.dilation;pad=d*(k-1)//2;tout=tin
 if co%8 or tout%8:raise SystemExit('cout and tout must be divisible by 8')
 rng=np.random.default_rng(39);x=rng.normal(0,.2,(ci,tin)).astype(np.float32);w=rng.normal(0,.03,(co,ci,k)).astype(np.float32);b=rng.normal(0,.02,co).astype(np.float32)
 xp=np.pad(x,((0,0),(pad,pad)));y=np.empty((co,tout),np.float32)
 for o in range(co):
  z=np.full(tout,b[o],np.float32)
  for c in range(ci):
   for q in range(k):z+=xp[c,q*d:q*d+tout]*w[o,c,q]
  y[o]=z
 kred=ci*k;k4=(kred+3)//4;tb=tout//8;ws=np.empty(co,np.float32);qw=np.zeros_like(w,dtype=np.int8);corr=np.empty(co,np.int32)
 for o in range(co):
  mx=float(np.max(np.abs(w[o])));sc=mx/127 if mx else 1.;ws[o]=sc;qq=np.rint(w[o]/sc).clip(-127,127).astype(np.int8);qw[o]=qq;corr[o]=-128*int(qq.astype(np.int32).sum())
 wp=np.zeros((co//8,k4,8,4),np.int8)
 for o in range(co):
  for r,v in enumerate(qw[o].reshape(-1)):wp[o//8,r//4,o%8,r%4]=v
 fp=np.zeros((co//8,ci,k,8),np.float32)
 for o in range(co):fp[o//8,:,:,o%8]=w[o]
 h=struct.pack('<8s12I8x',MAGIC,1,ci,co,k,tin,tout,pad,d,k4,tb,0,0)
 with open(a.out,'wb') as f:f.write(h);f.write(b.tobytes());f.write(ws.tobytes());f.write(corr.tobytes());f.write(wp.tobytes());f.write(fp.tobytes());f.write(x.tobytes());f.write(y.tobytes())
 print(f'M39 fake VNNI Conv: {ci}x{co} K{k} T{tin} d{d} -> {a.out}')
if __name__=='__main__':main()
