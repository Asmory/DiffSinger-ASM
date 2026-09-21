#!/usr/bin/env python3
import argparse, math
from pathlib import Path
import numpy as np
from dsv35_common import Builder

def conv_ref(x,w,b,pad=1):
    C,T=x.shape;O,_,K=w.shape;y=np.empty((O,T),np.float32);xp=np.pad(x,((0,0),(pad,pad)))
    for o in range(O):
        z=np.full(T,b[o],np.float32)
        for c in range(C):
            for k in range(K):z += xp[c,k:k+T]*w[o,c,k]
        y[o]=z
    return y

def ct_ref(x,w,b,stride=2,pad=1):
    Cin,T=x.shape;_,Cout,K=w.shape;Tout=(T-1)*stride-2*pad+K;y=np.repeat(b[:,None],Tout,1).astype(np.float32)
    for ci in range(Cin):
      for t in range(T):
       base=t*stride-pad
       for k in range(K):
        q=base+k
        if 0<=q<Tout:y[:,q]+=x[ci,t]*w[ci,:,k]
    return y

def main():
 ap=argparse.ArgumentParser();ap.add_argument('out');ap.add_argument('--work',default='build/m35_fake');ap.add_argument('--frames',type=int,default=16);ap.add_argument('--channels',type=int,default=8);a=ap.parse_args();rng=np.random.default_rng(35);T=a.frames;C=a.channels
 mel=rng.normal(-1,.3,(1,T,C)).astype(np.float32);f0=(220+np.arange(T)).reshape(1,T).astype(np.float32)
 W=rng.normal(0,.05,(C,C,3)).astype(np.float32);B=rng.normal(0,.02,C).astype(np.float32);WU=rng.normal(0,.03,(C,C,4)).astype(np.float32);BU=rng.normal(0,.01,C).astype(np.float32)
 b=Builder(T,C,T);tm=b.add_work(mel.shape);tf=b.add_work(f0.shape)
 t0=b.add_work((1,C,T));t1=b.add_work((1,C,T));t2=b.add_work((1,C,2*T));t3=b.add_work((1,C,T));t4=b.add_work((1,1,T));t5=b.add_work((1,T))
 # Source side: Div -> CumSum -> Mod -> Mul -> Sin -> Pad -> Slice.
 fu=b.add_work((1,1,T));fs=b.add_work((1,T));s0=b.add_work((1,T));s1=b.add_work((1,T));s2=b.add_work((1,T));s3=b.add_work((1,T));s4=b.add_work((1,T));s5=b.add_work((1,T+2));s6=b.add_work((1,T));s7=b.add_work((1,T));tout=b.add_work((1,T))
 c_sr=b.add_const(np.array(44100.,np.float32));c_one=b.add_const(np.array(1.,np.float32));c_tau=b.add_const(np.array(2*np.pi,np.float32))
 pk=np.zeros((1,C,3,8),np.float32)
 for oc in range(C):pk[0,:,:,oc]=W[oc]
 wid=b.add_const(pk);bid=b.add_const(B);wuid=b.add_const(np.transpose(WU,(1,0,2)).copy());buid=b.add_const(BU)
 b.add_op('Transpose',[tm],t0,p=[0,2,1]);b.add_op('Conv',[t0,wid,bid],t1,p=[C,C,C,3,T,1,1,8],f=[.1],flags=3,reserved=t0)
 b.add_op('ConvTranspose',[t1,wuid,buid],t2,p=[C,C,4,T,1,2])
 # Decimate the upsampled result back to T and take channel 0.
 b.add_op('Slice',[t2],t3,p=[0,0,0,0,1,1,2,1]); b.add_op('Slice',[t3],t4,p=[0,0,0,0,1,1,1,1]); b.add_op('Squeeze',[t4],t5)
 b.add_op('Unsqueeze',[tf],fu);b.add_op('Squeeze',[fu],fs);b.add_op('Div',[fs,c_sr],s0);b.add_op('CumSum',[s0],s1,p=[1]);b.add_op('Mod',[s1,c_one],s2);b.add_op('Mul',[s2,c_tau],s3);b.add_op('Sin',[s3],s4)
 b.add_op('Pad',[s4],s5,p=[0,1],f=[0.0]);b.add_op('Slice',[s5],s6,p=[0,1,0,0,1,1,1,1]);b.add_op('Add',[t5,s6],s7);b.add_op('Tanh',[s7],tout)
 x0=np.transpose(mel,(0,2,1))[0];x=np.where(x0>=0,x0,.1*x0);c=conv_ref(x,W,B)+x0;u=ct_ref(c,WU,BU);main=u[0,::2]
 src=np.sin((np.cumsum(f0[0]/44100.0)%1.0)*(2*np.pi)).astype(np.float32);gold=np.tanh(main+src).astype(np.float32)
 w=Path(a.work);w.mkdir(parents=True,exist_ok=True);mel.tofile(w/'mel.f32');f0.tofile(w/'f0.f32');gold.tofile(w/'golden_wave.f32')
 naive,planned=b.plan_arena_lifetimes(tm,tf,tout);b.write(a.out,tm,tf,tout);print('DSASM fake graph ->',a.out,'arena',naive,'->',planned,'golden',w/'golden_wave.f32')
if __name__=='__main__':main()
