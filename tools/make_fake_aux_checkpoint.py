#!/usr/bin/env python3
import argparse,torch
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('out',type=Path);ap.add_argument('--I',type=int,default=64);ap.add_argument('--C',type=int,default=64);ap.add_argument('--D',type=int,default=32);ap.add_argument('--L',type=int,default=2);a=ap.parse_args();g=torch.Generator().manual_seed(20260920)
f=lambda *s,scale=.02:torch.randn(s,generator=g)*scale
b='model.aux_decoder.decoder.';sd={b+'inconv.weight':f(a.C,a.I,7),b+'inconv.bias':f(a.C),b+'outconv.weight':f(a.D,a.C,7),b+'outconv.bias':f(a.D)}
for i in range(a.L):
 p=f'{b}conv.{i}.';sd[p+'dwconv.weight']=f(a.C,1,7);sd[p+'dwconv.bias']=f(a.C);sd[p+'norm.weight']=1+f(a.C);sd[p+'norm.bias']=f(a.C);sd[p+'pwconv1.weight']=f(4*a.C,a.C);sd[p+'pwconv1.bias']=f(4*a.C);sd[p+'pwconv2.weight']=f(a.C,4*a.C);sd[p+'pwconv2.bias']=f(a.C);sd[p+'gamma']=torch.full((a.C,),1e-6)
a.out.parent.mkdir(parents=True,exist_ok=True);torch.save({'state_dict':sd},a.out);print(a.out)
