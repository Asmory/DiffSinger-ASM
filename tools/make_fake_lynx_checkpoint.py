#!/usr/bin/env python3
"""Create a small realistic-scale LYNXNet2-style state_dict for bundle CI."""
from pathlib import Path
import argparse
import torch

ap=argparse.ArgumentParser()
ap.add_argument('out', type=Path)
ap.add_argument('--dim', type=int, default=64)
ap.add_argument('--hidden', type=int, default=64)
ap.add_argument('--seed', type=int, default=7)
a=ap.parse_args()
D,H=a.dim,a.hidden
p='model.diffusion.denoise_fn.residual_layers.0.net'
g=torch.Generator().manual_seed(a.seed)
def r(shape,scale): return torch.randn(shape,generator=g,dtype=torch.float32)*scale
sd={
 f'{p}.0.weight':1+r((D,),.08), f'{p}.0.bias':r((D,),.05),
 f'{p}.2.weight':r((D,1,31),.04), f'{p}.2.bias':r((D,),.03),
 f'{p}.4.weight':r((2*H,D),.04), f'{p}.4.bias':r((2*H,),.03),
 f'{p}.6.weight':r((2*H,H),.04), f'{p}.6.bias':r((2*H,),.03),
 f'{p}.8.weight':r((D,H),.04), f'{p}.8.bias':r((D,),.03),
}
a.out.parent.mkdir(parents=True,exist_ok=True)
torch.save({'state_dict':sd},a.out)
print(a.out)
