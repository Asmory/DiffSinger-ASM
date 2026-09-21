#!/usr/bin/env python3
from pathlib import Path
import argparse, subprocess, sys, tempfile, torch
ROOT=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('out',type=Path);ap.add_argument('--c',type=int,default=64);ap.add_argument('--layers-fs',type=int,default=2);ap.add_argument('--layers-aux',type=int,default=2);ap.add_argument('--layers-rf',type=int,default=2);a=ap.parse_args()
with tempfile.TemporaryDirectory() as td:
    td=Path(td); f=td/'fs.ckpt';u=td/'aux.ckpt';r=td/'rf.ckpt'
    subprocess.run([sys.executable,str(ROOT/'make_fake_fs2_m21_checkpoint.py'),str(f),'--c',str(a.c),'--layers',str(a.layers_fs)],check=True)
    subprocess.run([sys.executable,str(ROOT/'make_fake_aux_checkpoint.py'),str(u),'--I',str(a.c),'--C',str(a.c),'--D',str(a.c),'--L',str(a.layers_aux)],check=True)
    subprocess.run([sys.executable,str(ROOT/'make_fake_lynxnet2_backbone.py'),str(r),'--input',str(a.c),'--condition',str(a.c),'--channels',str(a.c),'--hidden',str(a.c),'--layers',str(a.layers_rf)],check=True)
    sd={}
    for p in (f,u,r):
        x=torch.load(p,map_location='cpu',weights_only=True); sd.update(x['state_dict'])
a.out.parent.mkdir(parents=True,exist_ok=True);torch.save({'state_dict':sd},a.out);print(a.out)
