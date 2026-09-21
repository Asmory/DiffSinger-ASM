#!/usr/bin/env python3
"""Upgrade a DSFS21 test bundle to DSFS25 with all deployment features.
Used only for loader/CLI smoke tests; values are deterministic synthetic data.
"""
from __future__ import annotations
import argparse,struct
from pathlib import Path
import numpy as np

ALIGN=64
LANG=1<<0;BREATH=1<<1;VOICE=1<<2;TENSION=1<<3;KEY=1<<4;SPEED=1<<5;SPK=1<<6;STRETCH=1<<7;LANGMASK=1<<8

def al(x): return (x+ALIGN-1)//ALIGN*ALIGN

def main():
    ap=argparse.ArgumentParser();ap.add_argument('src',type=Path);ap.add_argument('dst',type=Path);ap.add_argument('--languages',type=int,default=5);a=ap.parse_args()
    raw=a.src.read_bytes()
    if len(raw)<64: raise SystemExit('short DSFS21')
    magic,ver,V,C,L,H,K,inter,theta=struct.unpack('<8s7If24x',raw[:64])
    if magic!=b'DSFS21\0\0' or ver!=1: raise SystemExit(f'not DSFS21 v1: {magic!r} {ver}')
    flags=LANG|BREATH|VOICE|TENSION|KEY|SPEED|SPK|STRETCH|LANGMASK
    hdr=struct.pack('<8s9I4f4x',b'DSFS25\0\0',1,V,C,L,H,K,inter,a.languages,flags,theta,1/96,1/96,.1)
    out=bytearray(hdr);out.extend(raw[64:])
    rng=np.random.default_rng(252525)
    def add(x):
        nonlocal out
        off=al(len(out));out.extend(b'\0'*(off-len(out)));out.extend(np.ascontiguousarray(x,np.float32).tobytes())
    lang=(rng.standard_normal((a.languages,C))*0.01).astype(np.float32);lang[0]=0;add(lang)
    # Only a subset of token IDs are cross-lingual; deployment language IDs
    # must be masked to row 0 for all other tokens.
    lmask=np.zeros(V,np.float32);lmask[1::2]=1.0;add(lmask)
    for _ in range(5):
        add((rng.standard_normal(C)*0.005).astype(np.float32))
        add(np.zeros(C,np.float32))
    add(np.array([-1,1,12,1/12,.1,5,1],np.float32))
    # Exporter-folded stretch lookup. Zero is sufficient for loader/CLI smoke;
    # math equivalence is validated separately by validate_deploy_features_m25.py.
    add(np.zeros((1001,C),np.float32))
    a.dst.parent.mkdir(parents=True,exist_ok=True);a.dst.write_bytes(out)
    print(f'fake DSFS25: V={V} C={C} L={L} H={H} langs={a.languages} flags=0x{flags:x} bytes={len(out)} -> {a.dst}')
if __name__=='__main__':main()
