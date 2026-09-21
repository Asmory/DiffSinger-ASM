#!/usr/bin/env python3
"""Pack current DiffSinger shallow ConvNeXt aux decoder into DSAUX20.
No DiffSinger import is required; tensors are found by suffix in a checkpoint state_dict.
"""
from __future__ import annotations
import argparse,json,re,struct
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ALIGN=64
def align(x):return (x+63)//64*64
def unwrap(o):
    if not isinstance(o,dict):raise TypeError('checkpoint root must be dict')
    for k in ('state_dict','model_state_dict','model'):
        v=o.get(k)
        if isinstance(v,dict) and any(torch.is_tensor(x) for x in v.values()):return v
    if any(torch.is_tensor(x) for x in o.values()):return o
    raise ValueError('state_dict not found')
def find(sd,suffix,prefix=None):
    h=[k for k in sd if k.endswith(suffix) and (prefix is None or prefix in k)]
    if len(h)!=1:raise KeyError(f'{suffix}: expected 1 match, got {len(h)}; matches={h[:8]}')
    return h[0]
def arr(t):return np.ascontiguousarray(t.detach().cpu().float().numpy())
def pack16(w):
    w=np.ascontiguousarray(w,np.float32);n,k=w.shape
    if n%16:raise ValueError(f'N must be multiple of 16: {n}')
    return np.ascontiguousarray(w.reshape(n//16,16,k).transpose(0,2,1)).ravel()
def conv_flat(w):
    # PyTorch [N,Kin,7] -> im2col order [tap0 all Cin, tap1 all Cin, ...]
    return np.ascontiguousarray(w.transpose(0,2,1).reshape(w.shape[0],-1),np.float32)
def ref(x,p,bs):
    z=F.conv1d(x.T[None],p['inw'],p['inb'],padding=3)
    for b in bs:
        r=z
        q=F.conv1d(z,b['dw'],b['dwb'],padding=3,groups=z.shape[1]).transpose(1,2)
        q=F.layer_norm(q,(q.shape[-1],),b['g'],b['bb'],1e-6)
        q=F.gelu(F.linear(q,b['w1'],b['b1']))
        q=F.linear(q,b['w2'],b['b2'])*b['gamma']
        z=r+q.transpose(1,2)
    return F.conv1d(z,p['outw'],p['outb'],padding=3)[0].T

def main():
    ap=argparse.ArgumentParser();ap.add_argument('checkpoint',type=Path);ap.add_argument('--prefix',default='aux_decoder');ap.add_argument('--out',type=Path,default=Path('packed_aux'));ap.add_argument('--test-frames',type=int,default=37);ap.add_argument('--seed',type=int,default=20260920);a=ap.parse_args()
    sd=unwrap(torch.load(a.checkpoint,map_location='cpu',weights_only=True))
    k_in=find(sd,'aux_decoder.decoder.inconv.weight',a.prefix); base=k_in[:-len('inconv.weight')]
    def t(s):return sd[find(sd,base+s,a.prefix)].detach().cpu().float().contiguous()
    p={'inw':t('inconv.weight'),'inb':t('inconv.bias'),'outw':t('outconv.weight'),'outb':t('outconv.bias')}
    C,I,K=p['inw'].shape;D,C2,K2=p['outw'].shape
    if K!=7 or K2!=7 or C2!=C:raise ValueError('expected current ConvNeXt k7 shapes')
    rx=re.compile(re.escape(base)+r'conv\.(\d+)\.dwconv\.weight$');ids=[]
    for k in sd:
        m=rx.search(k)
        if m:ids.append(int(m.group(1)))
    ids=sorted(set(ids))
    if ids!=list(range(len(ids))) or not ids:raise ValueError(f'non-contiguous ConvNeXt layers: {ids}')
    bs=[];keys={}
    for i in ids:
        pref=f'{base}conv.{i}.'
        names={'dw':'dwconv.weight','dwb':'dwconv.bias','g':'norm.weight','bb':'norm.bias','w1':'pwconv1.weight','b1':'pwconv1.bias','w2':'pwconv2.weight','b2':'pwconv2.bias','gamma':'gamma'}
        ks={n:find(sd,pref+s,a.prefix) for n,s in names.items()};keys[str(i)]=ks
        b={n:sd[k].detach().cpu().float().contiguous() for n,k in ks.items()}
        if b['dw'].shape!=(C,1,7) or b['w1'].shape!=(4*C,C) or b['w2'].shape!=(C,4*C) or b['gamma'].shape!=(C,):raise ValueError(f'layer {i}: unexpected shape')
        bs.append(b)
    arrays=[]
    def add(n,x):arrays.append((n,np.ascontiguousarray(x,np.float32)))
    add('in_weight_m4n16',pack16(conv_flat(arr(p['inw']))));add('in_bias',arr(p['inb']))
    for i,b in enumerate(bs):
        add(f'b{i}.dw_weight_tap_major',arr(b['dw'])[:,0,:].T);add(f'b{i}.dw_bias',arr(b['dwb']));add(f'b{i}.ln_gamma',arr(b['g']));add(f'b{i}.ln_beta',arr(b['bb']));add(f'b{i}.pw1_weight_m4n16',pack16(arr(b['w1'])));add(f'b{i}.pw1_bias',arr(b['b1']));add(f'b{i}.pw2_weight_m4n16',pack16(arr(b['w2'])));add(f'b{i}.pw2_bias',arr(b['b2']));add(f'b{i}.gamma',arr(b['gamma']))
    add('out_weight_m4n16',pack16(conv_flat(arr(p['outw']))));add('out_bias',arr(p['outb']))
    a.out.mkdir(parents=True,exist_ok=True);blob=a.out/'aux_convnext.dsa';secs={}
    with blob.open('wb') as f:
        hdr=struct.pack('<8s8I24x',b'DSAUX20\0',1,I,C,D,len(bs),7,4,16);assert len(hdr)==64;f.write(hdr);off=64
        for name,x in arrays:
            al=align(off);f.write(b'\0'*(al-off));off=al;raw=x.tobytes();f.write(raw);secs[name]={'offset':off,'bytes':len(raw)};off+=len(raw)
    gen=torch.Generator().manual_seed(a.seed);x=torch.randn((a.test_frames,I),generator=gen)*.25
    with torch.inference_mode():y=ref(x,p,bs)
    x.numpy().astype(np.float32).tofile(a.out/'test_condition.f32');y.numpy().astype(np.float32).tofile(a.out/'test_aux_norm_pytorch.f32')
    m={'format':'DiffSinger-ASM shallow ConvNeXt aux decoder','magic':'DSAUX20','version':1,'input_dim':I,'channels':C,'output_dim':D,'num_layers':len(bs),'kernel_size':7,'base_key':base,'block_keys':keys,'sections':secs,'test_vector':{'frames':a.test_frames,'condition':'test_condition.f32','output_norm':'test_aux_norm_pytorch.f32'}}
    (a.out/'aux_convnext.json').write_text(json.dumps(m,indent=2,ensure_ascii=False)+'\n')
    print(f'packed DSAUX20: I={I} C={C} D={D} L={len(bs)} -> {blob}')
if __name__=='__main__':main()
