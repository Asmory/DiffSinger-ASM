#!/usr/bin/env python3
"""Pack a complete OpenVPI DiffSinger LYNXNet2 denoiser into DSLYNX7.

The converter intentionally does not import DiffSinger: it locates current
LYNXNet2 parameters by state-dict suffix, supports the official ATanGLU default
and SoftSignGLU, and emits deterministic PyTorch reference vectors.
"""
from __future__ import annotations
import argparse, json, math, re, struct
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ALIGN=64

def align_up(x): return (x+ALIGN-1)//ALIGN*ALIGN
def arr(t): return np.ascontiguousarray(t.detach().cpu().float().numpy())
def unwrap(o):
    if not isinstance(o,dict): raise TypeError('checkpoint root must be dict')
    for k in ('state_dict','model_state_dict','model'):
        v=o.get(k)
        if isinstance(v,dict) and any(torch.is_tensor(x) for x in v.values()): return v
    if any(torch.is_tensor(x) for x in o.values()): return o
    raise ValueError('tensor state_dict not found')
def find(sd,suffix,prefix):
    h=[k for k in sd if k.endswith(suffix) and (prefix is None or prefix in k)]
    if len(h)!=1: raise KeyError(f'{suffix}: expected 1 match, got {len(h)}; use --prefix. matches={h[:12]}')
    return h[0]
def pack16(w):
    w=np.ascontiguousarray(w,np.float32);n,k=w.shape
    if n%16: raise ValueError(f'linear N must be multiple of 16, got {n}')
    return np.ascontiguousarray(w.reshape(n//16,16,k).transpose(0,2,1)).ravel()
def pack_glu8(w):
    w=np.ascontiguousarray(w,np.float32);two,k=w.shape;n=two//2
    if two%2 or n%8: raise ValueError('GLU hidden dim must be multiple of 8')
    l=w[:n].reshape(n//8,8,k).transpose(0,2,1);g=w[n:].reshape(n//8,8,k).transpose(0,2,1)
    return np.ascontiguousarray(np.concatenate([l,g],2)).ravel()
def sinusoidal(x,c):
    half=c//2;s=-math.log(10000)/(half-1);e=torch.exp(torch.arange(half,dtype=torch.float32)*s)*float(x)
    return torch.cat((e.sin(),e.cos()))
def torch_ref(spec,cond,timestep,p,blocks,glu):
    x=F.linear(spec,p['iw'],p['ib'])+F.linear(cond,p['cw'],p['cb'])
    st=sinusoidal(timestep,x.shape[-1]);st=F.gelu(F.linear(st,p['tw1'],p['tb1']));st=F.linear(st,p['tw2'],p['tb2']);x=x+st
    for b in blocks:
        z=F.layer_norm(x,(x.shape[-1],),b['g'],b['b'],1e-5)
        z=F.conv1d(z.T[None],b['dw'][:,None,:],b['dwb'],padding=15,groups=x.shape[-1])[0].T
        z=F.linear(z,b['w1'],b['b1']);l,g=torch.chunk(z,2,-1);z=l*(torch.atan(g) if glu=='atan' else F.softsign(g))
        z=F.linear(z,b['w2'],b['b2']);l,g=torch.chunk(z,2,-1);z=l*(torch.atan(g) if glu=='atan' else F.softsign(g))
        x=x+F.linear(z,b['w3'],b['b3'])
    x=F.layer_norm(x,(x.shape[-1],),p['pg'],p['pb'],1e-5)
    return F.linear(x,p['ow'],p['ob'])

def main():
    a=argparse.ArgumentParser();a.add_argument('checkpoint',type=Path);a.add_argument('--prefix');a.add_argument('--glu-type',choices=['atan','softsign'],default='atan');a.add_argument('--out',type=Path,default=Path('packed_lynxnet2'));a.add_argument('--test-frames',type=int,default=37);a.add_argument('--seed',type=int,default=20260920);args=a.parse_args()
    sd=unwrap(torch.load(args.checkpoint,map_location='cpu',weights_only=True))
    def t(s): return sd[find(sd,s,args.prefix)].detach().cpu().float().contiguous()
    p={'iw':t('input_projection.weight'),'ib':t('input_projection.bias'),'cw':t('conditioner_projection.weight'),'cb':t('conditioner_projection.bias'),'tw1':t('diffusion_embedding.1.weight'),'tb1':t('diffusion_embedding.1.bias'),'tw2':t('diffusion_embedding.3.weight'),'tb2':t('diffusion_embedding.3.bias'),'pg':t('norm.weight'),'pb':t('norm.bias'),'ow':t('output_projection.weight'),'ob':t('output_projection.bias')}
    if p['cw'].ndim==3 and p['cw'].shape[-1]==1:p['cw']=p['cw'][:,:,0].contiguous()
    C,I=p['iw'].shape;C2,Q=p['cw'].shape
    if C2!=C or tuple(p['tw1'].shape)!=(4*C,C) or tuple(p['tw2'].shape)!=(C,4*C): raise ValueError('unexpected top-level LYNXNet2 shapes')
    if tuple(p['ow'].shape)!=(I,C): raise ValueError('output projection does not map C back to input dimension')
    # Discover layer indexes using the same selected prefix.
    rx=re.compile(r'residual_layers\.(\d+)\.net\.0\.weight$');ids=[]
    for k in sd:
        m=rx.search(k)
        if m and (args.prefix is None or args.prefix in k):ids.append(int(m.group(1)))
    ids=sorted(set(ids))
    if not ids or ids!=list(range(len(ids))):raise ValueError(f'could not discover contiguous residual layers: {ids}')
    blocks=[];keys={}
    for i in ids:
        base=f'residual_layers.{i}.net';names={'g':f'{base}.0.weight','b':f'{base}.0.bias','dw':f'{base}.2.weight','dwb':f'{base}.2.bias','w1':f'{base}.4.weight','b1':f'{base}.4.bias','w2':f'{base}.6.weight','b2':f'{base}.6.bias','w3':f'{base}.8.weight','b3':f'{base}.8.bias'}
        ks={n:find(sd,s,args.prefix) for n,s in names.items()};keys[str(i)]=ks;b={n:sd[k].detach().cpu().float().contiguous() for n,k in ks.items()}
        if b['dw'].ndim==3:b['dw']=b['dw'][:,0,:].contiguous()
        blocks.append(b)
    H=blocks[0]['w1'].shape[0]//2
    for b in blocks:
        if tuple(b['g'].shape)!=(C,) or tuple(b['dw'].shape)!=(C,31) or tuple(b['w1'].shape)!=(2*H,C) or tuple(b['w2'].shape)!=(2*H,H) or tuple(b['w3'].shape)!=(C,H):raise ValueError('unexpected residual block shapes')
    if C%16 or I%16 or H%8:raise ValueError(f'fast path requires C%16=0,I%16=0,H%8=0; got {C},{I},{H}')
    arrays=[]
    def add(name,x):arrays.append((name,np.ascontiguousarray(x,np.float32)))
    add('input_weight_m4n16',pack16(arr(p['iw'])));add('input_bias',arr(p['ib']));add('condition_weight_m4n16',pack16(arr(p['cw'])));add('condition_bias',arr(p['cb']));add('time1_weight_m4n16',pack16(arr(p['tw1'])));add('time1_bias',arr(p['tb1']));add('time2_weight_m4n16',pack16(arr(p['tw2'])));add('time2_bias',arr(p['tb2']))
    for i,b in enumerate(blocks):
        add(f'b{i}.ln_gamma',arr(b['g']));add(f'b{i}.ln_beta',arr(b['b']));add(f'b{i}.dw_weight_tap_major',arr(b['dw']).T);add(f'b{i}.dw_bias',arr(b['dwb']))
        pk=pack16 if args.glu_type=='atan' else pack_glu8
        add(f'b{i}.glu1_weight',pk(arr(b['w1'])));add(f'b{i}.glu1_bias',arr(b['b1']));add(f'b{i}.glu2_weight',pk(arr(b['w2'])));add(f'b{i}.glu2_bias',arr(b['b2']));add(f'b{i}.out_weight_m4n16',pack16(arr(b['w3'])));add(f'b{i}.out_bias',arr(b['b3']))
    add('post_norm_gamma',arr(p['pg']));add('post_norm_beta',arr(p['pb']));add('output_weight_m4n16',pack16(arr(p['ow'])));add('output_bias',arr(p['ob']))
    args.out.mkdir(parents=True,exist_ok=True);blob=args.out/'lynxnet2.dsn';manifest=args.out/'lynxnet2.json';secs={}
    with blob.open('wb') as f:
        hdr=struct.pack('<8s12I8x',b'DSLYNX7\0',1,I,Q,C,H,len(blocks),31,1 if args.glu_type=='atan' else 2,4,16,4,8);assert len(hdr)==64;f.write(hdr);off=64
        for name,x in arrays:
            al=align_up(off);f.write(b'\0'*(al-off));off=al;raw=x.tobytes();f.write(raw);secs[name]={'offset':off,'bytes':len(raw),'shape':list(x.shape)};off+=len(raw)
    gen=torch.Generator().manual_seed(args.seed);spec=torch.randn((args.test_frames,I),generator=gen)*.4;cond=torch.randn((args.test_frames,Q),generator=gen)*.25;timestep=torch.tensor(327.5,dtype=torch.float32)
    with torch.inference_mode():out=torch_ref(spec,cond,timestep,p,blocks,args.glu_type)
    spec.numpy().astype(np.float32).tofile(args.out/'test_spec.f32');cond.numpy().astype(np.float32).tofile(args.out/'test_condition.f32');np.array([timestep.item()],np.float32).tofile(args.out/'test_timestep.f32');out.numpy().astype(np.float32).tofile(args.out/'test_output_pytorch.f32')
    topkeys={n:find(sd,{'iw':'input_projection.weight','ib':'input_projection.bias','cw':'conditioner_projection.weight','cb':'conditioner_projection.bias','tw1':'diffusion_embedding.1.weight','tb1':'diffusion_embedding.1.bias','tw2':'diffusion_embedding.3.weight','tb2':'diffusion_embedding.3.bias','pg':'norm.weight','pb':'norm.bias','ow':'output_projection.weight','ob':'output_projection.bias'}[n],args.prefix) for n in p}
    m={'format':'DiffSinger-ASM complete LYNXNet2','magic':'DSLYNX7','version':1,'glu_type':args.glu_type,'input_dim':I,'condition_dim':Q,'channels':C,'hidden_dim':H,'num_layers':len(blocks),'kernel_size':31,'top_keys':topkeys,'block_keys':keys,'sections':secs,'test_vector':{'frames':args.test_frames,'seed':args.seed,'timestep':float(timestep),'spec':'test_spec.f32','condition':'test_condition.f32','output':'test_output_pytorch.f32'}}
    manifest.write_text(json.dumps(m,indent=2,ensure_ascii=False)+'\n');print(f'packed DSLYNX7: I={I} Q={Q} C={C} H={H} L={len(blocks)} glu={args.glu_type} -> {blob}')
if __name__=='__main__':main()
