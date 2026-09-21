#!/usr/bin/env python3
"""Pack current default DiffSinger FastSpeech2 acoustic conditioner into DSFS21.

Covers the inference path implemented by M21:
  txt_embed + dur_embed -> 4-layer RoPE Transformer -> gather/stretch MLP+GRU
  -> pitch embedding.
Optional language/speaker/variance/key-shift/speed branches are intentionally
not part of this current-default profile packer.
"""
from __future__ import annotations
import argparse,json,struct
from pathlib import Path
import numpy as np
import torch
ALIGN=64

def align(x): return (x+ALIGN-1)//ALIGN*ALIGN
def unwrap(o):
    if not isinstance(o,dict): raise TypeError('checkpoint root must be dict')
    for k in ('state_dict','model_state_dict','model'):
        v=o.get(k)
        if isinstance(v,dict) and any(torch.is_tensor(x) for x in v.values()): return v
    if any(torch.is_tensor(x) for x in o.values()): return o
    raise ValueError('state_dict not found')
def find(sd,suffix,prefix=None):
    h=[k for k in sd if k.endswith(suffix) and (prefix is None or prefix in k)]
    if len(h)!=1: raise KeyError(f'{suffix}: expected 1 match, got {len(h)}; matches={h[:8]}')
    return h[0]
def arr(t): return np.ascontiguousarray(t.detach().cpu().float().numpy(),np.float32)
def pack16(w):
    w=np.ascontiguousarray(w,np.float32);n,k=w.shape
    if n%16: raise ValueError(f'N must be multiple of 16, got {n}')
    return np.ascontiguousarray(w.reshape(n//16,16,k).transpose(0,2,1)).reshape(-1)
def conv3_flat(w):
    # torch Conv1d [N,C,3] -> tap-major im2col [N,3*C]
    return np.ascontiguousarray(w.transpose(0,2,1).reshape(w.shape[0],-1),np.float32)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('checkpoint',type=Path);ap.add_argument('--prefix',default='fs2');ap.add_argument('--out',type=Path,default=Path('packed_fs2'));a=ap.parse_args()
    sd=unwrap(torch.load(a.checkpoint,map_location='cpu',weights_only=True))
    def t(s): return sd[find(sd,s,a.prefix)].detach().cpu().float().contiguous()
    emb=t('fs2.txt_embed.weight'); V,C=emb.shape
    durw=t('fs2.dur_embed.weight');durb=t('fs2.dur_embed.bias')
    if tuple(durw.shape)!=(C,1): raise ValueError(f'dur_embed expected {(C,1)}, got {tuple(durw.shape)}')
    # Current acoustic profile is four layers; infer count rather than hard-code.
    ids=[]
    for k in sd:
        needle='fs2.encoder.layers.'
        if needle in k and k.endswith('.op.layer_norm1.weight'):
            try: ids.append(int(k.split(needle,1)[1].split('.',1)[0]))
            except ValueError: pass
    ids=sorted(set(ids))
    if not ids or ids!=list(range(len(ids))): raise ValueError(f'non-contiguous encoder layers: {ids}')
    arrays=[]
    def add(n,x): arrays.append((n,np.ascontiguousarray(x,np.float32)))
    add('token_embedding',arr(emb));add('dur_weight',arr(durw)[:,0]);add('dur_bias',arr(durb))
    keymap={}
    for i in ids:
        p=f'fs2.encoder.layers.{i}.op.'
        g1=t(p+'layer_norm1.weight');b1=t(p+'layer_norm1.bias');qkv=t(p+'self_attn.in_proj.weight');ow=t(p+'self_attn.out_proj.weight')
        g2=t(p+'layer_norm2.weight');b2=t(p+'layer_norm2.bias');f1=t(p+'ffn.ffn_1.weight');fb1=t(p+'ffn.ffn_1.bias');f2=t(p+'ffn.ffn_2.weight');fb2=t(p+'ffn.ffn_2.bias')
        if tuple(qkv.shape)!=(3*C,C) or tuple(ow.shape)!=(C,C): raise ValueError(f'layer {i}: attention shape mismatch')
        if tuple(f1.shape)!=(4*C,C,3) or tuple(f2.shape)!=(C,4*C): raise ValueError(f'layer {i}: FFN shape mismatch')
        add(f'l{i}.ln1_gamma',arr(g1));add(f'l{i}.ln1_beta',arr(b1));add(f'l{i}.qkv_weight_m4n16',pack16(arr(qkv)));add(f'l{i}.qkv_bias',np.zeros(3*C,np.float32))
        add(f'l{i}.out_weight_m4n16',pack16(arr(ow)));add(f'l{i}.out_bias',np.zeros(C,np.float32));add(f'l{i}.ln2_gamma',arr(g2));add(f'l{i}.ln2_beta',arr(b2))
        add(f'l{i}.ffn1_weight_m4n16',pack16(conv3_flat(arr(f1))));add(f'l{i}.ffn1_bias',arr(fb1));add(f'l{i}.ffn2_weight_m4n16',pack16(arr(f2)));add(f'l{i}.ffn2_bias',arr(fb2))
    add('final_ln_gamma',arr(t('fs2.encoder.layer_norm.weight')));add('final_ln_beta',arr(t('fs2.encoder.layer_norm.bias')))
    sw1=t('fs2.stretch_embed.1.weight');sb1=t('fs2.stretch_embed.1.bias');sw2=t('fs2.stretch_embed.3.weight');sb2=t('fs2.stretch_embed.3.bias')
    if tuple(sw1.shape)!=(4*C,C) or tuple(sw2.shape)!=(C,4*C):raise ValueError('stretch MLP shape mismatch')
    add('stretch_w1_m4n16',pack16(arr(sw1)));add('stretch_b1',arr(sb1));add('stretch_w2_m4n16',pack16(arr(sw2)));add('stretch_b2',arr(sb2))
    for n in ('weight_ih_l0','bias_ih_l0','weight_hh_l0','bias_hh_l0'):
        z=t('fs2.stretch_embed_rnn.'+n)
        if 'weight' in n:
            if tuple(z.shape)!=(3*C,C):raise ValueError(f'GRU {n} shape {tuple(z.shape)}')
            add('gru_'+n+'_m4n16',pack16(arr(z)))
        else:
            add('gru_'+n,arr(z))
    pw=t('fs2.pitch_embed.weight');pb=t('fs2.pitch_embed.bias')
    if tuple(pw.shape)!=(C,1):raise ValueError('pitch_embed shape mismatch')
    add('pitch_weight',arr(pw)[:,0]);add('pitch_bias',arr(pb))

    a.out.mkdir(parents=True,exist_ok=True);blob=a.out/'fs2_acoustic.dsfs';secs={}
    with blob.open('wb') as f:
        hdr=struct.pack('<8s7If24x',b'DSFS21\0\0',1,V,C,len(ids),2,3,0,10000.0);assert len(hdr)==64;f.write(hdr);off=64
        for name,x in arrays:
            al=align(off);f.write(b'\0'*(al-off));off=al;raw=x.tobytes();f.write(raw);secs[name]={'offset':off,'bytes':len(raw),'floats':x.size};off+=len(raw)
    man={'format':'DSFS21','version':1,'vocab_size':V,'hidden_size':C,'num_layers':len(ids),'num_heads':2,'ffn_kernel_size':3,'rope_interleaved':False,'rope_theta':10000.0,'sections':secs}
    (a.out/'fs2_acoustic.json').write_text(json.dumps(man,indent=2),encoding='utf8')
    print(f'packed DSFS21: vocab={V} C={C} L={len(ids)} H=2 -> {blob}')
    print(f'manifest: {a.out/"fs2_acoustic.json"}')
if __name__=='__main__':main()
