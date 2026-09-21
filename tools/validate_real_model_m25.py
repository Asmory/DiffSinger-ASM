#!/usr/bin/env python3
"""M25 real-model acceptance harness.

Given one supported DiffSinger acoustic checkpoint + acoustic yaml:
  1. pack M24 DSFS21/DSAUX20/DSLYNX7 bundles,
  2. synthesize a deterministic numeric phrase (tokens/durations/f0/noise),
  3. evaluate an independent PyTorch reference directly from checkpoint tensors,
  4. invoke the native dsasm-acoustic CLI with the *same* noise,
  5. compare the final raw mel elementwise and print quality/perf metrics.

This intentionally does not import DiffSinger, so the acceptance result is not
coupled to a separately-installed upstream checkout.
"""
from __future__ import annotations
import argparse, json, math, re, shutil, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import validate_pytorch_fs2_encoder_m21 as E
import validate_pytorch_fs2_condition_m21 as F2
import pack_aux_convnext_checkpoint as PA
import validate_pytorch_reflow as VR
import validate_pytorch_lynxnet2 as VL
import pack_acoustic_model_m24 as P24

torch.set_num_threads(1)

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

def load_fs2(sd,prefix='fs2'):
    def t(s): return sd[find(sd,s,prefix)].detach().cpu().float().contiguous()
    emb=arr(t('fs2.txt_embed.weight'));V,C=emb.shape
    dw=arr(t('fs2.dur_embed.weight'))[:,0];db=arr(t('fs2.dur_embed.bias'))
    ids=[]
    needle='fs2.encoder.layers.'
    for k in sd:
        if needle in k and k.endswith('.op.layer_norm1.weight') and prefix in k:
            try: ids.append(int(k.split(needle,1)[1].split('.',1)[0]))
            except ValueError: pass
    ids=sorted(set(ids))
    refs=[]
    for i in ids:
        p=f'fs2.encoder.layers.{i}.op.'
        refs.append(dict(
            g1=arr(t(p+'layer_norm1.weight')),b1=arr(t(p+'layer_norm1.bias')),
            qkv=arr(t(p+'self_attn.in_proj.weight')),ow=arr(t(p+'self_attn.out_proj.weight')),
            g2=arr(t(p+'layer_norm2.weight')),b2=arr(t(p+'layer_norm2.bias')),
            fw1=arr(t(p+'ffn.ffn_1.weight')),fb1=arr(t(p+'ffn.ffn_1.bias')),
            fw2=arr(t(p+'ffn.ffn_2.weight')),fb2=arr(t(p+'ffn.ffn_2.bias'))))
    fg=arr(t('fs2.encoder.layer_norm.weight'));fb=arr(t('fs2.encoder.layer_norm.bias'))
    q=dict(
        sw1=arr(t('fs2.stretch_embed.1.weight')),sb1=arr(t('fs2.stretch_embed.1.bias')),
        sw2=arr(t('fs2.stretch_embed.3.weight')),sb2=arr(t('fs2.stretch_embed.3.bias')),
        gwih=arr(t('fs2.stretch_embed_rnn.weight_ih_l0')),gbih=arr(t('fs2.stretch_embed_rnn.bias_ih_l0')),
        gwhh=arr(t('fs2.stretch_embed_rnn.weight_hh_l0')),gbhh=arr(t('fs2.stretch_embed_rnn.bias_hh_l0')),
        pw=arr(t('fs2.pitch_embed.weight'))[:,0],pb=arr(t('fs2.pitch_embed.bias')))
    return V,C,len(ids),refs,emb,dw,db,fg,fb,q

def load_aux(sd,prefix='aux_decoder'):
    kin=find(sd,'aux_decoder.decoder.inconv.weight',prefix);base=kin[:-len('inconv.weight')]
    def t(s): return sd[find(sd,base+s,prefix)].detach().cpu().float().contiguous()
    p={'inw':t('inconv.weight'),'inb':t('inconv.bias'),'outw':t('outconv.weight'),'outb':t('outconv.bias')}
    rx=re.compile(re.escape(base)+r'conv\.(\d+)\.dwconv\.weight$');ids=[]
    for k in sd:
        m=rx.search(k)
        if m: ids.append(int(m.group(1)))
    bs=[]
    for i in sorted(set(ids)):
        pref=f'{base}conv.{i}.'
        names={'dw':'dwconv.weight','dwb':'dwconv.bias','g':'norm.weight','bb':'norm.bias','w1':'pwconv1.weight','b1':'pwconv1.bias','w2':'pwconv2.weight','b2':'pwconv2.bias','gamma':'gamma'}
        bs.append({n:sd[find(sd,pref+s,prefix)].detach().cpu().float().contiguous() for n,s in names.items()})
    return p,bs

def load_rf(sd,prefix='diffusion.denoise_fn',glu='atan'):
    def t(s): return sd[find(sd,s,prefix)].detach().cpu().float().contiguous()
    p={'iw':t('input_projection.weight'),'ib':t('input_projection.bias'),'cw':t('conditioner_projection.weight'),'cb':t('conditioner_projection.bias'),
       'tw1':t('diffusion_embedding.1.weight'),'tb1':t('diffusion_embedding.1.bias'),'tw2':t('diffusion_embedding.3.weight'),'tb2':t('diffusion_embedding.3.bias'),
       'post_g':t('norm.weight'),'post_b':t('norm.bias'),'ow':t('output_projection.weight'),'ob':t('output_projection.bias')}
    if p['cw'].ndim==3 and p['cw'].shape[-1]==1:p['cw']=p['cw'][:,:,0].contiguous()
    rx=re.compile(r'residual_layers\.(\d+)\.net\.0\.weight$');ids=[]
    for k in sd:
        m=rx.search(k)
        if m and prefix in k: ids.append(int(m.group(1)))
    blocks=[]
    for i in sorted(set(ids)):
        base=f'residual_layers.{i}.net'
        names={'g':f'{base}.0.weight','b':f'{base}.0.bias','dw':f'{base}.2.weight','dwb':f'{base}.2.bias','w1':f'{base}.4.weight','b1':f'{base}.4.bias','w2':f'{base}.6.weight','b2':f'{base}.6.bias','w3':f'{base}.8.weight','b3':f'{base}.8.bias'}
        b={n:t(s) for n,s in names.items()}
        if b['dw'].ndim==3:b['dw']=b['dw'][:,0,:].contiguous()
        blocks.append({n:arr(v) for n,v in b.items()})
    q={n:arr(v) for n,v in p.items()};q['blocks']=blocks;q['glu']=glu
    return q

def spec_vec(v,D,default):
    if v is None:return np.array([default],np.float32)
    if isinstance(v,(list,tuple)):a=np.asarray(v,np.float32)
    else:a=np.asarray([v],np.float32)
    if len(a) not in (1,D): raise ValueError(f'spec range length {len(a)}, expected 1 or {D}')
    return np.ascontiguousarray(a)

def make_phrase(V,P,T,seed):
    if V<2: raise ValueError('vocab too small')
    P=min(P,max(1,T),max(1,V-1))
    tok=np.asarray([1+(i%(V-1)) for i in range(P)],np.int32)
    dur=np.full(P,T//P,np.int32);dur[:T%P]+=1
    mel=[]
    for i,d in enumerate(dur,1):mel.extend([i]*int(d))
    mel=np.asarray(mel,np.int32)
    x=np.arange(T,dtype=np.float32)
    f0=np.ascontiguousarray(220.0+55.0*np.sin(2*np.pi*x/max(T,1))+18.0*np.sin(6*np.pi*x/max(T,1)),np.float32)
    rng=np.random.default_rng(seed);noise=np.ascontiguousarray(rng.standard_normal((T,1)).astype(np.float32))
    return tok,dur,mel,f0,noise

def write_txt(path,a):path.write_text(' '.join(str(x) for x in a.tolist())+'\n')

def run_native(cli,model_dir,tok,dur,f0,noise,out,steps,threads=0):
    tmp=out.parent
    tp=tmp/'tokens.txt';dp=tmp/'durations.txt';fp=tmp/'f0.txt';npth=tmp/'noise.f32'
    write_txt(tp,tok);write_txt(dp,dur);fp.write_text(' '.join(f'{float(x):.9g}' for x in f0)+'\n');noise.astype(np.float32).tofile(npth)
    cmd=[str(cli),'infer',str(model_dir),'--tokens',str(tp),'--durations',str(dp),'--f0',str(fp),'--noise',str(npth),'--out',str(out),'--steps',str(steps)]
    if threads:cmd += ['--threads',str(threads)]
    p=subprocess.run(cmd,check=True,text=True,capture_output=True)
    print(p.stdout,end='')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('checkpoint',type=Path);ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--cli',type=Path,default=Path('build/dsasm-acoustic'));ap.add_argument('--out',type=Path,default=Path('build/m25_real_accept'))
    ap.add_argument('--frames',type=int,default=64);ap.add_argument('--tokens',type=int,default=32);ap.add_argument('--seed',type=int,default=2501);ap.add_argument('--steps',type=int)
    ap.add_argument('--threads',type=int,default=0);ap.add_argument('--fs2-prefix',default='fs2');ap.add_argument('--aux-prefix',default='aux_decoder');ap.add_argument('--rf-prefix',default='diffusion.denoise_fn');ap.add_argument('--glu-type',choices=['atan','softsign'],default='atan')
    ap.add_argument('--max-abs',type=float,default=1e-3)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    model=a.out/'model'
    if model.exists():shutil.rmtree(model)
    cmd=[sys.executable,str(HERE/'pack_acoustic_model_m24.py'),str(a.checkpoint),'--config',str(a.config),'--out',str(model),'--fs2-prefix',a.fs2_prefix,'--aux-prefix',a.aux_prefix,'--rf-prefix',a.rf_prefix,'--glu-type',a.glu_type]
    if a.steps is not None:cmd += ['--steps',str(a.steps)]
    print('+',' '.join(cmd),flush=True);subprocess.run(cmd,check=True)
    cfg=P24.read_config_loose(a.config);steps=int(a.steps if a.steps is not None else cfg.get('sampling_steps',20));t_start=float(cfg.get('T_start_infer',cfg.get('T_start',.4)));scale=float(cfg.get('time_scale_factor',1000.0))
    sd=unwrap(torch.load(a.checkpoint,map_location='cpu',weights_only=True))
    V,C,L,refs,emb,dw,db,fg,fb,fq=load_fs2(sd,a.fs2_prefix)
    ap_,abs_=load_aux(sd,a.aux_prefix);rq=load_rf(sd,a.rf_prefix,a.glu_type)
    D=rq['iw'].shape[1];Q=rq['cw'].shape[1]
    if C!=Q:raise SystemExit(f'FS2 hidden {C} != RF condition {Q}')
    tok,dur,mel2ph,f0,noise1=make_phrase(V,a.tokens,a.frames,a.seed);T=len(f0)
    noise=np.repeat(noise1,D,axis=1)
    # Decorrelate mel bins deterministically while retaining exact shared file.
    rng=np.random.default_rng(a.seed+1);noise=np.ascontiguousarray(rng.standard_normal((T,D)).astype(np.float32))
    cond=F2.ref_full(refs,emb,dw,db,fg,fb,tok,dur,mel2ph,f0,2,fq)
    with torch.inference_mode():aux_norm=PA.ref(torch.from_numpy(cond),ap_,abs_).numpy().astype(np.float32)
    lo=spec_vec(cfg.get('spec_min'),D,-12.0);hi=spec_vec(cfg.get('spec_max'),D,0.0)
    # shallow RF consumes normalized aux. torch_reflow accepts raw source, so denorm first.
    raw_aux=VR.torch_denorm(torch.from_numpy(aux_norm),lo,hi).numpy().astype(np.float32)
    rq['cond']=np.ascontiguousarray(cond,np.float32)
    norm=VR.torch_reflow(rq,noise,raw_aux if t_start>0 else None,lo,hi,t_start,steps,scale)
    ref=VR.torch_denorm(norm,lo,hi).numpy().astype(np.float32)
    ref_path=a.out/'reference_mel.f32';ref.tofile(ref_path)
    native_path=a.out/'native_mel.f32';run_native(a.cli.resolve(),model,tok,dur,f0,noise,native_path,steps,a.threads)
    got=np.fromfile(native_path,np.float32).reshape(T,D)
    diff=np.abs(got-ref);ma=float(diff.max());mr=float((diff/np.maximum(np.abs(ref),1e-5)).max());rmse=float(np.sqrt(np.mean((got-ref)**2)))
    den=float(np.linalg.norm(got)*np.linalg.norm(ref));cos=float(np.dot(got.ravel(),ref.ravel())/den) if den else 1.0
    sref=float(ref.sum(dtype=np.float64));sgot=float(got.sum(dtype=np.float64))
    ok=ma<a.max_abs
    report={'checkpoint':str(a.checkpoint),'frames':T,'tokens':len(tok),'mel_bins':D,'steps':steps,'t_start':t_start,'max_abs':ma,'max_rel':mr,'rmse':rmse,'cosine':cos,'reference_checksum':sref,'native_checksum':sgot,'pass':ok}
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'M25 real-model parity: P={len(tok)} T={T} mel={D} steps={steps} t_start={t_start:g}')
    print(f'  max_abs={ma:.8g} max_rel={mr:.8g} rmse={rmse:.8g} cosine={cos:.10f}')
    print(f'  checksum pytorch={sref:.12g} native={sgot:.12g} delta={sgot-sref:.6g}')
    print(f'  report={a.out/"report.json"} {"OK" if ok else "FAIL"}')
    if not ok: raise SystemExit(2)
if __name__=='__main__':main()
