#!/usr/bin/env python3
import argparse, json, math, os, subprocess, sys
from pathlib import Path
import numpy as np

TOOLS=Path(__file__).resolve().parent
sys.path.insert(0,str(TOOLS))
import validate_real_onnx_m27 as v27


def metrics(a,b): return v27.metrics(a,b)

def read_txt(path,dtype):
    return np.fromstring(Path(path).read_text(),sep=' ',dtype=dtype)

def load_sec(root, meta, name):
    s=meta['sections'][name]
    count=s['bytes']//4
    x=np.fromfile(root/'fs2_acoustic.dsfs',np.float32,count=count,offset=s['offset'])
    sh=s.get('shape') or [count]
    return x.reshape(sh)

def unpack16(a,N,K):
    return np.asarray(a,np.float32).reshape(N//16,K,16).transpose(0,2,1).reshape(N,K).copy()

def packed_torch_reference(packed, work, onnx_path, speaker_emb, depth, steps):
    import onnx
    import onnxruntime as ort
    import torch
    import torch.nn.functional as F
    from onnx import helper, TensorProto

    model=json.loads((packed/'model.json').read_text())
    fs=model['fs2']; C=int(fs['hidden_size']); L=int(fs['num_layers']); H=int(fs['num_heads']); hd=C//H
    toks=read_txt(work/'tokens.txt',np.int64); dur=read_txt(work/'durations.txt',np.int64); langs=read_txt(work/'languages.txt',np.int64)
    P=toks.size; T=int(dur.sum())
    emb=load_sec(packed,fs,'token_embedding').reshape(int(fs['vocab_size']),C)
    dw=load_sec(packed,fs,'dur_weight').reshape(C); db=load_sec(packed,fs,'dur_bias').reshape(C)
    le=load_sec(packed,fs,'language_embedding').reshape(int(fs['num_languages']),C)
    if 'language_token_mask' in fs.get('sections',{}):
        lm=load_sec(packed,fs,'language_token_mask').reshape(int(fs['vocab_size']))
        lids=np.where(lm[toks]>0.5,langs,0)
    else:
        lids=langs
    flags=int(fs.get('feature_flags',0))
    dur_value=dur.astype(np.float32) if flags&(1<<12) else np.log1p(dur.astype(np.float32))
    x=np.sqrt(np.float32(C))*emb[toks] + dur_value[:,None]*dw[None,:] + db[None,:] + le[lids]
    refs={'embedding':x.copy()}
    xt=torch.from_numpy(x.copy())
    if flags&(1<<11):
        cos=torch.from_numpy(load_sec(packed,fs,'rope_cos').reshape(-1,hd//2)[:P].copy())
        sin=torch.from_numpy(load_sec(packed,fs,'rope_sin').reshape(-1,hd//2)[:P].copy())
    else:
        pos=torch.arange(P,dtype=torch.float32)
        inv=1.0/(float(fs.get('rope_theta',10000.0)) ** (torch.arange(0,hd,2,dtype=torch.float32)/hd))
        freqs=torch.einsum('i,j->ij',pos,inv)
        cos=torch.cos(freqs); sin=torch.sin(freqs)
    def rot(z):
        # non-interleaved, as exported graph and DSFS25 metadata
        a,b=torch.chunk(z,2,dim=-1)
        co=cos[None,:,:]; si=sin[None,:,:]
        return torch.cat((a*co-b*si,b*co+a*si),dim=-1)
    for i in range(L):
        g1=torch.from_numpy(load_sec(packed,fs,f'l{i}.ln1_gamma').reshape(C)); b1=torch.from_numpy(load_sec(packed,fs,f'l{i}.ln1_beta').reshape(C))
        qkvw=torch.from_numpy(unpack16(load_sec(packed,fs,f'l{i}.qkv_weight_m4n16'),3*C,C))
        ow=torch.from_numpy(unpack16(load_sec(packed,fs,f'l{i}.out_weight_m4n16'),C,C))
        g2=torch.from_numpy(load_sec(packed,fs,f'l{i}.ln2_gamma').reshape(C)); b2=torch.from_numpy(load_sec(packed,fs,f'l{i}.ln2_beta').reshape(C))
        f1=torch.from_numpy(unpack16(load_sec(packed,fs,f'l{i}.ffn1_weight_m4n16'),4*C,3*C)).reshape(4*C,3,C).permute(0,2,1).contiguous()
        fb1=torch.from_numpy(load_sec(packed,fs,f'l{i}.ffn1_bias').reshape(4*C))
        f2=torch.from_numpy(unpack16(load_sec(packed,fs,f'l{i}.ffn2_weight_m4n16'),C,4*C)); fb2=torch.from_numpy(load_sec(packed,fs,f'l{i}.ffn2_bias').reshape(C))
        n=F.layer_norm(xt,(C,),g1,b1,1e-5)
        qkv=F.linear(n,qkvw,None).reshape(P,3,H,hd)
        q=rot(qkv[:,0].permute(1,0,2)); k=rot(qkv[:,1].permute(1,0,2)); vv=qkv[:,2].permute(1,0,2)
        att=torch.softmax(torch.matmul(q,k.transpose(-1,-2))/math.sqrt(hd),dim=-1)
        ctx=torch.matmul(att,vv).permute(1,0,2).reshape(P,C)
        xt=xt+F.linear(ctx,ow,None)
        n2=F.layer_norm(xt,(C,),g2,b2,1e-5)
        y=F.conv1d(n2.T[None,:,:],f1,fb1,padding=1)[0].T*(3.0**-0.5)
        y=F.gelu(y,approximate='none')
        xt=xt+F.linear(y,f2,fb2)
        refs[f'layer{i}']=xt.detach().numpy().copy()
    fg=torch.from_numpy(load_sec(packed,fs,'final_ln_gamma').reshape(C)); fb=torch.from_numpy(load_sec(packed,fs,'final_ln_beta').reshape(C))
    final=F.layer_norm(xt,(C,),fg,fb,1e-5).detach().numpy(); refs['final']=final

    # Add outputs to original ONNX. These are all before stochastic diffusion, so no noise patch needed.
    m=onnx.load(str(onnx_path),load_external_data=True)
    names=[('/fs2/encoder/Mul_1_output_0','embedding')]
    for i in range(L): names.append((f'/fs2/encoder/layers.{i}/op/Mul_1_output_0',f'layer{i}'))
    names.append(('/fs2/encoder/Mul_6_output_0','final'))
    produced={o for n in m.graph.node for o in n.output}
    for name,_ in names:
        if name not in produced: raise RuntimeError(f'M28 ONNX output not found: {name}')
        v27.add_output(m.graph,helper,TensorProto,name,[1,'tokens',C])
    pp=work/'m28_encoder_outputs.onnx'; onnx.save(m,str(pp))
    sess=ort.InferenceSession(str(pp),providers=['CPUExecutionProvider']); metas={x.name:x for x in sess.get_inputs()}
    f0=read_txt(work/'f0.txt',np.float32); breath=read_txt(work/'breathiness.txt',np.float32); voice=read_txt(work/'voicing.txt',np.float32); tension=read_txt(work/'tension.txt',np.float32); gender=read_txt(work/'gender.txt',np.float32); velocity=read_txt(work/'velocity.txt',np.float32)
    sp=np.fromfile(speaker_emb,np.float32); spk=np.broadcast_to(sp[None,:],(T,C)).copy()
    bases={'tokens':toks,'languages':langs,'durations':dur,'f0':f0,'breathiness':breath,'voicing':voice,'tension':tension,'gender':gender,'velocity':velocity,'spk_embed':spk}
    feed={}
    for name,x in bases.items():
        if name in metas: feed[name]=v27.shape_input(metas[name],x)
    if 'depth' in metas: feed['depth']=v27.shape_input(metas['depth'],np.array([depth]),scalar=True)
    if 'steps' in metas: feed['steps']=v27.shape_input(metas['steps'],np.array([steps]),scalar=True)
    outs=sess.run([n for n,_ in names],feed)
    out={}
    for (_,key),val in zip(names,outs): out[key]=metrics(v27.norm_out(val,C),refs[key])
    return out

def run_m27(a,packed,work,precise):
    cmd=[sys.executable,str(TOOLS/'validate_real_onnx_m27.py'),'--fs2-stages','--onnx',str(a.onnx),'--packed',str(packed),'--cli',str(a.cli),'--speaker-emb',str(a.speaker_emb),'--language-id',str(a.language_id),'--depth',str(a.depth),'--steps',str(a.steps),'--work',str(work)]
    env=os.environ.copy(); env['DSASM_FS2_PRECISE_LN']='1' if precise else '0'
    print(f"\n===== M28 {'PRECISE-LN' if precise else 'FAST-LN'} NATIVE =====")
    p=subprocess.run(cmd,env=env,text=True); 
    if p.returncode: raise SystemExit(p.returncode)
    return json.loads((work/'m27_report.json').read_text())

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--onnx',type=Path,required=True);ap.add_argument('--packed',type=Path,required=True);ap.add_argument('--cli',type=Path,default=Path('build/dsasm-acoustic'));ap.add_argument('--speaker-emb',type=Path,required=True);ap.add_argument('--work',type=Path,default=Path('build/m28_real'));ap.add_argument('--language-id',type=int,default=4);ap.add_argument('--depth',type=float,default=.6);ap.add_argument('--steps',type=int,default=20)
    a=ap.parse_args();a.work.mkdir(parents=True,exist_ok=True)
    fast=run_m27(a,a.packed,a.work/'fast',False)
    precise=run_m27(a,a.packed,a.work/'precise',True)
    ref=packed_torch_reference(a.packed,a.work/'fast',a.onnx,a.speaker_emb,a.depth,a.steps)
    print('\nM28 PACKED-TORCH ENCODER vs ONNX')
    for k in ['embedding','layer0','layer1','layer2','layer3','final']:
        q=ref[k];print(f"  {k:10s} max_abs={q['max_abs']:.9g} rmse={q['rmse']:.9g} cosine={q['cosine']:.9g}")
    print('\nM28 NATIVE FAST vs PRECISE-LN')
    for k in ['encoder_txt','gathered','stretch','gru','pitch','variance','key_shift','speed','speaker']:
        f=fast['fs2_stages'][k];p=precise['fs2_stages'][k]
        ratio=f['max_abs']/max(p['max_abs'],1e-30)
        print(f"  {k:12s} fast={f['max_abs']:.9g} precise={p['max_abs']:.9g} improvement={ratio:.3f}x")
    f=fast['fs2_stages']['encoder_txt']['max_abs'];p=precise['fs2_stages']['encoder_txt']['max_abs']; pt=ref['final']['max_abs']; emb=ref['embedding']['max_abs']
    print('\nM28 diagnosis:')
    print(f'  packed embedding vs ONNX max_abs={emb:.9g}')
    print(f'  packed PyTorch final encoder vs ONNX max_abs={pt:.9g}')
    print(f'  native encoder fast={f:.9g} precise={p:.9g}')
    if pt<5e-4 and p < f*0.35:
        print('  verdict: LayerNorm numerical cancellation strongly supported')
    elif pt<5e-4:
        print('  verdict: importer/encoder semantics look correct; remaining divergence is in native kernels/math')
    else:
        print('  verdict: packed reference already diverges; inspect importer/encoder semantics before ASM kernels')
    out={'packed_torch':ref,'fast':fast,'precise':precise}
    (a.work/'m28_report.json').write_text(json.dumps(out,indent=2)+'\n')
    print('M28 report=',a.work/'m28_report.json')
if __name__=='__main__': main()
