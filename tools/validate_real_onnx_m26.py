#!/usr/bin/env python3
import argparse, copy, json, math, subprocess, sys
from pathlib import Path
import numpy as np


def np_dtype(ort_type):
    if 'int64' in ort_type: return np.int64
    if 'int32' in ort_type: return np.int32
    if 'float' in ort_type: return np.float32
    raise TypeError(f'unsupported ORT input type: {ort_type}')

def rank_shape(meta):
    return len(meta.shape) if meta.shape is not None else None

def shape_input(meta, x, scalar=False):
    x=np.asarray(x,dtype=np_dtype(meta.type))
    r=rank_shape(meta)
    if scalar:
        if r==0:return np.asarray(x.reshape(-1)[0],dtype=x.dtype)
        if r==1:return np.asarray([x.reshape(-1)[0]],dtype=x.dtype)
    if r==1:return x.reshape(-1)
    if r==2:
        if x.ndim==1:return x[None,:]
        if x.ndim==2:return x
    if r==3:
        if x.ndim==2:return x[None,:,:]
    return x

def add_output(g,helper,TensorProto,name,shape):
    if any(o.name==name for o in g.output):return
    g.output.append(helper.make_tensor_value_info(name,TensorProto.FLOAT,shape))

def patch_onnx(src,dst,hidden,mel,debug_stages=False):
    import onnx
    from onnx import helper, TensorProto
    m=onnx.load(str(src),load_external_data=True)
    add_output(m.graph,helper,TensorProto,'condition',[1,'frames',hidden])
    add_output(m.graph,helper,TensorProto,'aux_mel',[1,'frames',mel])
    if debug_stages:
        stages=[
            ('/fs2/encoder/Mul_6_output_0',[1,'tokens',hidden]),
            ('/fs2/GatherElements_output_0',[1,'frames',hidden]),
            ('/fs2/Add_2_output_0',[1,'frames',hidden]),
            ('/fs2/Add_3_output_0',[1,'frames',hidden]),
            ('/fs2/Add_5_output_0',[1,'frames',hidden]),
            ('/fs2/Add_6_output_0',[1,'frames',hidden]),
            ('/fs2/Add_8_output_0',[1,'frames',hidden]),
            ('/fs2/Add_9_output_0',[1,'frames',hidden]),
        ]
        produced={o for n in m.graph.node for o in n.output}
        for name,shape in stages:
            if name not in produced:
                raise RuntimeError(f'M27 expected FS2 stage output not found: {name}')
            add_output(m.graph,helper,TensorProto,name,shape)
    rnd=[n for n in m.graph.node if n.op_type=='RandomNormalLike']
    if len(rnd)!=1:raise RuntimeError(f'expected 1 RandomNormalLike, got {len(rnd)}')
    n=rnd[0]
    old=list(n.input)
    n.op_type='Identity'; del n.attribute[:]; del n.input[:]; n.input.extend(['noise_external'])
    m.graph.input.append(helper.make_tensor_value_info('noise_external',TensorProto.FLOAT,[1,1,mel,'frames']))
    onnx.save(m,str(dst))
    print(f'M26 patched ONNX: {rnd[0].name} {old} -> noise_external; outputs += condition, aux_mel')

def parse_model_conf(path):
    d={}
    for line in path.read_text().splitlines():
        line=line.strip()
        if not line or line.startswith('#') or '=' not in line:continue
        k,v=line.split('=',1);d[k]=v
    return d

def metrics(a,b):
    a=np.asarray(a,dtype=np.float32).reshape(-1);b=np.asarray(b,dtype=np.float32).reshape(-1)
    if a.size!=b.size: return {'shape_mismatch':[int(a.size),int(b.size)]}
    d=np.abs(a-b); den=np.maximum(np.abs(a),1e-8)
    na=float(np.linalg.norm(a));nb=float(np.linalg.norm(b));cos=float(np.dot(a,b)/(na*nb)) if na and nb else 1.0
    return {'max_abs':float(d.max(initial=0)), 'max_rel':float((d/den).max(initial=0)), 'rmse':float(np.sqrt(np.mean((a-b)**2))), 'cosine':cos}

def read_vocab(packed):
    p=packed/'phonemes.json'
    if not p.exists():return [4,21,23,24,25,4]
    d=json.loads(p.read_text(encoding='utf8'))
    cand=[]
    for k,v in d.items():
        if k.startswith('zh/') and isinstance(v,int) and 0<v<206 and v not in cand:cand.append(v)
    if len(cand)<5:
        cand += [v for v in d.values() if isinstance(v,int) and 0<v<206 and v not in cand]
    return [4]+cand[:6]+[4]

def dump_txt(path,arr):
    a=np.asarray(arr).reshape(-1)
    path.write_text(' '.join(f'{float(x):.9g}' if np.issubdtype(a.dtype,np.floating) else str(int(x)) for x in a)+'\n')

def norm_out(x,last):
    x=np.asarray(x)
    while x.ndim>2 and x.shape[0]==1:x=x[0]
    if x.ndim==2 and x.shape[-1]==last:return x.astype(np.float32)
    if x.ndim==2 and x.shape[0]==last:return x.T.astype(np.float32)
    return x.reshape(-1,last).astype(np.float32)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--onnx',type=Path,required=True);ap.add_argument('--packed',type=Path,required=True)
    ap.add_argument('--cli',type=Path,default=Path('build/dsasm-acoustic'));ap.add_argument('--speaker-emb',type=Path,required=True)
    ap.add_argument('--work',type=Path,default=Path('build/m26_real'));ap.add_argument('--language-id',type=int,default=4)
    ap.add_argument('--depth',type=float,default=.6);ap.add_argument('--steps',type=int,default=20);ap.add_argument('--seed',type=int,default=2601)
    ap.add_argument('--threads',type=int,default=0);ap.add_argument('--mel-threshold',type=float,default=5e-3)
    ap.add_argument('--fs2-stages',action='store_true');ap.add_argument('--stage-threshold',type=float,default=5e-4);ap.add_argument('--strict',action='store_true')
    a=ap.parse_args();a.work.mkdir(parents=True,exist_ok=True)
    try:
        import onnxruntime as ort
    except Exception as e:
        raise SystemExit(f'onnxruntime is required: {e}')
    model=json.loads((a.packed/'model.json').read_text())
    hidden=int(model['fs2']['hidden_size']); mel=int(model['rf']['input_dim']); vocab=int(model['fs2']['vocab_size'])
    patched=a.work/'acoustic_m27_patched.onnx' if a.fs2_stages else a.work/'acoustic_m26_patched.onnx';patch_onnx(a.onnx,patched,hidden,mel,a.fs2_stages)
    toks=np.array([x for x in read_vocab(a.packed) if 0<x<vocab],np.int64)
    if toks.size<4:raise RuntimeError('could not choose valid tokens')
    dur=np.array([4+(i%4) for i in range(toks.size)],np.int64);T=int(dur.sum());P=int(toks.size)
    langs=np.full(P,a.language_id,np.int64)
    tt=np.arange(T,dtype=np.float32)
    f0=(220.0+18.0*np.sin(tt*0.17)).astype(np.float32)
    breath=(0.15+0.03*np.sin(tt*0.11)).astype(np.float32);voice=(0.75+0.05*np.cos(tt*0.07)).astype(np.float32)
    tension=(0.10+0.02*np.sin(tt*0.13)).astype(np.float32);gender=(0.08*np.sin(tt*0.09)).astype(np.float32);velocity=(1.0+0.04*np.cos(tt*0.05)).astype(np.float32)
    sp=np.fromfile(a.speaker_emb,np.float32)
    if sp.size!=hidden:raise RuntimeError(f'speaker emb has {sp.size} floats, expected {hidden}')
    spk=np.broadcast_to(sp[None,:],(T,hidden)).copy()
    rng=np.random.default_rng(a.seed);noise=rng.standard_normal((T,mel),dtype=np.float32)
    noise.tofile(a.work/'noise.f32')
    for name,arr in [('tokens',toks),('durations',dur),('f0',f0),('languages',langs),('breathiness',breath),('voicing',voice),('tension',tension),('gender',gender),('velocity',velocity)]:dump_txt(a.work/f'{name}.txt',arr)

    so=ort.SessionOptions();so.intra_op_num_threads=max(1,a.threads) if a.threads else 0
    sess=ort.InferenceSession(str(patched),so,providers=['CPUExecutionProvider'])
    metas={x.name:x for x in sess.get_inputs()};feed={}
    bases={'tokens':toks,'languages':langs,'durations':dur,'f0':f0,'breathiness':breath,'voicing':voice,'tension':tension,'gender':gender,'velocity':velocity,'spk_embed':spk}
    for name,x in bases.items():
        if name in metas:feed[name]=shape_input(metas[name],x)
    if 'depth' in metas:feed['depth']=shape_input(metas['depth'],np.array([a.depth]),scalar=True)
    if 'steps' in metas:feed['steps']=shape_input(metas['steps'],np.array([a.steps]),scalar=True)
    feed['noise_external']=shape_input(metas['noise_external'],noise.T[None,:,:] if len(metas['noise_external'].shape)==4 else noise.T)
    # Ensure exact [1,1,M,T] for patched input.
    feed['noise_external']=noise.T[None,None,:,:].astype(np.float32)
    print('M26 ORT inputs:')
    for k in sorted(feed):print(f'  {k:16s} {feed[k].dtype} {list(feed[k].shape)}')
    stage_onnx_names=[
        '/fs2/encoder/Mul_6_output_0','/fs2/GatherElements_output_0','/fs2/Add_2_output_0','/fs2/Add_3_output_0',
        '/fs2/Add_5_output_0','/fs2/Add_6_output_0','/fs2/Add_8_output_0','/fs2/Add_9_output_0'
    ]
    out_names=['mel','condition','aux_mel']+(stage_onnx_names if a.fs2_stages else [])
    vals=sess.run(out_names,feed)
    ort_mel,ort_cond,ort_aux=vals[:3]
    ort_mel=norm_out(ort_mel,mel);ort_aux=norm_out(ort_aux,mel);ort_cond=norm_out(ort_cond,hidden)
    ort_stages={}
    if a.fs2_stages:
        stage_keys=['encoder_txt','gathered','stretch','gru','pitch','variance','key_shift','speed']
        for key,val in zip(stage_keys,vals[3:]):ort_stages[key]=norm_out(val,hidden)
        ort_stages['speaker']=ort_cond
    if ort_mel.shape[0]!=T or ort_cond.shape[0]!=T or ort_aux.shape[0]!=T:raise RuntimeError(f'ORT T mismatch: mel={ort_mel.shape} cond={ort_cond.shape} aux={ort_aux.shape}, T={T}')

    native_mel=a.work/'native_mel.f32';native_cond=a.work/'native_condition.f32';native_aux=a.work/'native_aux_mel.f32'
    cmd=[str(a.cli),'infer',str(a.packed),'--tokens',str(a.work/'tokens.txt'),'--durations',str(a.work/'durations.txt'),'--f0',str(a.work/'f0.txt'),'--languages',str(a.work/'languages.txt'),'--speaker-emb',str(a.speaker_emb),'--breathiness',str(a.work/'breathiness.txt'),'--voicing',str(a.work/'voicing.txt'),'--tension',str(a.work/'tension.txt'),'--gender',str(a.work/'gender.txt'),'--velocity',str(a.work/'velocity.txt'),'--depth',str(a.depth),'--steps',str(a.steps),'--noise',str(a.work/'noise.f32'),'--dump-condition',str(native_cond),'--dump-aux-mel',str(native_aux),'--out',str(native_mel)]
    stage_dir=a.work/'native_fs2_stages'
    if a.fs2_stages:
        stage_dir.mkdir(parents=True,exist_ok=True);cmd += ['--dump-fs2-stages',str(stage_dir)]
    if a.threads:cmd += ['--threads',str(a.threads)]
    print(('M27' if a.fs2_stages else 'M26')+' native command:', ' '.join(cmd));p=subprocess.run(cmd,text=True,capture_output=True);print(p.stdout,end='');print(p.stderr,end='',file=sys.stderr)
    if p.returncode:raise SystemExit(p.returncode)
    nmel=np.fromfile(native_mel,np.float32).reshape(T,mel);naux=np.fromfile(native_aux,np.float32).reshape(T,mel);ncond=np.fromfile(native_cond,np.float32).reshape(T,hidden)
    report={'P':P,'T':T,'hidden':hidden,'mel_bins':mel,'condition':metrics(ort_cond,ncond),'aux_mel':metrics(ort_aux,naux),'mel':metrics(ort_mel,nmel)}
    first_bad=None
    if a.fs2_stages:
        report['fs2_stages']={}
        stage_keys=['encoder_txt','gathered','stretch','gru','pitch','variance','key_shift','speed','speaker']
        for key in stage_keys:
            rows=P if key=='encoder_txt' else T
            nv=np.fromfile(stage_dir/f'{key}.f32',np.float32).reshape(rows,hidden)
            q=metrics(ort_stages[key],nv);report['fs2_stages'][key]=q
            if first_bad is None and q['max_abs']>=a.stage_threshold:first_bad=key
    report_path=a.work/('m27_report.json' if a.fs2_stages else 'm26_report.json')
    report_path.write_text(json.dumps(report,indent=2)+'\n')
    if a.fs2_stages:
        print('M27 REAL FS2 STAGE PARITY')
        for k in ['encoder_txt','gathered','stretch','gru','pitch','variance','key_shift','speed','speaker']:
            q=report['fs2_stages'][k];print(f"  {k:12s} max_abs={q['max_abs']:.9g} rmse={q['rmse']:.9g} cosine={q['cosine']:.9g}")
        print('M27 first divergent stage:', first_bad or 'none', f'(threshold={a.stage_threshold:g})')
        print('M27 downstream context:')
    else:
        print('M26 REAL ONNX vs NATIVE PARITY')
    for k in ('condition','aux_mel','mel'):
        q=report[k];print(f"  {k:10s} max_abs={q['max_abs']:.9g} max_rel={q['max_rel']:.9g} rmse={q['rmse']:.9g} cosine={q['cosine']:.9g}")
    ok=report['condition']['max_abs']<5e-4 and report['aux_mel']['max_abs']<1e-3 and report['mel']['max_abs']<a.mel_threshold
    label='M27 parity' if a.fs2_stages else 'M26 parity'
    print(label+':', 'OK' if ok else 'FAIL', 'report=',report_path)
    if a.fs2_stages and not a.strict:
        raise SystemExit(0)
    raise SystemExit(0 if ok else 1)
if __name__=='__main__':main()
