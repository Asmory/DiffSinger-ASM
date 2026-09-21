#!/usr/bin/env python3
"""M29 real deployment language-mask fix validator.

Runs M27 full-stage parity with the current native runtime, then independently
compares the pre-Transformer FS2 front end against the original ONNX graph:
raw token embedding, duration log/linear, cross-lingual language masking,
language embedding, scaled token embedding and final encoder input.
"""
from __future__ import annotations
import argparse, json, math, subprocess, sys
from pathlib import Path
import numpy as np

TOOLS=Path(__file__).resolve().parent
sys.path.insert(0,str(TOOLS))
import validate_real_onnx_m27 as v27
import validate_real_onnx_m28 as v28


def read_txt(p,dtype):
    return np.fromstring(Path(p).read_text(),sep=' ',dtype=dtype)

def metric(a,b):
    return v27.metrics(np.asarray(a),np.asarray(b))

def squeeze_batch(x):
    x=np.asarray(x)
    while x.ndim>0 and x.shape[0]==1:
        x=x[0]
    return x

def run_m27(a,work):
    cmd=[sys.executable,str(TOOLS/'validate_real_onnx_m27.py'),'--fs2-stages',
         '--onnx',str(a.onnx),'--packed',str(a.packed),'--cli',str(a.cli),
         '--speaker-emb',str(a.speaker_emb),'--language-id',str(a.language_id),
         '--depth',str(a.depth),'--steps',str(a.steps),'--work',str(work)]
    if a.threads:cmd += ['--threads',str(a.threads)]
    print('===== M29 FIXED NATIVE FULL-STAGE PARITY =====')
    p=subprocess.run(cmd,text=True)
    if p.returncode: raise SystemExit(p.returncode)
    return json.loads((work/'m27_report.json').read_text())

def front_parity(a,work):
    import onnx
    import onnxruntime as ort
    from onnx import helper,TensorProto

    model=json.loads((a.packed/'model.json').read_text())
    fs=model['fs2']; C=int(fs['hidden_size']); V=int(fs['vocab_size']); NL=int(fs['num_languages'])
    toks=read_txt(work/'tokens.txt',np.int64);dur=read_txt(work/'durations.txt',np.int64);langs=read_txt(work/'languages.txt',np.int64)
    P=toks.size;T=int(dur.sum())

    emb=v28.load_sec(a.packed,fs,'token_embedding').reshape(V,C)
    dw=v28.load_sec(a.packed,fs,'dur_weight').reshape(C)
    db=v28.load_sec(a.packed,fs,'dur_bias').reshape(C)
    le=v28.load_sec(a.packed,fs,'language_embedding').reshape(NL,C)
    if 'language_token_mask' not in fs['sections']:
        raise RuntimeError('M29 packed model has no language_token_mask section')
    lmask=v28.load_sec(a.packed,fs,'language_token_mask').reshape(V)

    dur_eff=dur*(toks>0)
    ref={}
    ref['txt_raw']=emb[toks]
    ref['dur_log']=np.log1p(dur_eff.astype(np.float32))
    ref['dur_matmul']=ref['dur_log'][:,None]*dw[None,:]
    ref['dur_add']=ref['dur_matmul']+db[None,:]
    ref['lang_ids']=(langs*(lmask[toks]>0.5).astype(np.int64))
    ref['lang_embed']=le[ref['lang_ids']]
    ref['txt_scaled']=np.sqrt(np.float32(C))*ref['txt_raw']
    ref['encoder_add']=ref['txt_scaled']+ref['dur_add']+ref['lang_embed']
    ref['encoder_input']=ref['encoder_add']*(toks>0)[:,None]

    m=onnx.load(str(a.onnx),load_external_data=True)
    del m.graph.output[:]
    outs=[
      ('/fs2/txt_embed/Gather_output_0',TensorProto.FLOAT,[1,'P',C],'txt_raw'),
      ('/fs2/Log_output_0',TensorProto.FLOAT,[1,'P'],'dur_log'),
      ('/fs2/dur_embed/MatMul_output_0',TensorProto.FLOAT,[1,'P',C],'dur_matmul'),
      ('/fs2/dur_embed/Add_output_0',TensorProto.FLOAT,[1,'P',C],'dur_add'),
      ('/fs2/Mul_2_output_0',TensorProto.INT64,[1,'P'],'lang_ids'),
      ('/fs2/lang_embed/Gather_output_0',TensorProto.FLOAT,[1,'P',C],'lang_embed'),
      ('/fs2/encoder/Mul_output_0',TensorProto.FLOAT,[1,'P',C],'txt_scaled'),
      ('/fs2/encoder/Add_output_0',TensorProto.FLOAT,[1,'P',C],'encoder_add'),
      ('/fs2/encoder/Mul_1_output_0',TensorProto.FLOAT,[1,'P',C],'encoder_input'),
    ]
    produced={o for n in m.graph.node for o in n.output}
    for name,tp,sh,_ in outs:
        if name not in produced:raise RuntimeError(f'M29 expected ONNX output missing: {name}')
        m.graph.output.append(helper.make_tensor_value_info(name,tp,sh))
    pp=work/'m29_front.onnx';onnx.save(m,str(pp))
    sess=ort.InferenceSession(str(pp),providers=['CPUExecutionProvider']);metas={x.name:x for x in sess.get_inputs()}
    f0=read_txt(work/'f0.txt',np.float32);breath=read_txt(work/'breathiness.txt',np.float32);voice=read_txt(work/'voicing.txt',np.float32);tension=read_txt(work/'tension.txt',np.float32);gender=read_txt(work/'gender.txt',np.float32);velocity=read_txt(work/'velocity.txt',np.float32)
    sp=np.fromfile(a.speaker_emb,np.float32);spk=np.broadcast_to(sp[None,:],(T,C)).copy()
    bases={'tokens':toks,'languages':langs,'durations':dur,'f0':f0,'breathiness':breath,'voicing':voice,'tension':tension,'gender':gender,'velocity':velocity,'spk_embed':spk}
    feed={}
    for name,x in bases.items():
        if name in metas:feed[name]=v27.shape_input(metas[name],x)
    if 'depth' in metas:feed['depth']=v27.shape_input(metas['depth'],np.array([a.depth]),scalar=True)
    if 'steps' in metas:feed['steps']=v27.shape_input(metas['steps'],np.array([a.steps]),scalar=True)
    vals=sess.run([x[0] for x in outs],feed)

    rep={}
    print('\nM29 REAL FS2 FRONT-END PARITY')
    for (_,_,_,key),val in zip(outs,vals):
        ov=squeeze_batch(val)
        if key=='lang_ids':
            exact=bool(np.array_equal(ov.astype(np.int64),ref[key]))
            q={'exact':exact,'onnx':ov.astype(np.int64).tolist(),'packed':ref[key].astype(np.int64).tolist()}
            print(f"  {key:13s} exact={exact} onnx={q['onnx']} packed={q['packed']}")
        else:
            q=metric(ov.astype(np.float32),ref[key].astype(np.float32))
            print(f"  {key:13s} max_abs={q['max_abs']:.9g} rmse={q['rmse']:.9g} cosine={q['cosine']:.9g}")
        rep[key]=q
    cross=np.flatnonzero(lmask>0.5).astype(int).tolist()
    print(f'  cross-lingual mask: {len(cross)}/{V} ids={cross[:40]}' + (' ...' if len(cross)>40 else ''))
    return rep,cross

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--onnx',type=Path,required=True);ap.add_argument('--packed',type=Path,required=True);ap.add_argument('--cli',type=Path,default=Path('build/dsasm-acoustic'));ap.add_argument('--speaker-emb',type=Path,required=True);ap.add_argument('--work',type=Path,default=Path('build/m29_real'));ap.add_argument('--language-id',type=int,default=4);ap.add_argument('--depth',type=float,default=.6);ap.add_argument('--steps',type=int,default=20);ap.add_argument('--threads',type=int,default=0)
    a=ap.parse_args();a.work.mkdir(parents=True,exist_ok=True)
    stage=run_m27(a,a.work/'fixed')
    front,cross=front_parity(a,a.work/'fixed')
    enc=stage['fs2_stages']['encoder_txt']['max_abs'];cond=stage['condition']['max_abs'];aux=stage['aux_mel']['max_abs'];mel=stage['mel']['max_abs']
    front_bad=max((v.get('max_abs',0.0) for v in front.values() if isinstance(v,dict)),default=0.0)
    lang_ok=front['lang_ids']['exact']
    print('\nM29 diagnosis:')
    print(f'  front max_abs={front_bad:.9g} masked-language-ids exact={lang_ok}')
    print(f'  encoder_txt={enc:.9g} condition={cond:.9g} aux_mel={aux:.9g} mel={mel:.9g}')
    if front_bad<2e-5 and lang_ok and enc<5e-4:
        verdict='cross-lingual language-mask fix confirmed; Transformer/front-end parity restored'
    elif front_bad<2e-5 and lang_ok:
        verdict='language-mask/front-end fixed; remaining divergence starts inside Transformer'
    else:
        verdict='front-end still diverges; inspect the first non-exact M29 front stage'
    print('  verdict:',verdict)
    out={'front':front,'cross_lingual_token_idx':cross,'stage':stage,'verdict':verdict}
    (a.work/'m29_report.json').write_text(json.dumps(out,indent=2)+'\n')
    print('M29 report=',a.work/'m29_report.json')
if __name__=='__main__':main()
