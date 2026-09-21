#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, re, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np


def run(cmd, env=None):
    p=subprocess.run(cmd,text=True,capture_output=True,env=env)
    print(p.stdout,end=''); print(p.stderr,end='',file=sys.stderr)
    if p.returncode: raise SystemExit(p.returncode)
    return p.stdout+p.stderr

def parse_ms(text, pat):
    m=re.search(pat,text); return float(m.group(1)) if m else float('nan')

def pinned(cpus, cmd):
    return ['taskset','-c',cpus,*map(str,cmd)] if cpus else list(map(str,cmd))

def load_helpers(project:Path):
    sys.path.insert(0,str(project/'tools'))
    import run_real_voicebank_m30 as m30
    return m30

def build_fixture(a):
    m30=load_helpers(a.project)
    model=json.loads((a.packed_acoustic/'model.json').read_text())
    conf=m30.read_conf(a.packed_acoustic/'model.conf')
    C=int(model['fs2']['hidden_size']); M=int(model['rf']['input_dim'])
    sr=int(float(conf.get('sample_rate',44100))); hop=int(float(conf.get('hop_size',512)))
    base_names,base_tok=m30.pick_demo_phonemes(a.packed_acoustic)
    base_names = base_names if base_names is not None else [f'id:{x}' for x in base_tok]
    base_dur=m30.make_durations(base_names,len(base_tok))
    base_T=int(base_dur.sum())
    if a.frames % base_T:
        raise SystemExit(f'frames={a.frames} must be a multiple of base utterance T={base_T}')
    n=a.frames//base_T
    names=base_names*n
    tok=np.tile(np.asarray(base_tok,np.int64),n)
    dur=np.tile(base_dur,n)
    T=int(dur.sum()); P=int(tok.size)
    langs=np.full(P,a.language_id,np.int64)
    curves=m30.frame_curve(names,dur)
    a.fixture.mkdir(parents=True,exist_ok=True)
    for k,x in [('tokens',tok),('durations',dur),('languages',langs),*curves.items()]:
        m30.dump_txt(a.fixture/f'{k}.txt',x)
    curves['f0'].astype(np.float32).tofile(a.fixture/'f0.f32')
    rng=np.random.default_rng(a.seed+T)
    rng.standard_normal((T,M),dtype=np.float32).tofile(a.fixture/'noise.f32')
    meta=dict(P=P,T=T,M=M,sample_rate=sr,hop_size=hop,audio_ms=T*hop/sr*1000.0,repeats=n,
              phonemes=names,tokens=tok.tolist(),durations=dur.tolist())
    (a.fixture/'meta.json').write_text(json.dumps(meta,indent=2)+'\n')
    return meta,curves,m30

def acoustic_cmd(a,out):
    f=a.fixture
    cmd=[a.acoustic_cli,'infer',a.packed_acoustic,
         '--tokens',f/'tokens.txt','--durations',f/'durations.txt','--f0',f/'f0.txt','--languages',f/'languages.txt',
         '--speaker-emb',a.speaker_emb,'--breathiness',f/'breathiness.txt','--voicing',f/'voicing.txt',
         '--tension',f/'tension.txt','--gender',f/'gender.txt','--velocity',f/'velocity.txt',
         '--depth',str(a.depth),'--steps',str(a.steps),'--noise',f/'noise.f32','--threads',str(a.threads),
         '--profile-stages','--out',out]
    return pinned(a.cpus,cmd)

def prepare(a):
    meta,curves,m30=build_fixture(a)
    print(f"M53 long fixture: P={meta['P']} T={meta['T']} audio={meta['audio_ms']/1000:.3f}s repeats={meta['repeats']}")
    ref=a.fixture/'reference_mel.f32'
    txt=run(acoustic_cmd(a,ref)); ams=parse_ms(txt,r'time=([0-9.]+) ms')
    mel=np.fromfile(ref,np.float32).reshape(meta['T'],meta['M'])
    print('M53 ORT vocoder golden ONCE (outside measured E2E)')
    import onnxruntime as ort
    so=ort.SessionOptions(); so.intra_op_num_threads=max(1,a.threads)
    sess=ort.InferenceSession(str(a.vocoder_onnx),so,providers=['CPUExecutionProvider'])
    ins,_,mn,fn,wn=m30.find_io(sess)
    feed={mn:m30.adapt_mel(ins[mn],mel),fn:m30.adapt_f0(ins[fn],curves['f0'])}
    t0=time.perf_counter(); wave=m30.flat_wave(sess.run([wn],feed)[0]); oms=(time.perf_counter()-t0)*1000
    wave.astype(np.float32).tofile(a.fixture/'golden_wave.f32')
    meta.update(samples=int(wave.size),audio_ms=float(wave.size/meta['sample_rate']*1000),reference_acoustic_ms=ams,ort_golden_ms=oms)
    (a.fixture/'meta.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(f"M53 prepared: acoustic_ref={ams:.3f}ms ORT_golden={oms:.3f}ms samples={wave.size} audio={meta['audio_ms']/1000:.3f}s")

def measured(a):
    meta=json.loads((a.fixture/'meta.json').read_text())
    a.work.mkdir(parents=True,exist_ok=True)
    mel=a.work/'native_mel.f32'
    txt=run(acoustic_cmd(a,mel)); ams=parse_ms(txt,r'time=([0-9.]+) ms')
    ref=np.fromfile(a.fixture/'reference_mel.f32',np.float32); got=np.fromfile(mel,np.float32)
    if ref.shape!=got.shape: raise SystemExit(f'acoustic shape mismatch {got.shape} != {ref.shape}')
    bitdiff=int(np.count_nonzero(ref.view(np.uint32)!=got.view(np.uint32)))
    maxabs=float(np.abs(ref-got).max(initial=0))
    print(f'M53 acoustic determinism: max_abs={maxabs:.9g} bitdiff={bitdiff}')
    if bitdiff: raise SystemExit('acoustic output differs from prepared reference')
    env=os.environ.copy(); env.update(DSASM_GOLDEN_COS='0.999',DSASM_GOLDEN_SNR='25')
    out=a.work/'native_wave.f32'
    vcmd=[a.vocoder_cli,'infer',a.vocoder_bundle,'--mel',mel,'--f0',a.fixture/'f0.f32','--out',out,
          '--golden',a.fixture/'golden_wave.f32','--workers',str(a.threads),'--rounds','1']
    vtxt=run(pinned(a.cpus,vcmd),env=env); vms=parse_ms(vtxt,r'median=([0-9.]+) ms')
    total=ams+vms; audio=float(meta['audio_ms']); rtf=total/audio
    print(f'M53 LONG E2E: T={meta["T"]} audio={audio/1000:.3f}s acoustic={ams:.3f} vocoder={vms:.3f} total={total:.3f}ms RTF={rtf:.3f} speed={audio/total:.3f}x')
    rep=dict(T=meta['T'],audio_ms=audio,acoustic_ms=ams,vocoder_ms=vms,total_ms=total,rtf=rtf,
             cpus=a.cpus,threads=a.threads,bitdiff=bitdiff,max_abs=maxabs)
    (a.work/'result.json').write_text(json.dumps(rep,indent=2)+'\n')

def main():
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['prepare','run'])
    p.add_argument('--project',type=Path,required=True); p.add_argument('--packed-acoustic',type=Path,required=True)
    p.add_argument('--acoustic-cli',type=Path,required=True); p.add_argument('--vocoder-bundle',type=Path,required=True)
    p.add_argument('--vocoder-cli',type=Path,required=True); p.add_argument('--vocoder-onnx',type=Path,required=True)
    p.add_argument('--speaker-emb',type=Path,required=True); p.add_argument('--fixture',type=Path,required=True)
    p.add_argument('--work',type=Path,default=Path('build/m53/run')); p.add_argument('--frames',type=int,required=True)
    p.add_argument('--cpus',default='0,2,4,6'); p.add_argument('--threads',type=int,default=4)
    p.add_argument('--language-id',type=int,default=4); p.add_argument('--depth',type=float,default=.6); p.add_argument('--steps',type=int,default=4)
    p.add_argument('--seed',type=int,default=5301)
    a=p.parse_args(); a.project=a.project.resolve()
    if a.action=='prepare': prepare(a)
    else: measured(a)
if __name__=='__main__': main()
