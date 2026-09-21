#!/usr/bin/env python3
"""M30 real voicebank end-to-end smoke/parity.

Native packed acoustic -> ONNX NSF-HiFiGAN -> WAV, with an original ONNX
acoustic reference using the exact same inputs and external noise.  The
vocoder is intentionally kept in ORT for M30: this establishes a trustworthy
waveform golden reference before a native vocoder port.
"""
from __future__ import annotations
import argparse, json, math, re, subprocess, sys, time, wave
from pathlib import Path
import numpy as np

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import validate_real_onnx_m27 as v27


def read_conf(path: Path):
    d={}
    if not path.exists(): return d
    for line in path.read_text(errors='ignore').splitlines():
        line=line.strip()
        if line and not line.startswith('#') and '=' in line:
            k,v=line.split('=',1); d[k.strip()]=v.strip()
    return d


def pick_demo_phonemes(packed: Path):
    p=packed/'phonemes.json'
    if not p.exists():
        return None, [4,21,23,24,25,4]
    mp=json.loads(p.read_text(encoding='utf-8'))
    # Prefer a simple, sustained Chinese syllable sequence that is useful for
    # an audible vocoder smoke.  Inventories differ, so try several spellings.
    candidates=[
        ['SP','zh/l','zh/a','zh/l','zh/a','SP'],
        ['SP','zh/m','zh/a','zh/m','zh/a','SP'],
        ['SP','zh/l','zh/aa','zh/l','zh/aa','SP'],
        ['SP','zh/a','zh/a','SP'],
        ['SP','zh/aa','zh/aa','SP'],
    ]
    for seq in candidates:
        if all(x in mp for x in seq):
            return seq,[int(mp[x]) for x in seq]
    # Fall back to valid zh tokens selected the same way as the real parity
    # harness.  This may sound like nonsense, but is still a real voicebank
    # synthesis path rather than a synthetic model.
    ids=[x for x in v27.read_vocab(packed) if int(x)>0]
    inv={int(v):k for k,v in mp.items() if isinstance(v,int)}
    return [inv.get(int(x),f'id:{int(x)}') for x in ids], [int(x) for x in ids]


def make_durations(names, n):
    if names is None: return np.array([5 + (i%4)*2 for i in range(n)], np.int64)
    out=[]
    for name in names:
        base=name.split('/')[-1].upper()
        if base in ('SP','AP','EP','GS'): out.append(4)
        elif base in ('A','AA','AE','AH','AO','AX','E','EH','ER','I','IH','IY','O','OW','U','UH','UW'): out.append(12)
        else: out.append(8)
    return np.asarray(out,np.int64)


def frame_curve(names, dur):
    T=int(dur.sum()); f0=np.zeros(T,np.float32); pos=0
    for i,d in enumerate(dur.tolist()):
        name=(names[i] if names else '').split('/')[-1].upper()
        voiced=name not in ('SP','AP','EP','GS')
        if voiced:
            t=np.arange(d,dtype=np.float32)
            # A4-ish note with light vibrato; deterministic and within this
            # community vocoder's ordinary pitch range.
            f0[pos:pos+d]=220.0 + 4.5*np.sin(2*np.pi*t/max(1,d)*1.5)
        pos += d
    if not np.any(f0>0):
        tt=np.arange(T,dtype=np.float32); f0[:]=220.0+4.5*np.sin(tt*0.21)
    tt=np.arange(T,dtype=np.float32)
    curves={
        'f0':f0,
        'breathiness':(0.12+0.02*np.sin(tt*.11)).astype(np.float32),
        'voicing':(0.78+0.03*np.cos(tt*.07)).astype(np.float32),
        'tension':(0.10+0.015*np.sin(tt*.13)).astype(np.float32),
        'gender':np.zeros(T,np.float32),
        'velocity':np.ones(T,np.float32),
    }
    return curves


def dump_txt(path:Path, arr):
    a=np.asarray(arr).reshape(-1)
    path.write_text(' '.join(f'{float(x):.9g}' if np.issubdtype(a.dtype,np.floating) else str(int(x)) for x in a)+'\n')


def adapt_mel(meta, mel):
    x=np.asarray(mel,np.float32); T,M=x.shape; r=len(meta.shape or [])
    if r==3:
        sh=meta.shape
        # Standard DiffSinger/OpenVPI deployment is [B,T,M].
        if len(sh)>=3 and isinstance(sh[1],int) and sh[1]==M and not (isinstance(sh[2],int) and sh[2]==M):
            return x.T[None,:,:]
        return x[None,:,:]
    if r==2:
        sh=meta.shape
        if len(sh)>=2 and isinstance(sh[0],int) and sh[0]==M: return x.T
        return x
    raise RuntimeError(f'unsupported vocoder mel input rank/shape: {meta.shape}')


def adapt_f0(meta, f0):
    x=np.asarray(f0,np.float32).reshape(-1); r=len(meta.shape or [])
    if r==1:return x
    if r==2:return x[None,:]
    if r==3:
        sh=meta.shape
        if len(sh)>=3 and isinstance(sh[2],int) and sh[2]==1:return x[None,:,None]
        return x[None,None,:]
    raise RuntimeError(f'unsupported vocoder f0 input rank/shape: {meta.shape}')


def flat_wave(x):
    x=np.asarray(x,np.float32)
    while x.ndim>1 and x.shape[0]==1:x=x[0]
    while x.ndim>1 and x.shape[0]==1:x=x[0]
    return x.reshape(-1).astype(np.float32)


def write_wav(path:Path, x, sr:int):
    y=np.asarray(x,np.float32).reshape(-1)
    finite=np.isfinite(y)
    if not finite.all(): raise RuntimeError(f'non-finite waveform samples: {(~finite).sum()}')
    clipped=int(np.count_nonzero(np.abs(y)>1.0))
    pcm=np.round(np.clip(y,-1.0,1.0)*32767.0).astype('<i2')
    with wave.open(str(path),'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm.tobytes())
    return clipped,float(np.max(np.abs(y),initial=0.0))


def find_io(sess):
    ins={x.name:x for x in sess.get_inputs()}; outs={x.name:x for x in sess.get_outputs()}
    mel_name='mel' if 'mel' in ins else next((n for n in ins if 'mel' in n.lower()),None)
    f0_name='f0' if 'f0' in ins else next((n for n in ins if 'f0' in n.lower() or 'pitch' in n.lower()),None)
    wav_name='waveform' if 'waveform' in outs else next((n for n in outs if 'wave' in n.lower() or 'audio' in n.lower()),None)
    if not mel_name or not f0_name or not wav_name:
        raise RuntimeError(f'unsupported vocoder I/O: inputs={list(ins)} outputs={list(outs)}')
    extra=[n for n in ins if n not in (mel_name,f0_name)]
    if extra: raise RuntimeError(f'M30 vocoder has unsupported extra inputs: {extra}; all inputs={list(ins)}')
    return ins,outs,mel_name,f0_name,wav_name


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--acoustic-onnx',type=Path,required=True)
    ap.add_argument('--vocoder-onnx',type=Path,required=True)
    ap.add_argument('--packed',type=Path,required=True)
    ap.add_argument('--cli',type=Path,default=Path('build/dsasm-acoustic'))
    ap.add_argument('--speaker-emb',type=Path,required=True)
    ap.add_argument('--language-id',type=int,default=4); ap.add_argument('--depth',type=float,default=.6); ap.add_argument('--steps',type=int,default=20)
    ap.add_argument('--seed',type=int,default=3001); ap.add_argument('--threads',type=int,default=0)
    ap.add_argument('--work',type=Path,default=Path('build/m30_real'))
    a=ap.parse_args(); a.work.mkdir(parents=True,exist_ok=True)
    import onnxruntime as ort

    model=json.loads((a.packed/'model.json').read_text()); C=int(model['fs2']['hidden_size']); M=int(model['rf']['input_dim'])
    conf=read_conf(a.packed/'model.conf'); sr=int(float(conf.get('sample_rate',44100))); hop=int(float(conf.get('hop_size',512)))
    names,tok_list=pick_demo_phonemes(a.packed); tok=np.asarray(tok_list,np.int64); P=tok.size
    dur=make_durations(names,P); T=int(dur.sum()); langs=np.full(P,a.language_id,np.int64); curves=frame_curve(names,dur)
    sp=np.fromfile(a.speaker_emb,np.float32)
    if sp.size!=C: raise RuntimeError(f'speaker embedding has {sp.size} floats, expected {C}')
    spk=np.broadcast_to(sp[None,:],(T,C)).copy()
    rng=np.random.default_rng(a.seed); noise=rng.standard_normal((T,M),dtype=np.float32); noise.tofile(a.work/'noise.f32')
    for n,x in [('tokens',tok),('durations',dur),('languages',langs),*curves.items()]:dump_txt(a.work/f'{n}.txt',x)
    print('M30 demo phonemes:', names); print('M30 tokens:',tok.tolist()); print('M30 durations:',dur.tolist(),f'T={T} audio={T*hop/sr:.3f}s')

    # Original acoustic reference with deterministic external noise.
    patched=a.work/'acoustic_m30_patched.onnx'; v27.patch_onnx(a.acoustic_onnx,patched,C,M,False)
    so=ort.SessionOptions(); so.intra_op_num_threads=max(1,a.threads) if a.threads else 0
    asess=ort.InferenceSession(str(patched),so,providers=['CPUExecutionProvider']); metas={x.name:x for x in asess.get_inputs()}; feed={}
    bases={'tokens':tok,'languages':langs,'durations':dur,'f0':curves['f0'],'breathiness':curves['breathiness'],'voicing':curves['voicing'],'tension':curves['tension'],'gender':curves['gender'],'velocity':curves['velocity'],'spk_embed':spk}
    for n,x in bases.items():
        if n in metas:feed[n]=v27.shape_input(metas[n],x)
    if 'depth' in metas:feed['depth']=v27.shape_input(metas['depth'],np.array([a.depth]),scalar=True)
    if 'steps' in metas:feed['steps']=v27.shape_input(metas['steps'],np.array([a.steps]),scalar=True)
    feed['noise_external']=noise.T[None,None,:,:].astype(np.float32)
    t0=time.perf_counter(); ref_mel=v27.norm_out(asess.run(['mel'],feed)[0],M); ref_ms=(time.perf_counter()-t0)*1000
    ref_mel.astype(np.float32).tofile(a.work/'onnx_mel.f32')

    # Native acoustic.
    native_mel_p=a.work/'native_mel.f32'
    cmd=[str(a.cli),'infer',str(a.packed),'--tokens',str(a.work/'tokens.txt'),'--durations',str(a.work/'durations.txt'),'--f0',str(a.work/'f0.txt'),'--languages',str(a.work/'languages.txt'),'--speaker-emb',str(a.speaker_emb),'--breathiness',str(a.work/'breathiness.txt'),'--voicing',str(a.work/'voicing.txt'),'--tension',str(a.work/'tension.txt'),'--gender',str(a.work/'gender.txt'),'--velocity',str(a.work/'velocity.txt'),'--depth',str(a.depth),'--steps',str(a.steps),'--noise',str(a.work/'noise.f32'),'--out',str(native_mel_p)]
    if a.threads:cmd += ['--threads',str(a.threads)]
    p=subprocess.run(cmd,text=True,capture_output=True); print(p.stdout,end=''); print(p.stderr,end='',file=sys.stderr)
    if p.returncode: raise SystemExit(p.returncode)
    native_mel=np.fromfile(native_mel_p,np.float32).reshape(T,M)
    mm=re.search(r'time=([0-9.]+) ms',p.stdout); native_ms=float(mm.group(1)) if mm else float('nan')
    mel_metric=v27.metrics(ref_mel,native_mel)

    # Standard OpenVPI NSF-HiFiGAN deployment interface is mel + f0 -> waveform.
    vsess=ort.InferenceSession(str(a.vocoder_onnx),so,providers=['CPUExecutionProvider'])
    ins,outs,mel_name,f0_name,wav_name=find_io(vsess)
    print('M30 vocoder inputs:')
    for x in vsess.get_inputs():print(f'  {x.name:16s} {x.type:16s} {x.shape}')
    print('M30 vocoder outputs:')
    for x in vsess.get_outputs():print(f'  {x.name:16s} {x.type:16s} {x.shape}')
    def voc(mel):
        fd={mel_name:adapt_mel(ins[mel_name],mel),f0_name:adapt_f0(ins[f0_name],curves['f0'])}
        return flat_wave(vsess.run([wav_name],fd)[0])
    _=voc(native_mel)  # warmup
    ts=[]; native_wave=None
    for _i in range(3):
        q=time.perf_counter(); native_wave=voc(native_mel); ts.append((time.perf_counter()-q)*1000)
    voc_ms=float(np.median(ts)); ref_wave=voc(ref_mel)
    if native_wave.size!=ref_wave.size:raise RuntimeError(f'wave length mismatch native={native_wave.size} ref={ref_wave.size}')
    wave_metric=v27.metrics(ref_wave,native_wave)
    native_wave.tofile(a.work/'native_waveform.f32'); ref_wave.tofile(a.work/'onnx_waveform.f32')
    nc,npk=write_wav(a.work/'native_pipeline.wav',native_wave,sr); rc,rpk=write_wav(a.work/'onnx_reference.wav',ref_wave,sr)
    audio_ms=native_wave.size/sr*1000.0; total_ms=native_ms+voc_ms
    report={'P':int(P),'T':int(T),'sample_rate':sr,'hop_size':hop,'wave_samples':int(native_wave.size),'mel':mel_metric,'waveform':wave_metric,'timing_ms':{'native_acoustic':native_ms,'onnx_acoustic_reference':ref_ms,'vocoder_median':voc_ms,'native_plus_vocoder':total_ms,'audio':audio_ms},'wav':{'native_peak':npk,'native_clipped_samples':nc,'reference_peak':rpk,'reference_clipped_samples':rc},'phonemes':names,'tokens':tok.tolist(),'durations':dur.tolist()}
    (a.work/'m30_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('\nM30 REAL VOICEBANK -> WAVEFORM')
    print(f"  mel      max_abs={mel_metric['max_abs']:.9g} rmse={mel_metric['rmse']:.9g} cosine={mel_metric['cosine']:.9g}")
    print(f"  waveform max_abs={wave_metric['max_abs']:.9g} rmse={wave_metric['rmse']:.9g} cosine={wave_metric['cosine']:.9g}")
    print(f'  waveform samples={native_wave.size} duration={audio_ms/1000:.3f}s sr={sr} hop={hop}')
    print(f'  native acoustic={native_ms:.3f} ms  vocoder median={voc_ms:.3f} ms  total={total_ms:.3f} ms  total RTF={total_ms/audio_ms:.3f} speed={audio_ms/total_ms:.3f}x')
    print(f'  native peak={npk:.6g} clipped={nc}  reference peak={rpk:.6g} clipped={rc}')
    print('  native WAV   =',a.work/'native_pipeline.wav')
    print('  reference WAV=',a.work/'onnx_reference.wav')
    print('  report       =',a.work/'m30_report.json')
    ok=mel_metric['max_abs']<5e-4 and wave_metric['cosine']>0.999
    print('M30 end-to-end parity:', 'OK' if ok else 'WARN')

if __name__=='__main__': main()
