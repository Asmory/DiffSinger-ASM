#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re, subprocess, sys, time, wave
from pathlib import Path
import numpy as np
TOOLS=Path(__file__).resolve().parent;sys.path.insert(0,str(TOOLS))
import run_real_voicebank_m30 as m30

def parse_ms(text, pat):
    m=re.search(pat,text);return float(m.group(1)) if m else float('nan')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--packed-acoustic',type=Path,required=True);ap.add_argument('--acoustic-cli',type=Path,required=True)
    ap.add_argument('--vocoder-bundle',type=Path,required=True);ap.add_argument('--vocoder-cli',type=Path,required=True);ap.add_argument('--vocoder-onnx',type=Path,required=True)
    ap.add_argument('--speaker-emb',type=Path,required=True);ap.add_argument('--language-id',type=int,default=4);ap.add_argument('--depth',type=float,default=.6);ap.add_argument('--steps',type=int,default=4);ap.add_argument('--workers',type=int,default=8);ap.add_argument('--rounds',type=int,default=5);ap.add_argument('--seed',type=int,default=3501);ap.add_argument('--work',type=Path,required=True)
    a=ap.parse_args();a.work.mkdir(parents=True,exist_ok=True)
    model=json.loads((a.packed_acoustic/'model.json').read_text());C=int(model['fs2']['hidden_size']);M=int(model['rf']['input_dim']);conf=m30.read_conf(a.packed_acoustic/'model.conf');sr=int(float(conf.get('sample_rate',44100)));hop=int(float(conf.get('hop_size',512)))
    names,tok_list=m30.pick_demo_phonemes(a.packed_acoustic);tok=np.asarray(tok_list,np.int64);dur=m30.make_durations(names,tok.size);T=int(dur.sum());langs=np.full(tok.size,a.language_id,np.int64);curves=m30.frame_curve(names,dur)
    if T!=48: raise RuntimeError(f'M35 first fixed-shape bundle expects T=48 demo, got {T}')
    for n,x in [('tokens',tok),('durations',dur),('languages',langs),*curves.items()]:m30.dump_txt(a.work/f'{n}.txt',x)
    curves['f0'].astype(np.float32).tofile(a.work/'f0.f32')
    rng=np.random.default_rng(a.seed);noise=rng.standard_normal((T,M),dtype=np.float32);noise.tofile(a.work/'noise.f32')
    melp=a.work/'native_mel.f32'
    cmd=[str(a.acoustic_cli),'infer',str(a.packed_acoustic),'--tokens',str(a.work/'tokens.txt'),'--durations',str(a.work/'durations.txt'),'--f0',str(a.work/'f0.txt'),'--languages',str(a.work/'languages.txt'),'--speaker-emb',str(a.speaker_emb),'--breathiness',str(a.work/'breathiness.txt'),'--voicing',str(a.work/'voicing.txt'),'--tension',str(a.work/'tension.txt'),'--gender',str(a.work/'gender.txt'),'--velocity',str(a.work/'velocity.txt'),'--depth',str(a.depth),'--steps',str(a.steps),'--noise',str(a.work/'noise.f32'),'--profile-stages','--out',str(melp)]
    p=subprocess.run(cmd,text=True,capture_output=True);print(p.stdout,end='');print(p.stderr,end='',file=sys.stderr)
    if p.returncode:raise SystemExit(p.returncode)
    acoustic_ms=parse_ms(p.stdout,r'time=([0-9.]+) ms');mel=np.fromfile(melp,np.float32).reshape(T,M)
    # ORT is golden-only and explicitly excluded from the runtime budget.
    import onnxruntime as ort
    so=ort.SessionOptions();so.intra_op_num_threads=4
    sess=ort.InferenceSession(str(a.vocoder_onnx),so,providers=['CPUExecutionProvider']);ins,_,mn,fn,wn=m30.find_io(sess)
    feed={mn:m30.adapt_mel(ins[mn],mel),fn:m30.adapt_f0(ins[fn],curves['f0'])}
    t0=time.perf_counter();gold=m30.flat_wave(sess.run([wn],feed)[0]);gold_ms=(time.perf_counter()-t0)*1000;gold.astype(np.float32).tofile(a.work/'golden_wave.f32')
    wavep=a.work/'native_wave.f32';wavp=a.work/'native_pipeline.wav'
    vcmd=[str(a.vocoder_cli),'infer',str(a.vocoder_bundle),'--mel',str(melp),'--f0',str(a.work/'f0.f32'),'--out',str(wavep),'--wav',str(wavp),'--golden',str(a.work/'golden_wave.f32'),'--workers',str(a.workers),'--rounds',str(a.rounds)]
    q=subprocess.run(vcmd,text=True,capture_output=True);print(q.stdout,end='');print(q.stderr,end='',file=sys.stderr)
    if q.returncode:raise SystemExit(q.returncode)
    voc_ms=parse_ms(q.stdout,r'median=([0-9.]+) ms');samples=gold.size;audio_ms=samples/sr*1000.;total=acoustic_ms+voc_ms;rtf=total/audio_ms
    report={'T':T,'samples':samples,'sample_rate':sr,'hop_size':hop,'steps':a.steps,'workers':a.workers,'timing_ms':{'acoustic_native':acoustic_ms,'vocoder_native':voc_ms,'pure_native_total':total,'audio':audio_ms,'ort_vocoder_golden_excluded':gold_ms},'rtf':rtf,'speed':audio_ms/total,'realtime':rtf<1.0,'engineering_target':rtf<=0.8,'phonemes':names,'tokens':tok.tolist(),'durations':dur.tolist()}
    (a.work/'native_e2e_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('\nDSASM PURE CPU/C/ASM END-TO-END')
    print(f'  acoustic={acoustic_ms:.3f} ms  vocoder={voc_ms:.3f} ms  pure-native total={total:.3f} ms')
    print(f'  audio={audio_ms:.3f} ms  RTF={rtf:.3f} speed={audio_ms/total:.3f}x')
    print(f'  realtime RTF<1: {"CROSSED" if rtf<1 else "NOT YET"}')
    print(f'  engineering RTF<=0.8: {"CROSSED" if rtf<=.8 else "NOT YET"}')
    print('  WAV=',wavp);print('  report=',a.work/'native_e2e_report.json')
if __name__=='__main__':main()
