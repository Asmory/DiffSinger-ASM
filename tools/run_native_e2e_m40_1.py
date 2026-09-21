#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, re, subprocess, sys, time
from pathlib import Path
import numpy as np
TOOLS=Path(__file__).resolve().parent;sys.path.insert(0,str(TOOLS))
import run_real_voicebank_m30 as m30

def parse_ms(text, pat):
    m=re.search(pat,text);return float(m.group(1)) if m else float('nan')

def parse_quality(text):
    m=re.search(r'parity max_abs=([0-9.eE+-]+) rmse=([0-9.eE+-]+) cosine=([0-9.eE+-]+) SNR=([0-9.eE+-]+)',text)
    if not m:return {}
    return dict(max_abs=float(m.group(1)),rmse=float(m.group(2)),cosine=float(m.group(3)),snr_db=float(m.group(4)))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--packed-acoustic',type=Path,required=True);ap.add_argument('--acoustic-cli',type=Path,required=True)
    ap.add_argument('--vocoder-bundle',type=Path,required=True);ap.add_argument('--vocoder-cli',type=Path,required=True);ap.add_argument('--vocoder-onnx',type=Path,required=True)
    ap.add_argument('--speaker-emb',type=Path,required=True);ap.add_argument('--language-id',type=int,default=4);ap.add_argument('--depth',type=float,default=.6);ap.add_argument('--steps',type=int,default=4);ap.add_argument('--workers',type=int,default=4);ap.add_argument('--rounds',type=int,default=9);ap.add_argument('--seed',type=int,default=4001);ap.add_argument('--modes',default='k11,k117');ap.add_argument('--work',type=Path,required=True)
    a=ap.parse_args();a.work.mkdir(parents=True,exist_ok=True)
    model=json.loads((a.packed_acoustic/'model.json').read_text());M=int(model['rf']['input_dim']);conf=m30.read_conf(a.packed_acoustic/'model.conf');sr=int(float(conf.get('sample_rate',44100)));hop=int(float(conf.get('hop_size',512)))
    names,tok_list=m30.pick_demo_phonemes(a.packed_acoustic);tok=np.asarray(tok_list,np.int64);dur=m30.make_durations(names,tok.size);T=int(dur.sum());langs=np.full(tok.size,a.language_id,np.int64);curves=m30.frame_curve(names,dur)
    if T!=48:raise RuntimeError(f'M40.1 fixed-shape bundle expects T=48, got {T}')
    for n,x in [('tokens',tok),('durations',dur),('languages',langs),*curves.items()]:m30.dump_txt(a.work/f'{n}.txt',x)
    curves['f0'].astype(np.float32).tofile(a.work/'f0.f32')
    rng=np.random.default_rng(a.seed);rng.standard_normal((T,M),dtype=np.float32).tofile(a.work/'noise.f32')
    melp=a.work/'native_mel.f32'
    cmd=[str(a.acoustic_cli),'infer',str(a.packed_acoustic),'--tokens',str(a.work/'tokens.txt'),'--durations',str(a.work/'durations.txt'),'--f0',str(a.work/'f0.txt'),'--languages',str(a.work/'languages.txt'),'--speaker-emb',str(a.speaker_emb),'--breathiness',str(a.work/'breathiness.txt'),'--voicing',str(a.work/'voicing.txt'),'--tension',str(a.work/'tension.txt'),'--gender',str(a.work/'gender.txt'),'--velocity',str(a.work/'velocity.txt'),'--depth',str(a.depth),'--steps',str(a.steps),'--noise',str(a.work/'noise.f32'),'--profile-stages','--out',str(melp)]
    p=subprocess.run(cmd,text=True,capture_output=True);print(p.stdout,end='');print(p.stderr,end='',file=sys.stderr)
    if p.returncode:raise SystemExit(p.returncode)
    acoustic_ms=parse_ms(p.stdout,r'time=([0-9.]+) ms');mel=np.fromfile(melp,np.float32).reshape(T,-1)
    import onnxruntime as ort
    so=ort.SessionOptions();so.intra_op_num_threads=4
    sess=ort.InferenceSession(str(a.vocoder_onnx),so,providers=['CPUExecutionProvider']);ins,_,mn,fn,wn=m30.find_io(sess)
    feed={mn:m30.adapt_mel(ins[mn],mel),fn:m30.adapt_f0(ins[fn],curves['f0'])}
    t0=time.perf_counter();gold=m30.flat_wave(sess.run([wn],feed)[0]);gold_ms=(time.perf_counter()-t0)*1000
    goldp=a.work/'golden_wave.f32';gold.astype(np.float32).tofile(goldp)
    audio_ms=gold.size/sr*1000.;results={}
    for mode in [x.strip() for x in a.modes.split(',') if x.strip()]:
        md=a.work/mode;md.mkdir(exist_ok=True)
        env=os.environ.copy();env.update(DSASM_VNNI=mode,DSASM_GOLDEN_COS='0.999',DSASM_GOLDEN_SNR='25')
        vcmd=[str(a.vocoder_cli),'infer',str(a.vocoder_bundle),'--mel',str(melp),'--f0',str(a.work/'f0.f32'),'--out',str(md/'native_wave.f32'),'--wav',str(md/'native_pipeline.wav'),'--golden',str(goldp),'--workers',str(a.workers),'--rounds',str(a.rounds),'--profile']
        q=subprocess.run(vcmd,text=True,capture_output=True,env=env);print(f'\n===== M40.1 E2E DSASM_VNNI={mode} =====');print(q.stdout,end='');print(q.stderr,end='',file=sys.stderr)
        if q.returncode:raise SystemExit(q.returncode)
        voc_ms=parse_ms(q.stdout,r'median=([0-9.]+) ms');total=acoustic_ms+voc_ms;rtf=total/audio_ms
        qual=parse_quality(q.stdout+q.stderr)
        results[mode]={'vocoder_ms':voc_ms,'total_ms':total,'rtf':rtf,'speed':audio_ms/total,'quality':qual}
        print(f'M40.1 PURE CPU/C/ASM E2E [{mode}]: acoustic={acoustic_ms:.3f} vocoder={voc_ms:.3f} total={total:.3f} ms audio={audio_ms:.3f} RTF={rtf:.3f} speed={audio_ms/total:.3f}x')
        print(f'  realtime RTF<1: {"CROSSED" if rtf<1 else "NOT YET"}; engineering RTF<=0.8: {"CROSSED" if rtf<=.8 else "NOT YET"}')
    report={'T':T,'samples':gold.size,'sample_rate':sr,'hop_size':hop,'steps':a.steps,'workers':a.workers,'acoustic_ms':acoustic_ms,'audio_ms':audio_ms,'ort_vocoder_golden_excluded_ms':gold_ms,'results':results,'phonemes':names,'tokens':tok.tolist(),'durations':dur.tolist()}
    (a.work/'m40_1_e2e_report.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
