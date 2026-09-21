#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, re, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np


def parse_ms(text: str, pat: str) -> float:
    m = re.search(pat, text)
    return float(m.group(1)) if m else float('nan')


def parse_quality(text: str) -> dict:
    m = re.search(r'parity max_abs=([0-9.eE+-]+) rmse=([0-9.eE+-]+) cosine=([0-9.eE+-]+) SNR=([0-9.eE+-]+)', text)
    if not m:
        return {}
    return dict(max_abs=float(m.group(1)), rmse=float(m.group(2)), cosine=float(m.group(3)), snr_db=float(m.group(4)))


def load_m30(project: Path):
    tools = project / 'tools'
    sys.path.insert(0, str(tools))
    import run_real_voicebank_m30 as m30
    return m30


def make_inputs(a, m30, model: dict, conf: dict):
    a.fixture.mkdir(parents=True, exist_ok=True)
    M = int(model['rf']['input_dim'])
    names, tok_list = m30.pick_demo_phonemes(a.packed_acoustic)
    tok = np.asarray(tok_list, np.int64)
    dur = m30.make_durations(names, tok.size)
    T = int(dur.sum())
    if T != 48:
        raise RuntimeError(f'M52 fixed-shape bundle expects T=48, got {T}')
    langs = np.full(tok.size, a.language_id, np.int64)
    curves = m30.frame_curve(names, dur)
    for n, x in [('tokens', tok), ('durations', dur), ('languages', langs), *curves.items()]:
        m30.dump_txt(a.fixture / f'{n}.txt', x)
    curves['f0'].astype(np.float32).tofile(a.fixture / 'f0.f32')
    rng = np.random.default_rng(a.seed)
    rng.standard_normal((T, M), dtype=np.float32).tofile(a.fixture / 'noise.f32')
    sr = int(float(conf.get('sample_rate', 44100)))
    hop = int(float(conf.get('hop_size', 512)))
    return names, tok, dur, curves, T, M, sr, hop


def pinned(cpus: str, cmd: list[str]) -> list[str]:
    if not cpus:
        return cmd
    if not shutil.which('taskset'):
        raise SystemExit('M52 requires taskset (util-linux) for topology experiments')
    return ['taskset', '-c', cpus, *cmd]


def acoustic_cmd(a, out: Path):
    cmd = [str(a.acoustic_cli), 'infer', str(a.packed_acoustic),
           '--tokens', str(a.fixture/'tokens.txt'), '--durations', str(a.fixture/'durations.txt'),
           '--f0', str(a.fixture/'f0.txt'), '--languages', str(a.fixture/'languages.txt'),
           '--speaker-emb', str(a.speaker_emb), '--breathiness', str(a.fixture/'breathiness.txt'),
           '--voicing', str(a.fixture/'voicing.txt'), '--tension', str(a.fixture/'tension.txt'),
           '--gender', str(a.fixture/'gender.txt'), '--velocity', str(a.fixture/'velocity.txt'),
           '--depth', str(a.depth), '--steps', str(a.steps), '--noise', str(a.fixture/'noise.f32'),
           '--threads', str(a.acoustic_threads), '--profile-stages', '--out', str(out)]
    return pinned(a.acoustic_cpus, cmd)


def run_capture(cmd, env=None):
    p = subprocess.run(cmd, text=True, capture_output=True, env=env)
    print(p.stdout, end='')
    print(p.stderr, end='', file=sys.stderr)
    if p.returncode:
        raise SystemExit(p.returncode)
    return p.stdout + p.stderr


def prepare(a):
    m30 = load_m30(a.project)
    model = json.loads((a.packed_acoustic/'model.json').read_text())
    conf = m30.read_conf(a.packed_acoustic/'model.conf')
    names, tok, dur, curves, T, M, sr, hop = make_inputs(a, m30, model, conf)
    print('M52 prepare: native acoustic reference')
    ref_mel = a.fixture/'reference_mel.f32'
    txt = run_capture(acoustic_cmd(a, ref_mel))
    acoustic_ms = parse_ms(txt, r'time=([0-9.]+) ms')
    mel = np.fromfile(ref_mel, np.float32).reshape(T, -1)
    print('M52 prepare: ORT vocoder golden ONCE (outside measured native runs)')
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 4
    sess = ort.InferenceSession(str(a.vocoder_onnx), so, providers=['CPUExecutionProvider'])
    ins, _, mn, fn, wn = m30.find_io(sess)
    feed = {mn: m30.adapt_mel(ins[mn], mel), fn: m30.adapt_f0(ins[fn], curves['f0'])}
    t0 = time.perf_counter(); gold = m30.flat_wave(sess.run([wn], feed)[0]); gold_ms = (time.perf_counter()-t0)*1000.0
    gold.astype(np.float32).tofile(a.fixture/'golden_wave.f32')
    meta = {'T':T,'M':M,'sample_rate':sr,'hop_size':hop,'samples':int(gold.size),
            'audio_ms':float(gold.size/sr*1000.0),'reference_acoustic_ms':acoustic_ms,
            'ort_golden_ms':gold_ms,'phonemes':names,'tokens':tok.tolist(),'durations':dur.tolist()}
    (a.fixture/'meta.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(f"M52 PREPARED fixture={a.fixture} audio={meta['audio_ms']:.3f} ms ORT_golden_once={gold_ms:.3f} ms")


def run_once(a):
    meta = json.loads((a.fixture/'meta.json').read_text())
    a.work.mkdir(parents=True, exist_ok=True)
    melp = a.work/'native_mel.f32'
    print(f'M52 CLEAN PATH: acoustic CPUs={a.acoustic_cpus} threads={a.acoustic_threads} -> vocoder CPUs={a.vocoder_cpus} workers={a.vocoder_workers}')
    txt = run_capture(acoustic_cmd(a, melp))
    acoustic_ms = parse_ms(txt, r'time=([0-9.]+) ms')
    ref = np.fromfile(a.fixture/'reference_mel.f32', np.float32); got = np.fromfile(melp, np.float32)
    if ref.shape != got.shape: raise SystemExit(f'M52 acoustic shape mismatch {ref.shape} vs {got.shape}')
    bitdiff = int(np.count_nonzero(ref.view(np.uint32) != got.view(np.uint32)))
    max_abs = float(np.abs(ref-got).max(initial=0))
    print(f'M52 acoustic determinism: max_abs={max_abs:.9g} bitdiff={bitdiff}')
    if bitdiff: raise SystemExit('M52 acoustic output is not bit-exact to prepared reference')
    env=os.environ.copy(); env.update(DSASM_VNNI='k11',DSASM_GOLDEN_COS='0.999',DSASM_GOLDEN_SNR='25')
    wave=a.work/'native_wave.f32'
    vcmd=[str(a.vocoder_cli),'infer',str(a.vocoder_bundle),'--mel',str(melp),'--f0',str(a.fixture/'f0.f32'),
          '--out',str(wave),'--golden',str(a.fixture/'golden_wave.f32'),'--workers',str(a.vocoder_workers),'--rounds',str(a.rounds)]
    if a.profile: vcmd.append('--profile')
    vtxt=run_capture(pinned(a.vocoder_cpus,vcmd),env=env)
    voc_ms=parse_ms(vtxt,r'median=([0-9.]+) ms'); qual=parse_quality(vtxt)
    total=acoustic_ms+voc_ms; audio_ms=float(meta['audio_ms']); rtf=total/audio_ms
    print(f'M52 CLEAN E2E: acoustic={acoustic_ms:.3f} vocoder={voc_ms:.3f} total={total:.3f} ms audio={audio_ms:.3f} RTF={rtf:.3f} speed={audio_ms/total:.3f}x')
    print(f'  realtime RTF<1: {"CROSSED" if rtf<1 else "NOT YET"}; engineering RTF<=0.8: {"CROSSED" if rtf<=.8 else "NOT YET"}')
    report={'acoustic_ms':acoustic_ms,'vocoder_ms':voc_ms,'total_ms':total,'audio_ms':audio_ms,'rtf':rtf,
            'quality':qual,'rounds':a.rounds,'acoustic_threads':a.acoustic_threads,'vocoder_workers':a.vocoder_workers,
            'acoustic_cpus':a.acoustic_cpus,'vocoder_cpus':a.vocoder_cpus,'acoustic_bitdiff':bitdiff,
            'acoustic_max_abs':max_abs,'clean_path_no_ort_between':True}
    (a.work/'m52_e2e.json').write_text(json.dumps(report,indent=2)+'\n')


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('action',choices=['prepare','run'])
    ap.add_argument('--project',type=Path,required=True); ap.add_argument('--packed-acoustic',type=Path,required=True)
    ap.add_argument('--acoustic-cli',type=Path,required=True); ap.add_argument('--vocoder-bundle',type=Path,required=True)
    ap.add_argument('--vocoder-cli',type=Path,required=True); ap.add_argument('--vocoder-onnx',type=Path,required=True)
    ap.add_argument('--speaker-emb',type=Path,required=True); ap.add_argument('--fixture',type=Path,required=True)
    ap.add_argument('--work',type=Path,default=Path('build/m52/run')); ap.add_argument('--language-id',type=int,default=4)
    ap.add_argument('--depth',type=float,default=.6); ap.add_argument('--steps',type=int,default=4)
    ap.add_argument('--acoustic-threads',type=int,default=8); ap.add_argument('--vocoder-workers',type=int,default=8)
    ap.add_argument('--acoustic-cpus',default=''); ap.add_argument('--vocoder-cpus',default='')
    ap.add_argument('--rounds',type=int,default=7); ap.add_argument('--seed',type=int,default=4001); ap.add_argument('--profile',action='store_true')
    a=ap.parse_args(); a.project=a.project.resolve()
    if a.action=='prepare': prepare(a)
    else: run_once(a)
if __name__=='__main__': main()
