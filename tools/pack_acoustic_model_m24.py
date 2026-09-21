#!/usr/bin/env python3
"""Pack one DiffSinger acoustic checkpoint into an M24 native model directory.

The output directory contains:
  fs2_acoustic.dsfs   (DSFS21)
  aux_convnext.dsa    (DSAUX20)
  lynxnet2.dsn        (DSLYNX7)
  model.conf          (plain-text runtime sampling/mel config)
  model.json          (summary + bundle metadata)

No DiffSinger import is required. The three existing packers discover tensors by
suffix, so ordinary Lightning checkpoints/state_dict checkpoints are supported.
"""
from __future__ import annotations
import argparse, ast, json, re, shutil, struct, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def parse_scalar_or_list(text: str):
    text = text.strip()
    low = text.lower()
    if low in ('true','yes','on'): return True
    if low in ('false','no','off'): return False
    if low in ('null','none','~'): return None
    try:
        v = ast.literal_eval(text)
    except Exception:
        try:
            return float(text)
        except Exception:
            return text
    return v

def read_config_loose(path: Path):
    """Read only the few flat DiffSinger keys M24 needs, without PyYAML."""
    wanted = {
        'spec_min','spec_max','T_start','T_start_infer','time_scale_factor',
        'sampling_steps','audio_num_mel_bins','use_shallow_diffusion','diffusion_type',
        'use_lang_id','use_spk_id','use_energy_embed','use_breathiness_embed',
        'use_voicing_embed','use_tension_embed','use_key_shift_embed','use_speed_embed',
        'use_stretch_embed','use_variance_scaling','use_rope','rope_interleaved',
        'enc_ffn_kernel_size','num_heads','hidden_size','backbone_type','sampling_algorithm',
        'audio_sample_rate','hop_size'
    }
    out = {}
    for raw in path.read_text(encoding='utf8').splitlines():
        line = raw.split('#',1)[0].rstrip()
        m = re.match(r'^([A-Za-z_][A-Za-z0-9_]*):\s*(.*?)\s*$', line)
        if not m or m.group(1) not in wanted or not m.group(2):
            continue
        out[m.group(1)] = parse_scalar_or_list(m.group(2))
    return out

def csv(v):
    if isinstance(v, (list, tuple)):
        return ','.join(str(float(x)) for x in v)
    return str(float(v))

def read_header(path: Path):
    b = path.read_bytes()[:64]
    magic = b[:8]
    if magic == b'DSFS21\0\0':
        vals = struct.unpack_from('<7I', b, 8)
        return {'kind':'fs2','version':vals[0],'vocab_size':vals[1],'hidden_size':vals[2],
                'layers':vals[3],'heads':vals[4],'ffn_kernel':vals[5],'rope_interleaved':vals[6]}
    if magic == b'DSAUX20\0':
        vals = struct.unpack_from('<8I', b, 8)
        return {'kind':'aux','version':vals[0],'input_dim':vals[1],'channels':vals[2],
                'output_dim':vals[3],'layers':vals[4],'kernel':vals[5]}
    if magic == b'DSLYNX7\0':
        vals = struct.unpack_from('<12I', b, 8)
        return {'kind':'rf','version':vals[0],'input_dim':vals[1],'condition_dim':vals[2],
                'channels':vals[3],'hidden_dim':vals[4],'layers':vals[5],'kernel':vals[6],
                'glu':vals[7]}
    raise ValueError(f'unknown packed magic in {path}')

def run(*args):
    print('+', ' '.join(str(x) for x in args), flush=True)
    subprocess.run([str(x) for x in args], check=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoint', type=Path)
    ap.add_argument('--out', type=Path, default=Path('packed_acoustic_m24'))
    ap.add_argument('--config', type=Path, help='DiffSinger acoustic yaml; only runtime keys are read')
    ap.add_argument('--fs2-prefix', default='fs2')
    ap.add_argument('--aux-prefix', default='aux_decoder')
    ap.add_argument('--rf-prefix', default='diffusion.denoise_fn')
    ap.add_argument('--glu-type', choices=['atan','softsign'], default='atan')
    ap.add_argument('--spec-min', help='override scalar or CSV, e.g. -12 or -12,-11,...')
    ap.add_argument('--spec-max', help='override scalar or CSV')
    ap.add_argument('--t-start', type=float)
    ap.add_argument('--time-scale-factor', type=float)
    ap.add_argument('--steps', type=int)
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    tmp_fs = a.out / '_fs2'; tmp_aux = a.out / '_aux'; tmp_rf = a.out / '_rf'
    for p in (tmp_fs,tmp_aux,tmp_rf):
        if p.exists(): shutil.rmtree(p)

    run(sys.executable, ROOT/'pack_fs2_acoustic_checkpoint.py', a.checkpoint,
        '--prefix', a.fs2_prefix, '--out', tmp_fs)
    run(sys.executable, ROOT/'pack_aux_convnext_checkpoint.py', a.checkpoint,
        '--prefix', a.aux_prefix, '--out', tmp_aux, '--test-frames', '16')
    run(sys.executable, ROOT/'pack_lynxnet2_backbone.py', a.checkpoint,
        '--prefix', a.rf_prefix, '--glu-type', a.glu_type, '--out', tmp_rf, '--test-frames', '16')

    final = {
        tmp_fs/'fs2_acoustic.dsfs': a.out/'fs2_acoustic.dsfs',
        tmp_aux/'aux_convnext.dsa': a.out/'aux_convnext.dsa',
        tmp_rf/'lynxnet2.dsn': a.out/'lynxnet2.dsn',
    }
    for src,dst in final.items(): shutil.move(src,dst)
    for src,name in ((tmp_fs/'fs2_acoustic.json','fs2_acoustic.json'),
                     (tmp_aux/'aux_convnext.json','aux_convnext.json'),
                     (tmp_rf/'lynxnet2.json','lynxnet2.json')):
        if src.exists(): shutil.move(src,a.out/name)
    for p in (tmp_fs,tmp_aux,tmp_rf): shutil.rmtree(p, ignore_errors=True)

    cfg = read_config_loose(a.config) if a.config else {}

    fs = read_header(a.out/'fs2_acoustic.dsfs')
    au = read_header(a.out/'aux_convnext.dsa')
    rf = read_header(a.out/'lynxnet2.dsn')
    if fs['hidden_size'] != au['input_dim'] or au['input_dim'] != rf['condition_dim'] or au['output_dim'] != rf['input_dim']:
        raise SystemExit(f'incompatible bundles: fs2={fs}, aux={au}, rf={rf}')

    if a.config:
        if 'hidden_size' in cfg and int(cfg['hidden_size']) != fs['hidden_size']:
            raise SystemExit(f'config hidden_size={cfg["hidden_size"]} != packed FS2 hidden={fs["hidden_size"]}')
        if 'audio_num_mel_bins' in cfg and int(cfg['audio_num_mel_bins']) != rf['input_dim']:
            raise SystemExit(f'config audio_num_mel_bins={cfg["audio_num_mel_bins"]} != packed mel={rf["input_dim"]}')
        if 'sampling_algorithm' in cfg and str(cfg['sampling_algorithm']).strip("'\"") != 'euler':
            raise SystemExit(f'M24 currently supports sampling_algorithm=euler, got {cfg["sampling_algorithm"]!r}')

    if a.config:
        unsupported_true = [k for k in (
            'use_lang_id','use_spk_id','use_energy_embed','use_breathiness_embed',
            'use_voicing_embed','use_tension_embed','use_key_shift_embed','use_speed_embed'
        ) if bool(cfg.get(k, False))]
        if unsupported_true:
            raise SystemExit('M24 default-profile runtime does not yet support enabled branches: ' + ', '.join(unsupported_true))
        required = {
            'use_shallow_diffusion': True, 'use_stretch_embed': True,
            'use_variance_scaling': True, 'use_rope': True,
        }
        bad = [f'{k}={cfg.get(k)!r}' for k,v in required.items() if k in cfg and bool(cfg.get(k)) != v]
        if bad:
            raise SystemExit('M24 incompatible acoustic config: ' + ', '.join(bad))
        if 'rope_interleaved' in cfg and bool(cfg['rope_interleaved']):
            raise SystemExit('M24 currently matches acoustic rope_interleaved=false')
        for k,want in [('enc_ffn_kernel_size',3),('num_heads',2)]:
            if k in cfg and int(cfg[k]) != want:
                raise SystemExit(f'M24 currently requires {k}={want}, got {cfg[k]}')
        if 'diffusion_type' in cfg and str(cfg['diffusion_type']).strip("'\"") != 'reflow':
            raise SystemExit(f'M24 requires diffusion_type=reflow, got {cfg["diffusion_type"]!r}')
        if 'backbone_type' in cfg and str(cfg['backbone_type']).strip("'\"") != 'lynxnet2':
            raise SystemExit(f'M24 requires backbone_type=lynxnet2, got {cfg["backbone_type"]!r}')
    else:
        print('warning: --config not supplied; optional acoustic branches cannot be validated', file=sys.stderr)
    def parse_csv_arg(s): return [float(x) for x in s.split(',')]
    spec_min = parse_csv_arg(a.spec_min) if a.spec_min else cfg.get('spec_min', [-12.0])
    spec_max = parse_csv_arg(a.spec_max) if a.spec_max else cfg.get('spec_max', [0.0])
    if not isinstance(spec_min,(list,tuple)): spec_min=[float(spec_min)]
    if not isinstance(spec_max,(list,tuple)): spec_max=[float(spec_max)]
    if len(spec_min) not in (1,rf['input_dim']) or len(spec_max) not in (1,rf['input_dim']) or len(spec_min)!=len(spec_max):
        raise SystemExit(f'spec range must be scalar or mel-sized ({rf["input_dim"]}); got {len(spec_min)}/{len(spec_max)}')
    t_start = a.t_start if a.t_start is not None else float(cfg.get('T_start_infer', cfg.get('T_start', 0.4)))
    time_scale = a.time_scale_factor if a.time_scale_factor is not None else float(cfg.get('time_scale_factor', 1000.0))
    steps = a.steps if a.steps is not None else int(cfg.get('sampling_steps', 20))
    sample_rate = int(cfg.get('audio_sample_rate', 44100))
    hop_size = int(cfg.get('hop_size', 512))
    if not (0.0 <= t_start <= 1.0) or steps < 1 or time_scale <= 0 or sample_rate <= 0 or hop_size <= 0:
        raise SystemExit('invalid runtime sampling/audio settings')

    conf = [
        '# DiffSinger-ASM M24 native acoustic model config',
        f'spec_min={csv(spec_min)}', f'spec_max={csv(spec_max)}',
        f't_start={t_start}', f'time_scale_factor={time_scale}', f'steps={steps}',
        f'mel_bins={rf["input_dim"]}', f'hidden_size={fs["hidden_size"]}',
        f'sample_rate={sample_rate}', f'hop_size={hop_size}',
    ]
    (a.out/'model.conf').write_text('\n'.join(conf)+'\n',encoding='utf8')
    summary = {'format':'DiffSinger-ASM M24 acoustic directory','checkpoint':str(a.checkpoint),
               'fs2':fs,'aux':au,'rf':rf,'runtime':{'spec_min':spec_min,'spec_max':spec_max,
               't_start':t_start,'time_scale_factor':time_scale,'steps':steps,
               'sample_rate':sample_rate,'hop_size':hop_size}}
    (a.out/'model.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print(f'packed M24 acoustic model -> {a.out}')
    print(f'  V={fs["vocab_size"]} hidden={fs["hidden_size"]} mel={rf["input_dim"]} RF C={rf["channels"]} L={rf["layers"]}')
    print(f'  sampling: t_start={t_start} steps={steps} scale={time_scale} spec_range_dims={len(spec_min)}')

if __name__ == '__main__': main()
