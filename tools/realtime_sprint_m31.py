#!/usr/bin/env python3
"""M31 CPU realtime performance sprint for a real DiffSinger voicebank.

This deliberately keeps the already-validated acoustic math unchanged and asks
one concrete question: what combination of Rectified-Flow step count and CPU
vocoder threading gets closest to / across real time on the target CPU?

Outputs:
  * native acoustic FS2/Aux/RF timing for each RF step count
  * 20-step-relative mel and waveform quality deltas
  * NSF-HiFiGAN ORT 1/2/4/6/8-thread ABBA sweep
  * ONNX static op/parameter summary + best-thread ORT node profile
  * one WAV per RF step count for listening
  * JSON report suitable for later regression/optimization milestones
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import run_real_voicebank_m30 as m30
import validate_real_onnx_m27 as v27


def parse_csv_ints(s: str, *, allow_zero: bool = False) -> list[int]:
    out = []
    for x in s.split(','):
        x = x.strip()
        if x:
            v = int(x)
            if v < 0 or (v == 0 and not allow_zero):
                raise ValueError('integer sweep values must be >=0' if allow_zero else 'integer sweep values must be >0')
            out.append(v)
    if not out:
        raise ValueError('empty integer sweep')
    return list(dict.fromkeys(out))


def percentile(xs, p: float) -> float:
    a = np.asarray(xs, dtype=np.float64)
    return float(np.percentile(a, p)) if a.size else float('nan')


def metric_with_snr(ref, got):
    d = v27.metrics(np.asarray(ref, np.float32), np.asarray(got, np.float32))
    r = np.asarray(ref, np.float64).reshape(-1)
    e = np.asarray(got, np.float64).reshape(-1) - r
    rp = float(np.mean(r * r))
    ep = float(np.mean(e * e))
    if ep == 0.0:
        snr = float('inf')
    elif rp == 0.0:
        snr = float('-inf')
    else:
        snr = 10.0 * math.log10(rp / ep)
    d['snr_db'] = snr
    return d


def write_wav(path: Path, x, sr: int):
    y = np.asarray(x, np.float32).reshape(-1)
    if not np.isfinite(y).all():
        raise RuntimeError(f'non-finite waveform: {path}')
    pcm = np.round(np.clip(y, -1.0, 1.0) * 32767.0).astype('<i2')
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def parse_native_stdout(stdout: str) -> dict:
    m = re.search(r'time=([0-9.]+) ms .*?RTF=([0-9.]+) speed=([0-9.]+)x', stdout)
    s = re.search(
        r'stages: fs2=([0-9.]+) ms aux=([0-9.]+) ms rf=([0-9.]+) ms '
        r'rf_per_step=([0-9.]+) ms sum=([0-9.]+) ms', stdout)
    if not m or not s:
        raise RuntimeError('could not parse native --profile-stages output:\n' + stdout)
    return {
        'total_ms': float(m.group(1)),
        'rtf': float(m.group(2)),
        'speed_x': float(m.group(3)),
        'fs2_ms': float(s.group(1)),
        'aux_ms': float(s.group(2)),
        'rf_ms': float(s.group(3)),
        'rf_per_step_ms': float(s.group(4)),
        'stage_sum_ms': float(s.group(5)),
    }


def run_native(a, step: int, out_path: Path, common_paths: dict) -> dict:
    cmd = [
        str(a.cli), 'infer', str(a.packed),
        '--tokens', str(common_paths['tokens']),
        '--durations', str(common_paths['durations']),
        '--f0', str(common_paths['f0']),
        '--languages', str(common_paths['languages']),
        '--speaker-emb', str(a.speaker_emb),
        '--breathiness', str(common_paths['breathiness']),
        '--voicing', str(common_paths['voicing']),
        '--tension', str(common_paths['tension']),
        '--gender', str(common_paths['gender']),
        '--velocity', str(common_paths['velocity']),
        '--depth', str(a.depth),
        '--steps', str(step),
        '--noise', str(common_paths['noise']),
        '--out', str(out_path),
        '--threads', str(a.native_threads),
        '--profile-stages',
    ]
    p = subprocess.run(cmd, text=True, capture_output=True)
    if p.returncode:
        print(p.stdout, end='')
        print(p.stderr, end='', file=sys.stderr)
        raise RuntimeError(f'native CLI failed for steps={step}: rc={p.returncode}')
    d = parse_native_stdout(p.stdout)
    d['stdout'] = p.stdout
    return d


def set_affinity(cpus: list[int]):
    if not cpus:
        return None
    if not hasattr(os, 'sched_setaffinity'):
        print('M31 affinity: unsupported on this platform')
        return None
    before = sorted(os.sched_getaffinity(0))
    valid = [c for c in cpus if c in before]
    if not valid:
        print(f'M31 affinity: requested {cpus}, none in current allowed set {before}; unchanged')
        return before
    os.sched_setaffinity(0, set(valid))
    after = sorted(os.sched_getaffinity(0))
    print(f'M31 affinity: {before} -> {after}')
    return before


def make_vocoder_session(path: Path, threads: int, profile: bool = False):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(threads)
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if profile:
        so.enable_profiling = True
        so.profile_file_prefix = 'm31_vocoder_profile'
    return ort.InferenceSession(str(path), so, providers=['CPUExecutionProvider'])


def make_vocoder_runner(sess, mel_template, f0):
    ins, _outs, mel_name, f0_name, wav_name = m30.find_io(sess)
    f0_feed = m30.adapt_f0(ins[f0_name], f0)
    def run(mel):
        feed = {
            mel_name: m30.adapt_mel(ins[mel_name], mel),
            f0_name: f0_feed,
        }
        return m30.flat_wave(sess.run([wav_name], feed)[0])
    # Validate rank/layout once.
    _ = m30.adapt_mel(ins[mel_name], mel_template)
    return run


def vocoder_thread_sweep(path: Path, mel, f0, thread_counts: list[int], rounds: int):
    sessions = {}
    runners = {}
    for n in thread_counts:
        s = make_vocoder_session(path, n)
        sessions[n] = s
        runners[n] = make_vocoder_runner(s, mel, f0)
        _ = runners[n](mel)  # warmup before measured ABBA schedule
    times = {n: [] for n in thread_counts}
    last_wave = {}
    for r in range(rounds):
        order = thread_counts if r % 2 == 0 else list(reversed(thread_counts))
        # ABBA-ish second half in the opposite direction in every round.
        order = list(order) + list(reversed(order))
        for n in order:
            t0 = time.perf_counter()
            w = runners[n](mel)
            dt = (time.perf_counter() - t0) * 1000.0
            times[n].append(dt)
            last_wave[n] = w
    stat = {}
    for n, xs in times.items():
        stat[n] = {
            'samples_ms': xs,
            'median_ms': float(statistics.median(xs)),
            'p90_ms': percentile(xs, 90),
            'min_ms': float(min(xs)),
            'max_ms': float(max(xs)),
        }
    best = min(thread_counts, key=lambda n: stat[n]['median_ms'])
    return stat, best, sessions[best], runners[best], last_wave[best]


def onnx_static_summary(path: Path):
    import onnx
    from onnx import numpy_helper
    model = onnx.load(str(path), load_external_data=False)
    op_hist = collections.Counter()
    nodes = []
    def walk(g, prefix=''):
        for node in g.node:
            op_hist[node.op_type] += 1
            nodes.append((prefix + (node.name or node.op_type), node))
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    walk(attr.g, prefix + (node.name or node.op_type) + '/')
                elif attr.type == onnx.AttributeProto.GRAPHS:
                    for i, sg in enumerate(attr.graphs):
                        walk(sg, prefix + (node.name or node.op_type) + f'/{i}/')
    walk(model.graph)
    init = {x.name: x for x in model.graph.initializer}
    dtype_bytes = {1: 4, 2: 1, 3: 1, 4: 2, 5: 2, 6: 4, 7: 8, 10: 2, 11: 8, 12: 4, 13: 8, 16: 2}
    total_params = 0
    total_bytes = 0
    for t in model.graph.initializer:
        n = int(np.prod(t.dims, dtype=np.int64)) if t.dims else 1
        total_params += n
        total_bytes += n * dtype_bytes.get(t.data_type, 0)
    convs = []
    for name, node in nodes:
        if node.op_type not in ('Conv', 'ConvTranspose'):
            continue
        w = init.get(node.input[1]) if len(node.input) > 1 else None
        if w is None:
            continue
        shape = list(w.dims)
        params = int(np.prod(shape, dtype=np.int64))
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        convs.append({
            'name': name,
            'op': node.op_type,
            'weight': w.name,
            'weight_shape': shape,
            'weight_params': params,
            'kernel_shape': list(attrs.get('kernel_shape', [])),
            'strides': list(attrs.get('strides', [])),
            'dilations': list(attrs.get('dilations', [])),
            'group': int(attrs.get('group', 1)),
        })
    convs.sort(key=lambda x: x['weight_params'], reverse=True)
    return {
        'nodes_recursive': len(nodes),
        'operator_histogram': dict(sorted(op_hist.items(), key=lambda kv: (-kv[1], kv[0]))),
        'initializer_count': len(model.graph.initializer),
        'parameters': int(total_params),
        'parameter_bytes_estimate': int(total_bytes),
        'largest_convolutions': convs[:30],
    }


def ort_node_profile(path: Path, mel, f0, threads: int, runs: int, work: Path):
    sess = make_vocoder_session(path, threads, profile=True)
    runner = make_vocoder_runner(sess, mel, f0)
    # Profile exactly N complete invocations.  Results below are normalized by N.
    for _ in range(runs):
        _ = runner(mel)
    prof_path = Path(sess.end_profiling())
    try:
        events = json.loads(prof_path.read_text())
    finally:
        pass
    by_op = collections.defaultdict(float)
    by_provider = collections.defaultdict(float)
    by_node = collections.defaultdict(lambda: {'us': 0.0, 'op': '?', 'provider': '?'})
    node_events = 0
    for e in events:
        if e.get('cat') != 'Node' or 'dur' not in e:
            continue
        args = e.get('args') or {}
        op = str(args.get('op_name') or args.get('op_type') or '?')
        provider = str(args.get('provider') or '?')
        name = str(e.get('name') or '?')
        us = float(e.get('dur') or 0.0)
        by_op[op] += us
        by_provider[provider] += us
        d = by_node[name]
        d['us'] += us; d['op'] = op; d['provider'] = provider
        node_events += 1
    top_nodes = []
    for name, d in sorted(by_node.items(), key=lambda kv: kv[1]['us'], reverse=True)[:40]:
        top_nodes.append({
            'name': name,
            'op': d['op'],
            'provider': d['provider'],
            'mean_ms_per_run': d['us'] / runs / 1000.0,
        })
    op_mean = {k: v / runs / 1000.0 for k, v in sorted(by_op.items(), key=lambda kv: kv[1], reverse=True)}
    provider_mean = {k: v / runs / 1000.0 for k, v in sorted(by_provider.items(), key=lambda kv: kv[1], reverse=True)}
    dst = work / 'vocoder_ort_profile.json'
    # Preserve ORT trace alongside our compact summary.
    if prof_path.exists():
        raw_dst = work / 'vocoder_ort_trace.json'
        raw_dst.write_bytes(prof_path.read_bytes())
        try: prof_path.unlink()
        except OSError: pass
    compact = {
        'threads': threads,
        'profile_runs': runs,
        'node_events': node_events,
        'operator_mean_ms_per_run': op_mean,
        'provider_mean_ms_per_run': provider_mean,
        'top_nodes': top_nodes,
    }
    dst.write_text(json.dumps(compact, indent=2) + '\n')
    return compact


def fmt(x, n=3):
    if isinstance(x, float) and math.isinf(x):
        return 'inf'
    return f'{x:.{n}f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vocoder-onnx', type=Path, required=True)
    ap.add_argument('--packed', type=Path, required=True)
    ap.add_argument('--cli', type=Path, default=Path('build/dsasm-acoustic'))
    ap.add_argument('--speaker-emb', type=Path, required=True)
    ap.add_argument('--language-id', type=int, default=4)
    ap.add_argument('--depth', type=float, default=.6)
    ap.add_argument('--steps', default='20,16,12,10,8,6,4')
    ap.add_argument('--native-threads', type=int, default=8)
    ap.add_argument('--acoustic-rounds', type=int, default=2)
    ap.add_argument('--vocoder-threads', default='1,2,4,6,8')
    ap.add_argument('--vocoder-rounds', type=int, default=3)
    ap.add_argument('--profile-runs', type=int, default=3)
    ap.add_argument('--cpus', default='0,2,4,6,1,3,5,7')
    ap.add_argument('--seed', type=int, default=3101)
    ap.add_argument('--work', type=Path, default=Path('build/m31_realtime'))
    a = ap.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)
    steps = parse_csv_ints(a.steps)
    if 20 not in steps:
        steps = [20] + steps
    voc_threads = parse_csv_ints(a.vocoder_threads)
    cpus = parse_csv_ints(a.cpus, allow_zero=True) if a.cpus.strip() else []
    old_affinity = set_affinity(cpus)

    model = json.loads((a.packed / 'model.json').read_text())
    C = int(model['fs2']['hidden_size'])
    M = int(model['rf']['input_dim'])
    conf = m30.read_conf(a.packed / 'model.conf')
    sr = int(float(conf.get('sample_rate', 44100)))
    hop = int(float(conf.get('hop_size', 512)))

    names, tok_list = m30.pick_demo_phonemes(a.packed)
    tok = np.asarray(tok_list, np.int64)
    dur = m30.make_durations(names, tok.size)
    T = int(dur.sum())
    langs = np.full(tok.size, a.language_id, np.int64)
    curves = m30.frame_curve(names, dur)
    sp = np.fromfile(a.speaker_emb, np.float32)
    if sp.size != C:
        raise RuntimeError(f'speaker embedding {sp.size} floats, expected {C}')
    rng = np.random.default_rng(a.seed)
    noise = rng.standard_normal((T, M), dtype=np.float32)

    common = {'noise': a.work / 'noise.f32'}
    noise.tofile(common['noise'])
    vecs = {'tokens': tok, 'durations': dur, 'languages': langs, **curves}
    for k, v in vecs.items():
        p = a.work / f'{k}.txt'
        m30.dump_txt(p, v)
        common[k] = p
    audio_ms = T * hop / sr * 1000.0
    print('M31 demo phonemes:', names)
    print('M31 tokens:', tok.tolist())
    print('M31 durations:', dur.tolist(), f'T={T} audio={audio_ms/1000:.3f}s')
    print('M31 hard targets: CPU-only RTF < 1.0; engineering target RTF <= 0.8')

    # Acoustic sweep uses symmetric/reversed order to reduce simple phase-order bias.
    timing_samples = {s: [] for s in steps}
    mel_paths = {s: a.work / f'mel_steps{s:02d}.f32' for s in steps}
    for r in range(max(1, a.acoustic_rounds)):
        order = steps if r % 2 == 0 else list(reversed(steps))
        for s in order:
            d = run_native(a, s, mel_paths[s], common)
            timing_samples[s].append(d)
            print(f"M31 acoustic sample round={r+1} steps={s}: total={d['total_ms']:.3f} fs2={d['fs2_ms']:.3f} aux={d['aux_ms']:.3f} rf={d['rf_ms']:.3f} rf/step={d['rf_per_step_ms']:.3f} ms")

    acoustic = {}
    mels = {}
    for s in steps:
        mels[s] = np.fromfile(mel_paths[s], np.float32).reshape(T, M)
        xs = timing_samples[s]
        acoustic[s] = {}
        for key in ('total_ms', 'fs2_ms', 'aux_ms', 'rf_ms', 'rf_per_step_ms', 'stage_sum_ms'):
            vals = [x[key] for x in xs]
            acoustic[s][key] = float(statistics.median(vals))
            acoustic[s][key.replace('_ms', '_p90_ms')] = percentile(vals, 90)
        acoustic[s]['samples'] = xs

    ref_mel = mels[20]

    # Vocoder thread sweep uses the exact native 20-step reference mel.
    vstat, best_threads, best_sess, best_runner, _ = vocoder_thread_sweep(
        a.vocoder_onnx, ref_mel, curves['f0'], voc_threads, max(1, a.vocoder_rounds))
    print('\nM31 VOCODER THREAD SWEEP (ABBA/interleaved)')
    for n in voc_threads:
        st = vstat[n]
        print(f"  threads={n:2d} median={st['median_ms']:.3f} ms p90={st['p90_ms']:.3f} min={st['min_ms']:.3f} max={st['max_ms']:.3f}")
    best_voc_ms = float(vstat[best_threads]['median_ms'])
    print(f'  best threads={best_threads} median={best_voc_ms:.3f} ms')

    # Reference waveform and per-step audible outputs.  Timing budget uses the
    # interleaved median above; this avoids assigning random one-shot vocoder
    # jitter to one RF step count.
    ref_wave = best_runner(ref_mel)
    write_wav(a.work / 'steps20_reference.wav', ref_wave, sr)
    pareto = []
    for s in sorted(steps, reverse=True):
        wav = best_runner(mels[s])
        write_wav(a.work / f'steps{s:02d}.wav', wav, sr)
        mm = metric_with_snr(ref_mel, mels[s])
        wm = metric_with_snr(ref_wave, wav)
        total = acoustic[s]['total_ms'] + best_voc_ms
        rtf = total / audio_ms
        row = {
            'steps': s,
            'acoustic': acoustic[s],
            'vocoder_budget_ms': best_voc_ms,
            'total_budget_ms': total,
            'rtf': rtf,
            'realtime_speed_x': audio_ms / total,
            'meets_rtf_1': bool(rtf < 1.0),
            'meets_rtf_0_8': bool(rtf <= 0.8),
            'mel_vs_20': mm,
            'waveform_vs_20': wm,
            'wav': str(a.work / f'steps{s:02d}.wav'),
        }
        pareto.append(row)

    static = onnx_static_summary(a.vocoder_onnx)
    profile = ort_node_profile(a.vocoder_onnx, ref_mel, curves['f0'], best_threads, max(1, a.profile_runs), a.work)

    print('\nM31 RF STEP / END-TO-END PARETO (20-step native = quality reference)')
    print(' steps | FS2 ms | Aux ms | RF ms | RF/step | Acoustic | Vocoder | Total |  RTF | speed | mel max | wave cos | SNR dB | realtime')
    print('-------+--------+--------+-------+---------+----------+---------+-------+------+-------+---------+----------+--------+---------')
    for r in pareto:
        ac = r['acoustic']; mm = r['mel_vs_20']; wm = r['waveform_vs_20']
        goal = 'RTF<=0.8' if r['meets_rtf_0_8'] else ('RTF<1' if r['meets_rtf_1'] else '-')
        print(f" {r['steps']:5d} | {ac['fs2_ms']:6.1f} | {ac['aux_ms']:6.1f} | {ac['rf_ms']:5.1f} | {ac['rf_per_step_ms']:7.2f} | {ac['total_ms']:8.1f} | {best_voc_ms:7.1f} | {r['total_budget_ms']:5.1f} | {r['rtf']:4.2f} | {r['realtime_speed_x']:5.2f}x | {mm['max_abs']:.5f} | {wm['cosine']:.6f} | {wm['snr_db']:6.1f} | {goal}")

    print('\nM31 VOCODER STATIC GRAPH')
    print(f"  nodes(recursive)={static['nodes_recursive']} initializers={static['initializer_count']} params={static['parameters']:,} parameter_bytes~={static['parameter_bytes_estimate']/1024/1024:.2f} MiB")
    print('  operators:', ', '.join(f'{k}={v}' for k,v in list(static['operator_histogram'].items())[:16]))
    print('  largest conv weights:')
    for x in static['largest_convolutions'][:10]:
        print(f"    {x['op']:13s} params={x['weight_params']:9,d} shape={x['weight_shape']} stride={x['strides']} group={x['group']}  {x['name']}")

    print('\nM31 VOCODER ORT NODE PROFILE (mean per profiled run)')
    for op, ms in list(profile['operator_mean_ms_per_run'].items())[:12]:
        print(f'  op {op:18s} {ms:8.3f} ms')
    print('  top nodes:')
    for x in profile['top_nodes'][:15]:
        print(f"    {x['mean_ms_per_run']:8.3f} ms  {x['op']:16s} {x['name']}")

    speed_candidates = [r for r in pareto if r['meets_rtf_1']]
    goal_candidates = [r for r in pareto if r['meets_rtf_0_8']]
    fastest = min(pareto, key=lambda r: r['total_budget_ms'])
    report = {
        'target': {'realtime_rtf': 1.0, 'engineering_rtf': 0.8},
        'audio': {'frames': T, 'duration_ms': audio_ms, 'sample_rate': sr, 'hop_size': hop},
        'cpu_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
        'native_threads': a.native_threads,
        'vocoder_thread_sweep': {str(k): v for k, v in vstat.items()},
        'best_vocoder_threads': best_threads,
        'best_vocoder_median_ms': best_voc_ms,
        'pareto': pareto,
        'vocoder_static': static,
        'vocoder_profile': profile,
        'summary': {
            'realtime_steps': [r['steps'] for r in speed_candidates],
            'engineering_goal_steps': [r['steps'] for r in goal_candidates],
            'fastest_steps': fastest['steps'],
            'fastest_rtf': fastest['rtf'],
        },
        'phonemes': names,
        'tokens': tok.tolist(),
        'durations': dur.tolist(),
    }
    (a.work / 'm31_realtime_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print('\nM31 REALTIME VERDICT')
    if goal_candidates:
        print('  engineering target RTF<=0.8 CROSSED at steps:', [r['steps'] for r in goal_candidates])
    elif speed_candidates:
        print('  realtime RTF<1 CROSSED, but RTF<=0.8 not yet crossed; steps:', [r['steps'] for r in speed_candidates])
    else:
        need = fastest['rtf']
        print(f"  realtime NOT YET CROSSED. fastest measured budget: steps={fastest['steps']} RTF={fastest['rtf']:.3f} ({fastest['realtime_speed_x']:.3f}x realtime)")
        print(f"  remaining end-to-end speedup needed to RTF<1: {need:.3f}x; to RTF<=0.8: {need/0.8:.3f}x")
    print('  report:', a.work / 'm31_realtime_report.json')
    print('  listen:', ', '.join(str(a.work / f"steps{r['steps']:02d}.wav") for r in pareto))

    # Restore prior affinity for polite library/script behavior when invoked
    # interactively, although the process exits immediately afterward.
    if old_affinity is not None and hasattr(os, 'sched_setaffinity'):
        try: os.sched_setaffinity(0, set(old_affinity))
        except OSError: pass


if __name__ == '__main__':
    main()
