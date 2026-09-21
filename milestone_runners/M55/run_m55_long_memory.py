#!/usr/bin/env python3
from __future__ import annotations
import json, os, re, shutil, subprocess, sys
from pathlib import Path
from statistics import mean, median

PROJECT=Path(os.environ.get('PROJECT', str(Path.home()/'asm/diffsinger'))).resolve()
PYTHON=Path(os.environ.get('PY', str(PROJECT/'.venv/bin/python')))
HERE=Path(__file__).resolve().parent
MODEL=Path(os.environ.get('MODEL', str(PROJECT/'model/DongFangZhiZi_Nectar_DiffSinger_CE_26.08.31')))
F=384
D=PROJECT/'build/m53_long/f384'; FIX=D/'fixture'; BUNDLE=D/'nsf.dsv35'
OUT=PROJECT/'build/m55_long'; OUT.mkdir(parents=True,exist_ok=True)
RUNNER=HERE/'run_long_e2e_m53.py'


def sh(cmd,env=None):
    p=subprocess.run([str(x) for x in cmd],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=env)
    print(p.stdout,end='')
    if p.returncode: raise SystemExit(p.returncode)
    return p.stdout

def parse_cpulist(s):
    out=[]
    for q in s.strip().split(','):
        if not q: continue
        if '-' in q:
            a,b=map(int,q.split('-',1)); out.extend(range(a,b+1))
        else: out.append(int(q))
    return out

def p8_topology():
    txt=Path('/proc/self/status').read_text(); m=re.search(r'Cpus_allowed_list:\s*(.+)',txt)
    allowed=parse_cpulist(m.group(1)) if m else list(range(os.cpu_count() or 1)); aset=set(allowed)
    groups={}
    for c in allowed:
        p=Path(f'/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list')
        sib=tuple(sorted(x for x in parse_cpulist(p.read_text()) if x in aset)) if p.exists() else (c,)
        groups.setdefault(sib,[]).append(c)
    pg=sorted([list(k) for k in groups if len(k)>1],key=lambda x:min(x))
    cpus=[x for g in pg for x in g]
    if not cpus: cpus=allowed[:min(8,len(allowed))]
    return cpus,','.join(map(str,cpus)),len(cpus)

def require():
    for p in [PROJECT/'build/dsasm-vocoder-m40',PROJECT/'build/dsasm-acoustic',BUNDLE,FIX/'reference_mel.f32',FIX/'golden_wave.f32',FIX/'f0.f32',PROJECT/'build/m42_acoustic']:
        if not p.exists(): raise SystemExit(f'ERROR missing {p}; run M53 preparation first')

def base_env(add,leaky,tile,range24=0,profile=False):
    e=os.environ.copy(); e.update({
        'DSASM_2D':'1','DSASM_KSPEC':'0','DSASM_K3_TMODE':'24','DSASM_K7_T24':'1','DSASM_K11_T24':'1',
        'DSASM_RANGE_T24':str(range24),'DSASM_VOCODER_T_TILE':str(tile),'DSASM_RESIDUAL_T24':'1',
        'DSASM_VNNI':'k11','DSASM_VNNI_ASYM':'0','DSASM_VNNI_CIN':'128',
        'DSASM_PARALLEL_ADD':str(add),'DSASM_PARALLEL_LEAKY_COPY':str(leaky),
        'DSASM_GOLDEN_COS':'0.999','DSASM_GOLDEN_SNR':'25'})
    if profile: e['DSASM_PROFILE_SHAPES']='1'
    else: e.pop('DSASM_PROFILE_SHAPES',None)
    return e

def vocoder(cpulist,threads,tag,add,leaky,tile,range24=0,profile=False):
    out=OUT/f'{tag}.wave.f32'; env=base_env(add,leaky,tile,range24,profile)
    cmd=['taskset','-c',cpulist,PROJECT/'build/dsasm-vocoder-m40','infer',BUNDLE,
         '--mel',FIX/'reference_mel.f32','--f0',FIX/'f0.f32','--out',out,
         '--golden',FIX/'golden_wave.f32','--workers',str(threads),'--rounds','1']
    if profile: cmd.append('--profile')
    print(f'\n========== M55 VOCODER {tag} add={add} leaky={leaky} tile={tile} range24={range24} ==========',flush=True)
    txt=sh(cmd,env); m=re.search(r'median=([0-9.]+) ms',txt)
    return float(m.group(1)) if m else 1e99

def measured_e2e(cpulist,threads,tag,add,leaky,tile,range24):
    work=OUT/tag
    if work.exists(): shutil.rmtree(work)
    cmd=[PYTHON,RUNNER,'run','--project',PROJECT,'--packed-acoustic',PROJECT/'build/m42_acoustic',
         '--acoustic-cli',PROJECT/'build/dsasm-acoustic','--vocoder-bundle',BUNDLE,
         '--vocoder-cli',PROJECT/'build/dsasm-vocoder-m40','--vocoder-onnx',MODEL/'dsvocoder/nsf_hifigan.onnx',
         '--speaker-emb',MODEL/'dongfangzhizi-nectar-xiao.emb','--fixture',FIX,'--work',work,
         '--frames','384','--cpus',cpulist,'--threads',str(threads)]
    print(f'\n----- M55 LONG E2E {tag} -----',flush=True)
    sh(cmd,base_env(add,leaky,tile,range24,False))
    return json.loads((work/'result.json').read_text())

def main():
    os.chdir(PROJECT); require(); cpus,cpulist,threads=p8_topology()
    print(f'M55 long topology: P-SMT cpus={cpulist} workers={threads}')

    configs=[('base',0,0),('add',1,0),('leaky',0,1),('both',1,1)]
    rows={k:[] for k,_,_ in configs}
    print('\n========== M55 LONG MEMORY-PASS A/B ==========',flush=True)
    for k,a,l in configs: rows[k].append(vocoder(cpulist,threads,k+'_a',a,l,504,0))
    for k,a,l in reversed(configs): rows[k].append(vocoder(cpulist,threads,k+'_b',a,l,504,0))
    print('\n========== M55 MEMORY SUMMARY ==========',flush=True)
    rank=[]
    cmap={k:(a,l) for k,a,l in configs}
    for k in rows:
        x=rows[k]; av=mean(x); rank.append((av,k)); print(f'{k:6s} mean={av:.3f} ms samples={x}')
    rank.sort(); mem_name=rank[0][1]; add,leaky=cmap[mem_name]
    print(f'BEST_MEMORY {mem_name} add={add} leaky={leaky}')

    tiles=[504,1008,2016,4032]; tv={t:[] for t in tiles}
    print('\n========== M55 LONG 2-D TILE SWEEP ==========',flush=True)
    for t in tiles: tv[t].append(vocoder(cpulist,threads,f'tile{t}_a',add,leaky,t,0))
    for t in reversed(tiles): tv[t].append(vocoder(cpulist,threads,f'tile{t}_b',add,leaky,t,0))
    print('\n========== M55 TILE SUMMARY ==========',flush=True)
    tr=[]
    for t,x in tv.items(): av=mean(x);tr.append((av,t));print(f'tile={t:4d} mean={av:.3f} ms samples={x}')
    tr.sort(); tile=tr[0][1]; print(f'BEST_TILE {tile}')

    rv={0:[],1:[]}
    print('\n========== M55 LONG RANGE-T24 RETEST ==========',flush=True)
    for r in [0,1,1,0]: rv[r].append(vocoder(cpulist,threads,f'range{r}_{len(rv[r])+1}',add,leaky,tile,r))
    rmeans={r:mean(x) for r,x in rv.items()}
    for r in [0,1]: print(f'RANGE_T24={r} mean={rmeans[r]:.3f} ms samples={rv[r]}')
    range24=min(rmeans,key=rmeans.get); print(f'BEST_RANGE_T24 {range24}')

    print('\n========== M55 WINNER LONG VOCODER PROFILE ==========',flush=True)
    pms=vocoder(cpulist,threads,'winner_profile',add,leaky,tile,range24,True)

    print('\n========== M55 LONG E2E x3 ==========',flush=True)
    reps=[measured_e2e(cpulist,threads,f'e2e_{i}',add,leaky,tile,range24) for i in range(1,4)]
    rtfs=[x['rtf'] for x in reps]
    print('\n========== M55 FINAL ==========',flush=True)
    print(f'config memory={mem_name} add={add} leaky={leaky} tile={tile} range24={range24} cpus={cpulist} workers={threads}')
    print(f'profile_vocoder_ms={pms:.3f}')
    for i,x in enumerate(reps,1): print(f'run{i} acoustic={x["acoustic_ms"]:.3f} vocoder={x["vocoder_ms"]:.3f} total={x["total_ms"]:.3f} RTF={x["rtf"]:.3f}')
    print(f'median_RTF={median(rtfs):.3f} worst_RTF={max(rtfs):.3f}')
    print('long_audio_realtime_all3='+('PASS' if max(rtfs)<1 else 'FAIL'))
    print('long_audio_engineering_0.8_all3='+('PASS' if max(rtfs)<=0.8 else 'FAIL'))

if __name__=='__main__': main()
