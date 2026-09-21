#!/usr/bin/env python3
from __future__ import annotations
import json, os, re, subprocess, sys, time
from pathlib import Path
from statistics import mean

PROJECT=Path(os.environ.get('PROJECT', str(Path.home()/'asm/diffsinger'))).resolve()
PYTHON=Path(os.environ.get('PY', str(PROJECT/'.venv/bin/python')))
HERE=Path(__file__).resolve().parent
MODEL=Path(os.environ.get('MODEL', str(PROJECT/'model/DongFangZhiZi_Nectar_DiffSinger_CE_26.08.31')))
F=384
D=PROJECT/f'build/m53_long/f{F}'
FIX=D/'fixture'
BUNDLE=D/'nsf.dsv35'
RUNNER=HERE/'run_long_e2e_m53.py'
OUT=PROJECT/'build/m54_long'
OUT.mkdir(parents=True, exist_ok=True)


def sh(cmd, env=None, check=True):
    p=subprocess.run([str(x) for x in cmd], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    print(p.stdout, end='')
    if check and p.returncode:
        raise SystemExit(p.returncode)
    return p.stdout

def read_int(p):
    try: return int(Path(p).read_text().strip())
    except: return None

def parse_cpulist(s):
    out=[]
    for part in s.strip().split(','):
        if not part: continue
        if '-' in part:
            a,b=map(int,part.split('-',1)); out.extend(range(a,b+1))
        else: out.append(int(part))
    return out

def allowed_cpus():
    txt=Path('/proc/self/status').read_text()
    m=re.search(r'Cpus_allowed_list:\s*(.+)',txt)
    return parse_cpulist(m.group(1)) if m else list(range(os.cpu_count() or 1))

def topology():
    allowed=allowed_cpus(); aset=set(allowed)
    groups={}
    for c in allowed:
        p=Path(f'/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list')
        sib=tuple(sorted(x for x in parse_cpulist(p.read_text()) if x in aset)) if p.exists() else (c,)
        groups.setdefault(sib,[]).append(c)
    p_groups=sorted([list(k) for k in groups if len(k)>1], key=lambda x:min(x))
    used={x for g in p_groups for x in g}
    e=[c for c in allowed if c not in used]
    p_phys=[g[0] for g in p_groups]
    p_smt=[x for g in p_groups for x in g[1:]]
    specs=[]
    def add(name, cpus):
        cpus=[x for x in cpus if x in aset]
        if cpus: specs.append(dict(name=name,cpus=cpus,cpulist=','.join(map(str,cpus)),threads=len(cpus)))
    add('P4',p_phys)
    add('P6',p_phys+p_smt[:2])
    add('P8SMT',[x for g in p_groups for x in g])
    add('P4E4',p_phys+e[:4])
    add('ALL12',allowed)
    # unique cpu sets only
    uniq=[]; seen=set()
    for s in specs:
        k=tuple(s['cpus'])
        if k not in seen: seen.add(k); uniq.append(s)
    return dict(allowed=allowed,p_groups=p_groups,e_cpus=e,topologies=uniq)

def require():
    for p in [PROJECT/'build/dsasm-vocoder-m40',PROJECT/'build/dsasm-acoustic',BUNDLE,FIX/'reference_mel.f32',FIX/'golden_wave.f32',FIX/'f0.f32']:
        if not p.exists(): raise SystemExit(f'ERROR missing {p}')

def run_e2e(spec, tag):
    work=OUT/tag
    if work.exists():
        import shutil; shutil.rmtree(work)
    cmd=[PYTHON,RUNNER,'run','--project',PROJECT,'--packed-acoustic',PROJECT/'build/m42_acoustic',
         '--acoustic-cli',PROJECT/'build/dsasm-acoustic','--vocoder-bundle',BUNDLE,
         '--vocoder-cli',PROJECT/'build/dsasm-vocoder-m40','--vocoder-onnx',MODEL/'dsvocoder/nsf_hifigan.onnx',
         '--speaker-emb',MODEL/'dongfangzhizi-nectar-xiao.emb','--fixture',FIX,'--work',work,
         '--frames',str(F),'--cpus',spec['cpulist'],'--threads',str(spec['threads'])]
    print(f"\n========== M54 E2E {tag} cpus={spec['cpulist']} th={spec['threads']} ==========")
    sh(cmd)
    return json.loads((work/'result.json').read_text())

def run_vocoder(spec, d2, tag, profile=False):
    out=OUT/f'{tag}.wave.f32'
    env=os.environ.copy(); env['DSASM_2D']=str(int(d2)); env['DSASM_GOLDEN_COS']='0.999'; env['DSASM_GOLDEN_SNR']='25'
    if profile: env['DSASM_PROFILE_SHAPES']='1'
    cmd=['taskset','-c',spec['cpulist'],PROJECT/'build/dsasm-vocoder-m40','infer',BUNDLE,
         '--mel',FIX/'reference_mel.f32','--f0',FIX/'f0.f32','--out',out,'--golden',FIX/'golden_wave.f32',
         '--workers',str(spec['threads']),'--rounds','1']
    if profile: cmd.append('--profile')
    print(f"\n========== M54 VOCODER {tag} 2D={int(d2)} cpus={spec['cpulist']} th={spec['threads']} ==========")
    txt=sh(cmd,env=env)
    m=re.search(r'median=([0-9.]+) ms',txt); ms=float(m.group(1)) if m else 1e99
    return ms,txt

def main():
    os.chdir(PROJECT); require(); topo=topology(); print(json.dumps(topo,indent=2))
    specs=topo['topologies']
    rows=[]
    print('\n========== M54 BALANCED 4.458s LONG-TOPOLOGY SWEEP ==========')
    for s in specs:
        r=run_e2e(s,s['name']+'_a'); rows.append((s,r))
    for s in reversed(specs):
        r=run_e2e(s,s['name']+'_b'); rows.append((s,r))
    by={s['name']:[] for s in specs}
    smap={s['name']:s for s in specs}
    for s,r in rows: by[s['name']].append(r)
    print('\n========== M54 TOPOLOGY SUMMARY ==========')
    ranking=[]
    for name in by:
        xs=by[name]; ac=mean(x['acoustic_ms'] for x in xs); vo=mean(x['vocoder_ms'] for x in xs); tot=mean(x['total_ms'] for x in xs); rtf=mean(x['rtf'] for x in xs); worst=max(x['rtf'] for x in xs)
        ranking.append((tot,name,ac,vo,rtf,worst))
    ranking.sort()
    for tot,name,ac,vo,rtf,worst in ranking:
        s=smap[name]; print(f'{name:8s} cpus={s["cpulist"]:24s} th={s["threads"]:2d} acoustic={ac:8.1f} vocoder={vo:8.1f} total={tot:8.1f} mean_RTF={rtf:.3f} worst={worst:.3f}')
    winner=smap[ranking[0][1]]
    print(f'BEST_LONG_TOPOLOGY {winner["name"]} cpus={winner["cpulist"]} threads={winner["threads"]}')

    print('\n========== M54 LONG VOCODER 2D A/B ==========')
    vals={}
    for d2 in [1,0,0,1]:
        tag=f'd2_{d2}_{len([k for k in vals if k.startswith(str(d2))])+1}'
        ms,_=run_vocoder(winner,d2,tag,False)
        vals.setdefault(str(d2),[]).append(ms)
    for k in ['1','0']:
        print(f'DSASM_2D={k} mean_vocoder={mean(vals[k]):.3f} ms samples={vals[k]}')
    best2d=min((mean(v),int(k)) for k,v in vals.items())[1]
    print(f'BEST_2D {best2d}')

    print('\n========== M54 WINNER LONG VOCODER PROFILE ==========')
    ms,txt=run_vocoder(winner,best2d,'winner_profile',True)
    print('\n========== M54 FINAL ==========')
    print(f'topology={winner["name"]} cpus={winner["cpulist"]} workers={winner["threads"]} DSASM_2D={best2d}')
    print(f'profile_vocoder_ms={ms:.3f}')
    print('NEXT: inspect Conv/ConvTranspose/VNNI hot-shape distribution above; this is the sustained 4.458s optimization target.')

if __name__=='__main__': main()
