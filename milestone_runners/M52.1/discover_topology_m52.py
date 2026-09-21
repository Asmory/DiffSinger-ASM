#!/usr/bin/env python3
from __future__ import annotations
import json, os
from pathlib import Path
allowed=sorted(os.sched_getaffinity(0))
groups={}
for c in allowed:
    base=Path(f'/sys/devices/system/cpu/cpu{c}/topology')
    try: pkg=int((base/'physical_package_id').read_text()); core=int((base/'core_id').read_text())
    except Exception: pkg=0; core=c
    groups.setdefault((pkg,core),[]).append(c)
for v in groups.values(): v.sort()
p=[v for _,v in sorted(groups.items()) if len(v)>1]
e=[v[0] for _,v in sorted(groups.items()) if len(v)==1]
if not p:
    # non-hybrid fallback: all core groups are homogeneous; there is no E/P split.
    p=[v for _,v in sorted(groups.items())]
    e=[]
prim=[v[0] for v in p]
sibs=[c for v in p for c in v[1:]]
pall=[c for v in p for c in v]

def ent(name,cpus):
    cpus=list(dict.fromkeys(cpus)); return {'name':name,'cpus':cpus,'cpulist':','.join(map(str,cpus)),'threads':len(cpus)}
rows=[]
if prim: rows.append(ent(f'P{len(prim)}',prim))
if len(pall)>=6 and len(prim)<6: rows.append(ent('P6',prim+sibs[:max(0,6-len(prim))]))
if pall and set(pall)!=set(prim): rows.append(ent(f'P{len(pall)}SMT',pall))
if e and prim: rows.append(ent(f'P{len(prim)}E{len(e)}',prim+e))
allpe=pall+e
if allpe: rows.append(ent(f'ALL{len(allpe)}',allpe))
# dedupe exact CPU sets
out=[]; seen=set()
for r in rows:
    k=tuple(sorted(r['cpus']))
    if k not in seen: out.append(r); seen.add(k)
print(json.dumps({'allowed':allowed,'p_groups':p,'e_cpus':e,'topologies':out},indent=2))
