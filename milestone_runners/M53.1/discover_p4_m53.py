#!/usr/bin/env python3
import os,json
from pathlib import Path
allowed=sorted(os.sched_getaffinity(0)); seen=set(); groups=[]
for c in allowed:
    p=Path(f'/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list')
    if not p.exists(): continue
    vals=[]
    for q in p.read_text().strip().split(','):
        if '-' in q:
            a,b=map(int,q.split('-')); vals.extend(range(a,b+1))
        else: vals.append(int(q))
    t=tuple(sorted(x for x in vals if x in allowed))
    if len(t)>=2 and t not in seen: seen.add(t); groups.append(t)
if not groups: raise SystemExit('no SMT P-core groups found')
p4=[g[0] for g in groups]
print(json.dumps({'allowed':allowed,'p_groups':groups,'P4':p4,'cpulist':','.join(map(str,p4))},indent=2))
