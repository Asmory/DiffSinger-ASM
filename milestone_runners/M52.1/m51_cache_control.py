#!/usr/bin/env python3
import argparse, os, time
from pathlib import Path

def iter_files(paths):
    seen=set()
    for p0 in paths:
        p=Path(p0)
        if p.is_dir():
            it=(x for x in p.rglob('*') if x.is_file())
        elif p.is_file():
            it=(p,)
        else:
            continue
        for x in it:
            try: k=(x.stat().st_dev,x.stat().st_ino)
            except OSError: continue
            if k in seen: continue
            seen.add(k); yield x

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('action',choices=['drop','touch']); ap.add_argument('paths',nargs='+'); a=ap.parse_args()
    files=list(iter_files(a.paths)); total=0; t0=time.perf_counter(); errs=0
    for p in files:
        try:
            if a.action=='drop':
                fd=os.open(p,os.O_RDONLY)
                try:
                    if hasattr(os,'posix_fadvise') and hasattr(os,'POSIX_FADV_DONTNEED'):
                        os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
                finally: os.close(fd)
                total += p.stat().st_size
            else:
                with open(p,'rb',buffering=0) as f:
                    while True:
                        b=f.read(8*1024*1024)
                        if not b: break
                        total += len(b)
        except OSError as e:
            errs += 1; print(f'M51 cache {a.action} warning: {p}: {e}')
    ms=(time.perf_counter()-t0)*1000
    print(f'M51 CACHE {a.action}: files={len(files)} bytes={total} ms={ms:.3f} errors={errs}')
if __name__=='__main__': main()
