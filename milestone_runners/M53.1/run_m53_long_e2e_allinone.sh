#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
PROJECT="${PROJECT:-$HOME/asm/diffsinger}"
MODEL="${MODEL:-$PROJECT/model/DongFangZhiZi_Nectar_DiffSinger_CE_26.08.31}"
PY="${PY:-$PROJECT/.venv/bin/python}"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT"

printf '\n========== M53 TRUE LONG-AUDIO E2E =========='; printf '\n'
command -v taskset >/dev/null || { echo 'ERROR: taskset required'; exit 2; }
[[ -x build/dsasm-vocoder-m40 && -x build/dsasm-acoustic ]] || make -j"$(nproc)" build/dsasm-vocoder-m40 build/dsasm-acoustic
[[ -f build/m42_acoustic/model.json ]] || { echo 'ERROR: build/m42_acoustic missing'; exit 2; }
[[ -f "$MODEL/dsvocoder/nsf_hifigan.onnx" ]] || { echo 'ERROR: vocoder ONNX missing'; exit 2; }
mkdir -p build/m53_long tools
cp "$HERE/run_long_e2e_m53.py" tools/run_long_e2e_m53.py
cp "$HERE/discover_p4_m53.py" build/m53_long/discover_p4_m53.py
chmod +x tools/run_long_e2e_m53.py
"$PY" build/m53_long/discover_p4_m53.py > build/m53_long/topology.json
cat build/m53_long/topology.json
P4_CPUS="$($PY - <<'PY'
import json
print(json.load(open('build/m53_long/topology.json'))['cpulist'])
PY
)"
P4_TH="$($PY - <<'PY'
import json
print(len(json.load(open('build/m53_long/topology.json'))['P4']))
PY
)"
echo "M53 production topology: P4 cpus=$P4_CPUS threads=$P4_TH"

FRAMES_STR="${M53_FRAMES:-48 192 384}"
read -r -a FRAMES <<< "$FRAMES_STR"
REPS="${M53_REPS:-2}"

printf '\n========== M53 PACK + PREPARE LONG UTTERANCES (ORT OUTSIDE TIMING) =========='; printf '\n'
for F in "${FRAMES[@]}"; do
  D="build/m53_long/f$F"; mkdir -p "$D"
  if [[ ! -f "$D/nsf.dsv35" ]]; then
    rm -rf "$D/packer"
    "$PY" tools/pack_vocoder_graph_m35.py "$MODEL/dsvocoder/nsf_hifigan.onnx" \
      --frames "$F" --vnni-scope all-k711 --residual-scope k7ge128 \
      --out "$D/nsf.dsv35" --work "$D/packer"
  fi
  "$PY" tools/run_long_e2e_m53.py prepare \
    --project "$PROJECT" --packed-acoustic build/m42_acoustic --acoustic-cli ./build/dsasm-acoustic \
    --vocoder-bundle "$D/nsf.dsv35" --vocoder-cli ./build/dsasm-vocoder-m40 \
    --vocoder-onnx "$MODEL/dsvocoder/nsf_hifigan.onnx" --speaker-emb "$MODEL/dongfangzhizi-nectar-xiao.emb" \
    --fixture "$D/fixture" --frames "$F" --cpus "$P4_CPUS" --threads "$P4_TH"
done

# One-time service-style prefault, excluded from request timing.
printf '\n========== M53 ONE-TIME PREFAULT =========='; printf '\n'
"$PY" - "$MODEL" "$PROJECT/build/m42_acoustic" "$PROJECT/build/m53_long" <<'PY'
import os,sys,time
from pathlib import Path
roots=[Path(x) for x in sys.argv[1:]]; n=0; b=0; t=time.perf_counter()
for root in roots:
    for p in ([root] if root.is_file() else root.rglob('*')):
        if not p.is_file(): continue
        try:
            with p.open('rb',buffering=0) as f:
                while True:
                    x=f.read(8<<20)
                    if not x: break
                    b+=len(x)
            n+=1
        except OSError: pass
print(f'M53 prefault files={n} bytes={b} ms={(time.perf_counter()-t)*1000:.1f} (startup only)')
PY

run_case(){
  local F="$1" R="$2"
  local TAG="f${F}_r${R}"
  local D="build/m53_long/f${F}" W="build/m53_long/${TAG}"
  rm -rf "$W"
  printf '\n========== M53 LONG E2E frames=%s rep=%s ==========' "$F" "$R"; printf '\n'
  "$PY" tools/run_long_e2e_m53.py run \
    --project "$PROJECT" --packed-acoustic build/m42_acoustic --acoustic-cli ./build/dsasm-acoustic \
    --vocoder-bundle "$D/nsf.dsv35" --vocoder-cli ./build/dsasm-vocoder-m40 \
    --vocoder-onnx "$MODEL/dsvocoder/nsf_hifigan.onnx" --speaker-emb "$MODEL/dongfangzhizi-nectar-xiao.emb" \
    --fixture "$D/fixture" --work "$W" --frames "$F" --cpus "$P4_CPUS" --threads "$P4_TH"
  "$PY" - "$F" "$R" "$W/result.json" >> build/m53_long/results.tsv <<'PY'
import json,sys
r=json.load(open(sys.argv[3])); print('\t'.join(map(str,[sys.argv[1],sys.argv[2],r['audio_ms'],r['acoustic_ms'],r['vocoder_ms'],r['total_ms'],r['rtf']])))
PY
}

: > build/m53_long/results.tsv
printf '\n========== M53 BALANCED LONG-AUDIO E2E SWEEP =========='; printf '\n'
for ((R=1;R<=REPS;R++)); do
  if (( R % 2 )); then
    for F in "${FRAMES[@]}"; do run_case "$F" "$R"; done
  else
    for ((I=${#FRAMES[@]}-1;I>=0;I--)); do run_case "${FRAMES[$I]}" "$R"; done
  fi
done

printf '\n========== M53 LONG-AUDIO SUMMARY =========='; printf '\n'
"$PY" - <<'PY'
from pathlib import Path
from collections import defaultdict
from statistics import mean,median
G=defaultdict(list)
for l in Path('build/m53_long/results.tsv').read_text().splitlines():
    f,r,a,ac,v,t,rt=l.split('\t'); G[int(f)].append(tuple(map(float,(a,ac,v,t,rt))))
print('frames  audio_s  acoustic_ms  vocoder_ms  total_ms  median_RTF  worst_RTF')
for f in sorted(G):
    x=G[f]
    print(f'{f:6d} {median(z[0] for z in x)/1000:8.3f} {median(z[1] for z in x):11.1f} {median(z[2] for z in x):10.1f} {median(z[3] for z in x):9.1f} {median(z[4] for z in x):11.3f} {max(z[4] for z in x):10.3f}')
base=min(G); longest=max(G)
print(f'LONGEST frames={longest} audio={median(z[0] for z in G[longest])/1000:.3f}s median_RTF={median(z[4] for z in G[longest]):.3f} worst_RTF={max(z[4] for z in G[longest]):.3f}')
PY

LONGEST="${FRAMES[-1]}"
printf '\n========== M53 LONGEST TRUE BACK-TO-BACK x3 frames=%s ==========' "$LONGEST"; printf '\n'
: > build/m53_long/sustained.tsv
for R in 1 2 3; do
  F="$LONGEST"; D="build/m53_long/f$F"; W="build/m53_long/sustain_$R"; rm -rf "$W"
  "$PY" tools/run_long_e2e_m53.py run \
    --project "$PROJECT" --packed-acoustic build/m42_acoustic --acoustic-cli ./build/dsasm-acoustic \
    --vocoder-bundle "$D/nsf.dsv35" --vocoder-cli ./build/dsasm-vocoder-m40 \
    --vocoder-onnx "$MODEL/dsvocoder/nsf_hifigan.onnx" --speaker-emb "$MODEL/dongfangzhizi-nectar-xiao.emb" \
    --fixture "$D/fixture" --work "$W" --frames "$F" --cpus "$P4_CPUS" --threads "$P4_TH"
  "$PY" - "$R" "$W/result.json" >> build/m53_long/sustained.tsv <<'PY'
import json,sys
r=json.load(open(sys.argv[2])); print('\t'.join(map(str,[sys.argv[1],r['acoustic_ms'],r['vocoder_ms'],r['total_ms'],r['audio_ms'],r['rtf']])))
PY
done

printf '\n========== M53 FINAL =========='; printf '\n'
"$PY" - <<'PY'
from pathlib import Path
from statistics import median,mean
R=[]
for l in Path('build/m53_long/sustained.tsv').read_text().splitlines():
    i,a,v,t,au,r=l.split('\t'); R.append((int(i),*map(float,(a,v,t,au,r))))
for i,a,v,t,au,r in R: print(f'long_request{i} audio={au/1000:.3f}s acoustic={a:.1f} vocoder={v:.1f} total={t:.1f}ms RTF={r:.3f}')
rt=[x[-1] for x in R]
print(f'median_RTF={median(rt):.3f} worst_RTF={max(rt):.3f}')
print('long_audio_realtime_all3=' + ('PASS' if max(rt)<1 else 'FAIL'))
print('long_audio_engineering_0.8_all3=' + ('PASS' if max(rt)<=.8 else 'FAIL'))
PY

echo 'M53 note: this is real long-output E2E: acoustic T grows with audio length and native vocoder runs once with --rounds 1. ORT is preparation only.'
