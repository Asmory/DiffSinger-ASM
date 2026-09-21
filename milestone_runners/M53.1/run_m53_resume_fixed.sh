#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
PROJECT="${PROJECT:-$HOME/asm/diffsinger}"
MODEL="${MODEL:-$PROJECT/model/DongFangZhiZi_Nectar_DiffSinger_CE_26.08.31}"
PY="${PY:-$PROJECT/.venv/bin/python}"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT"

printf '\n========== M53.1 RESUME LONG-AUDIO SWEEP / TAG HOTFIX ==========\n'
command -v taskset >/dev/null || { echo 'ERROR: taskset required'; exit 2; }
[[ -x build/dsasm-vocoder-m40 && -x build/dsasm-acoustic ]] || { echo 'ERROR: native binaries missing'; exit 2; }
mkdir -p build/m53_long tools
cp "$HERE/run_long_e2e_m53.py" tools/run_long_e2e_m53.py
cp "$HERE/discover_p4_m53.py" build/m53_long/discover_p4_m53.py
chmod +x tools/run_long_e2e_m53.py

if [[ ! -f build/m53_long/topology.json ]]; then
  "$PY" build/m53_long/discover_p4_m53.py > build/m53_long/topology.json
fi
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
echo "M53.1 production topology: P4 cpus=$P4_CPUS threads=$P4_TH"

FRAMES_STR="${M53_FRAMES:-48 192 384}"
read -r -a FRAMES <<< "$FRAMES_STR"
REPS="${M53_REPS:-2}"

# Resume requires the already prepared fixture/bundle for every requested frame.
for F in "${FRAMES[@]}"; do
  D="build/m53_long/f${F}"
  [[ -f "$D/nsf.dsv35" ]] || { echo "ERROR: missing $D/nsf.dsv35; run full fixed script once"; exit 2; }
  [[ -f "$D/fixture/reference_mel.f32" ]] || { echo "ERROR: missing $D/fixture/reference_mel.f32; run full fixed script once"; exit 2; }
  [[ -f "$D/fixture/golden_wave.f32" ]] || { echo "ERROR: missing $D/fixture/golden_wave.f32; run full fixed script once"; exit 2; }
done

echo "M53.1 resume: reusing prepared bundles/fixtures/goldens and prior one-time prefault"

run_case(){
  local F="$1" R="$2"
  local TAG="f${F}_r${R}"
  local D="build/m53_long/f${F}" W="build/m53_long/${TAG}"
  rm -rf "$W"
  printf '\n========== M53 LONG E2E frames=%s rep=%s ==========\n' "$F" "$R"
  "$PY" tools/run_long_e2e_m53.py run \
    --project "$PROJECT" --packed-acoustic build/m42_acoustic --acoustic-cli ./build/dsasm-acoustic \
    --vocoder-bundle "$D/nsf.dsv35" --vocoder-cli ./build/dsasm-vocoder-m40 \
    --vocoder-onnx "$MODEL/dsvocoder/nsf_hifigan.onnx" --speaker-emb "$MODEL/dongfangzhizi-nectar-xiao.emb" \
    --fixture "$D/fixture" --work "$W" --frames "$F" --cpus "$P4_CPUS" --threads "$P4_TH"
  "$PY" - "$F" "$R" "$W/result.json" >> build/m53_long/results.tsv <<'PY'
import json,sys
r=json.load(open(sys.argv[3]))
print('\t'.join(map(str,[sys.argv[1],sys.argv[2],r['audio_ms'],r['acoustic_ms'],r['vocoder_ms'],r['total_ms'],r['rtf']])))
PY
}

: > build/m53_long/results.tsv
printf '\n========== M53 BALANCED LONG-AUDIO E2E SWEEP ==========\n'
for ((R=1;R<=REPS;R++)); do
  if (( R % 2 )); then
    for F in "${FRAMES[@]}"; do run_case "$F" "$R"; done
  else
    for ((I=${#FRAMES[@]}-1;I>=0;I--)); do run_case "${FRAMES[$I]}" "$R"; done
  fi
done

printf '\n========== M53 LONG-AUDIO SUMMARY ==========\n'
"$PY" - <<'PY'
from pathlib import Path
from collections import defaultdict
from statistics import median
G=defaultdict(list)
for l in Path('build/m53_long/results.tsv').read_text().splitlines():
    f,r,a,ac,v,t,rt=l.split('\t'); G[int(f)].append(tuple(map(float,(a,ac,v,t,rt))))
print('frames  audio_s  acoustic_ms  vocoder_ms  total_ms  median_RTF  worst_RTF')
for f in sorted(G):
    x=G[f]
    print(f'{f:6d} {median(z[0] for z in x)/1000:8.3f} {median(z[1] for z in x):11.1f} {median(z[2] for z in x):10.1f} {median(z[3] for z in x):9.1f} {median(z[4] for z in x):11.3f} {max(z[4] for z in x):10.3f}')
longest=max(G)
print(f'LONGEST frames={longest} audio={median(z[0] for z in G[longest])/1000:.3f}s median_RTF={median(z[4] for z in G[longest]):.3f} worst_RTF={max(z[4] for z in G[longest]):.3f}')
PY

LONGEST="${FRAMES[-1]}"
printf '\n========== M53 LONGEST TRUE BACK-TO-BACK x3 frames=%s ==========\n' "$LONGEST"
: > build/m53_long/sustained.tsv
for R in 1 2 3; do
  F="$LONGEST"; D="build/m53_long/f${F}"; W="build/m53_long/sustain_${R}"; rm -rf "$W"
  "$PY" tools/run_long_e2e_m53.py run \
    --project "$PROJECT" --packed-acoustic build/m42_acoustic --acoustic-cli ./build/dsasm-acoustic \
    --vocoder-bundle "$D/nsf.dsv35" --vocoder-cli ./build/dsasm-vocoder-m40 \
    --vocoder-onnx "$MODEL/dsvocoder/nsf_hifigan.onnx" --speaker-emb "$MODEL/dongfangzhizi-nectar-xiao.emb" \
    --fixture "$D/fixture" --work "$W" --frames "$F" --cpus "$P4_CPUS" --threads "$P4_TH"
  "$PY" - "$R" "$W/result.json" >> build/m53_long/sustained.tsv <<'PY'
import json,sys
r=json.load(open(sys.argv[2]))
print('\t'.join(map(str,[sys.argv[1],r['audio_ms'],r['acoustic_ms'],r['vocoder_ms'],r['total_ms'],r['rtf']])))
PY
done

printf '\n========== M53 FINAL ==========\n'
"$PY" - <<'PY'
from pathlib import Path
from statistics import median
rows=[]
for l in Path('build/m53_long/sustained.tsv').read_text().splitlines():
    r,a,ac,v,t,rt=l.split('\t'); rows.append(tuple(map(float,(r,a,ac,v,t,rt))))
for r,a,ac,v,t,rt in rows:
    print(f'long_request{int(r)} audio={a/1000:.3f}s acoustic={ac:.3f} vocoder={v:.3f} total={t:.3f} RTF={rt:.3f}')
rt=[x[5] for x in rows]
print(f'median_RTF={median(rt):.3f} worst_RTF={max(rt):.3f}')
print('long_audio_realtime_all3=' + ('PASS' if max(rt)<1.0 else 'FAIL'))
print('long_audio_engineering_0.8_all3=' + ('PASS' if max(rt)<=0.8 else 'FAIL'))
PY
