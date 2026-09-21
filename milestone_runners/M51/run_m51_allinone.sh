#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
PROJECT="${PROJECT:-$HOME/asm/diffsinger}"
MODEL="${MODEL:-$PROJECT/model/DongFangZhiZi_Nectar_DiffSinger_CE_26.08.31}"
PY="${PY:-$PROJECT/.venv/bin/python}"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT"

printf '\n========== M51 CLEAN E2E + COLD-START DIAGNOSIS ==========\n'
mkdir -p build/m51 tools scripts docs
cp "$HERE/run_native_e2e_m51_clean.py" tools/run_native_e2e_m51_clean.py
cp "$HERE/m51_cache_control.py" tools/m51_cache_control.py
chmod +x tools/run_native_e2e_m51_clean.py tools/m51_cache_control.py
cc -O3 -mavx2 -mfma -pthread "$HERE/m51_cpu_ramp.c" -o build/m51/cpu_ramp

BUNDLE=build/m50/nsf_release.dsv35
ACOUSTIC=build/m42_acoustic
if [[ ! -x build/dsasm-vocoder-m40 || ! -x build/dsasm-acoustic ]]; then
  echo 'M51: native binaries missing; rebuilding.'
  make -j"$(nproc)" build/dsasm-vocoder-m40 build/dsasm-acoustic
fi
if [[ ! -f "$BUNDLE" ]]; then
  echo 'M51: M50 release bundle missing; packing it.'
  rm -rf build/m50/packer; mkdir -p build/m50
  "$PY" tools/pack_vocoder_graph_m35.py "$MODEL/dsvocoder/nsf_hifigan.onnx" \
    --frames 48 --vnni-scope all-k711 --residual-scope k7ge128 \
    --out "$BUNDLE" --work build/m50/packer
fi
if [[ ! -f "$ACOUSTIC/model.json" ]]; then
  echo 'M51: packed acoustic missing; packing it.'
  rm -rf "$ACOUSTIC"
  "$PY" tools/pack_acoustic_onnx_m25.py "$MODEL/acoustic.onnx" --model-dir "$MODEL" --out "$ACOUSTIC"
fi

FIXTURE=build/m51/fixture
rm -rf "$FIXTURE"
printf '\n========== M51 PREPARE FIXTURE / GOLDEN ONCE ==========\n'
"$PY" tools/run_native_e2e_m51_clean.py prepare \
  --project "$PROJECT" --packed-acoustic "$ACOUSTIC" --acoustic-cli ./build/dsasm-acoustic \
  --vocoder-bundle "$BUNDLE" --vocoder-cli ./build/dsasm-vocoder-m40 \
  --vocoder-onnx "$MODEL/dsvocoder/nsf_hifigan.onnx" \
  --speaker-emb "$MODEL/dongfangzhizi-nectar-xiao.emb" \
  --fixture "$FIXTURE" --language-id 4 --depth .6 --steps 4 --workers 8 --rounds 1

# Files whose page residency can influence a native first run.  The ORT model is deliberately excluded
# because ORT is no longer in the measured native path.
CACHE_PATHS=("$ACOUSTIC" "$BUNDLE" "$MODEL/dongfangzhizi-nectar-xiao.emb" ./build/dsasm-acoustic ./build/dsasm-vocoder-m40)

# Discover P-core logical CPUs by highest advertised max frequency.
P_CPUS="$($PY - <<'PY2'
from pathlib import Path
rows=[]
for p in Path('/sys/devices/system/cpu').glob('cpu[0-9]*/cpufreq/cpuinfo_max_freq'):
    try: rows.append((int(p.parent.parent.name[3:]), int(p.read_text())))
    except Exception: pass
if rows:
    m=max(v for _,v in rows); print(','.join(str(c) for c,v in rows if v==m))
else: print('0-7')
PY2
)"
P_THREADS="$($PY - "$P_CPUS" <<'PY2'
import sys
s=sys.argv[1]; n=0
for x in s.split(','):
    if '-' in x:
        a,b=map(int,x.split('-',1)); n+=b-a+1
    elif x.strip(): n+=1
print(max(1,min(n,8)))
PY2
)"
echo "M51 P-core cpus=$P_CPUS ramp_threads=$P_THREADS"

freq_avg(){
  "$PY" - "$P_CPUS" <<'PY2'
from pathlib import Path
import sys
cp=[]
for x in sys.argv[1].split(','):
    if '-' in x:
        a,b=map(int,x.split('-',1)); cp += list(range(a,b+1))
    elif x.strip(): cp.append(int(x))
v=[]
for c in cp:
    p=Path(f'/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_cur_freq')
    try:v.append(int(p.read_text())/1000)
    except:pass
print(f'{sum(v)/len(v):.0f}' if v else 'nan')
PY2
}

cache_drop(){ "$PY" tools/m51_cache_control.py drop "${CACHE_PATHS[@]}"; }
cache_touch(){ "$PY" tools/m51_cache_control.py touch "${CACHE_PATHS[@]}"; }
cpu_ramp(){
  local ms="$1"
  if command -v taskset >/dev/null 2>&1; then taskset -c "$P_CPUS" build/m51/cpu_ramp "$ms" "$P_THREADS"
  else build/m51/cpu_ramp "$ms" "$P_THREADS"; fi
}

: > build/m51/diag.tsv
run_case(){
  local tag="$1" touch="$2" ramp_ms="$3"
  local idle="${M51_IDLE_SECONDS:-4}" rounds="${M51_ROUNDS:-7}"
  printf '\n========== M51 %s touch=%s ramp_ms=%s ==========\n' "$tag" "$touch" "$ramp_ms"
  cache_drop
  sleep "$idle"
  local f0 f1 f2 prep0 prep1 prep_ms
  f0="$(freq_avg)"; prep0="$(date +%s%N)"
  if [[ "$touch" == 1 ]]; then cache_touch; fi
  if (( ramp_ms > 0 )); then cpu_ramp "$ramp_ms"; fi
  prep1="$(date +%s%N)"; prep_ms="$(( (prep1-prep0)/1000000 ))"; f1="$(freq_avg)"
  local work="build/m51/$tag" log="build/m51/$tag.log"
  rm -rf "$work"
  env DSASM_GOLDEN_COS=.999 DSASM_GOLDEN_SNR=25 \
    "$PY" tools/run_native_e2e_m51_clean.py run \
      --project "$PROJECT" --packed-acoustic "$ACOUSTIC" --acoustic-cli ./build/dsasm-acoustic \
      --vocoder-bundle "$BUNDLE" --vocoder-cli ./build/dsasm-vocoder-m40 \
      --vocoder-onnx "$MODEL/dsvocoder/nsf_hifigan.onnx" \
      --speaker-emb "$MODEL/dongfangzhizi-nectar-xiao.emb" \
      --fixture "$FIXTURE" --work "$work" --language-id 4 --depth .6 --steps 4 --workers 8 --rounds "$rounds" 2>&1 | tee "$log"
  f2="$(freq_avg)"
  local line acoustic vocoder total rtf audio
  line="$(grep 'M51 CLEAN E2E:' "$log" | tail -1)"
  acoustic="$(sed -n 's/.*acoustic=\([0-9.]*\).*/\1/p' <<<"$line")"
  vocoder="$(sed -n 's/.*vocoder=\([0-9.]*\).*/\1/p' <<<"$line")"
  total="$(sed -n 's/.*total=\([0-9.]*\) ms.*/\1/p' <<<"$line")"
  audio="$(sed -n 's/.*audio=\([0-9.]*\).*/\1/p' <<<"$line")"
  rtf="$(sed -n 's/.*RTF=\([0-9.]*\).*/\1/p' <<<"$line")"
  if [[ -z "$total" || -z "$rtf" ]]; then echo "M51 parse failure $tag" >&2; exit 5; fi
  local incl_rtf
  incl_rtf="$($PY - "$prep_ms" "$total" "$audio" <<'PY2'
import sys
p,t,a=map(float,sys.argv[1:]); print(f'{(p+t)/a:.4f}')
PY2
)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$tag" "$touch" "$ramp_ms" "$prep_ms" "$acoustic" "$vocoder" "$total" "$rtf" "$incl_rtf" "$f0" "$f1/$f2" >> build/m51/diag.tsv
  echo "M51 CASE $tag: prep=${prep_ms}ms freq=${f0}->${f1}->${f2}MHz compute_RTF=$rtf inclusive_RTF=$incl_rtf"
}

printf '\n========== M51 BALANCED COLD-START MATRIX ==========\n'
run_case cold_a     0 0
run_case prefault_a 1 0
run_case ramp_a     0 "${M51_RAMP_MS:-250}"
run_case both_a     1 "${M51_RAMP_MS:-250}"
run_case both_b     1 "${M51_RAMP_MS:-250}"
run_case ramp_b     0 "${M51_RAMP_MS:-250}"
run_case prefault_b 1 0
run_case cold_b     0 0

printf '\n========== M51 DIAGNOSIS SUMMARY ==========\n'
"$PY" - <<'PY2'
from pathlib import Path
from collections import defaultdict
from statistics import mean
rows=[]
for l in Path('build/m51/diag.tsv').read_text().splitlines():
    tag,touch,ramp,prep,a,v,total,rtf,incl,f0,f12=l.split('\t')
    base=tag.rsplit('_',1)[0]
    rows.append(dict(tag=tag,base=base,touch=int(touch),ramp=int(ramp),prep=float(prep),a=float(a),v=float(v),total=float(total),rtf=float(rtf),incl=float(incl),f0=f0,f12=f12))
g=defaultdict(list)
for r in rows:g[r['base']].append(r)
for name in ['cold','prefault','ramp','both']:
    x=g[name]
    print(f"{name:8s} mean_total={mean(r['total'] for r in x):.3f} ms mean_RTF={mean(r['rtf'] for r in x):.3f} worst_RTF={max(r['rtf'] for r in x):.3f} mean_prep={mean(r['prep'] for r in x):.1f} ms mean_inclusive_RTF={mean(r['incl'] for r in x):.3f}")
# Selection: first prefer strategies whose forced-cold worst compute RTF meets 0.8,
# then least startup work. Otherwise choose lowest worst compute RTF.
cands=[]
for name,x in g.items():
    c=dict(name=name,worst=max(r['rtf'] for r in x),avg=mean(r['rtf'] for r in x),prep=mean(r['prep'] for r in x),touch=x[0]['touch'],ramp=x[0]['ramp'])
    cands.append(c)
passing=[c for c in cands if c['worst']<=.8]
if passing: best=min(passing,key=lambda c:(c['prep'],c['worst'],c['avg']))
else: best=min(cands,key=lambda c:(c['worst'],c['avg'],c['prep']))
print(f"SELECTED {best['name']} touch={best['touch']} ramp_ms={best['ramp']} worst_RTF={best['worst']:.3f} avg_RTF={best['avg']:.3f} prep_ms={best['prep']:.1f}")
Path('build/m51/selected.env').write_text(f"M51_SELECTED={best['name']}\nM51_TOUCH={best['touch']}\nM51_SELECTED_RAMP_MS={best['ramp']}\n")
PY2

# shellcheck disable=SC1091
source build/m51/selected.env
printf '\n========== M51 FORCED-COLD VALIDATION selected=%s touch=%s ramp=%s ==========\n' "$M51_SELECTED" "$M51_TOUCH" "$M51_SELECTED_RAMP_MS"
: > build/m51/validation.tsv
for i in 1 2 3; do
  # run_case writes diag.tsv too; parse its emitted case and copy row to validation.
  run_case "validate_$i" "$M51_TOUCH" "$M51_SELECTED_RAMP_MS"
  tail -1 build/m51/diag.tsv >> build/m51/validation.tsv
done

printf '\n========== M51 FINAL ==========\n'
"$PY" - <<'PY2'
from pathlib import Path
from statistics import median
rows=[]
for l in Path('build/m51/validation.tsv').read_text().splitlines():
    tag,touch,ramp,prep,a,v,total,rtf,incl,f0,f12=l.split('\t')
    rows.append((tag,float(prep),float(a),float(v),float(total),float(rtf),float(incl),f0,f12))
for r in rows:
    print(f'{r[0]} prep={r[1]:.0f}ms acoustic={r[2]:.3f} vocoder={r[3]:.3f} total={r[4]:.3f} compute_RTF={r[5]:.3f} inclusive_RTF={r[6]:.3f} freq={r[7]}->{r[8]}MHz')
rt=[r[5] for r in rows]; inc=[r[6] for r in rows]; tot=[r[4] for r in rows]
print(f'median_total={median(tot):.3f} ms median_compute_RTF={median(rt):.3f} worst_compute_RTF={max(rt):.3f}')
print(f'worst_inclusive_first_request_RTF={max(inc):.3f}')
print('forced_cold_realtime_all3=' + ('PASS' if max(rt)<1 else 'FAIL'))
print('forced_cold_engineering_0.8_all3=' + ('PASS' if max(rt)<=.8 else 'FAIL'))
print('NOTE: inclusive_RTF includes one-time prefault/ramp startup work; compute_RTF is native acoustic+vocoder only.')
PY2
