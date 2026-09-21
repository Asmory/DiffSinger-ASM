#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
PROJECT="${PROJECT:-$HOME/asm/diffsinger}"
MODEL="${MODEL:-$PROJECT/model/DongFangZhiZi_Nectar_DiffSinger_CE_26.08.31}"
PY="${PY:-$PROJECT/.venv/bin/python}"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT"

printf '\n========== M52 STEADY-STATE TOPOLOGY / SMT SWEEP ==========\n'
command -v taskset >/dev/null 2>&1 || { echo 'ERROR: taskset is required (util-linux).'; exit 2; }
mkdir -p build/m52 tools
cp "$HERE/run_native_e2e_m52_topology.py" tools/run_native_e2e_m52_topology.py
cp "$HERE/m51_cache_control.py" tools/m52_cache_control.py
chmod +x tools/run_native_e2e_m52_topology.py tools/m52_cache_control.py

BUNDLE=build/m50/nsf_release.dsv35
ACOUSTIC=build/m42_acoustic
[[ -x build/dsasm-vocoder-m40 && -x build/dsasm-acoustic ]] || make -j"$(nproc)" build/dsasm-vocoder-m40 build/dsasm-acoustic
[[ -f "$BUNDLE" ]] || { echo "ERROR: missing $BUNDLE; run M50/M50.1 first."; exit 2; }
[[ -f "$ACOUSTIC/model.json" ]] || { echo "ERROR: missing $ACOUSTIC; run M42+ acoustic pack first."; exit 2; }

"$PY" "$HERE/discover_topology_m52.py" > build/m52/topology.json
cat build/m52/topology.json
mapfile -t TOPO < <("$PY" - <<'PY2'
import json
j=json.load(open('build/m52/topology.json'))
for x in j['topologies']:
    print(f"{x['name']}\t{x['cpulist']}\t{x['threads']}")
PY2
)
((${#TOPO[@]}>=2)) || { echo 'ERROR: M52 could not discover enough topology candidates.'; exit 2; }

FIXTURE=build/m52/fixture
if [[ -f build/m51/fixture/meta.json && -f build/m51/fixture/golden_wave.f32 ]]; then
  rm -rf "$FIXTURE"; cp -a build/m51/fixture "$FIXTURE"
  echo 'M52 fixture: reused clean M51 fixture/golden (ORT stays outside measured path).'
else
  rm -rf "$FIXTURE"
  # reference on all allowed CPUs; output is expected bit-exact across worker topology.
  first_name="${TOPO[0]%%$'\t'*}"; rest="${TOPO[0]#*$'\t'}"; first_cpus="${rest%%$'\t'*}"; first_threads="${rest##*$'\t'}"
  "$PY" tools/run_native_e2e_m52_topology.py prepare \
    --project "$PROJECT" --packed-acoustic "$ACOUSTIC" --acoustic-cli ./build/dsasm-acoustic \
    --vocoder-bundle "$BUNDLE" --vocoder-cli ./build/dsasm-vocoder-m40 --vocoder-onnx "$MODEL/dsvocoder/nsf_hifigan.onnx" \
    --speaker-emb "$MODEL/dongfangzhizi-nectar-xiao.emb" --fixture "$FIXTURE" \
    --acoustic-cpus "$first_cpus" --acoustic-threads "$first_threads" --vocoder-cpus "$first_cpus" --vocoder-workers "$first_threads" --rounds 1
fi

printf '\n========== M52 ONE-TIME MODEL INITIALIZATION / PREFAULT ==========\n'
CACHE_PATHS=("$ACOUSTIC" "$BUNDLE" "$MODEL/dongfangzhizi-nectar-xiao.emb" ./build/dsasm-acoustic ./build/dsasm-vocoder-m40 "$FIXTURE")
START_NS="$(date +%s%N)"
"$PY" tools/m52_cache_control.py touch "${CACHE_PATHS[@]}"
END_NS="$(date +%s%N)"; STARTUP_MS="$(( (END_NS-START_NS)/1000000 ))"
echo "M52 startup prefault=${STARTUP_MS} ms (one-time model initialization, excluded from request latency)"

: > build/m52/matrix.tsv
run_case(){
  local pass="$1" name="$2" cpus="$3" threads="$4"
  local idle="${M52_IDLE_SECONDS:-2}" rounds="${M52_ROUNDS:-7}"
  sleep "$idle"
  local tag="${name}_${pass}"
  local work="build/m52/$tag"
  local log="build/m52/$tag.log"
  rm -rf "$work"
  printf '\n========== M52 %s cpus=%s threads=%s ==========\n' "$tag" "$cpus" "$threads"
  "$PY" tools/run_native_e2e_m52_topology.py run \
    --project "$PROJECT" --packed-acoustic "$ACOUSTIC" --acoustic-cli ./build/dsasm-acoustic \
    --vocoder-bundle "$BUNDLE" --vocoder-cli ./build/dsasm-vocoder-m40 --vocoder-onnx "$MODEL/dsvocoder/nsf_hifigan.onnx" \
    --speaker-emb "$MODEL/dongfangzhizi-nectar-xiao.emb" --fixture "$FIXTURE" --work "$work" \
    --acoustic-cpus "$cpus" --acoustic-threads "$threads" --vocoder-cpus "$cpus" --vocoder-workers "$threads" \
    --rounds "$rounds" 2>&1 | tee "$log"
  local line a v t r
  line="$(grep 'M52 CLEAN E2E:' "$log" | tail -1)"
  a="$(sed -n 's/.*acoustic=\([0-9.]*\).*/\1/p' <<<"$line")"
  v="$(sed -n 's/.*vocoder=\([0-9.]*\).*/\1/p' <<<"$line")"
  t="$(sed -n 's/.*total=\([0-9.]*\) ms.*/\1/p' <<<"$line")"
  r="$(sed -n 's/.*RTF=\([0-9.]*\).*/\1/p' <<<"$line")"
  [[ -n "$r" ]] || { echo "M52 parse failure $tag"; exit 5; }
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$tag" "$name" "$cpus" "$threads" "$a" "$v" "$t" "$r" >> build/m52/matrix.tsv
}

printf '\n========== M52 BALANCED TOPOLOGY MATRIX ==========\n'
for row in "${TOPO[@]}"; do IFS=$'\t' read -r n c th <<<"$row"; run_case a "$n" "$c" "$th"; done
for ((i=${#TOPO[@]}-1;i>=0;i--)); do IFS=$'\t' read -r n c th <<<"${TOPO[$i]}"; run_case b "$n" "$c" "$th"; done

printf '\n========== M52 TOPOLOGY SUMMARY ==========\n'
"$PY" - "$STARTUP_MS" <<'PY2'
import sys
from pathlib import Path
from collections import defaultdict
from statistics import mean
rows=[]
for l in Path('build/m52/matrix.tsv').read_text().splitlines():
    tag,name,cpus,threads,a,v,t,r=l.split('\t')
    rows.append(dict(tag=tag,name=name,cpus=cpus,threads=int(threads),a=float(a),v=float(v),t=float(t),r=float(r)))
g=defaultdict(list)
for r in rows:g[r['name']].append(r)
stats=[]
for name,x in g.items():
    s=dict(name=name,cpus=x[0]['cpus'],threads=x[0]['threads'],ma=mean(z['a'] for z in x),mv=mean(z['v'] for z in x),mt=mean(z['t'] for z in x),mr=mean(z['r'] for z in x),worst=max(z['r'] for z in x))
    stats.append(s)
for s in sorted(stats,key=lambda z:(z['worst'],z['mr'])):
    print(f"{s['name']:8s} cpus={s['cpus']:18s} th={s['threads']:2d} acoustic={s['ma']:.3f} vocoder={s['mv']:.3f} total={s['mt']:.3f} mean_RTF={s['mr']:.3f} worst_RTF={s['worst']:.3f}")
best_same=min(stats,key=lambda z:(z['worst'],z['mr']))
best_a=min(stats,key=lambda z:(z['ma'],z['worst']))
best_v=min(stats,key=lambda z:(z['mv'],z['worst']))
print(f"BEST_SAME {best_same['name']} cpus={best_same['cpus']} threads={best_same['threads']}")
print(f"BEST_ACOUSTIC {best_a['name']} cpus={best_a['cpus']} threads={best_a['threads']}")
print(f"BEST_VOCODER {best_v['name']} cpus={best_v['cpus']} threads={best_v['threads']}")
Path('build/m52/candidates.env').write_text(
    f"M52_STARTUP_MS={sys.argv[1]}\n"
    f"SAME_NAME={best_same['name']}\nSAME_CPUS={best_same['cpus']}\nSAME_THREADS={best_same['threads']}\n"
    f"A_NAME={best_a['name']}\nA_CPUS={best_a['cpus']}\nA_THREADS={best_a['threads']}\n"
    f"V_NAME={best_v['name']}\nV_CPUS={best_v['cpus']}\nV_THREADS={best_v['threads']}\n")
PY2
# shellcheck disable=SC1091
source build/m52/candidates.env

: > build/m52/probe.tsv
run_mixed(){
  local tag="$1" acpus="$2" ath="$3" vcpus="$4" vth="$5"
  sleep "${M52_IDLE_SECONDS:-2}"
  local work="build/m52/$tag"
  local log="build/m52/$tag.log"
  rm -rf "$work"
  printf '\n========== M52 PROBE %s A=%s/%s V=%s/%s ==========\n' "$tag" "$acpus" "$ath" "$vcpus" "$vth"
  "$PY" tools/run_native_e2e_m52_topology.py run \
    --project "$PROJECT" --packed-acoustic "$ACOUSTIC" --acoustic-cli ./build/dsasm-acoustic \
    --vocoder-bundle "$BUNDLE" --vocoder-cli ./build/dsasm-vocoder-m40 --vocoder-onnx "$MODEL/dsvocoder/nsf_hifigan.onnx" \
    --speaker-emb "$MODEL/dongfangzhizi-nectar-xiao.emb" --fixture "$FIXTURE" --work "$work" \
    --acoustic-cpus "$acpus" --acoustic-threads "$ath" --vocoder-cpus "$vcpus" --vocoder-workers "$vth" \
    --rounds "${M52_ROUNDS:-7}" 2>&1 | tee "$log"
  local line a v t r; line="$(grep 'M52 CLEAN E2E:' "$log" | tail -1)"
  a="$(sed -n 's/.*acoustic=\([0-9.]*\).*/\1/p' <<<"$line")"; v="$(sed -n 's/.*vocoder=\([0-9.]*\).*/\1/p' <<<"$line")"; t="$(sed -n 's/.*total=\([0-9.]*\) ms.*/\1/p' <<<"$line")"; r="$(sed -n 's/.*RTF=\([0-9.]*\).*/\1/p' <<<"$line")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$tag" "$acpus" "$ath" "$vcpus" "$vth" "$a" "$v" "$t" "$r" >> build/m52/probe.tsv
}

printf '\n========== M52 SAME-vs-MIXED PROBE ==========\n'
run_mixed same_a  "$SAME_CPUS" "$SAME_THREADS" "$SAME_CPUS" "$SAME_THREADS"
run_mixed mixed_a "$A_CPUS" "$A_THREADS" "$V_CPUS" "$V_THREADS"
run_mixed mixed_b "$A_CPUS" "$A_THREADS" "$V_CPUS" "$V_THREADS"
run_mixed same_b  "$SAME_CPUS" "$SAME_THREADS" "$SAME_CPUS" "$SAME_THREADS"

"$PY" - <<'PY2'
from pathlib import Path
from collections import defaultdict
from statistics import mean
rows=[]
for l in Path('build/m52/probe.tsv').read_text().splitlines():
    tag,ac,at,vc,vt,a,v,t,r=l.split('\t'); base=tag.rsplit('_',1)[0]
    rows.append(dict(base=base,ac=ac,at=at,vc=vc,vt=vt,a=float(a),v=float(v),t=float(t),r=float(r)))
g=defaultdict(list)
for x in rows:g[x['base']].append(x)
for k in ('same','mixed'):
    x=g[k]; print(f"{k:5s} mean_total={mean(z['t'] for z in x):.3f} mean_RTF={mean(z['r'] for z in x):.3f} worst_RTF={max(z['r'] for z in x):.3f}")
best=min(g,key=lambda k:(max(z['r'] for z in g[k]),mean(z['r'] for z in g[k])))
x=g[best][0]
print(f"SELECTED {best} A={x['ac']}/{x['at']} V={x['vc']}/{x['vt']}")
Path('build/m52/selected.env').write_text(f"SELECTED={best}\nSEL_A_CPUS={x['ac']}\nSEL_A_THREADS={x['at']}\nSEL_V_CPUS={x['vc']}\nSEL_V_THREADS={x['vt']}\n")
PY2
# shellcheck disable=SC1091
source build/m52/selected.env

printf '\n========== M52 SUSTAINED VALIDATION (5 BACK-TO-BACK REQUESTS) ==========\n'
: > build/m52/validation.tsv
for i in 1 2 3 4 5; do
  tag="validate_$i"; work="build/m52/$tag"; log="build/m52/$tag.log"; rm -rf "$work"
  printf '\n----- M52 request %d/5 A=%s/%s V=%s/%s -----\n' "$i" "$SEL_A_CPUS" "$SEL_A_THREADS" "$SEL_V_CPUS" "$SEL_V_THREADS"
  "$PY" tools/run_native_e2e_m52_topology.py run \
    --project "$PROJECT" --packed-acoustic "$ACOUSTIC" --acoustic-cli ./build/dsasm-acoustic \
    --vocoder-bundle "$BUNDLE" --vocoder-cli ./build/dsasm-vocoder-m40 --vocoder-onnx "$MODEL/dsvocoder/nsf_hifigan.onnx" \
    --speaker-emb "$MODEL/dongfangzhizi-nectar-xiao.emb" --fixture "$FIXTURE" --work "$work" \
    --acoustic-cpus "$SEL_A_CPUS" --acoustic-threads "$SEL_A_THREADS" --vocoder-cpus "$SEL_V_CPUS" --vocoder-workers "$SEL_V_THREADS" \
    --rounds "${M52_VALIDATE_ROUNDS:-7}" 2>&1 | tee "$log"
  line="$(grep 'M52 CLEAN E2E:' "$log" | tail -1)"; a="$(sed -n 's/.*acoustic=\([0-9.]*\).*/\1/p' <<<"$line")"; v="$(sed -n 's/.*vocoder=\([0-9.]*\).*/\1/p' <<<"$line")"; t="$(sed -n 's/.*total=\([0-9.]*\) ms.*/\1/p' <<<"$line")"; r="$(sed -n 's/.*RTF=\([0-9.]*\).*/\1/p' <<<"$line")"
  printf '%s\t%s\t%s\t%s\t%s\n' "$i" "$a" "$v" "$t" "$r" >> build/m52/validation.tsv
done

printf '\n========== M52 FINAL ==========\n'
"$PY" - "$M52_STARTUP_MS" "$SELECTED" "$SEL_A_CPUS" "$SEL_A_THREADS" "$SEL_V_CPUS" "$SEL_V_THREADS" <<'PY2'
import sys
from pathlib import Path
from statistics import median,mean
rows=[]
for l in Path('build/m52/validation.tsv').read_text().splitlines():
    i,a,v,t,r=l.split('\t'); rows.append((int(i),float(a),float(v),float(t),float(r)))
for x in rows: print(f'request{x[0]} acoustic={x[1]:.3f} vocoder={x[2]:.3f} total={x[3]:.3f} RTF={x[4]:.3f}')
rt=[x[4] for x in rows]; tot=[x[3] for x in rows]
print(f'selected={sys.argv[2]} acoustic_cpus={sys.argv[3]} acoustic_threads={sys.argv[4]} vocoder_cpus={sys.argv[5]} vocoder_workers={sys.argv[6]}')
print(f'one_time_model_prefault={float(sys.argv[1]):.0f} ms (excluded from request latency)')
print(f'mean_total={mean(tot):.3f} ms median_total={median(tot):.3f} ms median_RTF={median(rt):.3f} worst_RTF={max(rt):.3f}')
print('steady_realtime_all5=' + ('PASS' if max(rt)<1 else 'FAIL'))
print('steady_engineering_0.8_all5=' + ('PASS' if max(rt)<=.8 else 'FAIL'))
PY2
