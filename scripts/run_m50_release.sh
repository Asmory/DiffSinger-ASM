#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
PROJECT="${PROJECT:-$HOME/asm/diffsinger}"
MODEL="${MODEL:-$PROJECT/model/DongFangZhiZi_Nectar_DiffSinger_CE_26.08.31}"
PY="${PY:-$PROJECT/.venv/bin/python}"
cd "$PROJECT"

printf '\n========== M50 BUILD / WARNING GATE ==========\n'
make clean >/dev/null
mkdir -p build/m50
# Capture build output and fail if the old DSAsmJob vnni_fn warning returns.
make -j"$(nproc)" build/dsasm-vocoder-m40 build/dsasm-acoustic 2>&1 | tee build/m50/build.log
if grep -q "missing initializer for field 'vnni_fn'" build/m50/build.log; then
  echo 'M50 ERROR: DSAsmJob vnni_fn warning regressed' >&2; exit 2
fi

printf '\n========== M50 PACK RELEASE BUNDLE ==========\n'
rm -rf build/m50/packer
"$PY" tools/pack_vocoder_graph_m35.py "$MODEL/dsvocoder/nsf_hifigan.onnx" \
  --frames 48 --vnni-scope all-k711 --residual-scope k7ge128 \
  --out build/m50/nsf_release.dsv35 --work build/m50/packer
BUNDLE=build/m50/nsf_release.dsv35
MEL=build/m50/packer/mel.f32
F0=build/m50/packer/f0.f32
GOLD=build/m50/packer/golden_wave.f32

PERF_ARGS=()
if command -v perf >/dev/null 2>&1; then
  if perf stat -e cpu_core/cycles/u -e cpu_core/instructions/u -- true >/dev/null 2>&1; then
    PERF_ARGS=(-e cpu_core/cycles/u -e cpu_core/instructions/u)
  elif perf stat -e cycles:u -e instructions:u -- true >/dev/null 2>&1; then
    PERF_ARGS=(-e cycles:u -e instructions:u)
  fi
fi

run_case(){
  local tag="$1" explicit="$2" log="build/m50/${tag}.log" pf="build/m50/${tag}.perf"
  local -a E=(DSASM_GOLDEN_COS=0.999 DSASM_GOLDEN_SNR=25 DSASM_PROFILE_SHAPES=0)
  if [[ "$explicit" == 1 ]]; then
    E+=(DSASM_PARALLEL_ADD=1 DSASM_PARALLEL_LEAKY_COPY=1 DSASM_CONVT_S8K16=1 DSASM_VNNI=k117 DSASM_VNNI_ASYM=1 DSASM_VNNI_CIN=128,64
       DSASM_KSPEC=0 DSASM_K3_TMODE=24 DSASM_K7_T24=1 DSASM_K11_T24=1
       DSASM_RANGE_T24=1 DSASM_RANGE_RESIDUAL_T24=1 DSASM_VOCODER_T_TILE=2016 DSASM_RESIDUAL_T24=1)
  else
    # Prove the release defaults work without the tuning variables.
    unset DSASM_PARALLEL_ADD DSASM_PARALLEL_LEAKY_COPY DSASM_CONVT_S8K16 DSASM_VNNI DSASM_VNNI_ASYM DSASM_VNNI_CIN \
          DSASM_KSPEC DSASM_K3_TMODE DSASM_K7_T24 DSASM_K11_T24 DSASM_RANGE_T24 \
          DSASM_RANGE_RESIDUAL_T24 DSASM_VOCODER_T_TILE DSASM_RESIDUAL_T24 || true
  fi
  printf '\n========== M50 %s explicit=%s ==========\n' "$tag" "$explicit"
  if ((${#PERF_ARGS[@]})); then
    env "${E[@]}" perf stat -x, -o "$pf" "${PERF_ARGS[@]}" -- \
      ./build/dsasm-vocoder-m40 infer "$BUNDLE" --mel "$MEL" --f0 "$F0" \
        --out "build/m50/${tag}.f32" --golden "$GOLD" --workers 8 --rounds 7 2>&1 | tee "$log"
  else
    : > "$pf"
    env "${E[@]}" ./build/dsasm-vocoder-m40 infer "$BUNDLE" --mel "$MEL" --f0 "$F0" \
      --out "build/m50/${tag}.f32" --golden "$GOLD" --workers 8 --rounds 7 2>&1 | tee "$log"
  fi
}
run_case defaults_a 0
run_case explicit_a 1
run_case explicit_b 1
run_case defaults_b 0

printf '\n========== M50 DEFAULT/EXPLICIT BIT PARITY ==========\n'
"$PY" - <<'PY2'
import numpy as np
from pathlib import Path
pairs=[('defaults_a','explicit_a'),('defaults_b','explicit_b')]
for a,b in pairs:
    x=np.fromfile(Path('build/m50')/(a+'.f32'),np.float32)
    y=np.fromfile(Path('build/m50')/(b+'.f32'),np.float32)
    if x.shape!=y.shape: raise SystemExit(f'shape mismatch {a} {b}')
    d=np.abs(x-y); bit=np.count_nonzero(x.view(np.uint32)!=y.view(np.uint32))
    print(f'{a} vs {b}: max_abs={d.max(initial=0):.9g} bitdiff={bit}')
    if bit: raise SystemExit(3)
PY2

printf '\n========== M50 RELEASE PROFILE ==========\n'
env DSASM_PROFILE_SHAPES=1 DSASM_GOLDEN_COS=0.999 DSASM_GOLDEN_SNR=25 \
  ./build/dsasm-vocoder-m40 infer "$BUNDLE" --mel "$MEL" --f0 "$F0" \
    --out build/m50/profile.f32 --golden "$GOLD" --workers 8 --rounds 9 --profile 2>&1 | tee build/m50/profile.log

ACOUSTIC=build/m42_acoustic
if [[ ! -f "$ACOUSTIC/model.json" ]]; then
  rm -rf "$ACOUSTIC"
  "$PY" tools/pack_acoustic_onnx_m25.py "$MODEL/acoustic.onnx" --model-dir "$MODEL" --out "$ACOUSTIC"
fi
printf '\n========== M50 TRUE E2E / STEADY-STATE DEFAULT PROFILE ==========\n'
: > build/m50/e2e_summary.tsv
for i in 1 2 3; do
  work="build/m50_e2e_$i"; log="build/m50/e2e_$i.log"
  rm -rf "$work"
  printf '\n----- M50 E2E run %s/3 -----\n' "$i"
  # No tuning env here except quality gates. run_native_e2e no longer forces KSPEC.
  env DSASM_GOLDEN_COS=0.999 DSASM_GOLDEN_SNR=25 \
    "$PY" tools/run_native_e2e_m40_1.py \
      --packed-acoustic "$ACOUSTIC" --acoustic-cli ./build/dsasm-acoustic \
      --vocoder-bundle "$BUNDLE" --vocoder-cli ./build/dsasm-vocoder-m40 \
      --vocoder-onnx "$MODEL/dsvocoder/nsf_hifigan.onnx" \
      --speaker-emb "$MODEL/dongfangzhizi-nectar-xiao.emb" \
      --language-id 4 --depth 0.6 --steps 4 --workers 8 --rounds "${M50_E2E_ROUNDS:-11}" \
      --modes k11 --work "$work" 2>&1 | tee "$log"
  line=$(grep 'M40.1 PURE CPU/C/ASM E2E' "$log" | tail -1 || true)
  total=$(printf '%s\n' "$line" | sed -n 's/.*total=\([0-9.]*\) ms.*/\1/p')
  rtf=$(printf '%s\n' "$line" | sed -n 's/.*RTF=\([0-9.]*\).*/\1/p')
  acoustic=$(printf '%s\n' "$line" | sed -n 's/.*acoustic=\([0-9.]*\).*/\1/p')
  vocoder=$(printf '%s\n' "$line" | sed -n 's/.*vocoder=\([0-9.]*\).*/\1/p')
  printf '%s\t%s\t%s\t%s\t%s\n' "$i" "$acoustic" "$vocoder" "$total" "$rtf" >> build/m50/e2e_summary.tsv
  sleep "${M50_SETTLE_SECONDS:-2}"
done

printf '\n========== M50 STEADY E2E SUMMARY ==========\n'
"$PY" - <<'PY3'
from pathlib import Path
from statistics import median
rows=[]
for l in Path('build/m50/e2e_summary.tsv').read_text().splitlines():
    i,a,v,t,r=l.split('\t');rows.append((int(i),float(a),float(v),float(t),float(r)))
for x in rows: print(f'run{x[0]} acoustic={x[1]:.3f} vocoder={x[2]:.3f} total={x[3]:.3f} RTF={x[4]:.3f}')
r=[x[4] for x in rows];t=[x[3] for x in rows]
print(f'median_total={median(t):.3f} ms median_RTF={median(r):.3f} worst_RTF={max(r):.3f}')
print('stable_realtime_all3=' + ('PASS' if max(r)<1.0 else 'FAIL'))
print('stable_engineering_0.8_all3=' + ('PASS' if max(r)<=0.8 else 'FAIL'))
PY3

printf '\n========== M50 FINAL ==========\n'
grep -E '^  Add|^  Conv[[:space:]]|^  ConvTranspose|^  VNNI-|PURE-ASM vocoder:|parity max_abs=' build/m50/profile.log || true
for f in build/m50/e2e_*.log; do grep -E 'M40\.1 PURE CPU/C/ASM E2E|realtime RTF|PURE-ASM vocoder:|parity max_abs=' "$f" || true; done
printf 'release-defaults: KSPEC=0 K3=24 K7=24 K11=24 RANGE=1 TILE=2016 RESIDUAL=1 ADD/LEAKY=1 VNNI=k117@Cin128,64 asymmetric\n'
