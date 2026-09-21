#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
PROJECT="${PROJECT:-$HOME/asm/diffsinger}"
MODEL="${MODEL:-$PROJECT/model/DongFangZhiZi_Nectar_DiffSinger_CE_26.08.31}"
PY="${PY:-$PROJECT/.venv/bin/python}"
cd "$PROJECT"
mkdir -p build/m49

make -j"$(nproc)" build/dsasm-vocoder-m40 build/dsasm-acoustic
cc -O3 -march=native -Iinclude tools/test_residual_t24_m49.c \
  build/conv1d_m39_kspec.o build/add_m33.o -lm -o build/test_residual_t24_m49
printf '\n========== M49 RESIDUAL T24 SYNTHETIC / MICRO ==========\n'
./build/test_residual_t24_m49
[[ "${M49_TEST_ONLY:-0}" == 1 ]] && exit 0

printf '\n========== M49 PACK BASE + SELECTIVE RESIDUAL ==========\n'
rm -rf build/m49/packer_base build/m49/packer_res
"$PY" tools/pack_vocoder_graph_m35.py "$MODEL/dsvocoder/nsf_hifigan.onnx" \
  --frames 48 --vnni-scope all-k711 --residual-scope none \
  --out build/m49/nsf_base.dsv35 --work build/m49/packer_base
"$PY" tools/pack_vocoder_graph_m35.py "$MODEL/dsvocoder/nsf_hifigan.onnx" \
  --frames 48 --vnni-scope all-k711 --residual-scope k7ge128 \
  --out build/m49/nsf_res.dsv35 --work build/m49/packer_res
BASE=build/m49/nsf_base.dsv35
RES=build/m49/nsf_res.dsv35
MEL=build/m49/packer_base/mel.f32
F0=build/m49/packer_base/f0.f32
GOLD=build/m49/packer_base/golden_wave.f32

BASE_ENV=(
  DSASM_PARALLEL_ADD=0 DSASM_CONVT_S8K16=1
  DSASM_VNNI=k11 DSASM_VNNI_ASYM=0 DSASM_VNNI_CIN=128
  DSASM_GOLDEN_COS=0.999 DSASM_GOLDEN_SNR=25 DSASM_PROFILE_SHAPES=0
  DSASM_KSPEC=0 DSASM_K3_TMODE=24 DSASM_K7_T24=1 DSASM_K11_T24=1
  DSASM_RANGE_T24=0 DSASM_VOCODER_T_TILE=504
)
PERF_ARGS=()
if command -v perf >/dev/null 2>&1; then
  if perf stat -e cpu_core/cycles/u -e cpu_core/instructions/u -- true >/dev/null 2>&1; then
    PERF_ARGS=(-e cpu_core/cycles/u -e cpu_core/instructions/u)
  elif perf stat -e cycles:u -e instructions:u -- true >/dev/null 2>&1; then
    PERF_ARGS=(-e cycles:u -e instructions:u)
  fi
fi
: > build/m49/cases.tsv
run_case(){
  local tag="$1" bundle="$2" rt="$3"
  local log="build/m49/${tag}.log" pf="build/m49/${tag}.perf"
  printf '\n========== M49 %s RESIDUAL_T24=%s ==========\n' "$tag" "$rt"
  if ((${#PERF_ARGS[@]})); then
    env "${BASE_ENV[@]}" DSASM_RESIDUAL_T24="$rt" \
      perf stat -x, -o "$pf" "${PERF_ARGS[@]}" -- \
      ./build/dsasm-vocoder-m40 infer "$bundle" --mel "$MEL" --f0 "$F0" \
        --out "build/m49/${tag}.f32" --golden "$GOLD" --workers 8 --rounds 5 \
        2>&1 | tee "$log"
  else
    : > "$pf"
    env "${BASE_ENV[@]}" DSASM_RESIDUAL_T24="$rt" \
      ./build/dsasm-vocoder-m40 infer "$bundle" --mel "$MEL" --f0 "$F0" \
        --out "build/m49/${tag}.f32" --golden "$GOLD" --workers 8 --rounds 5 \
        2>&1 | tee "$log"
  fi
  local med cyc ins
  med=$(sed -n 's/.*median=\([0-9.]*\) ms.*/\1/p' "$log" | tail -1)
  cyc=$(awk -F, '$3 ~ /cycles/ {gsub(/[[:space:]]/,"",$1); print $1; exit}' "$pf" 2>/dev/null || true)
  ins=$(awk -F, '$3 ~ /instructions/ {gsub(/[[:space:]]/,"",$1); print $1; exit}' "$pf" 2>/dev/null || true)
  [[ -z "$cyc" ]] && cyc=NA; [[ -z "$ins" ]] && ins=NA
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$tag" "$bundle" "$rt" "${med:-NA}" "$cyc" "$ins" >> build/m49/cases.tsv
  printf 'M49 sample: tag=%s median=%s ms Pcycles=%s Pinsns=%s\n' "$tag" "${med:-NA}" "$cyc" "$ins"
  sleep 1
}
run_case base_a "$BASE" 0
run_case oldres_a "$RES" 0
run_case t24res_a "$RES" 1
run_case t24res_b "$RES" 1
run_case oldres_b "$RES" 0
run_case base_b "$BASE" 0

printf '\n========== M49 CONTROLLED SUMMARY ==========\n'
BEST=$("$PY" - <<'PY2'
from pathlib import Path
from statistics import mean
rows={}
for line in Path('build/m49/cases.tsv').read_text().splitlines():
    tag,bundle,rt,med,cyc,ins=line.split('\t'); key=tag.rsplit('_',1)[0]
    rows.setdefault(key,[]).append((bundle,rt,float(med),None if cyc=='NA' else int(cyc),None if ins=='NA' else int(ins)))
for k,v in rows.items():
    m=mean(x[2] for x in v); cs=[x[3] for x in v if x[3] is not None]; ii=[x[4] for x in v if x[4] is not None]
    c=mean(cs) if cs else float('nan'); ins=mean(ii) if ii else float('nan'); ipc=ins/c if cs and ii else float('nan')
    print(f'{k:8s} mean_median={m:.3f} ms mean_Pcycles={c:.0f} mean_Pinsns={ins:.0f} IPC={ipc:.3f} rt24={v[0][1]}')
valid=[]
for k,v in rows.items():
    cs=[x[3] for x in v]
    score=mean(cs) if all(x is not None for x in cs) else mean(x[2] for x in v)
    valid.append((score,k,v[0][0],v[0][1]))
_,k,b,r=min(valid)
print(f'WINNER\t{k}\t{b}\t{r}')
PY2
)
echo "$BEST"
IFS=$'\t' read -r _ WIN_TAG WIN_BUNDLE WIN_RT <<< "$(printf '%s\n' "$BEST" | tail -1)"
printf 'M49 winner: %s bundle=%s RESIDUAL_T24=%s\n' "$WIN_TAG" "$WIN_BUNDLE" "$WIN_RT"

sleep 3
printf '\n========== M49 WINNER PROFILE ==========\n'
env "${BASE_ENV[@]}" DSASM_RESIDUAL_T24="$WIN_RT" DSASM_PROFILE_SHAPES=1 \
  ./build/dsasm-vocoder-m40 infer "$WIN_BUNDLE" --mel "$MEL" --f0 "$F0" \
    --out build/m49/winner.f32 --golden "$GOLD" --workers 8 --rounds 7 --profile \
    2>&1 | tee build/m49/winner_profile.log

ACOUSTIC=build/m42_acoustic
if [[ ! -f "$ACOUSTIC/model.json" ]]; then
  rm -rf "$ACOUSTIC"
  "$PY" tools/pack_acoustic_onnx_m25.py "$MODEL/acoustic.onnx" --model-dir "$MODEL" --out "$ACOUSTIC"
fi
sleep "${M49_SETTLE_SECONDS:-2}"
rm -rf build/m49_e2e
printf '\n========== M49 TRUE E2E ==========\n'
env "${BASE_ENV[@]}" DSASM_RESIDUAL_T24="$WIN_RT" \
  "$PY" tools/run_native_e2e_m40_1.py \
    --packed-acoustic "$ACOUSTIC" --acoustic-cli ./build/dsasm-acoustic \
    --vocoder-bundle "$WIN_BUNDLE" --vocoder-cli ./build/dsasm-vocoder-m40 \
    --vocoder-onnx "$MODEL/dsvocoder/nsf_hifigan.onnx" \
    --speaker-emb "$MODEL/dongfangzhizi-nectar-xiao.emb" \
    --language-id 4 --depth 0.6 --steps 4 --workers 8 --rounds "${M49_E2E_ROUNDS:-9}" \
    --modes k11 --work build/m49_e2e 2>&1 | tee build/m49/e2e.log

printf '\n========== M49 FINAL ==========\n'
grep -E '^  Add|^  Conv[[:space:]]|^  ConvTranspose|^  VNNI-|PURE-ASM vocoder:|parity max_abs=' build/m49/winner_profile.log || true
grep -E 'M40\.1 PURE CPU/C/ASM E2E|realtime RTF|PURE-ASM vocoder:|parity max_abs=' build/m49/e2e.log || true
printf 'winner=%s RESIDUAL_T24=%s bundle=%s\n' "$WIN_TAG" "$WIN_RT" "$WIN_BUNDLE"
