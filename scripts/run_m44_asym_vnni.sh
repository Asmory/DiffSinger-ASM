#!/usr/bin/env bash
set -euo pipefail
PROJECT="${PROJECT:-/home/fuurin/asm/diffsinger}"
MODEL="${MODEL:-$PROJECT/model/DongFangZhiZi_Nectar_DiffSinger_CE_26.08.31}"
PY="${PY:-$PROJECT/.venv/bin/python}"
SCREEN_ROUNDS="${M44_SCREEN_ROUNDS:-5}"
FINAL_ROUNDS="${M44_FINAL_ROUNDS:-9}"
cd "$PROJECT"

make -j"$(nproc)" build/dsasm-vocoder-m40 build/dsasm-acoustic
rm -rf build/m44
mkdir -p build/m44

printf '\n========== M44 SYNTHETIC QUANTIZER CHECK ==========\n'
"$PY" tools/make_fake_vnni_graph_m40.py build/m44/fake.dsv35 --work build/m44/fake
for asym in 0 1; do
  echo "--- fake asym=$asym ---"
  DSASM_PARALLEL_ADD=0 DSASM_VNNI=k117 DSASM_VNNI_ASYM="$asym" DSASM_VNNI_CIN=all \
  DSASM_GOLDEN_COS=0.999 DSASM_GOLDEN_SNR=25 \
  ./build/dsasm-vocoder-m40 infer build/m44/fake.dsv35 \
    --mel build/m44/fake/mel.f32 --f0 build/m44/fake/f0.f32 \
    --out "build/m44/fake_${asym}.f32" --golden build/m44/fake/golden.f32 \
    --workers 4 --rounds 5 --profile 2>&1 | tee "build/m44/fake_asym${asym}.log"
done

printf '\n========== M44 PACK ALL VNNI CANDIDATES ONCE ==========\n'
"$PY" tools/pack_vocoder_graph_m35.py \
  "$MODEL/dsvocoder/nsf_hifigan.onnx" --frames 48 --vnni-scope all-k711 \
  --out build/m44/nsf_allk.dsv35 --work build/m44/packer
MEL=build/m44/packer/mel.f32
F0=build/m44/packer/f0.f32
GOLD=build/m44/packer/golden_wave.f32
BUNDLE=build/m44/nsf_allk.dsv35
: > build/m44/candidates.tsv

run_case(){
  local tag="$1" mode="$2" workers="$3" asym="$4" mask="$5" rounds="${6:-$SCREEN_ROUNDS}" profile="${7:-0}"
  local extra=()
  [[ "$profile" == 1 ]] && extra+=(--profile)
  echo
  echo "========== M44 $tag  mode=$mode workers=$workers asym=$asym cin=$mask =========="
  set +e
  DSASM_PARALLEL_ADD=0 DSASM_CONVT_S8K16=1 DSASM_VNNI="$mode" \
  DSASM_VNNI_ASYM="$asym" DSASM_VNNI_CIN="$mask" DSASM_KSPEC=1 \
  DSASM_PROFILE_SHAPES=0 DSASM_GOLDEN_COS=0.999 DSASM_GOLDEN_SNR=25 \
  ./build/dsasm-vocoder-m40 infer "$BUNDLE" \
    --mel "$MEL" --f0 "$F0" --out "build/m44/${tag}.f32" --golden "$GOLD" \
    --workers "$workers" --rounds "$rounds" "${extra[@]}" \
    2>&1 | tee "build/m44/${tag}.log"
  local rc=${PIPESTATUS[0]}
  set -e
  local med cos snr
  med=$(sed -n 's/.*median=\([0-9.]*\) ms.*/\1/p' "build/m44/${tag}.log" | tail -1)
  cos=$(sed -n 's/.*cosine=\([0-9.eE+-]*\).*/\1/p' "build/m44/${tag}.log" | tail -1)
  snr=$(sed -n 's/.*SNR=\([0-9.eE+-]*\) dB.*/\1/p' "build/m44/${tag}.log" | tail -1)
  if [[ $rc -eq 0 && -n "$med" ]] && grep -q 'DSASM parity: OK' "build/m44/${tag}.log"; then
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$tag" "$mode" "$workers" "$asym" "$mask" "$med" "${cos:-NA}" "${snr:-NA}" >> build/m44/candidates.tsv
    return 0
  fi
  echo "M44 rejected: $tag rc=$rc median=${med:-NA} cosine=${cos:-NA} SNR=${snr:-NA}"
  return 1
}

# Known-safe reference and the direct asymmetric replacement.
run_case sym_128_k11_w8  k11  8 0 128 "$SCREEN_ROUNDS" 1
run_case asym_128_k11_w8 k11  8 1 128 "$SCREEN_ROUNDS" 1
run_case asym_128_k117_w8 k117 8 1 128 "$SCREEN_ROUNDS" 0 || true

# Greedy quality-gated stage expansion.  The all-k bundle has every VNNI blob,
# but only stages named by DSASM_VNNI_CIN actually execute quantized kernels.
MASK=128
REMAINING=(256 64 32 16)
ROUND=0
while ((${#REMAINING[@]})); do
  ROUND=$((ROUND+1))
  : > "build/m44/greedy_${ROUND}.tsv"
  NEXT=()
  for st in "${REMAINING[@]}"; do
    cand="$MASK,$st"
    tag="greedy_r${ROUND}_cin_${cand//,/_}"
    if run_case "$tag" k11 8 1 "$cand" "$SCREEN_ROUNDS" 0; then
      med=$(awk -F '\t' -v t="$tag" '$1==t{print $6}' build/m44/candidates.tsv | tail -1)
      printf '%s\t%s\t%s\n' "$med" "$st" "$cand" >> "build/m44/greedy_${ROUND}.tsv"
    fi
  done
  if [[ ! -s "build/m44/greedy_${ROUND}.tsv" ]]; then
    echo "M44 greedy: no remaining stage passes quality gate after mask=$MASK"
    break
  fi
  read -r BEST_MED BEST_STAGE BEST_MASK < <(sort -n -k1,1 "build/m44/greedy_${ROUND}.tsv" | head -1)
  printf 'M44 greedy round %d: admit Cin=%s -> mask=%s median=%s ms\n' "$ROUND" "$BEST_STAGE" "$BEST_MASK" "$BEST_MED"
  MASK="$BEST_MASK"
  for st in "${REMAINING[@]}"; do [[ "$st" != "$BEST_STAGE" ]] && NEXT+=("$st"); done
  REMAINING=("${NEXT[@]}")
done

printf '\nM44 quality-safe greedy mask: %s\n' "$MASK"

# Re-race finalists with longer medians.  k117 is allowed only if it still
# passes the same waveform gate with the selected stage mask.
run_case final_k11_w8  k11  8 1 "$MASK" "$FINAL_ROUNDS" 1
run_case final_k117_w8 k117 8 1 "$MASK" "$FINAL_ROUNDS" 0 || true
run_case final_k11_w10 k11 10 1 "$MASK" "$FINAL_ROUNDS" 0
run_case final_k117_w10 k117 10 1 "$MASK" "$FINAL_ROUNDS" 0 || true

BEST=$(python - <<'PY'
from pathlib import Path
rows=[]
for line in Path('build/m44/candidates.tsv').read_text().splitlines():
    tag,mode,w,asym,mask,med,cos,snr=line.split('\t')
    if tag.startswith('final_'):
        rows.append((float(med),tag,mode,w,asym,mask,cos,snr))
if not rows: raise SystemExit('M44: no quality-passing final candidate')
print('\t'.join(map(str,min(rows))))
PY
)
IFS=$'\t' read -r BEST_MS BEST_TAG BEST_MODE BEST_WORKERS BEST_ASYM BEST_MASK BEST_COS BEST_SNR <<< "$BEST"
printf '\n========== M44 WINNER ==========\n'
printf 'tag=%s median=%s ms mode=%s workers=%s asym=%s cin=%s cosine=%s SNR=%s\n' \
  "$BEST_TAG" "$BEST_MS" "$BEST_MODE" "$BEST_WORKERS" "$BEST_ASYM" "$BEST_MASK" "$BEST_COS" "$BEST_SNR"

printf '\n========== M44 VOCODER SUMMARY ==========\n'
for f in build/m44/final_*.log build/m44/sym_128_k11_w8.log build/m44/asym_128_k11_w8.log; do
  [[ -f "$f" ]] || continue
  echo "--- $(basename "$f") ---"
  grep -E '^  Add[[:space:]]|^  Conv[[:space:]]|^  ConvTranspose|^  VNNI-|PURE-ASM vocoder:|parity max_abs=' "$f" || true
done

ACOUSTIC=build/m42_acoustic
if [[ ! -f "$ACOUSTIC/model.json" ]]; then
  rm -rf "$ACOUSTIC"
  "$PY" tools/pack_acoustic_onnx_m25.py "$MODEL/acoustic.onnx" --model-dir "$MODEL" --out "$ACOUSTIC"
fi

# Run E2E from a short settled state.  Acoustic keeps its own proven default
# pool; --workers only controls the vocoder in run_native_e2e_m40_1.py.
sleep "${M44_SETTLE_SECONDS:-2}"
rm -rf build/m44_e2e
set +e
DSASM_PARALLEL_ADD=0 DSASM_CONVT_S8K16=1 DSASM_VNNI_ASYM="$BEST_ASYM" DSASM_VNNI_CIN="$BEST_MASK" \
"$PY" tools/run_native_e2e_m40_1.py \
  --packed-acoustic "$ACOUSTIC" --acoustic-cli ./build/dsasm-acoustic \
  --vocoder-bundle "$BUNDLE" --vocoder-cli ./build/dsasm-vocoder-m40 \
  --vocoder-onnx "$MODEL/dsvocoder/nsf_hifigan.onnx" \
  --speaker-emb "$MODEL/dongfangzhizi-nectar-xiao.emb" \
  --language-id 4 --depth 0.6 --steps 4 --workers "$BEST_WORKERS" \
  --rounds "$FINAL_ROUNDS" --modes "$BEST_MODE" --work build/m44_e2e \
  2>&1 | tee build/m44/e2e.log
E2E_RC=${PIPESTATUS[0]}
set -e

# The real acoustic mel is the final authority.  If the greedy synthetic/sample
# search over-quantized it, finish this same command on the established M42-safe
# symmetric stage128/k11/w8 path.
if [[ $E2E_RC -ne 0 ]]; then
  echo
  echo 'M44 winner failed real-acoustic E2E quality gate; fallback = symmetric Cin128/k11/w8.'
  sleep "${M44_SETTLE_SECONDS:-2}"
  rm -rf build/m44_e2e_fallback
  DSASM_PARALLEL_ADD=0 DSASM_CONVT_S8K16=1 DSASM_VNNI_ASYM=0 DSASM_VNNI_CIN=128 \
  "$PY" tools/run_native_e2e_m40_1.py \
    --packed-acoustic "$ACOUSTIC" --acoustic-cli ./build/dsasm-acoustic \
    --vocoder-bundle "$BUNDLE" --vocoder-cli ./build/dsasm-vocoder-m40 \
    --vocoder-onnx "$MODEL/dsvocoder/nsf_hifigan.onnx" \
    --speaker-emb "$MODEL/dongfangzhizi-nectar-xiao.emb" \
    --language-id 4 --depth 0.6 --steps 4 --workers 8 \
    --rounds "$FINAL_ROUNDS" --modes k11 --work build/m44_e2e_fallback \
    2>&1 | tee build/m44/e2e_fallback.log
fi

printf '\n========== M44 FINAL ==========\n'
for f in build/m44/e2e.log build/m44/e2e_fallback.log; do
  [[ -f "$f" ]] || continue
  echo "--- $(basename "$f") ---"
  grep -E 'M40\.1 PURE CPU/C/ASM E2E|realtime RTF|PURE-ASM vocoder:|parity max_abs=|^  Add[[:space:]]|^  Conv[[:space:]]|^  ConvTranspose|^  VNNI-' "$f" || true
done
