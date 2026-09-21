#!/usr/bin/env bash
set -euo pipefail
PROJECT="${PROJECT:-/home/fuurin/asm/diffsinger}"
MODEL="${MODEL:-$PROJECT/model/DongFangZhiZi_Nectar_DiffSinger_CE_26.08.31}"
PY="${PY:-$PROJECT/.venv/bin/python}"
ROUNDS="${M43_ROUNDS:-9}"
cd "$PROJECT"

make -j"$(nproc)" build/dsasm-vocoder-m40 build/dsasm-acoustic
mkdir -p build/m43

# Small correctness check for the new public thread-pool Add path.  threadpool.o
# references the normal runtime kernels, so link the same kernel set as the CLI.
cc -Iinclude -Isrc/runtime -O3 -march=native tools/test_parallel_add_m43.c \
  build/threadpool.o build/vnni_pack_m40_1.o build/conv1d_m32.o \
  build/conv1d_m36.o build/conv1d_m37_range.o build/conv1d_m39_kspec.o \
  build/conv1d_m38_residual.o build/conv1d_m38_range_residual.o \
  build/convtranspose_m33.o build/glu_m6.o build/linear_m4n16.o \
  build/linear_residual_m4n16.o build/linear_m4n16_strided.o \
  build/linear_residual_m4n16_strided.o build/linear_m4n16_idxstrided.o \
  build/linear_residual_m4n16_idxstrided.o build/linear_n16_kblock.o \
  build/dwconv_k31_tc.o build/dwconv_k31_tc_cstrided.o \
  build/atan_glu_m7.o build/atan_glu_m9_strided.o -lm -pthread \
  -o build/m43/test_parallel_add
./build/m43/test_parallel_add

pack_scope(){
  local scope="$1" tag="$2"
  local dir="build/m43/packer_${tag}" bundle="build/m43/nsf_${tag}.dsv35"
  rm -rf "$dir"; mkdir -p "$dir"
  "$PY" tools/pack_vocoder_graph_m35.py \
    "$MODEL/dsvocoder/nsf_hifigan.onnx" --frames 48 --vnni-scope "$scope" \
    --out "$bundle" --work "$dir"
}
pack_scope stage128 stage128
pack_scope all-k711 allk

MEL=build/m43/packer_stage128/mel.f32
F0=build/m43/packer_stage128/f0.f32
GOLD=build/m43/packer_stage128/golden_wave.f32
: > build/m43/candidates.tsv

run_case(){
  local tag="$1" bundle="$2" mode="$3" workers="$4" padd="$5" profile="${6:-0}"
  local extra=()
  [[ "$profile" == 1 ]] && extra+=(--profile)
  echo
  echo "========== M43 $tag =========="
  set +e
  DSASM_PARALLEL_ADD="$padd" DSASM_CONVT_S8K16=1 DSASM_VNNI="$mode" \
  DSASM_KSPEC=1 DSASM_PROFILE_SHAPES=0 DSASM_GOLDEN_COS=0.999 DSASM_GOLDEN_SNR=25 \
  ./build/dsasm-vocoder-m40 infer "$bundle" \
    --mel "$MEL" --f0 "$F0" --out "build/m43/${tag}.f32" --golden "$GOLD" \
    --workers "$workers" --rounds "$ROUNDS" "${extra[@]}" \
    2>&1 | tee "build/m43/${tag}.log"
  local rc=${PIPESTATUS[0]}
  set -e
  local med
  med=$(sed -n 's/.*median=\([0-9.]*\) ms.*/\1/p' "build/m43/${tag}.log" | tail -1)
  if [[ $rc -eq 0 && -n "$med" ]] && grep -q 'DSASM parity: OK' "build/m43/${tag}.log"; then
    printf '%s\t%s\t%s\t%s\t%s\n' "$tag" "$bundle" "$mode" "$workers" "$med" >> build/m43/candidates.tsv
  else
    echo "M43 candidate rejected: $tag rc=$rc median=${med:-NA}"
  fi
}

# Isolate the code change first, with op profiling on the exact M42 production path.
run_case stage128_k11_w8_add_off build/m43/nsf_stage128.dsv35 k11 8 0 1
run_case stage128_k11_w8_add_on  build/m43/nsf_stage128.dsv35 k11 8 1 1

# Let the target machine decide whether the new Add scheduling stays enabled.
ADD_BEST=$(python -c '
from pathlib import Path
rows=[]
for line in Path("build/m43/candidates.tsv").read_text().splitlines():
    tag,bundle,mode,w,med=line.split("\t")
    if tag in ("stage128_k11_w8_add_off","stage128_k11_w8_add_on"):
        rows.append((float(med),tag))
if not rows: raise SystemExit("M43 Add A/B produced no quality-passing result")
print("\t".join(map(str,min(rows))))
')
IFS=$'\t' read -r ADD_MS ADD_TAG <<< "$ADD_BEST"
if [[ "$ADD_TAG" == *_add_off ]]; then ADD_TOGGLE=0; else ADD_TOGGLE=1; fi
printf 'M43 Add policy: %s (%s ms) -> DSASM_PARALLEL_ADD=%s\n' "$ADD_TAG" "$ADD_MS" "$ADD_TOGGLE"

# Race the two quantization scopes/modes using the winning Add policy.
run_case stage128_k117_w8 build/m43/nsf_stage128.dsv35 k117 8 "$ADD_TOGGLE" 0
run_case allk_k11_w8       build/m43/nsf_allk.dsv35     k11  8 "$ADD_TOGGLE" 0
run_case allk_k117_w8      build/m43/nsf_allk.dsv35     k117 8 "$ADD_TOGGLE" 0

# Pick the best 8-worker production candidate. Keep only the winning baseline
# Add variant so the loser cannot sneak back in through timing noise.
BEST8=$(ADD_TAG="$ADD_TAG" python -c '
from pathlib import Path
import os
rows=[]; add_tag=os.environ["ADD_TAG"]
for line in Path("build/m43/candidates.tsv").read_text().splitlines():
    tag,bundle,mode,w,med=line.split("\t")
    if tag in ("stage128_k11_w8_add_off","stage128_k11_w8_add_on") and tag!=add_tag: continue
    if w=="8": rows.append((float(med),tag,bundle,mode,w))
if not rows: raise SystemExit("no quality-passing M43 8-worker candidate")
print("\t".join(map(str,min(rows))))
')
IFS=$'\t' read -r BEST8_MS BEST8_TAG BEST8_BUNDLE BEST8_MODE _ <<< "$BEST8"
echo
printf 'M43 best @8: %s  %s ms  bundle=%s mode=%s\n' "$BEST8_TAG" "$BEST8_MS" "$BEST8_BUNDLE" "$BEST8_MODE"

# Only the 8-worker winner gets the E-core experiment.
run_case "${BEST8_TAG%_w8}_w10" "$BEST8_BUNDLE" "$BEST8_MODE" 10 "$ADD_TOGGLE" 0
run_case "${BEST8_TAG%_w8}_w12" "$BEST8_BUNDLE" "$BEST8_MODE" 12 "$ADD_TOGGLE" 0

BEST=$(ADD_TAG="$ADD_TAG" python -c '
from pathlib import Path
import os
rows=[]; add_tag=os.environ["ADD_TAG"]
for line in Path("build/m43/candidates.tsv").read_text().splitlines():
    tag,bundle,mode,w,med=line.split("\t")
    if tag in ("stage128_k11_w8_add_off","stage128_k11_w8_add_on") and tag!=add_tag: continue
    rows.append((float(med),tag,bundle,mode,w))
if not rows: raise SystemExit("no quality-passing M43 candidate")
print("\t".join(map(str,min(rows))))
')
IFS=$'\t' read -r BEST_MS BEST_TAG BEST_BUNDLE BEST_MODE BEST_WORKERS <<< "$BEST"

echo
printf '========== M43 WINNER ==========\n'
printf 'tag=%s median=%s ms bundle=%s mode=%s workers=%s add=%s\n' \
  "$BEST_TAG" "$BEST_MS" "$BEST_BUNDLE" "$BEST_MODE" "$BEST_WORKERS" "$ADD_TOGGLE"

printf '\n========== M43 A/B SUMMARY ==========\n'
for f in build/m43/*.log; do
  [[ -f "$f" ]] || continue
  echo "--- $(basename "$f") ---"
  grep -E '^  Add[[:space:]]|^  Conv[[:space:]]|^  ConvTranspose|^  VNNI-|PURE-ASM vocoder:|parity max_abs=' "$f" || true
done

ACOUSTIC=build/m42_acoustic
if [[ ! -f "$ACOUSTIC/model.json" ]]; then
  rm -rf "$ACOUSTIC"
  "$PY" tools/pack_acoustic_onnx_m25.py "$MODEL/acoustic.onnx" \
    --model-dir "$MODEL" --out "$ACOUSTIC"
fi
rm -rf build/m43_e2e
set +e
DSASM_PARALLEL_ADD="$ADD_TOGGLE" DSASM_CONVT_S8K16=1 \
"$PY" tools/run_native_e2e_m40_1.py \
  --packed-acoustic "$ACOUSTIC" --acoustic-cli ./build/dsasm-acoustic \
  --vocoder-bundle "$BEST_BUNDLE" --vocoder-cli ./build/dsasm-vocoder-m40 \
  --vocoder-onnx "$MODEL/dsvocoder/nsf_hifigan.onnx" \
  --speaker-emb "$MODEL/dongfangzhizi-nectar-xiao.emb" \
  --language-id 4 --depth 0.6 --steps 4 --workers "$BEST_WORKERS" \
  --rounds "$ROUNDS" --modes "$BEST_MODE" --work build/m43_e2e \
  2>&1 | tee build/m43/e2e.log
E2E_RC=${PIPESTATUS[0]}
set -e

# Aggressive all-K quantization is allowed to fail the real-acoustic quality
# gate. If that happens, finish the one-shot run with the known-safe M42 path
# instead of forcing a second user command.
if [[ $E2E_RC -ne 0 ]]; then
  echo
  echo 'M43 winner failed real E2E gate; falling back to stage128/k11/w8.'
  rm -rf build/m43_e2e_fallback
  DSASM_PARALLEL_ADD="$ADD_TOGGLE" DSASM_CONVT_S8K16=1 \
  "$PY" tools/run_native_e2e_m40_1.py \
    --packed-acoustic "$ACOUSTIC" --acoustic-cli ./build/dsasm-acoustic \
    --vocoder-bundle build/m43/nsf_stage128.dsv35 --vocoder-cli ./build/dsasm-vocoder-m40 \
    --vocoder-onnx "$MODEL/dsvocoder/nsf_hifigan.onnx" \
    --speaker-emb "$MODEL/dongfangzhizi-nectar-xiao.emb" \
    --language-id 4 --depth 0.6 --steps 4 --workers 8 \
    --rounds "$ROUNDS" --modes k11 --work build/m43_e2e_fallback \
    2>&1 | tee build/m43/e2e_fallback.log
fi

printf '\n========== M43 FINAL ==========\n'
for f in build/m43/e2e.log build/m43/e2e_fallback.log; do
  [[ -f "$f" ]] || continue
  echo "--- $(basename "$f") ---"
  grep -E 'M40\.1 PURE CPU/C/ASM E2E|realtime RTF|PURE-ASM vocoder:|parity max_abs=|^  Add[[:space:]]|^  Conv[[:space:]]|^  ConvTranspose|^  VNNI-' "$f" || true
done
