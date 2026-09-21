#!/usr/bin/env bash
set -euo pipefail
PROJECT="${PROJECT:-/home/fuurin/asm/diffsinger}"
MODEL="${MODEL:-$PROJECT/model/DongFangZhiZi_Nectar_DiffSinger_CE_26.08.31}"
PY="${PY:-$PROJECT/.venv/bin/python}"
ROUNDS="${M41_ROUNDS:-7}"
cd "$PROJECT"

make -j"$(nproc)" build/dsasm-vocoder-m40
mkdir -p build/m41_profile

BUNDLE=build/m41_profile/nsf_hifigan.dsv35
WORK=build/m41_profile/packer
if [[ ! -f "$BUNDLE" ]]; then
  rm -rf "$WORK"
  mkdir -p "$WORK"
  "$PY" tools/pack_vocoder_graph_m35.py \
    "$MODEL/dsvocoder/nsf_hifigan.onnx" \
    --frames 48 --vnni-scope stage128 \
    --out "$BUNDLE" --work "$WORK"
fi

MEL="$WORK/mel.f32"
F0="$WORK/f0.f32"
GOLD="$WORK/golden_wave.f32"

run_case() {
  local tag="$1" workers="$2"; shift 2
  echo
  echo "========== M41 $tag workers=$workers =========="
  env "$@" \
    DSASM_PROFILE_SHAPES=1 \
    DSASM_GOLDEN_COS=0.999 DSASM_GOLDEN_SNR=25 \
    ./build/dsasm-vocoder-m40 infer "$BUNDLE" \
      --mel "$MEL" --f0 "$F0" \
      --out "build/m41_profile/${tag}.f32" --golden "$GOLD" \
      --workers "$workers" --rounds "$ROUNDS" --profile \
    2>&1 | tee "build/m41_profile/${tag}.log"
}

# Current M40.1 winner first, then isolate scheduler/kernel effects.
run_case baseline_k11_w4 4 DSASM_VNNI=k11 DSASM_KSPEC=1
run_case genericK_k11_w4 4 DSASM_VNNI=k11 DSASM_KSPEC=0
run_case staticOC_k11_w4 4 DSASM_VNNI=k11 DSASM_KSPEC=1 DSASM_2D=0
run_case baseline_k11_w6 6 DSASM_VNNI=k11 DSASM_KSPEC=1
run_case baseline_k11_w8 8 DSASM_VNNI=k11 DSASM_KSPEC=1

printf '\n========== M41 SUMMARY =========='; printf '\n'
for f in build/m41_profile/*.log; do
  printf '\n--- %s ---\n' "$(basename "$f")"
  grep -E 'PURE-ASM vocoder:|parity max_abs=|^(  Conv|  ConvTranspose|  VNNI-)|^  #[0-9]+ (Conv |ConvT)' "$f" | head -35 || true
done

printf '\nM41 logs: %s\n' "$PROJECT/build/m41_profile"
