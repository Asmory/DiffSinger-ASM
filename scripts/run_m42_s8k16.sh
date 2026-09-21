#!/usr/bin/env bash
set -euo pipefail
PROJECT="${PROJECT:-/home/fuurin/asm/diffsinger}"
MODEL="${MODEL:-$PROJECT/model/DongFangZhiZi_Nectar_DiffSinger_CE_26.08.31}"
PY="${PY:-$PROJECT/.venv/bin/python}"
ROUNDS="${M42_ROUNDS:-9}"
cd "$PROJECT"

make -j"$(nproc)" build/dsasm-vocoder-m40 build/dsasm-acoustic
mkdir -p build/m42

# Direct kernel parity against the old generic scatter kernel.
cc -Iinclude -O3 -march=native tools/test_convtranspose_s8k16_m42.c \
  build/convtranspose_m33.o -lm -o build/m42/test_s8k16
./build/m42/test_s8k16

BUNDLE=build/m41_profile/nsf_hifigan.dsv35
WORK=build/m41_profile/packer
if [[ ! -f "$BUNDLE" || ! -f "$WORK/mel.f32" ]]; then
  mkdir -p build/m41_profile
  rm -rf "$WORK"; mkdir -p "$WORK"
  "$PY" tools/pack_vocoder_graph_m35.py \
    "$MODEL/dsvocoder/nsf_hifigan.onnx" \
    --frames 48 --vnni-scope stage128 --out "$BUNDLE" --work "$WORK"
fi
MEL="$WORK/mel.f32"; F0="$WORK/f0.f32"; GOLD="$WORK/golden_wave.f32"

run_vocoder(){
  local tag="$1" toggle="$2"
  echo
  echo "========== M42 $tag =========="
  DSASM_CONVT_S8K16="$toggle" DSASM_VNNI=k11 DSASM_KSPEC=1 \
  DSASM_PROFILE_SHAPES=1 DSASM_GOLDEN_COS=0.999 DSASM_GOLDEN_SNR=25 \
  ./build/dsasm-vocoder-m40 infer "$BUNDLE" \
    --mel "$MEL" --f0 "$F0" --out "build/m42/${tag}.f32" --golden "$GOLD" \
    --workers 8 --rounds "$ROUNDS" --profile 2>&1 | tee "build/m42/${tag}.log"
}
run_vocoder generic_s8k16_off 0
run_vocoder asm_s8k16_on 1

printf '\n========== M42 VOCODER A/B SUMMARY ==========\n'
for f in build/m42/generic_s8k16_off.log build/m42/asm_s8k16_on.log; do
  echo "--- $(basename "$f") ---"
  grep -E 'PURE-ASM vocoder:|parity max_abs=|^  ConvTranspose|#060 ConvT|#027 ConvT|#095 ConvT' "$f" || true
done

# Re-run the actual CPU/C/ASM end-to-end path with the new 8-worker vocoder.
ACOUSTIC=build/m42_acoustic
if [[ ! -f "$ACOUSTIC/model.json" ]]; then
  rm -rf "$ACOUSTIC"
  "$PY" tools/pack_acoustic_onnx_m25.py "$MODEL/acoustic.onnx" \
    --model-dir "$MODEL" --out "$ACOUSTIC"
fi
rm -rf build/m42_e2e
DSASM_CONVT_S8K16=1 "$PY" tools/run_native_e2e_m40_1.py \
  --packed-acoustic "$ACOUSTIC" \
  --acoustic-cli ./build/dsasm-acoustic \
  --vocoder-bundle "$BUNDLE" \
  --vocoder-cli ./build/dsasm-vocoder-m40 \
  --vocoder-onnx "$MODEL/dsvocoder/nsf_hifigan.onnx" \
  --speaker-emb "$MODEL/dongfangzhizi-nectar-xiao.emb" \
  --language-id 4 --depth 0.6 --steps 4 --workers 8 --rounds "$ROUNDS" --modes k11 \
  --work build/m42_e2e 2>&1 | tee build/m42/e2e.log

printf '\n========== M42 FINAL ==========\n'
grep -E 'M40\.1 PURE CPU/C/ASM E2E|realtime RTF|PURE-ASM vocoder:|parity max_abs=' build/m42/e2e.log || true
