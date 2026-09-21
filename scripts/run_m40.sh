#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/fuurin/asm/diffsinger}"
MODEL="${MODEL:-$PROJECT/model/DongFangZhiZi_Nectar_DiffSinger_CE_26.08.31}"
PY="${PY:-$PROJECT/.venv/bin/python}"
SPEAKER_EMB="${SPEAKER_EMB:-$MODEL/dongfangzhizi-nectar-xiao.emb}"

cd "$PROJECT"
[ -x "$PY" ] || { echo "missing python: $PY"; exit 2; }
[ -f "$MODEL/acoustic.onnx" ] || { echo "missing acoustic.onnx"; exit 2; }
[ -f "$MODEL/dsvocoder/nsf_hifigan.onnx" ] || { echo "missing nsf_hifigan.onnx"; exit 2; }

if ! "$PY" -c 'import onnx, onnxruntime, torch, numpy' >/dev/null 2>&1; then
  uv pip install --python "$PY" onnx onnxruntime torch numpy
fi

make clean
make -j"$(nproc)" build/dsasm-vocoder-m40 build/dsasm-acoustic build/libdsasm_m25.so

echo
echo '========== M40 FULL-EXECUTOR VNNI CHECK =========='
make PYTHON="$PY" m40-check

echo
echo '========== M40 ACOUSTIC SAFETY REGRESSION =========='
make PYTHON="$PY" m29-check

echo
echo '========== VERIFY PURE CPU/ASM RUNTIME =========='
for exe in build/dsasm-acoustic build/dsasm-vocoder-m40; do
  echo "----- $exe -----"
  ldd "$exe"
  if ldd "$exe" | grep -Eqi 'onnx|onnxruntime|dnnl|onednn|openvino|mkl|blas'; then
    echo 'ERROR: forbidden runtime backend detected'
    exit 1
  fi
done

echo
echo '========== VERIFY VEX AVX-VNNI =========='
objdump -d -M intel build/conv1d_m39_vnni.o | grep 'vpdpbusd' | head -16
if objdump -d -M intel build/conv1d_m39_vnni.o | grep 'vpdpbusd' | grep -qE '^[[:space:]]*[0-9a-f]+:[[:space:]]+62 '; then
  echo 'ERROR: EVEX/AVX-512 encoding detected'
  exit 1
fi

echo
echo '========== M40 REAL VOCODER: FP32 vs K11 vs K7+K11 =========='
make PYTHON="$PY" m40-real-vocoder MODEL_DIR="$MODEL" M40_FRAMES=48 M40_ROUNDS="${M40_ROUNDS:-9}"

echo
echo '========== M40 PURE ASM END-TO-END =========='
make PYTHON="$PY" m40-real-e2e \
  MODEL_DIR="$MODEL" \
  SPEAKER_EMB="$SPEAKER_EMB" \
  M40_STEPS="${M40_STEPS:-4}" \
  M40_E2E_WORKERS="${M40_E2E_WORKERS:-4}" \
  M40_E2E_ROUNDS="${M40_E2E_ROUNDS:-9}"

echo
echo '========== M40 RESULTS =========='
find build/m40_real build/m40_e2e -maxdepth 3 -type f \
  \( -name '*.json' -o -name '*.wav' -o -name '*.dsv35' \) \
  -printf '%p\t%k KiB\n' 2>/dev/null | sort || true
