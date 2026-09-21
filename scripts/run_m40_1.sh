#!/usr/bin/env bash
set -euo pipefail
PROJECT="${PROJECT:-/home/fuurin/asm/diffsinger}"
MODEL="${MODEL:-$PROJECT/model/DongFangZhiZi_Nectar_DiffSinger_CE_26.08.31}"
PY="${PY:-$PROJECT/.venv/bin/python}"
cd "$PROJECT"
make clean
make -j"$(nproc)" build/dsasm-vocoder-m40 build/dsasm-acoustic build/libdsasm_m25.so
printf '\n========== M40.1 FAST VNNI PACK CHECK ==========\n'
make PYTHON="$PY" m40-1-check
printf '\n========== M40.1 ACOUSTIC REGRESSION ==========\n'
make PYTHON="$PY" m29-check
printf '\n========== M40.1 PURE RUNTIME ==========\n'
for exe in build/dsasm-acoustic build/dsasm-vocoder-m40; do
  ldd "$exe"
  ! ldd "$exe" | grep -Eqi 'onnx|onnxruntime|dnnl|onednn|openvino|mkl|blas'
done
printf '\n========== M40.1 REAL VOCODER ==========\n'
make PYTHON="$PY" m40-1-real-vocoder MODEL_DIR="$MODEL" M40_1_ROUNDS="${M40_1_ROUNDS:-9}"
printf '\n========== M40.1 END TO END ==========\n'
make PYTHON="$PY" m40-1-real-e2e MODEL_DIR="$MODEL" SPEAKER_EMB="${SPEAKER_EMB:-$MODEL/dongfangzhizi-nectar-xiao.emb}" M40_1_STEPS="${M40_1_STEPS:-4}" M40_1_E2E_WORKERS="${M40_1_E2E_WORKERS:-4}" M40_1_E2E_ROUNDS="${M40_1_E2E_ROUNDS:-9}"
printf '\n========== M40.1 DONE ==========\n'
find build/m40_1_real build/m40_1_e2e -maxdepth 4 -type f \( -name '*.json' -o -name '*.wav' -o -name '*.dsv35' \) -printf '%p\t%k KiB\n' 2>/dev/null | sort || true
