#!/usr/bin/env bash
set -euo pipefail
PROJECT="${PROJECT:-$HOME/asm/diffsinger}"
PY="${PY:-$PROJECT/.venv/bin/python}"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT"
printf '\n========== M55 LONG MEMORY-PASS APPLY ==========\n'
if ! grep -q 'DS_JOB_LEAKY_COPY_NCT' src/runtime/threadpool_internal.h; then
  patch -p1 < "$HERE/diffsinger_m55_long_memory_passes.patch"
else
  echo 'M55 source patch already applied.'
fi
printf '\n========== M55 BUILD ==========\n'
make -j"$(nproc)" build/dsasm-vocoder-m40 build/dsasm-acoustic
printf '\n========== M55 LEAKY-COPY BIT PARITY / MICRO ==========\n'
cc -Iinclude -Isrc/runtime -O3 -Wall -Wextra -std=c11 -march=native \
  tools/test_parallel_leaky_m55.c \
  build/threadpool.o build/leaky_copy_m36.o build/conv1d_m36.o build/conv1d_m37_range.o \
  build/conv1d_m39_kspec.o build/conv1d_m38_residual.o build/conv1d_m38_range_residual.o \
  build/linear_m4n16.o build/linear_residual_m4n16.o build/linear_m4n16_strided.o \
  build/linear_residual_m4n16_strided.o build/linear_m4n16_idxstrided.o \
  build/linear_residual_m4n16_idxstrided.o build/linear_n16_kblock.o \
  build/dwconv_k31_tc_cstrided.o build/glu_m6.o build/atan_glu_m9_strided.o \
  build/conv1d_m39_vnni.o build/vnni_pack_m40_1.o build/convtranspose_m33.o build/conv1d_m32.o \
  -o build/test_parallel_leaky_m55 -lm -pthread
./build/test_parallel_leaky_m55
printf '\n========== M55 TARGET LONG-AUDIO A/B ==========\n'
"$PY" "$HERE/run_m55_long_memory.py"
