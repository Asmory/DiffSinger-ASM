#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

PROJECT="${PROJECT:-$(cd "$(dirname "$0")/.." && pwd)}"
CPU="${K3_CPU:-0}"
ROUNDS="${K3_ROUNDS:-7}"
WARMUPS="${K3_WARMUPS:-2}"
OUT="$PROJECT/build/k3_range"
mkdir -p "$OUT"
cd "$PROJECT"

make -j"$(nproc)" build/test_m58_range_residual
: > "$OUT/samples.tsv"
{
  date --iso-8601=seconds
  uname -sr
  powerprofilesctl get 2>/dev/null || true
  awk '/model name/ {print; exit}' /proc/cpuinfo
  awk '/Cpus_allowed_list/ {print}' /proc/self/status
  cat /proc/loadavg
} > "$OUT/environment.txt"

for ((i=1; i<=WARMUPS; i++)); do
  taskset -c "$CPU" ./build/test_m58_range_residual > /dev/null
done
for ((i=1; i<=ROUNDS; i++)); do
  log="$OUT/micro_${i}.log"
  taskset -c "$CPU" ./build/test_m58_range_residual | tee "$log"
  awk '
    /^M58 range-residual K3 C=(32|64) / {
      case_name=$3 "_" $4 "_" $5
      old=$0; sub(/^.* old=/,"",old); sub(/ms.*$/,"",old)
      new=$0; sub(/^.* new=/,"",new); sub(/ms.*$/,"",new)
      print case_name "\tbaseline\t" old
      print case_name "\tcandidate\t" new
    }
  ' "$log" >> "$OUT/samples.tsv"
done

python tools/check_perf_gate.py "$OUT/samples.tsv" --minimum 3
