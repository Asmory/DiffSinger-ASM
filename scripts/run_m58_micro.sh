#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

PROJECT="${PROJECT:-$(cd "$(dirname "$0")/.." && pwd)}"
CPU="${M58_CPU:-0}"
ROUNDS="${M58_ROUNDS:-7}"
OUT="$PROJECT/build/m58"
mkdir -p "$OUT"
cd "$PROJECT"

make -j"$(nproc)" build/test_m58_range_residual
: > "$OUT/samples.tsv"

for ((i=1; i<=ROUNDS; i++)); do
  log="$OUT/micro_${i}.log"
  taskset -c "$CPU" ./build/test_m58_range_residual | tee "$log"
  awk '
    /^M58 range-residual/ {
      case_name=$3 "_" $4 "_" $5
      old=$0; sub(/^.* old=/,"",old); sub(/ms.*$/,"",old)
      new=$0; sub(/^.* new=/,"",new); sub(/ms.*$/,"",new)
      print case_name "\tbaseline\t" old
      print case_name "\tcandidate\t" new
    }
  ' "$log" >> "$OUT/samples.tsv"
done

gate_rc=0
python tools/check_perf_gate.py "$OUT/samples.tsv" --minimum 3 || gate_rc=$?

if command -v perf >/dev/null 2>&1 && perf stat -e cpu_core/cycles/u -- true >/dev/null 2>&1; then
  taskset -c "$CPU" perf stat -x, -o "$OUT/perf-stat.csv" \
    -e cpu_core/cycles/u -e cpu_core/instructions/u -e cpu_core/cache-misses/u -- \
    ./build/test_m58_range_residual > "$OUT/perf-run.log"
  echo "hardware counters: $OUT/perf-stat.csv"
fi
exit "$gate_rc"
