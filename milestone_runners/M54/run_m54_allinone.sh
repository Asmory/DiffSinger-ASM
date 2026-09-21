#!/usr/bin/env bash
set -euo pipefail
PROJECT="${PROJECT:-$HOME/asm/diffsinger}"
PY="${PY:-$PROJECT/.venv/bin/python}"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT"
printf '\n========== M54 LONG-AUDIO TOPOLOGY + VOCODER PROFILE ==========\n'
"$PY" "$HERE/run_m54_long_topology_profile.py"
