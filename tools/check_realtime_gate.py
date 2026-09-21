#!/usr/bin/env python3
"""Gate paired real-time runs without median-based promotion."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input", type=Path,
        help="TSV rows: baseline|candidate, cpu_rtf, worst_rtf, deadline_misses",
    )
    parser.add_argument("--step", type=float, default=5.0)
    args = parser.parse_args()

    runs: dict[str, list[tuple[float, float, int]]] = {
        "baseline": [], "candidate": []
    }
    for line_number, raw in enumerate(args.input.read_text().splitlines(), 1):
        if not raw or raw.startswith("#"):
            continue
        fields = raw.split("\t")
        if len(fields) != 4 or fields[0] not in runs:
            raise SystemExit(
                f"{args.input}:{line_number}: expected "
                "baseline|candidate<TAB>cpu_rtf<TAB>worst_rtf<TAB>misses"
            )
        runs[fields[0]].append(
            (float(fields[1]), float(fields[2]), int(fields[3]))
        )

    if min(map(len, runs.values())) < 3:
        print("REALTIME: FAIL need at least three baseline and candidate runs")
        return 1
    service_ok = all(
        worst_rtf < 1.0 and misses == 0
        for values in runs.values()
        for _, worst_rtf, misses in values
    )
    if service_ok:
        baseline_value = max(value[0] for value in runs["baseline"])
        candidate_value = max(value[0] for value in runs["candidate"])
        metric = "cpu_RTF"
        misses_ok = True
    else:
        baseline_value = max(value[1] for value in runs["baseline"])
        candidate_value = max(value[1] for value in runs["candidate"])
        metric = "RTF"
        baseline_misses = max(value[2] for value in runs["baseline"])
        candidate_misses = max(value[2] for value in runs["candidate"])
        misses_ok = candidate_misses <= baseline_misses
    reduction = (1.0 - candidate_value / baseline_value) * 100.0
    passed = reduction >= args.step and misses_ok
    detail = "service=PASS" if service_ok else (
        f"worst_misses={baseline_misses}->{candidate_misses}"
    )
    print(
        f"REALTIME: {'PASS' if passed else 'FAIL'} phase="
        f"{'cpu-load' if service_ok else 'pre-service'} "
        f"worst_{metric}={baseline_value:.6f}->{candidate_value:.6f} "
        f"reduction={reduction:.2f}% {detail}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
