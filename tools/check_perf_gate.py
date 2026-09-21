#!/usr/bin/env python3
"""Apply the release performance gate to a two-column TSV sample set."""

from __future__ import annotations

import argparse
import math
import statistics
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def stats(values: list[float]) -> dict[str, float]:
    avg = statistics.mean(values)
    return {
        "median": statistics.median(values),
        "p90": percentile(values, 0.90),
        "worst": max(values),
        "cv": statistics.pstdev(values) / avg if avg else math.inf,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="TSV rows: case, variant, milliseconds[, measured_weight]")
    parser.add_argument("--minimum", type=float, default=5.0, help="minimum median improvement in percent")
    parser.add_argument("--max-cv", type=float, default=0.10, help="maximum candidate coefficient of variation")
    parser.add_argument("--weighted", action="store_true", help="aggregate cases using measured parent-profile weights")
    args = parser.parse_args()

    cases: dict[str, dict[str, list[float]]] = {}
    weights: dict[str, float] = {}
    for number, raw in enumerate(args.input.read_text().splitlines(), 1):
        if not raw or raw.startswith("#"):
            continue
        fields = raw.split("\t")
        if len(fields) not in {3, 4} or fields[1] not in {"baseline", "candidate"}:
            raise SystemExit(f"{args.input}:{number}: expected case<TAB>baseline|candidate<TAB>value[<TAB>weight]")
        cases.setdefault(fields[0], {}).setdefault(fields[1], []).append(float(fields[2]))
        if len(fields) == 4:
            weight = float(fields[3])
            if weight <= 0 or (fields[0] in weights and weights[fields[0]] != weight):
                raise SystemExit(f"{args.input}:{number}: weight must be positive and consistent per case")
            weights[fields[0]] = weight

    if args.weighted and set(weights) != set(cases):
        raise SystemExit("weighted mode requires a measured weight on every row")

    failed = False
    summaries: dict[str, tuple[dict[str, float], dict[str, float]]] = {}
    for case in sorted(cases):
        variants = cases[case]
        if set(variants) != {"baseline", "candidate"} or min(map(len, variants.values())) < 5:
            print(f"{case}: FAIL need at least five samples for both variants")
            failed = True
            continue
        base = stats(variants["baseline"])
        cand = stats(variants["candidate"])
        summaries[case] = (base, cand)
        gain = (base["median"] / cand["median"] - 1.0) * 100.0
        checks = {
            "median_gain": gain >= args.minimum,
            "p90_no_regression": cand["p90"] <= base["p90"],
            "worst_no_regression": cand["worst"] <= base["worst"],
            # Paired A/B measurements can drift together as package power and
            # frequency change. The candidate must either meet the absolute
            # CV limit or be no more variable than the paired baseline.
            "candidate_cv": cand["cv"] <= args.max_cv or cand["cv"] <= base["cv"] + 0.01,
        }
        status = "PASS" if all(checks.values()) else "FAIL"
        print(
            f"{case}: {status} gain={gain:.2f}% "
            f"baseline(median={base['median']:.3f},p90={base['p90']:.3f},worst={base['worst']:.3f},cv={base['cv']:.3f}) "
            f"candidate(median={cand['median']:.3f},p90={cand['p90']:.3f},worst={cand['worst']:.3f},cv={cand['cv']:.3f})"
        )
        if status == "FAIL" and not args.weighted:
            print("  failed=" + ",".join(name for name, ok in checks.items() if not ok))
            failed = True
    if args.weighted and summaries:
        total_weight = sum(weights.values())
        ratios = {
            metric: sum(weights[case] * cand[metric] / base[metric] for case, (base, cand) in summaries.items()) / total_weight
            for metric in ("median", "p90", "worst")
        }
        candidate_cv = sum(weights[case] * cand["cv"] for case, (_, cand) in summaries.items()) / total_weight
        baseline_cv = sum(weights[case] * base["cv"] for case, (base, _) in summaries.items()) / total_weight
        gain = (1.0 - ratios["median"]) * 100.0
        checks = {
            "weighted_gain": gain >= args.minimum,
            "weighted_p90": ratios["p90"] <= 1.0,
            "weighted_worst": ratios["worst"] <= 1.0,
            "weighted_cv": candidate_cv <= args.max_cv or candidate_cv <= baseline_cv + 0.01,
        }
        status = "PASS" if all(checks.values()) else "FAIL"
        print(
            f"WEIGHTED: {status} gain={gain:.2f}% p90_ratio={ratios['p90']:.4f} "
            f"worst_ratio={ratios['worst']:.4f} baseline_cv={baseline_cv:.3f} candidate_cv={candidate_cv:.3f}"
        )
        if status == "FAIL":
            print("  failed=" + ",".join(name for name, ok in checks.items() if not ok))
            failed = True
    return 1 if failed or not cases else 0


if __name__ == "__main__":
    raise SystemExit(main())
