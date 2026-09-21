#!/usr/bin/env python3
"""Compare only the verified M55 long-audio baseline with the M58 candidate."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path


PROJECT = Path(os.environ.get("PROJECT", Path(__file__).resolve().parents[1])).resolve()
MODEL = Path(os.environ.get("MODEL", PROJECT / "models/DongFangZhiZi_Nectar_DiffSinger_CE_26.08.31"))
PYTHON = Path(os.environ.get("PY", PROJECT / ".venv/bin/python"))
BASE = PROJECT / "build/m53_long/f384"
FIXTURE = BASE / "fixture"
BASELINE_BUNDLE = BASE / "nsf.dsv35"
OUT = PROJECT / "build/m58_e2e"


def run(command: list[object], env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=PROJECT, env=env, check=True)


def parse_cpu_list(value: str) -> list[int]:
    result: list[int] = []
    for part in value.strip().split(","):
        if "-" in part:
            first, last = map(int, part.split("-", 1))
            result.extend(range(first, last + 1))
        elif part:
            result.append(int(part))
    return result


def p_core_smt() -> str:
    override = os.environ.get("M58_CPUS")
    if override:
        return override
    match = re.search(r"Cpus_allowed_list:\s*(.+)", Path("/proc/self/status").read_text())
    allowed = parse_cpu_list(match.group(1)) if match else list(range(os.cpu_count() or 1))
    allowed_set = set(allowed)
    groups: set[tuple[int, ...]] = set()
    for cpu in allowed:
        path = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
        siblings = tuple(x for x in parse_cpu_list(path.read_text()) if x in allowed_set)
        if len(siblings) > 1:
            groups.add(siblings)
    cpus = [cpu for group in sorted(groups, key=min) for cpu in group]
    if not cpus:
        cpus = allowed[: min(8, len(allowed))]
    return ",".join(map(str, cpus))


def require_inputs() -> None:
    required = [
        PYTHON,
        MODEL / "dsvocoder/nsf_hifigan.onnx",
        MODEL / "dongfangzhizi-nectar-xiao.emb",
        PROJECT / "build/dsasm-vocoder-m40",
        PROJECT / "build/dsasm-acoustic",
        PROJECT / "build/m42_acoustic/model.json",
        BASELINE_BUNDLE,
        FIXTURE / "reference_mel.f32",
        FIXTURE / "golden_wave.f32",
        FIXTURE / "f0.f32",
        FIXTURE / "noise.f32",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing M55 comparison prerequisites:\n  " + "\n  ".join(missing))


def benchmark_env(label: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        DSASM_2D="1",
        DSASM_KSPEC="0",
        DSASM_PARALLEL_ADD="1",
        DSASM_PARALLEL_LEAKY_COPY="1",
        DSASM_VOCODER_T_TILE="2016",
        DSASM_RANGE_T24="1",
        DSASM_RANGE_RESIDUAL_T24="1",
        DSASM_RESIDUAL_T24="1",
        DSASM_K3_TMODE="24",
        DSASM_K7_T24="1",
        DSASM_K11_T24="1",
        DSASM_VNNI="k11" if label == "baseline" else "k117",
        DSASM_VNNI_ASYM="0" if label == "baseline" else "1",
        DSASM_VNNI_CIN="128",
        DSASM_GOLDEN_COS="0.999",
        DSASM_GOLDEN_SNR="25",
    )
    return env


def measure(label: str, bundle: Path, index: int, cpus: str) -> list[float]:
    work = OUT / f"{index:02d}_{label}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    command = [
        "taskset", "-c", cpus,
        PROJECT / "build/persistent-e2e",
        "--model", PROJECT / "build/m42_acoustic",
        "--vocoder", bundle,
        "--fixture", FIXTURE,
        "--speaker-emb", MODEL / "dongfangzhizi-nectar-xiao.emb",
        "--workers", str(len(parse_cpu_list(cpus))),
        "--warmup", "2",
        "--requests", "5",
        "--steps", "4",
        "--depth", "0.6",
    ]
    print("+ " + " ".join(map(str, command)), flush=True)
    process = subprocess.run(list(map(str, command)), cwd=PROJECT, env=benchmark_env(label), text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
    print(process.stdout, end="")
    (work / "run.log").write_text(process.stdout)
    values = [float(value) for value in re.findall(r"^E2E request\d+ .* total=([0-9.]+)", process.stdout, re.MULTILINE)]
    if len(values) != 5:
        raise SystemExit(f"expected five persistent samples, got {len(values)}")
    return values


def main() -> None:
    run(["make", "-j" + str(os.cpu_count() or 1), "build/persistent-e2e", "build/dsasm-vocoder-m40"])
    require_inputs()
    OUT.mkdir(parents=True, exist_ok=True)
    candidate = OUT / "m58-all3711.dsv35"
    pack_work = OUT / "packer"
    run(
        [
            PYTHON,
            "tools/pack_vocoder_graph_m35.py",
            MODEL / "dsvocoder/nsf_hifigan.onnx",
            "--frames", "384",
            "--vnni-scope", "all-k711",
            "--residual-scope", "all3711",
            "--out", candidate,
            "--work", pack_work,
        ]
    )

    cpus = p_core_smt()
    samples: list[tuple[str, float]] = []
    sequence = ["baseline", "candidate", "candidate", "baseline"]
    bundles = {"baseline": BASELINE_BUNDLE, "candidate": candidate}
    for index, label in enumerate(sequence, 1):
        samples.extend((label, value) for value in measure(label, bundles[label], index, cpus))

    sample_file = OUT / "samples.tsv"
    sample_file.write_text("".join(f"e2e\t{label}\t{value:.6f}\n" for label, value in samples))
    run(["python", "tools/check_perf_gate.py", sample_file, "--minimum", "3"])


if __name__ == "__main__":
    main()
