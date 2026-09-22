#!/usr/bin/env python3
"""Versioned offline model protocol for DiffSinger-ASM product bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA = "org.openutau.dsasm-model-tool"
SCHEMA_VERSION = 1
MANIFEST_SCHEMA = "org.openutau.dsasm-bundle"
MANIFEST_VERSION = 1
CONVERTER_REVISION = "m25-m35-v1"
ISA_PROFILE = "avx2-fma"
PACKERS = [
    {"role": "acoustic", "revision": "M25-onnx-deployment"},
    {"role": "vocoder", "revision": "M40-dsvoc35-all-k711-all3711"},
]
FORMATS = [
    {"role": "acoustic.aux", "version": "DSAUX20/1"},
    {"role": "acoustic.fs2", "version": "DSFS25/1"},
    {"role": "acoustic.reflow", "version": "DSLYNX7/1"},
    {"role": "vocoder.32", "version": "DSVOC35/1"},
    {"role": "vocoder.384", "version": "DSVOC35/1"},
]
REQUIRED_ARTIFACTS = (
    ("acoustic.config", "acoustic/model.conf"),
    ("acoustic.fs2", "acoustic/fs2_acoustic.dsfs"),
    ("acoustic.aux", "acoustic/aux_convnext.dsa"),
    ("acoustic.reflow", "acoustic/lynxnet2.dsn"),
    ("vocoder.32", "vocoder/32.dsv35"),
    ("vocoder.384", "vocoder/384.dsv35"),
)
VOCODER_OPS = {
    "Add", "Sub", "Mul", "Div", "Mod", "LeakyRelu", "Tanh", "Sin", "CumSum",
    "Reshape", "Squeeze", "Transpose", "Slice", "Pad", "Conv", "ConvTranspose", "Unsqueeze",
}


class ProtocolError(Exception):
    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code
        self.detail = detail


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("ascii")


def provider_build_identity() -> str:
    explicit = os.environ.get("DSASM_PROVIDER_BUILD_ID")
    if explicit:
        return explicit
    root = Path(__file__).resolve().parent.parent
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], check=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        ).stdout
        return f"git:{commit}{'-dirty' if dirty else ''}"
    except (OSError, subprocess.SubprocessError):
        return "unversioned"


def status_for(code: str) -> str:
    if code == "ok":
        return "ok"
    if code == "cancelled":
        return "cancelled"
    if code == "source.changed":
        return "error"
    if code.startswith("source."):
        return "incompatible"
    if code.startswith("bundle."):
        return "invalid"
    return "error"


def exit_for(status: str, code: str) -> int:
    if status == "ok":
        return 0
    if status == "incompatible":
        return 3
    if status == "invalid":
        return 4
    if status == "cancelled":
        return 7
    if code == "source.changed":
        return 9
    if code in ("request.invalid", "protocol.unsupported"):
        return 2
    if code.startswith("toolchain."):
        return 5
    if code.startswith("io.") or code == "space.insufficient":
        return 6
    return 8


def envelope(operation: str, status: str, code: str, diagnostic: str = "", event: str = "result", **payload: Any) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "operation": operation,
        "event": event,
        "status": status,
        "reason_code": code,
        "diagnostic": diagnostic,
        "converter_revision": CONVERTER_REVISION,
        "provider_build": provider_build_identity(),
        **payload,
    }


def load_singer_config(singer_root: Path) -> dict[str, Any]:
    config_path = singer_root / "dsconfig.yaml"
    if not config_path.is_file():
        raise ProtocolError("source.missing", "dsconfig.yaml")
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:
        raise ProtocolError("toolchain.missing_dependency", exc.name or "yaml") from exc
    try:
        value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ProtocolError("source.invalid_config", str(exc)) from exc
    if not isinstance(value, dict):
        raise ProtocolError("source.invalid_config", "dsconfig.yaml root must be a mapping")
    return value


def contained_source(singer_root: Path, value: str, label: str) -> Path:
    pure = PurePosixPath(value)
    if (not value or pure.is_absolute() or "\\" in value or
            any(part in ("", ".", "..") for part in pure.parts)):
        raise ProtocolError("source.invalid_path", f"{label}: {value!r}")
    root = singer_root.resolve()
    candidate = root.joinpath(*pure.parts)

    cursor = root
    for part in pure.parts:
        cursor /= part
        if cursor.is_symlink() and not cursor.exists():
            raise ProtocolError("source.invalid_path", f"{label}: broken symlink: {value}")

    if not candidate.exists():
        try:
            candidate.resolve(strict=False).relative_to(root)
        except (OSError, ValueError) as exc:
            raise ProtocolError("source.invalid_path", f"{label}: {value}") from exc
        raise ProtocolError("source.missing", f"{label}: {value}")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise ProtocolError("source.invalid_path", f"{label}: {value}") from exc
    if not resolved.is_file():
        raise ProtocolError("source.invalid_path", f"{label}: not a regular file")
    return resolved


def relative_source_path(singer_root: Path, path: Path) -> str:
    return path.resolve().relative_to(singer_root.resolve()).as_posix()


def resolve_sources(singer_root: Path, config: dict[str, Any]) -> tuple[Path, Path]:
    acoustic_name = config.get("acoustic", "acoustic.onnx")
    vocoder_name = config.get("vocoder", "nsf_hifigan")
    if not isinstance(acoustic_name, str) or not isinstance(vocoder_name, str):
        raise ProtocolError("source.invalid_config", "acoustic and vocoder must be paths")
    acoustic = contained_source(singer_root, acoustic_name, "acoustic")
    vocoder_relative = Path(vocoder_name)
    if vocoder_relative.suffix != ".onnx":
        vocoder_relative = vocoder_relative.with_suffix(".onnx")
    vocoder_value = (Path("dsvocoder") / vocoder_relative).as_posix()
    fallback = singer_root / "dsvocoder" / "nsf_hifigan.onnx"
    try:
        vocoder = contained_source(singer_root, vocoder_value, "vocoder")
    except ProtocolError as exc:
        if exc.code != "source.missing" or not fallback.is_file():
            raise
        vocoder = contained_source(singer_root, "dsvocoder/nsf_hifigan.onnx", "vocoder")
    return acoustic, vocoder


def onnx_external_sources(singer_root: Path, onnx_path: Path, prefix: str) -> list[tuple[str, Path]]:
    try:
        import onnx  # type: ignore
        import onnxruntime  # type: ignore  # noqa: F401
        from onnx import AttributeProto, TensorProto  # type: ignore
    except ModuleNotFoundError as exc:
        raise ProtocolError("toolchain.missing_dependency", exc.name or "onnx") from exc
    model = onnx.load(str(onnx_path), load_external_data=False)
    tensors = []

    def walk(graph: Any) -> None:
        tensors.extend(graph.initializer)
        for node in graph.node:
            for attribute in node.attribute:
                if attribute.type == AttributeProto.GRAPH:
                    walk(attribute.g)
                elif attribute.type == AttributeProto.GRAPHS:
                    for child in attribute.graphs:
                        walk(child)

    walk(model.graph)
    values: list[tuple[str, Path]] = []
    for tensor in tensors:
        if tensor.data_location != TensorProto.EXTERNAL:
            continue
        location = next((item.value for item in tensor.external_data if item.key == "location"), "")
        external = PurePosixPath(location)
        if (not location or external.is_absolute() or "\\" in location or
                any(part in ("", ".", "..") for part in external.parts)):
            raise ProtocolError("source.invalid_path", f"external data: {location!r}")
        parent = onnx_path.resolve().parent.relative_to(singer_root.resolve())
        source_path = parent.joinpath(*external.parts).as_posix()
        resolved = contained_source(singer_root, source_path, "external data")
        values.append((f"{prefix}.external/{external.as_posix()}", resolved))
    return values


def source_entries(singer_root: Path, acoustic: Path, vocoder: Path, config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    config = config or load_singer_config(singer_root)
    candidates: list[tuple[str, Path]] = [
        ("config.dsconfig", contained_source(singer_root, "dsconfig.yaml", "config")),
        ("acoustic.onnx", acoustic),
        ("vocoder.onnx", vocoder),
    ]
    for key, role in (("phonemes", "tokens.phonemes"), ("languages", "tokens.languages")):
        value = config.get(key)
        if isinstance(value, str):
            candidates.append((role, contained_source(singer_root, value, key)))
    for path in sorted(singer_root.glob("*.emb")):
        resolved = contained_source(singer_root, path.name, "speaker embedding")
        candidates.append((f"speaker.embedding/{path.name}", resolved))
    candidates.extend(onnx_external_sources(singer_root, acoustic, "acoustic"))
    candidates.extend(onnx_external_sources(singer_root, vocoder, "vocoder"))
    entries = []
    seen: set[str] = set()
    for role, path in candidates:
        if role in seen:
            continue
        seen.add(role)
        entries.append({
            "role": role, "path": relative_source_path(singer_root, path),
            "size": path.stat().st_size, "sha256": sha256_file(path),
        })
    return sorted(entries, key=lambda item: item["role"])


def source_fingerprint(entries: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256(b"dsasm-source-v1\0")
    seen: set[str] = set()
    for item in sorted(entries, key=lambda value: value["role"]):
        role = item["role"]
        if role in seen:
            raise ProtocolError("source.invalid_config", f"duplicate source role: {role}")
        seen.add(role)
        digest.update(role.encode("utf-8") + b"\0")
        digest.update(str(item["size"]).encode("ascii") + b"\0")
        digest.update(item["sha256"].encode("ascii") + b"\0")
    return digest.hexdigest()


def validate_source_records(entries: Any) -> list[dict[str, Any]]:
    if not isinstance(entries, list):
        raise ProtocolError("bundle.invalid.manifest", "source_artifacts must be an array")
    seen_roles: set[str] = set()
    seen_paths: set[str] = set()
    for item in entries:
        if not isinstance(item, dict):
            raise ProtocolError("bundle.invalid.manifest", "source artifact must be an object")
        role, path = item.get("role"), item.get("path")
        size, digest = item.get("size"), item.get("sha256")
        if not isinstance(role, str) or not role or not isinstance(path, str):
            raise ProtocolError("bundle.invalid.manifest", "source artifact role/path")
        pure = PurePosixPath(path)
        if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts) or "\\" in path:
            raise ProtocolError("bundle.invalid.path", path)
        if role in seen_roles or path in seen_paths:
            raise ProtocolError("bundle.invalid.path", f"duplicate source role or path: {role} {path}")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ProtocolError("bundle.invalid.manifest", f"invalid source size: {role}")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ProtocolError("bundle.invalid.manifest", f"invalid source digest: {role}")
        seen_roles.add(role)
        seen_paths.add(path)
    return entries


def inspect_source(singer_root: Path, config: dict[str, Any], acoustic: Path, vocoder: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    missing = [label for label, path in (("acoustic.onnx", acoustic), ("vocoder.onnx", vocoder)) if not path.is_file()]
    entries = source_entries(singer_root, acoustic, vocoder, config)
    if missing:
        raise ProtocolError("source.missing", ", ".join(missing))
    dimensions = {
        "sample_rate": config.get("sample_rate"),
        "hop_size": config.get("hop_size"),
        "mel_bins": config.get("num_mel_bins"),
    }
    expected = {"sample_rate": 44100, "hop_size": 512, "mel_bins": 128}
    for key, wanted in expected.items():
        try:
            actual = int(dimensions[key])
        except (TypeError, ValueError) as exc:
            raise ProtocolError("source.invalid_config", f"{key} is missing or invalid") from exc
        dimensions[key] = actual
        if actual != wanted:
            raise ProtocolError(f"source.unsupported.{key}", f"expected {wanted}, got {actual}")
    if bool(config.get("use_energy_embed", False)):
        raise ProtocolError("source.unsupported.energy_embedding", "use_energy_embed is true")
    try:
        import onnx  # type: ignore
        acoustic_model = onnx.load(str(acoustic), load_external_data=False)
        vocoder_model = onnx.load(str(vocoder), load_external_data=False)
        onnx.checker.check_model(acoustic_model, full_check=False)
        onnx.checker.check_model(vocoder_model, full_check=False)
        unsupported = sorted({node.op_type for node in vocoder_model.graph.node if node.op_type not in VOCODER_OPS})
        inputs = {item.name for item in vocoder_model.graph.input}
        if unsupported:
            raise ProtocolError("source.unsupported.vocoder_graph", ",".join(unsupported))
        if not {"mel", "f0"}.issubset(inputs):
            raise ProtocolError("source.unsupported.vocoder_graph", f"inputs={sorted(inputs)}")
        import pack_acoustic_onnx_m25 as acoustic_packer  # type: ignore
        acoustic_packer.preflight_onnx(acoustic)
        packer = Path(__file__).resolve().parent / "pack_vocoder_graph_m35.py"
        for frames in (32, 384):
            result = subprocess.run(
                [sys.executable, str(packer), str(vocoder), "--frames", str(frames), "--preflight",
                 "--vnni-scope", "all-k711", "--residual-scope", "all3711"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            if result.returncode:
                raise ProtocolError("source.unsupported.vocoder_graph", result.stdout.strip()[-8000:])
    except ModuleNotFoundError as exc:
        raise ProtocolError("toolchain.missing_dependency", exc.name or "onnx") from exc
    except ProtocolError:
        raise
    except Exception as exc:
        raise ProtocolError("source.unsupported.acoustic_graph", f"{type(exc).__name__}: {exc}") from exc
    return dimensions, entries


def parse_config(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    except (OSError, UnicodeError) as exc:
        raise ProtocolError("bundle.invalid.dimensions", str(exc)) from exc
    return values


def read_exact_header(path: Path, size: int) -> tuple[bytes, int]:
    try:
        file_size = path.stat().st_size
        with path.open("rb") as stream:
            data = stream.read(size)
    except OSError as exc:
        raise ProtocolError("bundle.incomplete", f"{path}: {exc}") from exc
    if len(data) != size:
        raise ProtocolError("bundle.invalid.format", f"{path}: {file_size} bytes, header requires {size}")
    return data, file_size


class PackedF32Layout:
    def __init__(self, path: Path, file_size: int):
        self.path = path
        self.file_size = file_size
        self.offset = 64

    def take(self, count: int) -> int:
        start = (self.offset + 63) & ~63
        end = start + count * 4
        if count < 0 or end > self.file_size:
            raise ProtocolError(
                "bundle.invalid.format",
                f"{self.path.name}: section ending at {end} exceeds {self.file_size} bytes",
            )
        self.offset = end
        return start

    def finish(self) -> None:
        if self.offset != self.file_size:
            raise ProtocolError(
                "bundle.invalid.format",
                f"{self.path.name}: expected {self.offset} bytes, got {self.file_size}",
            )


def parse_acoustic_headers(bundle: Path) -> dict[str, Any]:
    acoustic = bundle / "acoustic"
    fs_path = acoustic / "fs2_acoustic.dsfs"
    fs_raw, fs_size = read_exact_header(fs_path, 64)
    if fs_raw[:8] != b"DSFS25\0\0":
        raise ProtocolError("bundle.invalid.format", repr(fs_raw[:8]))
    fs = struct.unpack_from("<9I4fI", fs_raw, 8)
    fs_version, vocab, fs_hidden, fs_layers, heads, kernel, interleaved, languages, flags = fs[:9]
    rope_max = fs[13]
    if (fs_version != 1 or not all((vocab, fs_hidden, fs_layers, heads)) or kernel != 3 or
            fs_hidden % 16 or fs_hidden % heads or interleaved not in (0, 1) or flags & ~((1 << 14) - 1)):
        raise ProtocolError("bundle.invalid.format", repr(fs[:9]))
    if (flags & (1 << 0)) and languages < 2:
        raise ProtocolError("bundle.invalid.format", "DSFS25 language table has fewer than two rows")
    if (flags & (1 << 11)) and not rope_max:
        raise ProtocolError("bundle.invalid.format", "DSFS25 exact RoPE has zero capacity")

    fs_layout = PackedF32Layout(fs_path, fs_size)
    for count in (vocab * fs_hidden, fs_hidden, fs_hidden):
        fs_layout.take(count)
    adaptive: tuple[float, ...] = ()
    if flags & (1 << 13):
        adaptive_offset = fs_layout.take(2 * fs_layers)
        try:
            with fs_path.open("rb") as stream:
                stream.seek(adaptive_offset)
                adaptive_raw = stream.read(2 * fs_layers * 4)
            adaptive = struct.unpack(f"<{2 * fs_layers}f", adaptive_raw)
        except (OSError, struct.error) as exc:
            raise ProtocolError("bundle.invalid.format", f"DSFS25 adaptive mask: {exc}") from exc
        if any(value not in (0.0, 1.0) for value in adaptive):
            raise ProtocolError("bundle.invalid.format", "DSFS25 adaptive mask must contain only 0 or 1")
    for layer in range(fs_layers):
        fs_layout.take(fs_hidden)
        fs_layout.take(fs_hidden)
        if adaptive and adaptive[2 * layer]:
            fs_layout.take(2 * fs_hidden * fs_hidden)
            fs_layout.take(2 * fs_hidden)
        for count in (
            3 * fs_hidden * fs_hidden, 3 * fs_hidden,
            fs_hidden * fs_hidden, fs_hidden, fs_hidden, fs_hidden,
        ):
            fs_layout.take(count)
        if adaptive and adaptive[2 * layer + 1]:
            fs_layout.take(2 * fs_hidden * fs_hidden)
            fs_layout.take(2 * fs_hidden)
        for count in (12 * fs_hidden * fs_hidden, 4 * fs_hidden, 4 * fs_hidden * fs_hidden, fs_hidden):
            fs_layout.take(count)
    for count in (
        fs_hidden, fs_hidden,
        4 * fs_hidden * fs_hidden, 4 * fs_hidden, 4 * fs_hidden * fs_hidden, fs_hidden,
        3 * fs_hidden * fs_hidden, 3 * fs_hidden, 3 * fs_hidden * fs_hidden, 3 * fs_hidden,
        fs_hidden, fs_hidden,
    ):
        fs_layout.take(count)
    if flags & (1 << 0):
        fs_layout.take(languages * fs_hidden)
    if flags & (1 << 8):
        fs_layout.take(vocab)
    for feature in (1, 2, 3, 4, 5):
        if flags & (1 << feature):
            fs_layout.take(fs_hidden)
            fs_layout.take(fs_hidden)
    fs_layout.take(7)
    if flags & (1 << 10):
        fs_layout.take(fs_hidden)
    if flags & (1 << 11):
        rope_width = (fs_hidden // heads) // 2
        fs_layout.take(rope_max * rope_width)
        fs_layout.take(rope_max * rope_width)
    if flags & (1 << 7):
        fs_layout.take(1001 * fs_hidden)
    fs_layout.finish()

    aux_path = acoustic / "aux_convnext.dsa"
    aux_raw, aux_size = read_exact_header(aux_path, 64)
    if aux_raw[:8] != b"DSAUX20\0":
        raise ProtocolError("bundle.invalid.format", repr(aux_raw[:8]))
    aux = struct.unpack_from("<8I", aux_raw, 8)
    aux_version, aux_input, aux_channels, aux_output, aux_layers, aux_kernel, aux_m, aux_n = aux
    if (aux_version != 1 or not all((aux_input, aux_channels, aux_output, aux_layers)) or
            aux_kernel != 7 or aux_channels % 16 or aux_output % 16 or (aux_m, aux_n) != (4, 16)):
        raise ProtocolError("bundle.invalid.format", repr(aux))
    aux_layout = PackedF32Layout(aux_path, aux_size)
    aux_layout.take(aux_channels * 7 * aux_input)
    aux_layout.take(aux_channels)
    for _ in range(aux_layers):
        for count in (
            7 * aux_channels, aux_channels, aux_channels, aux_channels,
            4 * aux_channels * aux_channels, 4 * aux_channels,
            4 * aux_channels * aux_channels, aux_channels, aux_channels,
        ):
            aux_layout.take(count)
    aux_layout.take(aux_output * 7 * aux_channels)
    aux_layout.take(aux_output)
    aux_layout.finish()

    rf_path = acoustic / "lynxnet2.dsn"
    rf_raw, rf_size = read_exact_header(rf_path, 64)
    if rf_raw[:8] != b"DSLYNX7\0":
        raise ProtocolError("bundle.invalid.format", repr(rf_raw[:8]))
    rf = struct.unpack_from("<12I", rf_raw, 8)
    rf_version, rf_input, rf_condition, rf_channels, rf_hidden, rf_layers, rf_kernel, glu = rf[:8]
    if (rf_version != 1 or not all((rf_input, rf_condition, rf_channels, rf_hidden, rf_layers)) or
            rf_kernel != 31 or glu not in (1, 2, 3) or rf_channels % 16 or rf_input % 16 or
            rf_hidden % 8 or tuple(rf[8:12]) != (4, 16, 4, 8)):
        raise ProtocolError("bundle.invalid.format", repr(rf))
    rf_layout = PackedF32Layout(rf_path, rf_size)
    for count in (
        rf_channels * rf_input, rf_channels, rf_channels * rf_condition, rf_channels,
        4 * rf_channels * rf_channels, 4 * rf_channels, 4 * rf_channels * rf_channels, rf_channels,
    ):
        rf_layout.take(count)
    for _ in range(rf_layers):
        for count in (
            rf_channels, rf_channels, 31 * rf_channels, rf_channels,
            2 * rf_hidden * rf_channels, 2 * rf_hidden,
            2 * rf_hidden * rf_hidden, 2 * rf_hidden,
            rf_channels * rf_hidden, rf_channels,
        ):
            rf_layout.take(count)
    for count in (rf_channels, rf_channels, rf_input * rf_channels, rf_input):
        rf_layout.take(count)
    rf_layout.finish()

    if fs_hidden != aux_input or aux_input != rf_condition or aux_output != rf_input:
        raise ProtocolError(
            "bundle.invalid.dimensions",
            f"fs_hidden={fs_hidden} aux={aux_input}->{aux_output} rf={rf_input}/{rf_condition}",
        )
    config = parse_config(acoustic / "model.conf")
    try:
        config_mel = int(config["mel_bins"])
        sample_rate = int(config["sample_rate"])
        hop_size = int(config["hop_size"])
    except (KeyError, ValueError) as exc:
        raise ProtocolError("bundle.invalid.dimensions", f"missing or invalid audio dimensions: {exc}") from exc
    if config_mel != rf_input or sample_rate <= 0 or hop_size <= 0:
        raise ProtocolError(
            "bundle.invalid.dimensions",
            f"mel_bins={config_mel}, reflow_input={rf_input}, sample_rate={sample_rate}, hop_size={hop_size}",
        )
    return {
        "sample_rate": sample_rate, "hop_size": hop_size, "mel_bins": config_mel,
        "acoustic_hidden_size": fs_hidden, "acoustic_output_dim": aux_output,
    }


def parse_vocoder_header(path: Path) -> dict[str, int]:
    raw, file_size = read_exact_header(path, 128)
    values = struct.unpack("<8s7I8Q28x", raw)
    if values[0] != b"DSVOC35\0" or values[1] != 1:
        raise ProtocolError("bundle.invalid.format", f"{path.name}: magic={values[0]!r} version={values[1]}")
    tensor_count, op_count = values[2], values[3]
    input_mel, input_f0, output = values[4], values[5], values[6]
    const_bytes = values[9]
    tensor_offset, op_offset, const_offset = values[10], values[11], values[12]
    frames, mel_bins, samples = values[13], values[14], values[15]
    if not tensor_count or not op_count or not frames or not mel_bins or not samples:
        raise ProtocolError("bundle.invalid.format", f"{path.name}: zero dimension or table count")
    if any(identifier >= tensor_count for identifier in (input_mel, input_f0, output)):
        raise ProtocolError("bundle.invalid.format", f"{path.name}: invalid input/output tensor id")
    if (tensor_offset != 128 or op_offset != tensor_offset + tensor_count * 80 or
            const_offset < op_offset + op_count * 192 or const_offset % 64 or
            const_offset + const_bytes != file_size):
        raise ProtocolError("bundle.invalid.format", path.name)
    return {"frames": frames, "mel_bins": mel_bins, "samples": samples}


def validate_formats(bundle: Path) -> dict[str, Any]:
    missing = [path for _, path in REQUIRED_ARTIFACTS if not (bundle / path).is_file()]
    if missing:
        raise ProtocolError("bundle.incomplete", ", ".join(missing))
    audio = parse_acoustic_headers(bundle)
    buckets = []
    for expected in (32, 384):
        header = parse_vocoder_header(bundle / "vocoder" / f"{expected}.dsv35")
        if header["frames"] != expected:
            raise ProtocolError("bundle.invalid.dimensions", f"{expected}.dsv35 declares {header['frames']} frames")
        if header["mel_bins"] != audio["mel_bins"]:
            raise ProtocolError("bundle.invalid.dimensions", f"bucket={expected}")
        if header["samples"] != expected * audio["hop_size"]:
            raise ProtocolError(
                "bundle.invalid.dimensions",
                f"bucket={expected} samples={header['samples']} hop={audio['hop_size']}",
            )
        buckets.append(expected)
    return {**audio, "required_buckets": buckets}


def artifact_records(bundle: Path) -> list[dict[str, Any]]:
    return [
        {"role": role, "path": relative, "size": (bundle / relative).stat().st_size, "sha256": sha256_file(bundle / relative)}
        for role, relative in REQUIRED_ARTIFACTS
    ]


def bundle_fingerprint(manifest: dict[str, Any]) -> str:
    digest = hashlib.sha256(b"dsasm-bundle-v1\0")
    for value in (manifest["source_fingerprint"], manifest["converter_revision"], manifest["isa_profile"]):
        digest.update(value.encode("utf-8") + b"\0")
    seen_formats: set[str] = set()
    for item in sorted(manifest["formats"], key=lambda value: value["role"]):
        role = item["role"]
        if role in seen_formats:
            raise ProtocolError("bundle.invalid.manifest", f"duplicate format role: {role}")
        seen_formats.add(role)
        digest.update(role.encode("utf-8") + b"\0")
        digest.update(item["version"].encode("utf-8") + b"\0")
    seen_artifacts: set[str] = set()
    for item in sorted(manifest["artifacts"], key=lambda value: value["role"]):
        role = item["role"]
        if role in seen_artifacts:
            raise ProtocolError("bundle.invalid.manifest", f"duplicate artifact role: {role}")
        seen_artifacts.add(role)
        for value in (role, item["path"], str(item["size"]), item["sha256"]):
            digest.update(value.encode("utf-8") + b"\0")
    return digest.hexdigest()


def build_manifest(bundle: Path, sources: list[dict[str, Any]], audio: dict[str, Any]) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "schema_version": MANIFEST_VERSION,
        "converter_revision": CONVERTER_REVISION,
        "provider_build": provider_build_identity(),
        "source_fingerprint": source_fingerprint(sources),
        "source_artifacts": sources,
        "packers": PACKERS,
        "formats": FORMATS,
        "isa_profile": ISA_PROFILE,
        "audio": {
            "sample_rate": audio["sample_rate"], "hop_size": audio["hop_size"], "mel_bins": audio["mel_bins"],
        },
        "product_buckets": [32, 384],
        "artifacts": artifact_records(bundle),
    }
    manifest["bundle_fingerprint"] = bundle_fingerprint(manifest)
    return manifest


def validate_manifest(bundle: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = bundle / "bundle.json"
    if not path.is_file():
        raise ProtocolError("bundle.missing", "bundle.json")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("bundle.invalid.manifest", str(exc)) from exc
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("schema_version") != MANIFEST_VERSION:
        raise ProtocolError("bundle.invalid.manifest", repr((manifest.get("schema"), manifest.get("schema_version"))))
    try:
        for child in bundle.rglob("*"):
            if child.is_symlink() or (not child.is_dir() and not child.is_file()):
                raise ProtocolError("bundle.invalid.path", child.relative_to(bundle).as_posix())
    except OSError as exc:
        raise ProtocolError("io.failed", str(exc)) from exc
    listed = manifest.get("artifacts")
    if not isinstance(listed, list):
        raise ProtocolError("bundle.invalid.manifest", "artifacts must be an array")
    by_role: dict[str, dict[str, Any]] = {}
    paths: set[str] = set()
    for item in listed:
        if not isinstance(item, dict) or not isinstance(item.get("role"), str) or not isinstance(item.get("path"), str):
            raise ProtocolError("bundle.invalid.manifest", "artifact entries require role and path")
        role, relative = item["role"], item["path"]
        pure = PurePosixPath(relative)
        if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts) or "\\" in relative:
            raise ProtocolError("bundle.invalid.path", relative)
        if role in by_role or relative in paths:
            raise ProtocolError("bundle.invalid.path", f"duplicate role or path: {role} {relative}")
        if (not isinstance(item.get("size"), int) or isinstance(item.get("size"), bool) or item["size"] < 0 or
                not isinstance(item.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])):
            raise ProtocolError("bundle.invalid.manifest", f"artifact size/digest: {role}")
        by_role[role] = item
        paths.add(relative)
    required = dict(REQUIRED_ARTIFACTS)
    if set(by_role) != set(required) or any(by_role[role]["path"] != path for role, path in required.items()):
        raise ProtocolError("bundle.incomplete", repr(sorted(by_role)))
    for role, relative in REQUIRED_ARTIFACTS:
        path = bundle / relative
        item = by_role[role]
        if path.is_symlink() or not path.is_file():
            raise ProtocolError("bundle.incomplete", relative)
        if path.stat().st_size != item.get("size") or sha256_file(path) != item.get("sha256"):
            raise ProtocolError("bundle.invalid.digest", relative)
    sources = validate_source_records(manifest.get("source_artifacts"))
    if manifest.get("source_fingerprint") != source_fingerprint(sources):
        raise ProtocolError("bundle.invalid.digest", "source fingerprint")
    actual_fingerprint = bundle_fingerprint(manifest)
    if manifest.get("bundle_fingerprint") != actual_fingerprint:
        raise ProtocolError("bundle.invalid.digest", "bundle fingerprint")
    audio = validate_formats(bundle)
    declared_audio = manifest.get("audio")
    expected_audio = {key: audio[key] for key in ("sample_rate", "hop_size", "mel_bins")}
    if declared_audio != expected_audio:
        raise ProtocolError("bundle.invalid.dimensions", repr(declared_audio))
    if (manifest.get("formats") != FORMATS or manifest.get("isa_profile") != ISA_PROFILE or
            manifest.get("product_buckets") != [32, 384] or manifest.get("packers") != PACKERS or
            manifest.get("converter_revision") != CONVERTER_REVISION):
        raise ProtocolError("bundle.invalid.manifest", "converter, packers, formats, ISA profile, or product buckets")
    return manifest, audio


def bundle_report(bundle: Path | None) -> dict[str, Any]:
    if bundle is None or not bundle.exists():
        return {"bundle_state": "missing", "bundle_fingerprint": None, "bundle_reason_code": "bundle.missing", "bundle_diagnostic": ""}
    try:
        manifest, audio = validate_manifest(bundle)
        return {
            "bundle_state": "ready", "bundle_reason_code": "ok", "bundle_diagnostic": "",
            "bundle_fingerprint": manifest["bundle_fingerprint"], "audio": audio,
        }
    except ProtocolError as exc:
        return {"bundle_state": "invalid", "bundle_fingerprint": None, "bundle_reason_code": exc.code, "bundle_diagnostic": exc.detail}


def plan_payload(singer_root: Path, acoustic: Path, vocoder: Path) -> dict[str, Any]:
    entries = source_entries(singer_root, acoustic, vocoder)
    source_bytes = sum(item["size"] for item in entries)
    basis = max(source_bytes, 1024 * 1024)
    return {
        "artifacts": [{"role": role, "path": path} for role, path in REQUIRED_ARTIFACTS],
        "space": {
            "estimate_kind": "upper_bound",
            "staging_bytes": basis * 3,
            "final_bytes": basis * 2,
            "work_peak_bytes": basis * 5,
        },
        "stages": ["source.inspect", "acoustic.pack", "vocoder.32.pack", "vocoder.384.pack", "bundle.validate", "manifest.write"],
    }


def emit(value: dict[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True), flush=True)


def progress(operation: str, stage: str, state: str, completed: int, total: int) -> dict[str, Any]:
    del state
    return envelope(operation, "ok", "ok", event="progress", stage=stage, progress=completed / total)


def run_packer(command: list[str], stage: str, json_lines: bool, completed: int, total: int) -> None:
    if json_lines:
        emit(progress("convert", stage, "started", completed, total))
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        detail = result.stdout.strip()[-8000:]
        reason_code = "source.unsupported.acoustic_graph" if stage == "acoustic.pack" else "source.unsupported.vocoder_graph"
        raise ProtocolError(reason_code, detail)
    if json_lines:
        emit(progress("convert", stage, "completed", completed + 1, total))


def published_bundle(singer_root: Path) -> tuple[Path | None, ProtocolError | None]:
    dsasm = singer_root / "dsasm"
    pointer = dsasm / "current.json"
    if not pointer.exists():
        return None, None
    try:
        value = json.loads(pointer.read_text(encoding="utf-8"))
        generation = value["generation"]
        if (value.get("schema") != "org.openutau.dsasm-current" or value.get("schema_version") != 1 or
                not isinstance(generation, str)):
            raise ValueError("unsupported current.json schema")
        if not re.fullmatch(r"[0-9a-f]{32}", generation):
            raise ValueError("invalid generation identifier")
        candidate = dsasm / "generations" / generation
        if candidate.is_symlink():
            raise ValueError("generation must not be a symlink")
        return candidate, None
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, ValueError) as exc:
        return None, ProtocolError("bundle.invalid.manifest", f"current.json: {exc}")


def require_protocol(args: argparse.Namespace) -> None:
    if args.protocol != SCHEMA_VERSION:
        raise ProtocolError("protocol.unsupported", str(args.protocol))


def command_inspect(args: argparse.Namespace) -> dict[str, Any]:
    require_protocol(args)
    singer_root = args.singer_root.resolve()
    config = load_singer_config(singer_root)
    acoustic, vocoder = resolve_sources(singer_root, config)
    dimensions, entries = inspect_source(singer_root, config, acoustic, vocoder)
    bundle_path, pointer_error = published_bundle(singer_root)
    bundle = bundle_report(bundle_path)
    if pointer_error:
        bundle = {"bundle_state": "missing", "bundle_fingerprint": None,
                  "bundle_reason_code": "bundle.missing", "bundle_diagnostic": pointer_error.detail}
    elif bundle_path is not None and bundle["bundle_state"] == "ready":
        pointer_path = singer_root / "dsasm" / "current.json"
        if pointer_path.is_file():
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            manifest, _ = validate_manifest(bundle_path)
            if (pointer.get("source_fingerprint") != manifest["source_fingerprint"] or
                    pointer.get("bundle_fingerprint") != manifest["bundle_fingerprint"]):
                bundle = {"bundle_state": "missing", "bundle_fingerprint": None,
                          "bundle_reason_code": "bundle.missing", "bundle_diagnostic": "current.json identity mismatch"}
    return envelope(
        "inspect", "ok", "ok", source_fingerprint=source_fingerprint(entries),
        source_artifacts=entries, model_state="compatible", source=dimensions, **bundle,
    )


def command_plan(args: argparse.Namespace) -> dict[str, Any]:
    require_protocol(args)
    singer_root = args.singer_root.resolve()
    config = load_singer_config(singer_root)
    acoustic, vocoder = resolve_sources(singer_root, config)
    dimensions, entries = inspect_source(singer_root, config, acoustic, vocoder)
    return envelope(
        "plan", "ok", "ok", source_fingerprint=source_fingerprint(entries),
        source_artifacts=entries, model_state="compatible", source=dimensions,
        **plan_payload(singer_root, acoustic, vocoder),
    )


def command_convert(args: argparse.Namespace) -> dict[str, Any]:
    require_protocol(args)
    singer_root = args.singer_root.resolve()
    config = load_singer_config(singer_root)
    acoustic, vocoder = resolve_sources(singer_root, config)
    _, entries = inspect_source(singer_root, config, acoustic, vocoder)
    if not re.fullmatch(r"[0-9a-f]{64}", args.expected_source_fingerprint):
        raise ProtocolError("request.invalid", "expected source fingerprint must be 64 lowercase hexadecimal digits")
    initial_fingerprint = source_fingerprint(entries)
    if initial_fingerprint != args.expected_source_fingerprint:
        raise ProtocolError("source.changed", f"expected {args.expected_source_fingerprint}, got {initial_fingerprint}")
    output, work = args.staging.resolve(), args.work.resolve()
    if output.exists() and any(output.iterdir()):
        raise ProtocolError("request.invalid", f"staging is not empty: {output}")
    acoustic_out, vocoder_out = output / "acoustic", output / "vocoder"
    acoustic_out.mkdir(parents=True, exist_ok=True)
    vocoder_out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parent
    stages = ["source.inspect", "acoustic.pack", "vocoder.32.pack", "vocoder.384.pack", "bundle.validate", "manifest.write"]
    total = 5
    if args.jsonl:
        emit(envelope("convert", "ok", "ok", event="started", source_fingerprint=source_fingerprint(entries), stages=stages))
    run_packer(
        [sys.executable, str(root / "pack_acoustic_onnx_m25.py"), str(acoustic), "--model-dir", str(singer_root), "--out", str(acoustic_out)],
        "acoustic.pack", args.jsonl, 0, total,
    )
    for completed, frames in ((1, 32), (2, 384)):
        run_packer(
            [
                sys.executable, str(root / "pack_vocoder_graph_m35.py"), str(vocoder), "--frames", str(frames),
                "--out", str(vocoder_out / f"{frames}.dsv35"), "--work", str(work / str(frames)),
                "--vnni-scope", "all-k711", "--residual-scope", "all3711",
            ],
            f"vocoder.{frames}.pack", args.jsonl, completed, total,
        )
    if args.jsonl:
        emit(progress("convert", "bundle.validate", "started", 3, total))
    audio = validate_formats(output)
    final_entries = source_entries(singer_root, acoustic, vocoder, config)
    final_fingerprint = source_fingerprint(final_entries)
    if final_fingerprint != initial_fingerprint:
        raise ProtocolError("source.changed", f"source changed during conversion: {final_fingerprint}")
    if args.jsonl:
        emit(progress("convert", "bundle.validate", "completed", 4, total))
        for item in artifact_records(output):
            emit(envelope("convert", "ok", "ok", event="artifact", **item))
        emit(progress("convert", "manifest.write", "started", 4, total))
    manifest = build_manifest(output, final_entries, audio)
    manifest_path = output / "bundle.json"
    temporary = output / ".bundle.json.new"
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n", encoding="ascii")
    temporary.replace(manifest_path)
    validate_manifest(output)
    if args.jsonl:
        emit(progress("convert", "manifest.write", "completed", 5, total))
    return envelope(
        "convert", "ok", "ok", source_fingerprint=final_fingerprint, source_artifacts=final_entries,
        bundle_state="ready",
        bundle_fingerprint=manifest["bundle_fingerprint"], bundle=str(output), artifacts=manifest["artifacts"],
    )


def command_validate(args: argparse.Namespace) -> dict[str, Any]:
    require_protocol(args)
    bundle = args.bundle.resolve()
    manifest, audio = validate_manifest(bundle)
    return envelope(
        "validate", "ok", "ok", bundle=str(bundle), bundle_state="ready",
        source_fingerprint=manifest["source_fingerprint"],
        bundle_fingerprint=manifest["bundle_fingerprint"], artifacts=manifest["artifacts"],
        audio=audio,
    )


class ProtocolArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ProtocolError("request.invalid", message)


def add_common_arguments(parser: argparse.ArgumentParser, output_flag: str) -> None:
    parser.add_argument("--protocol", type=int, required=True)
    parser.add_argument(output_flag, action="store_true", required=True)


def add_source_arguments(parser: argparse.ArgumentParser, output_flag: str = "--json") -> None:
    add_common_arguments(parser, output_flag)
    parser.add_argument("--singer-root", type=Path, required=True)


def make_parser() -> argparse.ArgumentParser:
    parser = ProtocolArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    inspect_parser = subparsers.add_parser("inspect", help="Inspect source compatibility and optional bundle state")
    add_source_arguments(inspect_parser)
    plan_parser = subparsers.add_parser("plan", help="Return output files and conservative space estimates")
    add_source_arguments(plan_parser)
    convert_parser = subparsers.add_parser("convert", help="Convert into a caller-owned staging directory")
    add_source_arguments(convert_parser, "--jsonl")
    convert_parser.add_argument("--expected-source-fingerprint", required=True)
    convert_parser.add_argument("--staging", type=Path, required=True)
    convert_parser.add_argument("--work", type=Path, required=True)
    validate_parser = subparsers.add_parser("validate", help="Validate bundle without loading the native runtime")
    add_common_arguments(validate_parser, "--json")
    validate_parser.add_argument("--bundle", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    operation = raw[0] if raw and raw[0] in ("inspect", "plan", "convert", "validate") else "unknown"
    try:
        args = make_parser().parse_args(raw)
        operation = args.operation
        result = {
            "inspect": command_inspect,
            "plan": command_plan,
            "convert": command_convert,
            "validate": command_validate,
        }[args.operation](args)
    except ProtocolError as exc:
        if operation == "convert" and "--staging" in raw:
            index = raw.index("--staging") + 1
            if index < len(raw):
                (Path(raw[index]).resolve() / "bundle.json").unlink(missing_ok=True)
        status = status_for(exc.code)
        payload: dict[str, Any] = {}
        if operation == "inspect":
            payload = {
                "model_state": "incompatible" if status == "incompatible" else "unknown",
                "bundle_state": "missing", "source_fingerprint": None,
                "bundle_fingerprint": None, "source_artifacts": [], "source": None,
            }
        result = envelope(operation, status, exc.code, exc.detail, **payload)
    except KeyboardInterrupt:
        if operation == "convert":
            staging_index = raw.index("--staging") + 1 if "--staging" in raw and raw.index("--staging") + 1 < len(raw) else -1
            if staging_index > 0:
                (Path(raw[staging_index]).resolve() / "bundle.json").unlink(missing_ok=True)
        result = envelope(operation, "cancelled", "cancelled")
    except Exception as exc:
        if operation == "convert" and "--staging" in raw:
            index = raw.index("--staging") + 1
            if index < len(raw):
                (Path(raw[index]).resolve() / "bundle.json").unlink(missing_ok=True)
        result = envelope(operation, "error", "internal.error", f"{type(exc).__name__}: {exc}")
    emit(result)
    return exit_for(result["status"], result["reason_code"])


if __name__ == "__main__":
    raise SystemExit(main())
