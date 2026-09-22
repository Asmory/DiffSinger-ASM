#!/usr/bin/env python3

import json
import io
import os
import struct
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import dsasm_model_tool as tool  # noqa: E402


def write_fixture_bundle(root: Path) -> dict:
    acoustic = root / "acoustic"
    vocoder = root / "vocoder"
    acoustic.mkdir(parents=True)
    vocoder.mkdir(parents=True)

    def blob(header, counts):
        value = bytearray(header)
        for count in counts:
            value.extend(bytes((-len(value)) % 64))
            value.extend(bytes(count * 4))
        return bytes(value)

    vocab, hidden, layers = 100, 16, 2
    fs_counts = [vocab * hidden, hidden, hidden]
    for _ in range(layers):
        fs_counts.extend([
            hidden, hidden, 3 * hidden * hidden, 3 * hidden,
            hidden * hidden, hidden, hidden, hidden,
            12 * hidden * hidden, 4 * hidden, 4 * hidden * hidden, hidden,
        ])
    fs_counts.extend([
        hidden, hidden,
        4 * hidden * hidden, 4 * hidden, 4 * hidden * hidden, hidden,
        3 * hidden * hidden, 3 * hidden, 3 * hidden * hidden, 3 * hidden,
        hidden, hidden, 7,
    ])
    fs = blob(
        struct.pack("<8s9I4fI", b"DSFS25\0\0", 1, vocab, hidden, layers, 4, 3, 0, 0, 0,
                    10000.0, 1.0, 1.0, 1.0, 0),
        fs_counts,
    )

    aux_input, aux_channels, aux_output, aux_layers = 16, 16, 128, 2
    aux_counts = [aux_channels * 7 * aux_input, aux_channels]
    for _ in range(aux_layers):
        aux_counts.extend([
            7 * aux_channels, aux_channels, aux_channels, aux_channels,
            4 * aux_channels * aux_channels, 4 * aux_channels,
            4 * aux_channels * aux_channels, aux_channels, aux_channels,
        ])
    aux_counts.extend([aux_output * 7 * aux_channels, aux_output])
    aux = blob(
        struct.pack("<8s8I24x", b"DSAUX20\0", 1, aux_input, aux_channels, aux_output,
                    aux_layers, 7, 4, 16),
        aux_counts,
    )

    rf_input, rf_condition, rf_channels, rf_hidden, rf_layers = 128, 16, 16, 16, 2
    reflow_counts = [
        rf_channels * rf_input, rf_channels, rf_channels * rf_condition, rf_channels,
        4 * rf_channels * rf_channels, 4 * rf_channels,
        4 * rf_channels * rf_channels, rf_channels,
    ]
    for _ in range(rf_layers):
        reflow_counts.extend([
            rf_channels, rf_channels, 31 * rf_channels, rf_channels,
            2 * rf_hidden * rf_channels, 2 * rf_hidden,
            2 * rf_hidden * rf_hidden, 2 * rf_hidden,
            rf_channels * rf_hidden, rf_channels,
        ])
    reflow_counts.extend([rf_channels, rf_channels, rf_input * rf_channels, rf_input])
    reflow = blob(
        struct.pack("<8s12I8x", b"DSLYNX7\0", 1, rf_input, rf_condition, rf_channels,
                    rf_hidden, rf_layers, 31, 1, 4, 16, 4, 8),
        reflow_counts,
    )
    (acoustic / "fs2_acoustic.dsfs").write_bytes(fs)
    (acoustic / "aux_convnext.dsa").write_bytes(aux)
    (acoustic / "lynxnet2.dsn").write_bytes(reflow)
    (acoustic / "model.conf").write_text("mel_bins=128\nsample_rate=44100\nhop_size=512\n", encoding="ascii")

    for frames in (32, 384):
        header = struct.pack(
            "<8s7I8Q28x", b"DSVOC35\0", 1, 1, 1, 0, 0, 0, 0,
            16, 0, 128, 208, 448, frames, 128, frames * 512,
        )
        (vocoder / f"{frames}.dsv35").write_bytes(header + bytes(320))

    sources = [{"role": "acoustic.onnx", "path": "acoustic.onnx", "size": 3, "sha256": "0" * 64}]
    audio = tool.validate_formats(root)
    manifest = tool.build_manifest(root, sources, audio)
    (root / "bundle.json").write_text(json.dumps(manifest, sort_keys=True), encoding="ascii")
    return manifest


class ModelToolTests(unittest.TestCase):
    def setUp(self):
        self.identity = mock.patch.dict(os.environ, {"DSASM_PROVIDER_BUILD_ID": "test-build"})
        self.identity.start()

    def tearDown(self):
        self.identity.stop()

    def test_source_fingerprint_is_path_independent(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            fingerprints = []
            for directory in (Path(first), Path(second)):
                source = directory / "acoustic.onnx"
                source.write_bytes(b"acoustic")
                entries = [{"role": "acoustic.onnx", "path": "acoustic.onnx",
                            "size": source.stat().st_size, "sha256": tool.sha256_file(source)}]
                fingerprints.append(tool.source_fingerprint(entries))
            self.assertEqual(fingerprints[0], fingerprints[1])

    def test_valid_manifest_and_fingerprint_are_deterministic(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_manifest = write_fixture_bundle(Path(first))
            second_manifest = write_fixture_bundle(Path(second))
            self.assertEqual(first_manifest["bundle_fingerprint"], second_manifest["bundle_fingerprint"])
            checked, audio = tool.validate_manifest(Path(first))
            self.assertEqual(checked["bundle_fingerprint"], first_manifest["bundle_fingerprint"])
            self.assertEqual(audio["required_buckets"], [32, 384])

    def test_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture_bundle(root)
            with (root / "acoustic" / "model.conf").open("ab") as stream:
                stream.write(b"# changed\n")
            with self.assertRaises(tool.ProtocolError) as raised:
                tool.validate_manifest(root)
            self.assertEqual(raised.exception.code, "bundle.invalid.digest")

    def test_manifest_cannot_bless_truncated_packed_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = write_fixture_bundle(root)
            packed = root / "acoustic" / "fs2_acoustic.dsfs"
            packed.write_bytes(packed.read_bytes()[:-4])
            artifact = next(item for item in manifest["artifacts"] if item["role"] == "acoustic.fs2")
            artifact["size"] = packed.stat().st_size
            artifact["sha256"] = tool.sha256_file(packed)
            manifest["bundle_fingerprint"] = tool.bundle_fingerprint(manifest)
            (root / "bundle.json").write_text(json.dumps(manifest, sort_keys=True), encoding="ascii")
            with self.assertRaises(tool.ProtocolError) as raised:
                tool.validate_manifest(root)
            self.assertEqual(raised.exception.code, "bundle.invalid.format")

    def test_missing_product_bucket_has_stable_reason(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture_bundle(root)
            (root / "vocoder" / "384.dsv35").unlink()
            report = tool.bundle_report(root)
            self.assertEqual(report["bundle_state"], "invalid")
            self.assertEqual(report["bundle_reason_code"], "bundle.incomplete")

    def test_validator_does_not_load_native_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = write_fixture_bundle(root)
            with mock.patch("ctypes.CDLL", side_effect=AssertionError("native runtime loaded")):
                checked, _ = tool.validate_manifest(root)
            self.assertEqual(checked["bundle_fingerprint"], manifest["bundle_fingerprint"])

    def test_plan_has_required_outputs_and_nonzero_estimates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "dsvocoder").mkdir()
            acoustic = root / "acoustic.onnx"
            vocoder = root / "dsvocoder" / "nsf_hifigan.onnx"
            acoustic.write_bytes(b"a")
            vocoder.write_bytes(b"v")
            entries = [
                {"role": "acoustic.onnx", "path": "acoustic.onnx", "size": 1, "sha256": tool.sha256_file(acoustic)},
                {"role": "vocoder.onnx", "path": "dsvocoder/nsf_hifigan.onnx", "size": 1, "sha256": tool.sha256_file(vocoder)},
            ]
            with mock.patch.object(tool, "source_entries", return_value=entries):
                payload = tool.plan_payload(root, acoustic, vocoder)
            self.assertEqual(payload["artifacts"], [
                {"role": role, "path": path} for role, path in tool.REQUIRED_ARTIFACTS
            ])
            self.assertTrue(all(payload["space"][key] > 0 for key in ("staging_bytes", "final_bytes", "work_peak_bytes")))

    def test_envelope_is_versioned(self):
        value = tool.envelope("validate", "ok", "ok")
        self.assertEqual(value["schema"], tool.SCHEMA)
        self.assertEqual(value["schema_version"], 1)
        self.assertEqual(value["converter_revision"], "m25-m35-v1")
        self.assertEqual(value["event"], "result")
        self.assertEqual(value["reason_code"], "ok")

    def test_fingerprint_vectors(self):
        vectors = json.loads((ROOT / "tests" / "fixtures" / "model_protocol_v1_fingerprints.json").read_text())
        self.assertEqual(tool.source_fingerprint(vectors["source"]["artifacts"]), vectors["source"]["fingerprint"])
        bundle = vectors["bundle"]
        manifest = {
            "source_fingerprint": bundle["source_fingerprint"],
            "converter_revision": bundle["converter_revision"],
            "isa_profile": bundle["isa_profile"],
            "formats": bundle["formats"],
            "artifacts": bundle["artifacts"],
        }
        self.assertEqual(tool.bundle_fingerprint(manifest), bundle["fingerprint"])

    def test_manifest_rejects_symlink_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture_bundle(root)
            target = root / "acoustic" / "model.conf"
            saved = root / "saved.conf"
            target.replace(saved)
            target.symlink_to(saved)
            with self.assertRaises(tool.ProtocolError) as raised:
                tool.validate_manifest(root)
            self.assertEqual(raised.exception.code, "bundle.invalid.path")

    def test_source_path_cannot_escape_singer_root(self):
        with tempfile.TemporaryDirectory() as singer, tempfile.TemporaryDirectory() as outside:
            singer_root = Path(singer)
            external = Path(outside) / "model.onnx"
            external.write_bytes(b"model")
            (singer_root / "model.onnx").symlink_to(external)
            with self.assertRaises(tool.ProtocolError) as raised:
                tool.contained_source(singer_root, "model.onnx", "acoustic")
            self.assertEqual(raised.exception.code, "source.invalid_path")

    def test_missing_source_has_stable_reason(self):
        with tempfile.TemporaryDirectory() as singer:
            singer_root = Path(singer)
            with self.assertRaises(tool.ProtocolError) as raised:
                tool.contained_source(singer_root, "acoustic.onnx", "acoustic")
            self.assertEqual(raised.exception.code, "source.missing")

    def test_unsafe_configured_vocoder_does_not_fall_back(self):
        for configured in ("/outside/model", "../outside/model"):
            with self.subTest(configured=configured), tempfile.TemporaryDirectory() as singer:
                singer_root = Path(singer)
                (singer_root / "acoustic.onnx").write_bytes(b"acoustic")
                fallback = singer_root / "dsvocoder" / "nsf_hifigan.onnx"
                fallback.parent.mkdir()
                fallback.write_bytes(b"vocoder")
                with self.assertRaises(tool.ProtocolError) as raised:
                    tool.resolve_sources(singer_root, {"acoustic": "acoustic.onnx", "vocoder": configured})
                self.assertEqual(raised.exception.code, "source.invalid_path")

    def test_external_data_path_escape_has_stable_reason(self):
        import onnx
        from onnx import TensorProto, helper

        with tempfile.TemporaryDirectory() as singer:
            singer_root = Path(singer)
            tensor = TensorProto()
            tensor.name = "weight"
            tensor.data_type = TensorProto.FLOAT
            tensor.dims.append(1)
            tensor.data_location = TensorProto.EXTERNAL
            location = tensor.external_data.add()
            location.key = "location"
            location.value = "../outside.bin"
            graph = helper.make_graph([], "external", [], [], [tensor])
            model_path = singer_root / "model.onnx"
            model_path.write_bytes(helper.make_model(graph).SerializeToString())
            with mock.patch.dict(sys.modules, {"onnxruntime": mock.Mock()}):
                with self.assertRaises(tool.ProtocolError) as raised:
                    tool.onnx_external_sources(singer_root, model_path, "acoustic")
            self.assertEqual(raised.exception.code, "source.invalid_path")

    def test_invalid_inspect_has_complete_result_shape(self):
        with tempfile.TemporaryDirectory() as singer:
            singer_root = Path(singer)
            (singer_root / "dsconfig.yaml").write_text("acoustic: missing.onnx\n", encoding="ascii")
            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = tool.main([
                    "inspect", "--protocol", "1", "--singer-root", str(singer_root), "--json",
                ])
            result = json.loads(output.getvalue())
            self.assertEqual(exit_code, 3)
            self.assertEqual(result["model_state"], "incompatible")
            self.assertEqual(result["bundle_state"], "missing")
            self.assertIsNone(result["source_fingerprint"])
            self.assertIsNone(result["bundle_fingerprint"])
            self.assertEqual(result["reason_code"], "source.missing")

    def test_bundle_without_current_pointer_is_not_published(self):
        with tempfile.TemporaryDirectory() as singer:
            singer_root = Path(singer)
            direct = singer_root / "dsasm"
            direct.mkdir()
            (direct / "bundle.json").write_text("{}", encoding="ascii")
            bundle, error = tool.published_bundle(singer_root)
            self.assertIsNone(bundle)
            self.assertIsNone(error)

    def test_manifest_rejects_traversal_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = write_fixture_bundle(root)
            manifest["artifacts"][0]["path"] = "../model.conf"
            (root / "bundle.json").write_text(json.dumps(manifest), encoding="ascii")
            with self.assertRaises(tool.ProtocolError) as raised:
                tool.validate_manifest(root)
            self.assertEqual(raised.exception.code, "bundle.invalid.path")

    def test_exit_code_mapping(self):
        self.assertEqual(tool.exit_for("ok", "ok"), 0)
        self.assertEqual(tool.exit_for("error", "request.invalid"), 2)
        self.assertEqual(tool.exit_for("incompatible", "source.unsupported.acoustic_graph"), 3)
        self.assertEqual(tool.exit_for("invalid", "bundle.invalid.digest"), 4)
        self.assertEqual(tool.exit_for("error", "toolchain.missing_dependency"), 5)
        self.assertEqual(tool.exit_for("error", "io.failed"), 6)
        self.assertEqual(tool.exit_for("cancelled", "cancelled"), 7)
        self.assertEqual(tool.exit_for("error", "internal.error"), 8)
        self.assertEqual(tool.exit_for("error", "source.changed"), 9)


if __name__ == "__main__":
    unittest.main()
