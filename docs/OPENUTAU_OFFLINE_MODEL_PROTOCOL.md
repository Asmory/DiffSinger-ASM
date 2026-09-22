# DiffSinger-ASM Offline Model Protocol for OpenUtau

Status: agreed implementation draft; not a formal release contract
Protocol version: 1
Date: 2026-09-22

This document is the single shared specification for the offline model boundary
between DiffSinger-ASM and OpenUtau. DiffSinger-ASM owns the tool and model
semantics. OpenUtau owns orchestration, UI, publication, and recovery.

`MUST`, `MUST NOT`, `SHOULD`, and `MAY` are normative within this draft.

## 1. Scope

The protocol supplies four operations:

```text
inspect -> plan -> convert -> validate
```

It covers source compatibility, conversion planning, staging conversion,
prepared-bundle validation, stable reason codes, progress, and authoritative
model identity. It does not change native engine ABI v3.

The tool never selects an OpenUtau renderer, publishes a singer generation,
updates `current.json`, writes canonical PCM, or starts a conversion during an
`inspect` operation.

## 2. Packaged Command

The provider package MUST include one independently callable command named
`dsasm-model-tool`. It MUST NOT require a DiffSinger-ASM source checkout or a
hidden repository `.venv` layout. Its runtime dependencies MUST be contained in
or declared by the provider package.

The production Linux x86-64 package layout is fixed as:

```text
<package>/bin/dsasm-model-tool
<package>/lib/libdsasm.so
<package>/share/dsasm-model-tool/...
```

OpenUtau discovers the production tool relative to the selected provider
package root and MUST NOT search `PATH`. A developer override, when exposed,
must be an explicit absolute executable path. The provider package MUST include
either a self-contained tool runtime or CPython 3.12 plus an exact dependency
lock with hashes. An unpinned requirements file is not a product package.

Version 1 commands are:

```text
dsasm-model-tool inspect  --protocol 1 --singer-root PATH --json
dsasm-model-tool plan     --protocol 1 --singer-root PATH --json
dsasm-model-tool convert  --protocol 1 --singer-root PATH \
  --expected-source-fingerprint SHA256 --staging PATH --work PATH --jsonl
dsasm-model-tool validate --protocol 1 --bundle PATH --json
```

`--singer-root`, `--staging`, `--work`, and `--bundle` are passed as separate
process arguments. Paths may contain spaces and MUST NOT be interpreted by a
shell. `inspect`, `plan`, and `validate` emit exactly one JSON object to stdout.
`convert` emits JSON Lines, one complete JSON object per line. Human diagnostics
go to stderr and are never parsed for product decisions.

## 3. Common Envelope

Every JSON object contains:

```json
{
  "schema": "org.openutau.dsasm-model-tool",
  "schema_version": 1,
  "operation": "inspect",
  "event": "result",
  "status": "ok",
  "reason_code": "ok",
  "diagnostic": "",
  "converter_revision": "m25-m35-v1",
  "provider_build": "build-id"
}
```

Required common fields are:

| Field | Rule |
| --- | --- |
| `schema` | Exact value `org.openutau.dsasm-model-tool` |
| `schema_version` | Integer `1` |
| `operation` | `inspect`, `plan`, `convert`, or `validate` |
| `event` | `result`, or a convert event from section 7 |
| `status` | `ok`, `incompatible`, `invalid`, `cancelled`, or `error` |
| `reason_code` | Stable machine-readable code from section 9 |
| `diagnostic` | UTF-8 developer detail; not localized and not parsed |
| `converter_revision` | Stable revision of compatibility and packing behavior |
| `provider_build` | Provider build identity for diagnostics and capability invalidation |

Consumers MUST reject an unsupported `schema_version`, missing required field,
wrong field type, malformed JSON, mismatched operation, or a `result` that does
not agree with the process exit code. Consumers MUST ignore unknown fields in a
supported version.

`converter_revision` is an opaque nonempty ASCII identifier matching
`[A-Za-z0-9._-]+`. It MUST change when compatibility decisions, default packing
semantics, required artifacts, packed bytes for unchanged source input, or the
default ISA profile can change. It need not change for diagnostics, progress
wording, packaging, or performance-only changes that preserve all generated
bytes. Consumers compare it for exact equality and do not impose ordering.

## 4. Inspect

`inspect` is read-only. It MUST NOT create work directories, convert files,
download dependencies, or modify the singer. It reports source compatibility
and any currently published prepared bundle visible below the singer root.

The result adds:

```json
{
  "model_state": "compatible",
  "bundle_state": "missing",
  "source_fingerprint": "sha256-lowercase-hex-or-null",
  "bundle_fingerprint": null,
  "source": {
    "sample_rate": 44100,
    "hop_size": 512,
    "mel_bins": 128
  }
}
```

`model_state` is `unknown`, `compatible`, or `incompatible`. `bundle_state` is
`missing`, `invalid`, or `ready`. Runtime availability, tool discovery,
conversion activity, allowed actions, and localized messages are deliberately
absent; OpenUtau Core combines those dimensions into its `CapabilityReport`.

A missing or invalid bundle MUST NOT change a compatible source model into an
incompatible model. An unavailable native runtime MUST NOT affect `inspect`.

`compatible` means the source passed the complete read-only preflight used by
the production packers: source closure, configuration, graph structure,
required initializers, supported features and operations, dimensions, product
buckets, and cross-file constraints. A shallow ONNX parse or checker-only pass
is insufficient. Compatibility logic MUST be shared with conversion so a known
unsupported topology is reported by `inspect`, not first discovered by
`convert`.

The result also returns `source_artifacts`, with `role`, singer-root-relative
`path`, `size`, and `sha256` for every item in the source closure. OpenUtau may
persist this list with PCM provenance and recompute the source fingerprint
without the provider tool.

## 5. Plan

`plan` performs no conversion and no singer mutation. For a compatible source
it returns the same source identity as `inspect` plus:

```json
{
  "artifacts": [
    {"role": "acoustic.fs2", "path": "acoustic/fs2_acoustic.dsfs"},
    {"role": "vocoder.32", "path": "vocoder/32.dsv35"},
    {"role": "vocoder.384", "path": "vocoder/384.dsv35"}
  ],
  "space": {
    "estimate_kind": "upper_bound",
    "staging_bytes": 0,
    "final_bytes": 0,
    "work_peak_bytes": 0
  },
  "stages": [
    "source.inspect",
    "acoustic.pack",
    "vocoder.32.pack",
    "vocoder.384.pack",
    "bundle.validate",
    "manifest.write"
  ]
}
```

Artifact paths are POSIX-style paths relative to the bundle root. They MUST be
unique, MUST NOT be absolute, and MUST NOT contain `.` or `..` segments.

All byte counts are non-negative integers. `staging_bytes` is the peak space
for all output and temporary files under `--staging`; `final_bytes` is the
complete committed generation size; `work_peak_bytes` is the simultaneous peak
for all acoustic and both vocoder conversion children under `--work`.
Source/external-data read size is not counted as new free-space consumption.
Product `plan` MUST return `estimate_kind` as `exact` or `upper_bound`; an
unbounded estimate is not sufficient to enable conversion.

OpenUtau places staging on the singer generation filesystem and requires free
space of at least `staging_bytes + max(512 MiB, ceil(staging_bytes * 0.10))`.
It checks the work filesystem independently using the same margin over
`work_peak_bytes`. The retained previous generation already occupies space and
is not added again. OpenUtau owns these checks; the provider owns estimate
completeness.

Stage identifiers are stable machine values. New stages may be added without a
schema-major change; OpenUtau MUST display unknown stages using a generic
localized conversion message.

## 6. Convert

`convert` writes only below caller-provided `--staging` and `--work` paths. The
staging path MUST be absent or empty. The tool MUST NOT read or modify an
OpenUtau `current.json`, an existing committed generation, or PCM cache.

The provider MUST generate both product vocoder buckets, 32 and 384 frames.
`bundle.json` is written at the `--staging` root last, only after all artifacts
have been packed
and offline validation has succeeded. Its presence therefore marks a complete
staging result, but publication still belongs to OpenUtau.

Before writing work, `convert` MUST recompute the source fingerprint and compare
it with `--expected-source-fingerprint`. It MUST recompute it again after all
packing and before validation/manifest creation. Any mismatch returns
`source.changed`, writes no valid manifest, and cannot be committed. The final
result repeats the actual source fingerprint. OpenUtau compares it with the
plan and rechecks the current source closure before replacing `current.json`.

On failure or cancellation the tool MUST NOT leave a valid `bundle.json` in
staging. OpenUtau owns deletion of staging and work paths after the process
returns. The tool MUST handle SIGINT as a cancellation request and SHOULD emit a
cancelled result promptly at a stage boundary. OpenUtau waits 5 seconds after
SIGINT and then terminates the entire process tree. Forced termination, EOF
without a final result, or an exit/result mismatch is a tool failure; OpenUtau
synthesizes `tool.cancel_timeout` or `tool.result_missing` for diagnostics and
never publishes the staging directory.

## 7. Convert Events

Before the final result, `convert` may emit these JSONL events:

```json
{"schema":"org.openutau.dsasm-model-tool","schema_version":1,
 "operation":"convert","event":"progress","stage":"acoustic.pack",
 "progress":0.35,"status":"ok","reason_code":"ok","diagnostic":"",
 "converter_revision":"m25-m35-v1","provider_build":"build-id"}
```

| Event | Additional fields |
| --- | --- |
| `started` | `source_fingerprint`, `stages` |
| `progress` | `stage`, `progress` |
| `artifact` | `role`, `path`, `size`, `sha256` |
| `result` | final status and identities |

`progress` is finite, between 0 and 1 inclusive, and MUST NOT decrease. Artifact
paths follow section 5. Exactly one `result` MUST be the final stdout object.
EOF before a valid result is a tool failure. Progress events are advisory;
OpenUtau determines success only from the final result, exit code, and later
validation.

## 8. Validate

`validate` is read-only and accepts any bundle directory, including staging.
It MUST NOT require the machine to satisfy native product runtime OS, CPU, ISA,
affinity, or worker gates. It validates:

- required 32- and 384-frame product artifacts;
- packed headers and format versions;
- audio dimensions and cross-file dimensions;
- manifest paths, sizes, and SHA-256 digests;
- source, converter, packer, ISA, and bundle identity fields;
- the recomputed authoritative bundle fingerprint.

On success it returns `bundle_state: "ready"`, the source fingerprint, and the
bundle fingerprint. Native `dsasm_engine_create_mode()` remains a separate
target-runtime check that OpenUtau performs for both modes when runtime support
is available.

## 9. Stable Reason Codes

Version 1 reserves these codes:

```text
ok
request.invalid
protocol.unsupported
source.missing
source.invalid_config
source.invalid_path
source.changed
source.unsupported.sample_rate
source.unsupported.hop_size
source.unsupported.mel_bins
source.unsupported.energy_embedding
source.unsupported.acoustic_graph
source.unsupported.vocoder_graph
bundle.missing
bundle.incomplete
bundle.invalid.manifest
bundle.invalid.path
bundle.invalid.digest
bundle.invalid.format
bundle.invalid.dimensions
space.insufficient
io.denied
io.failed
toolchain.missing_dependency
tool.cancel_timeout
tool.result_missing
cancelled
internal.error
```

The provider MAY add codes within these namespaces. Existing code meanings MUST
NOT change within protocol version 1. OpenUtau maps known codes to localized
messages and uses namespace-level fallback for unknown codes. `diagnostic` may
change and never controls actions.

## 10. Exit Codes

| Exit | Meaning |
| ---: | --- |
| 0 | `ok` result |
| 2 | invalid request or unsupported protocol |
| 3 | source incompatible |
| 4 | bundle missing or invalid |
| 5 | toolchain or dependency unavailable |
| 6 | I/O or insufficient space |
| 7 | cancelled |
| 8 | internal tool failure |
| 9 | source changed since inspect/plan; caller must re-inspect and re-plan |

For every normal outcome, the final JSON status, reason code, and exit code
MUST agree. A crash, signal termination, missing final result, or disagreement
is reported by OpenUtau as a tool failure.

`source.changed` uses status `error` and exit 9. It MUST NOT be mapped to model
incompatibility. `tool.cancel_timeout` and `tool.result_missing` are synthesized
by OpenUtau when no trustworthy provider result exists and therefore have no
provider process exit code.

## 11. Fingerprints

SHA-256 values are lowercase hexadecimal. Source files are assigned stable
logical roles rather than absolute paths. The source fingerprint is SHA-256 of
this byte sequence:

```text
"dsasm-source-v1\0"
for each source artifact sorted by role:
  role UTF-8 + "\0" + decimal size ASCII + "\0" + sha256 ASCII + "\0"
```

The bundle fingerprint is SHA-256 of:

```text
"dsasm-bundle-v1\0"
source_fingerprint + "\0"
converter_revision + "\0"
ISA profile + "\0"
for each packed format sorted by role:
  role + "\0" + format version + "\0"
for each required artifact sorted by role:
  role + "\0" + relative path + "\0" + decimal size + "\0" + sha256 + "\0"
```

All literal separators above are single zero bytes. Roles and paths are UTF-8.
Integers have no sign or leading zeroes. Duplicate roles are invalid. Absolute
source paths, timestamps, diagnostics, progress, and `provider_build` do not
participate in either fingerprint.

### 11.1 Source Closure

The source closure contains every file whose bytes can affect compatibility or
generated artifacts. Version 1 defines these stable roles:

```text
config.dsconfig                         dsconfig.yaml
config.vocoder                          dsvocoder/vocoder.yaml, when read
acoustic.onnx                           configured acoustic ONNX
acoustic.external/<normalized-path>     every acoustic ONNX external-data file
vocoder.onnx                            configured vocoder ONNX
vocoder.external/<normalized-path>      every vocoder ONNX external-data file
tokens.phonemes                         phonemes.json, when read
tokens.languages                        languages.json, when read
speaker.embedding/<normalized-path>     every speaker embedding read or copied
provider.input/<normalized-path>        any other file read by a production packer
```

Metadata that cannot affect compatibility or generated bytes, such as display
name, portrait, and free-form character text, MUST NOT enter the source
fingerprint. If a future packer starts reading another file, it must add that
file through `provider.input/` and bump `converter_revision`.

Artifact `path` values are normalized POSIX paths relative to the singer root.
The tool may follow a source symlink only when its fully resolved regular-file
target remains inside the fully resolved singer root. Absolute references,
cycles, broken links, and any resolved escape are `source.invalid_path`. ONNX
external-data locations are resolved relative to their ONNX file, must be
relative, and obey the same containment rule. Their normalized locations form
the suffix of the corresponding `*.external/` role.

The provider tool returns the complete closure in `source_artifacts`. OpenUtau
MUST implement the fingerprint algorithm over a previously stored closure so
canonical PCM lookup does not require the toolchain. It validates paths with
the same containment rules, rehashes every listed file, and treats a missing or
changed item as a cache miss.

## 12. Authoritative Bundle Manifest

The completed staging bundle contains `bundle.json` at the bundle root. It
includes:

```json
{
  "schema": "org.openutau.dsasm-bundle",
  "schema_version": 1,
  "source_fingerprint": "...",
  "bundle_fingerprint": "...",
  "converter_revision": "m25-m35-v1",
  "provider_build": "build-id",
  "isa_profile": "avx2-fma",
  "audio": {"sample_rate": 44100, "hop_size": 512, "mel_bins": 128},
  "product_buckets": [32, 384],
  "formats": [
    {"role": "acoustic.fs2", "version": "DSFS25/1"},
    {"role": "vocoder.32", "version": "DSVOC35/1"}
  ],
  "source_artifacts": [
    {"role": "acoustic.onnx", "path": "acoustic.onnx",
     "size": 0, "sha256": "..."}
  ],
  "artifacts": [
    {"role": "acoustic.fs2", "path": "acoustic/fs2_acoustic.dsfs",
     "size": 0, "sha256": "..."}
  ]
}
```

The manifest itself is not listed as an artifact and does not hash itself.
Every required packed file is listed exactly once. Validators MUST reject
absolute paths, traversal, symlinks, duplicate paths or roles, unexpected file
types, size mismatches, digest mismatches, and fingerprint mismatches.

## 13. OpenUtau Aggregation and Publication

OpenUtau combines provider results with these local dimensions:

```text
runtime:   available | unavailable
toolchain: available | unavailable
activity:  idle | detecting | converting
actions:   consumer-owned allowed actions
```

Cache lookup occurs before runtime probe or engine creation. PCM provenance
stores the source and bundle fingerprints. A deleted prepared bundle does not
invalidate matching PCM when the current source fingerprint still matches and
no published bundle contradicts the stored bundle fingerprint.

PCM provenance also stores the version-1 `source_artifacts` closure. OpenUtau
uses that stored closure to perform the cache-first source check when the
provider tool or runtime is unavailable.

OpenUtau publication is:

```text
plan -> staging convert -> offline validate -> acquire singer maintenance lane
-> wait for active inference -> release old engines -> atomically replace
current.json -> release lane
```

Generation directories are immutable after publication. `current.json` is the
single visibility pointer and is atomically replaced on the singer filesystem.
The commit critical section ignores cancellation. Failure before pointer
replacement preserves the previous generation. New requests use the new
generation; an active playback session never changes model or PCM midway.

The consumer-owned published layout is:

```text
<singer>/dsasm/current.json
<singer>/dsasm/generations/<generation>/bundle.json
<singer>/dsasm/generations/<generation>/acoustic/...
<singer>/dsasm/generations/<generation>/vocoder/32.dsv35
<singer>/dsasm/generations/<generation>/vocoder/384.dsv35
```

`generation` is a consumer-generated lowercase 32-digit hexadecimal identifier.
`current.json` contains exactly one visible generation selection:

```json
{
  "schema": "org.openutau.dsasm-current",
  "schema_version": 1,
  "generation": "0123456789abcdef0123456789abcdef",
  "source_fingerprint": "...",
  "bundle_fingerprint": "..."
}
```

Before pointer replacement, OpenUtau moves the validated staging directory to
the final immutable generation path on the same filesystem. It writes the new
pointer to a sibling temporary file and atomically replaces `current.json`.
Readers snapshot `current.json` once and resolve only that generation. A pointer
with an invalid identifier, missing generation, or mismatched manifest identity
is treated as no published bundle. Unreferenced staging or generation paths may
be reclaimed only after active engine and conversion ownership has ended.

OpenUtau owns per-singer single-flight and MUST NOT run two
inspect/plan/convert or publication mutations concurrently for the same singer.
The provider tool does not lock singer state. Read-only work for a different
singer and the globally serial first batch-conversion policy remain consumer
scheduling decisions.

## 14. Compatibility and Acceptance

Protocol version 1 acceptance requires:

- deterministic inspect and plan results for unchanged inputs;
- tests for every stable reason-code family and exit-code mapping;
- cancellation with no valid staging manifest;
- validation on a machine that fails native runtime support;
- manifest path traversal, symlink, truncation, digest, and format tamper tests;
- deterministic source and bundle fingerprint vectors shared by provider and
  consumer tests;
- conversion of a supported real singer into both product buckets;
- natural cache miss after a different bundle fingerprint is committed;
- recovery from orphan staging and an interrupted pre-commit conversion.

Schema version 1 remains an implementation draft until both repositories pass
these tests. Realtime canonical compatibility and long-phrase cancellation are
separate release gates and do not block implementation of this offline
protocol.

The normative aggregate fingerprint vector is stored at
`tests/fixtures/model_protocol_v1_fingerprints.json` in the provider repository.
Provider and consumer tests MUST load the same values or an exact vendored copy;
recomputing a different expected value inside each implementation does not
satisfy the cross-language gate.

## 15. Next Implementation Order

1. Provider and consumer land the shared fingerprint vector tests and source
   closure/path-validation tests.
2. Provider completes packaged inspect and plan, including full shared
   preflight, bounded space estimates, and deterministic identities.
3. Provider completes convert and validate, cancellation, manifest-last
   behavior, tamper tests, and package smoke tests outside the source checkout.
4. OpenUtau lands protocol DTOs, fixed package discovery, process supervision,
   source rehashing, and Core capability aggregation.
5. OpenUtau lands per-track renderer IDs, cache-before-capability lookup,
   immutable generation publication, maintenance admission, and recovery.
6. OpenUtau lands current/all conversion UI and both repositories run a real
   singer end-to-end acceptance pass.

## 16. Protocol Lifecycle

This file is the only active offline model protocol. Repositories MUST NOT
maintain another document that defines competing command lines, schemas, reason
codes, fingerprints, manifests, or publication rules.

Version 1 is implemented only when the provider tool, OpenUtau integration,
shared vectors, package smoke tests, and real-singer acceptance tests in this
document all pass. At that point the parties move this file to:

```text
docs/contracts/OPENUTAU_OFFLINE_MODEL_PROTOCOL_V1.md
```

Both repositories then update their references to that archived path. An
archived protocol is immutable. Any semantic change after archival starts one
new active `OPENUTAU_OFFLINE_MODEL_PROTOCOL_V2_DRAFT.md`; it does not edit v1 or
create an alternative v1 specification.
