# DiffSinger-ASM / OpenUtau Supply Contract (Round 4 Draft)

Status: provider proposal; not accepted or archived as a contract
Date: 2026-09-22
Provider: DiffSinger-ASM
Consumer: OpenUtau DiffSinger renderer

This draft states what the current DiffSinger-ASM implementation supplies and
the next implementation proposed in response to the OpenUtau round 3 draft. It
does not claim final acceptance. A formal contract may be archived only after
both repositories adopt the same normative terms and pass the agreed gates.

`MUST`, `MUST NOT`, `SHOULD`, and `MAY` are normative within this proposal.

## 1. Scope

This draft covers:

- the public native ABI and its two product modes;
- model artifacts and engine lifetime;
- phrase requests, PCM callbacks, cancellation, and errors;
- canonical PCM compatibility and publication;
- the boundary between native execution and OpenUtau orchestration;
- correctness, latency, and release acceptance evidence.

It does not specify OpenUtau UI design, project editing commands, waveform
display, or the internal representation of project ranges.

## 2. Supplied artifacts and platform

DiffSinger-ASM supplies:

- `libdsasm.so` for Linux x86-64;
- `include/dsasm_engine.h` as the authoritative C declaration;
- offline acoustic and vocoder packers;
- `docs/OPENUTAU_OFFLINE_MODEL_PROTOCOL.md` as the agreed implementation
  specification for offline model integration;
- `docs/ENGINE_ABI.md` and this draft;
- fingerprinted correctness and performance evidence for each mode.

The online library requires AVX2 and FMA. Optional AVX-VNNI paths are selected
only after runtime CPU detection. The online library MUST NOT load Python,
PyTorch, ONNX Runtime, or another unapproved inference runtime. Python remains
offline-only for conversion and validation.

The current product artifact does not promise Windows, macOS, ARM64, or a GPU
backend. Those require separate artifacts and workload strata.

## 3. ABI discovery and layout

The consumer MUST load the library by absolute path and resolve
`dsasm_engine_abi_version()` before any other symbol. This draft requires ABI
version `3` exactly. Any other version is incompatible.

The consumer MUST call `dsasm_engine_is_supported()` before engine creation.

The ABI v3 layouts on Linux x86-64 are:

| Type | Size | Alignment |
| --- | ---: | ---: |
| `dsasm_mode_config` | 36 bytes | 4 bytes |
| `dsasm_request` | 200 bytes | 8 bytes |

The C# binding MUST use sequential layout and test both sizes. Pointers and
`size_t` fields are 64-bit. Enums and error codes are 32-bit.

OpenUtau uses these product symbols:

```text
dsasm_engine_abi_version
dsasm_engine_is_supported
dsasm_engine_mode_config
dsasm_engine_create_mode
dsasm_engine_render
dsasm_engine_cancel
dsasm_engine_destroy
dsasm_engine_last_error
dsasm_engine_last_create_error
dsasm_engine_sample_rate
dsasm_engine_hop_size
dsasm_engine_mel_bins
```

`dsasm_engine_create(..., workers)` remains available for controlled provider
experiments. It is not a product entry point for OpenUtau.

## 4. Product modes

ABI v3 exposes exactly two product modes:

| Mode | Value | Workers | Region | Bucket | Overlap | Profile | Output compatibility |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Real-time streaming | 1 | 4 | 32 | 32 | 8 | 2 | 0 |
| Block batch | 2 | 8 | 384 | 384 | 0 | 2 | 1 |

Before calling `dsasm_engine_mode_config()`, the consumer sets:

```text
config.struct_size = 36
config.abi_version = 3
```

The provider fills every other field. The consumer MUST query these values and
MUST NOT maintain a second production copy of worker, region, bucket, overlap,
or revision constants. Unknown modes return `DSASM_E_UNSUPPORTED`; a bad ABI or
layout returns `DSASM_E_INVALID`.

`dsasm_request_init_mode()` contains matching constants as a C convenience
helper. It is not a second production configuration authority. OpenUtau MUST
still query `dsasm_engine_mode_config()` and populate requests from the returned
configuration.

`profile_revision` identifies supplied execution behavior without changing ABI
layout. Revision 2 adds the cancellation boundaries in section 8. It does not
change successful PCM output.

`output_compatibility_revision` identifies a canonical PCM quality class:

- zero means session-only output and forbids canonical commit;
- equal nonzero values permit logical cache reuse without claiming bit identity;
- unequal values forbid cross-mode reuse.

The current realtime value remains zero. The current batch value is one.

## 5. Mode selection and engine lifetime

OpenUtau selects a mode from the operation:

- playback with a live PCM consumer uses real-time streaming;
- pre-render, dedicated render, mixdown, export, and cache fill use block batch;
- `DIFFSINGER` selects ONNX and `DIFFSINGER-ASM` selects ASM explicitly;
- project state does not use `Auto`; a global preference MAY choose only the
  default renderer for a newly created track;
- explicit ASM selection reports failure and MUST NOT change mode,
  bucket, overlap, or worker count silently.

Each singer retains at most one resident engine per mode. A realtime engine
MUST NOT serve batch requests, and a batch engine MUST NOT serve realtime
requests. Engine creation fails if the exact worker count is unavailable or
the required fixed bucket is missing.

OpenUtau owns one inference admission lane per singer. It MUST NOT run the two
mode engines concurrently for that singer under this contract. Model files may
still share operating-system file cache pages.

Reset, singer unload, and shutdown wait for active synchronous calls before
destroying both engines.

## 6. Request boundary

One functional native call contains one complete OpenUtau `RenderPhrase`,
including its normal context and padding. Native acoustic inference covers the
complete request. Only vocoder work is divided into the selected fixed bucket.

For every product-mode request:

```text
request.struct_size = 200
request.abi_version = 3
request.mode = config.mode
request.vocoder_bucket_frames = config.vocoder_bucket_frames
request.overlap_frames = config.overlap_frames
```

The runtime rejects a mismatch with `DSASM_E_INVALID`. It does not grow a
request to another bucket or switch architecture implicitly.

All input arrays remain pinned and valid until `dsasm_engine_render()` returns.
`mel2ph` is 1-based. When `mel2ph` is null, durations are required and sum
exactly to `mel_frames`. Optional features must match the packed graph.

## 7. PCM callback and publication

The native callback is synchronous. PCM is mono native-endian float32 and is
borrowed only for the callback duration. OpenUtau copies it before return,
keeps the delegate rooted for the native call, verifies contiguous offsets,
and accepts exactly one final callback on success.

Realtime chunks may be published progressively only to the active playback
generation. Batch chunks remain staged until the complete phrase succeeds.

A result may enter canonical cache only after:

1. `dsasm_engine_render()` returns success;
2. offsets are contiguous and exactly one callback is final;
3. sample count and values are valid;
4. the phrase generation is still current;
5. `output_compatibility_revision` is nonzero.

Cancelled, callback-aborted, stale, non-finite, incomplete, or revision-zero
output MUST NOT enter canonical cache.

## 8. Cancellation supply

`dsasm_engine_cancel()` is lock-free and may be called from another thread. It
invalidates the active render epoch. The consumer waits for the synchronous
render call to return before reusing or destroying the engine.

Profile revision 2 checks cancellation:

- after FS2 conditioning;
- after the auxiliary decoder;
- after Rectified Flow conditioner projection;
- before and after every Euler denoiser step;
- before and after every vocoder bucket.

The runtime does not asynchronously interrupt an assembly kernel, threadpool
task, or backend call already executing. Return latency is bounded by reaching
the next listed boundary, subject to the workload and target machine.

When cancellation is observed, the call returns `DSASM_E_CANCELLED`. Internal
partial tensors are discarded. PCM already copied by a realtime callback
cannot be recalled; OpenUtau generation checks prevent it from updating a
superseding playback session or cache entry.

### 8.1 Reference measurements

On the fingerprinted Intel Core i5-13420H reference target, the provider offers
the proposed p99 cancel-to-return gate of `371.52 ms` for exact 32-frame
realtime and 384-frame batch requests:

| Mode | Cancel delay | Samples | p99 | Worst | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Realtime streaming | 10 ms | 100 | 19.007 ms | 20.381 ms | PASS |
| Block batch | 10 ms | 100 | 58.225 ms | 58.691 ms | PASS |
| Block batch | 100 ms | 25 | 176.767 ms | 178.101 ms | PASS |
| Block batch | 250 ms | 25 | 210.443 ms | 214.869 ms | PASS |

Every measured request returned `DSASM_E_CANCELLED` and emitted zero callbacks.
Raw samples and fingerprints are stored under
`benchmarks/artifacts/2026-09-22-engine-cancel-stage-checks/` as performance
evidence, not as an archived contract.

### 8.2 Cancellation limitations

The reference measurements do not establish a universal wall-clock guarantee.
They do not transfer automatically across CPU, affinity, power state, backend,
precision, model shape, worker profile, or request stratum.

A complete phrase longer than `region_frames` is functionally valid but does
not inherit the fixed-region cancellation result. An individual neural stage
grows with request shape and remains uninterruptible while its kernel is
running. Long-phrase batch preemption therefore remains unbounded until a
representative length sweep passes or OpenUtau adopts a quality-approved
segmentation policy.

The stored samples requested cancellation during acoustic inference and
observed zero callbacks. They do not prove zero callbacks when cancelling a
later multi-bucket request. Batch callbacks must remain staged; realtime
callbacks remain generation-scoped session state.

The 100 ms and 250 ms delay samples improve phase coverage but are not an
exhaustive adversarial phase sweep. They do not establish hard real-time
behavior under arbitrary system load. A target device must repeat the gate
before OpenUtau relies on active batch preemption there.

## 9. Model bundle

A prepared singer contains:

```text
dsasm/acoustic/model.conf
dsasm/acoustic/fs2_acoustic.dsfs
dsasm/acoustic/aux_convnext.dsa
dsasm/acoustic/lynxnet2.dsn
dsasm/vocoder/32.dsv35
dsasm/vocoder/384.dsv35
```

The converter generates both product buckets. Missing either bucket makes the
product bundle incomplete. Legacy 64/128/256 buckets are experimental and do
not satisfy product readiness.

OpenUtau verifies native sample rate, hop size, and mel bins against singer
configuration before retaining either engine. Prepared-model identity includes
source model hashes, packer version, packed format version, and ISA profile.
Singer name alone is not an identity.

Bundle validity and runtime availability are independent. A bundle can be
valid when the current machine cannot execute the runtime because of OS, CPU,
affinity, or worker constraints. Runtime unavailability MUST NOT be reported as
source-model incompatibility.

### 9.1 Agreed offline tool protocol

The parties agree to implement
[`OPENUTAU_OFFLINE_MODEL_PROTOCOL.md`](OPENUTAU_OFFLINE_MODEL_PROTOCOL.md) as
the single authoritative offline model boundary. The current provider
implementation exposes these logical operations through the packaged
`dsasm-model-tool` command:

```text
inspect -> plan -> convert -> validate
```

Each operation supports the JSON or JSONL transport, exit codes, stable reason
codes, fingerprint rules, path constraints, and cancellation behavior defined
by that shared protocol. The schema includes:

- schema version, converter revision, and provider build identity;
- source fingerprint and prepared bundle fingerprint;
- model compatibility, bundle state, and a stable reason code plus diagnostic;
- required output files and estimated staging, final, and work space;
- conversion stages, structured progress, and structured errors;
- packed format versions, ISA profile, file sizes, and artifact digests.

OpenUtau Core combines this provider data with local runtime, toolchain, and
activity state. OpenUtau owns allowed UI actions and localization. The native
ABI does not grow UI, toolchain, activity, or conversion-plan concepts.

The provider writes an authoritative `bundle.json` at the root of each staging
and immutable prepared generation. OpenUtau publishes it through the
consumer-owned `dsasm/current.json` pointer defined by the shared protocol. Its
bundle fingerprint is computed from canonical manifest fields and every
required packed artifact digest. The manifest also records source hashes,
converter and packer revisions, packed format versions, ISA profile, audio
dimensions, and required product buckets.

Offline `validate` MUST validate an arbitrary staging bundle without requiring
the current machine to satisfy product runtime worker, affinity, or ISA
execution gates. It validates formats, dimensions, digests, cross-file
consistency, and product-mode completeness.

`dsasm_engine_create_mode()` accepts arbitrary acoustic and vocoder directory
paths, including staging paths. It performs a target-runtime validation and is
therefore subject to the current machine's OS, CPU, affinity, and exact worker
requirements. OpenUtau SHOULD run offline `validate` first, then create and
destroy both product-mode engines on the target before publication when the
target runtime is available.

The provider implementation is in `tools/dsasm_model_tool.py`, with the fixed
package launcher at `bin/dsasm-model-tool`. The package carries CPython 3.12,
an exact dependency lock with hashes, and the production packers under
`share/dsasm-model-tool`. `inspect` uses the same acoustic topology extraction
and fixed-shape 32/384 vocoder graph construction as conversion, without
creating output or work files.

Provider acceptance evidence on the supported Yousa V1.56 singer records:

- deterministic source fingerprint
  `e456cf48bdfafee3b9568d79bdff3aba1fe5601bf113b545fff50704be8e2875`;
- successful 32/384 staging conversion and offline validation;
- authoritative bundle fingerprint
  `29bc1b9164c330fd8bf4c6ad28fa2ea97ea92224ef39c8fb383e65e62c3e96ed`;
- successful create/destroy of both ABI v3 product mode engines directly from
  the staging acoustic and vocoder paths;
- SIGINT result `cancelled`, exit code 7, and no staging `bundle.json`;
- stale expected source identity result `source.changed`, exit code 9, and no
  staging `bundle.json`;
- package validation outside the source checkout with a nonempty embedded
  provider build identity.

## 10. Canonical cache

The minimum canonical WAV cache identity is:

```text
phrase hash
+ explicit backend
+ ABI
+ output_compatibility_revision
+ authoritative source fingerprint
+ authoritative bundle fingerprint
+ depth
+ steps
```

Mode and profile revision are provenance metadata. They do not split a cache
entry when the output compatibility revision is the same nonzero value.

Current behavior is:

| Completed source | Later playback | Later pre-render/export |
| --- | --- | --- |
| Batch revision 1 | reuse | reuse |
| Realtime revision 0 | rerender after active session | rerender |

The native runtime does not read, write, publish, or replace canonical WAV
files. OpenUtau owns memory cache, disk cache, staging files, commit manifests,
generation publication, cleanup, and recovery. A failed or cancelled
replacement leaves the previous complete generation intact.

Canonical ASM PCM remains reusable after its prepared bundle is deleted when:

1. PCM provenance was atomically committed with provider-generated source and
   bundle fingerprints;
2. the current source-model fingerprint still equals the stored source
   fingerprint;
3. phrase identity, explicit backend, ABI, output compatibility revision,
   depth, steps, and PCM validation all still match; and
4. no currently published bundle contradicts the stored bundle fingerprint.

Cache lookup MUST NOT require the native runtime, conversion toolchain, or
prepared bundle to be available. When a current bundle exists, its authoritative
fingerprint MUST equal the PCM provenance fingerprint. Re-conversion producing
a different bundle fingerprint therefore causes a natural cache miss. Deleting
a bundle does not by itself invalidate an already valid PCM artifact.

## 11. Overwrite rendering

Overwrite rendering is owned by OpenUtau. OpenUtau resolves the requested
half-open project range against a stable phrase snapshot and rerenders every
intersecting phrase in full. Unaffected phrases retain their cache entries.

OpenUtau owns range selection, scope, cache bypass level, phrase generation,
scheduling, staged publication, and atomic replacement. These concepts do not
add ABI fields and do not change the operation-selected native mode.

The previous complete result remains valid until its replacement succeeds.
Cancellation, failure, or stale generation discards staged replacement data.

Prepared-bundle publication is also consumer-owned. The proposed transaction
is:

```text
plan -> staging convert -> offline validate -> acquire singer maintenance lane
-> release old engines -> atomically switch one generation pointer
```

The provider writes and validates staging artifacts but does not own the singer
directory generation pointer. Publication cannot change models inside an
active playback session. OpenUtau may validate both product modes directly
from staging paths before switching the pointer.

## 12. Performance claim boundary

The promoted performance results apply only when request frames equal the
queried `region_frames`. Complete-phrase requests are functionally conformant,
but do not inherit fixed-region latency, throughput, or cancellation claims.

Exact 32/384 segmentation is a separate performance-conformance feature. It
must preserve phrase context and pass a joint real-song quality gate before it
becomes the default OpenUtau scheduler. Padding and trimming a final short
region are allowed only after that policy passes the same gate.

ASM, oneDNN, GGML, and GPU results remain separate strata whenever backend,
device, precision, graph partition, or scheduling differs.

## 13. Errors and fallback

| Code | Meaning | Consumer action |
| ---: | --- | --- |
| `0` | success | accept only with one final callback |
| `-1` | invalid ABI, request, or mode fields | integration error; do not retry unchanged |
| `-2` | I/O failure | report model or filesystem failure |
| `-3` | invalid packed model | mark prepared bundle incompatible |
| `-4` | allocation failure | fail the render |
| `-5` | unsupported mode, graph, or bucket | explicit ASM reports; consumer may offer ONNX selection |
| `-6` | cancelled | translate to managed cancellation |
| `-7` | callback aborted | surface callback exception or cancellation |
| `-8` | internal inference failure | report native `last_error` |

The consumer copies the UTF-8 error before another call on the same engine or
thread-local creation-error channel.

## 14. Acceptance gates

Provider acceptance requires:

```sh
make -j"$(nproc)" engine-check
make engine-real-stream-check \
  ENGINE_REAL_ACOUSTIC=/path/to/acoustic \
  ENGINE_REAL_VOCODER=/path/to/vocoder
```

It also requires the current M18, M20, M21, and M22 numerical parity gates,
mode-specific golden gates, and the cancellation measurements in section 8.
Instrumented localization runs cannot be used as promotion timing.

Consumer acceptance requires:

- a successful OpenUtau build;
- managed 36-byte config and 200-byte request layout tests;
- exact ABI and symbol-loading rejection tests;
- mode-config query tests without duplicated profile constants;
- separate mode-engine lifecycle and one-lane scheduling tests;
- request mode, bucket, and overlap propagation tests;
- revision-zero canonical exclusion and compatible cache lookup tests;
- cancellation, callback ownership, generation, and atomic replacement tests;
- converter completeness tests for both product buckets.

The next offline protocol additionally requires schema compatibility tests,
stable reason-code tests, staging validation on a machine that cannot satisfy
runtime worker gates, manifest tamper tests, and bundle-fingerprint invalidation
tests.

Joint acceptance requires real singer renders in both modes with finite output,
contiguous callbacks, exact sample count, and agreed waveform quality gates.
Timing from a failed correctness run is inadmissible.

## 15. Proposed ownership freeze

DiffSinger-ASM owns:

- the native inference ABI, product modes, execution profiles, and errors;
- source-model compatibility rules;
- the versioned offline inspect, plan, convert, and validate protocol;
- packed formats, authoritative bundle manifests, and artifact fingerprints;
- provider correctness and performance evidence.

OpenUtau owns:

- one singer identity regardless of backend;
- the per-track `DIFFSINGER` or `DIFFSINGER-ASM` selection;
- capability aggregation, UI actions, localization, and user-triggered flows;
- cache lookup and publication, scheduling, cancellation policy, and phrase
  generations;
- resident dual-engine lifetime, singer admission and maintenance lanes;
- prepared-bundle generation pointers and canonical PCM generation commits.

Selecting ASM performs asynchronous read-only detection. It MUST NOT start a
conversion. Conversion requires an explicit user action. Capability errors are
shown only on a compatible cache miss when inference is actually required.

The first batch-conversion implementation is globally serial across all singer
roots. Each singer is independently staged, validated, and committed. A failed
or cancelled singer does not invalidate previously committed singers, and the
batch operation does not change existing track backend selections.

## 16. Open items before a formal contract

This provider draft does not become a formal or archived contract until both
sides resolve and record the following:

1. **Realtime canonical compatibility.** Realtime remains revision zero. A
   completed realtime render therefore does not satisfy later playback,
   pre-render, mixdown, or export. Promotion needs an agreed real-song corpus,
   perceptual and numerical thresholds, archived cross-mode evidence, and a
   shared nonzero compatibility revision. The existing single 384-frame
   comparison is insufficient.
2. **Long-phrase cancellation scope.** The current `371.52 ms` offer covers
   only exact fixed-region reference strata. The parties must decide whether
   formal acceptance requires a representative complete-phrase length sweep,
   a maximum native request length, or quality-approved exact-region
   segmentation.
3. **Target-device enablement.** The parties must decide how OpenUtau records
   or verifies that a target device belongs to a passing cancellation stratum
   before it enables active batch preemption.
4. **Offline protocol consumer acceptance.** The provider tool, shared
   fingerprint vectors, real-singer conversion, cancellation, staging mode
   creation, and package smoke gates pass. OpenUtau integration must consume
   the same vectors and pass its process supervision, source rehash,
   publication, recovery, and cache invalidation gates before protocol v1 is
   moved to its immutable contract path.
5. **Consumer gates.** OpenUtau must finish its implementation review, build,
   and managed tests against the final proposed ABI library.

Until these items are accepted in matching documents, this file remains a
negotiation draft and MUST NOT be presented as the first formal supply
contract.
