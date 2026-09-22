# DSASM engine ABI v3

The provider/consumer obligations shared with OpenUtau are specified in the
[supply contract draft](OPENUTAU_SUPPLY_CONTRACT_DRAFT.md).

`build/libdsasm.so` is the product-facing library. ABI v3 exposes two explicit
execution modes backed by independently promoted performance profiles:

| Mode | Workers | Measured region | Bucket | Overlap | Profile | Output compatibility |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `DSASM_MODE_REALTIME_STREAMING` | 4 | 32 | 32 | 8 | 2 | 0 (session only) |
| `DSASM_MODE_BLOCK_BATCH` | 8 | 384 | 384 | 0 | 2 | 1 (canonical) |

The values are returned by `dsasm_engine_mode_config()`. Integrations must use
the returned values instead of duplicating them. `profile_revision` identifies
the promoted execution profile. An `output_compatibility_revision` of zero
forbids canonical PCM cache commit; equal nonzero revisions declare that the
complete outputs may share a product cache entry. The two modes remain separate
performance strata.

## Build and model layout

```sh
make engine-check
```

The acoustic directory contains:

```text
fs2_acoustic.dsfs
aux_convnext.dsa
lynxnet2.dsn
model.conf
```

The vocoder directory must contain both product buckets:

```text
32.dsv35
384.dsv35
```

Each bucket must match the acoustic model's mel-bin count and the
`model.conf` hop size. A mode engine fails creation if its required bucket is
missing or the process CPU affinity cannot provide its exact worker count.

## Creating mode engines

Call `dsasm_engine_is_supported()` first, initialize the versioned config, and
query the selected profile:

```c
dsasm_mode_config config = {
    .struct_size = sizeof(config),
    .abi_version = DSASM_ENGINE_ABI_VERSION,
};
if (dsasm_engine_mode_config(DSASM_MODE_REALTIME_STREAMING, &config) != DSASM_OK)
    /* reject the backend */;

dsasm_engine *engine = dsasm_engine_create_mode(
    acoustic_dir, vocoder_dir, DSASM_MODE_REALTIME_STREAMING);
```

`dsasm_engine_create_mode()` creates the profile's fixed worker pool and keeps
the models and inference buffers resident. Product integrations should keep one
engine per singer and mode. The older `dsasm_engine_create(..., workers)` entry
point remains available for controlled benchmark experiments; it does not
claim either promoted mode.

One engine serializes render calls. Cancellation is lock-free:
`dsasm_engine_cancel()` invalidates the active request, and destruction is only
valid after the render call returns. Profile revision 2 checks cancellation
after FS2, after the aux decoder, after Rectified Flow conditioner projection,
before and after each Euler step, and between vocoder buckets. It does not
interrupt an assembly kernel that is already running.

## Request contract

Initialize requests with `dsasm_request_init_mode(mode)`. The resulting
`mode`, `vocoder_bucket_frames`, and `overlap_frames` fields exactly match the
queried profile. A mode engine rejects a request when any of those fields
differs, so it cannot silently switch architecture or grow to another bucket.

`region_frames` records the request size used by the accepted performance
gate. Longer requests are valid and are divided into fixed vocoder buckets,
but they do not inherit the champion's latency or throughput claim. A caller
that needs the measured real-time cadence should submit successive 32-frame
requests to one resident real-time engine. Batch scheduling should submit
384-frame regions to one resident batch engine. The final short region may be
padded by the integration and trimmed after PCM publication.

Request pointers need to remain valid only for the synchronous call. `mel2ph`
is 1-based. Passing `durations` instead lets the engine construct it. A single
speaker embedding row is broadcast to every acoustic frame.

PCM callbacks borrow mono float32 memory that is valid only during the call.
Offsets are contiguous within one request. The caller must copy a callback's
samples before returning. Returning nonzero aborts with `DSASM_E_CALLBACK`.

## OpenUtau contract

The OpenUtau binding must:

1. Resolve `dsasm_engine_abi_version` first and require version 3.
2. Query both mode configs. Retain mode and profile revision as provenance;
   use a nonzero output compatibility revision in canonical WAV cache keys.
3. Lazily retain separate real-time and batch engines for each singer.
4. Use real-time mode when PCM is consumed progressively during playback, and
   batch mode for ordinary complete rendering and pre-rendering.
5. Copy `mode`, `vocoder_bucket_frames`, and `overlap_frames` from the selected
   config into every native request.
6. Require `32.dsv35` and `384.dsv35` when deciding whether a prepared singer
   bundle is complete.
7. Keep the native callback delegate and all pinned request arrays alive until
   `dsasm_engine_render()` returns.

`Auto` may fall back to ONNX Runtime when the library, CPU, packed graph, or
mode profile is unavailable. An explicitly selected ASM backend must report the
native error instead of changing mode or bucket.

Cache keys for packed bundles should include source model hashes, packer and
format versions, and the ISA profile. Singer names are not sufficient.
