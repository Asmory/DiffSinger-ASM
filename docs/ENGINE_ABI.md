# DSASM engine ABI v1

`build/libdsasm.so` is the product-facing library. Its exported surface is
limited to `dsasm_engine_*`; model, graph, and thread-pool structs remain
implementation details.

## Build and package

```sh
make engine-check
```

Package the library beside the OpenUtau native runtime. Prepare models before
playback. The acoustic directory contains:

```text
fs2_acoustic.dsfs
aux_convnext.dsa
lynxnet2.dsn
model.conf
```

The vocoder directory contains numeric fixed-shape buckets. Each bucket must
have the acoustic model's mel-bin count and `model.conf` hop size:

```text
64.dsv35
128.dsv35
256.dsv35
384.dsv35
```

Call `dsasm_engine_is_supported()` for the `Auto` backend check; ABI v1 requires
Linux x86-64 with AVX2 and FMA. `dsasm_engine_create()` mmaps every model once,
creates one persistent thread pool shared by acoustic inference and every
vocoder bucket, and retains grow-on-demand inference buffers between calls. One engine
serializes its render calls, so OpenUtau should keep one engine per active
singer and schedule phrases through a singer-level queue.

## Rendering contract

Initialize `dsasm_request` with `DSASM_REQUEST_INIT`, then fill pointers and
dimensions. Set override flags for depth, steps, time scale, or spec range;
otherwise `model.conf` values are used. `mel2ph` is 1-based. Passing durations
instead lets the engine construct it. A single speaker embedding row is
broadcast without requiring C# to duplicate it.

The first vocoder call always uses the smallest bucket. Later calls use the
smallest bucket that fits the remaining frames, or the largest bucket for a
long remainder. Eight frames overlap by default (set `overlap_frames` to a
nonzero custom value). The engine withholds each block's overlap tail, blends
it with the next block, and only then publishes immutable mono float32 PCM.
Callback offsets are contiguous output sample offsets and the last callback
has `is_final=1`.

Cancellation is lock-free. `dsasm_engine_cancel()` invalidates the active
request; the render call observes it before and after acoustic inference and
between vocoder buckets. Destroy an engine only after its render call returns.

## C# loading

Use `NativeLibrary.Load()` with an absolute path, resolve
`dsasm_engine_abi_version` first, and require version 1 before resolving the
remaining delegates. Keep delegates and the PCM callback rooted for the whole
native call. The callback buffer is borrowed and must be copied into an owned
array before returning.

Publish each owned array to the mixer immediately. Since the native callback
is synchronous, do not call `Render` again or destroy the engine from inside
it. Return nonzero from the callback to abort; call `dsasm_engine_cancel()`
from the cancellation-token thread for normal cancellation.

Cache keys for prepared bundles should include the source model hash, packer
version, format version, and ISA profile. Singer names are not sufficient.
