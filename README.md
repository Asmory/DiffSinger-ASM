# DiffSinger-ASM M40.1 — Fast AVX-VNNI Pack + Relative Quality Gate

Current M55/M58 fusion status, the 3% stability gate, instrumented results and
reproduction commands are recorded in [docs/PERFORMANCE.md](docs/PERFORMANCE.md).

M40.1 keeps M40 selective stage128 K7/K11 AVX-VNNI, but replaces the scalar activation im2col packer with a handwritten x86-64 pack kernel. Quantized activations are written into a zero-point-padded NCT buffer; a precomputed reduction-offset map lets the packer load four contiguous 8-byte vectors and transpose them directly into each 32-byte `vpdpbusd` block.

It also changes INT8 quality validation from a fixed absolute max-error gate to configurable cosine/SNR gates (`DSASM_GOLDEN_COS`, `DSASM_GOLDEN_SNR`). The e2e harness now runs acoustic and ORT golden once, then benchmarks k11-only and k7+k11 on the exact same mel/f0. Production runtime remains C + handwritten x86-64 ASM only.

Run with `./scripts/run_m40_1.sh`.

# M40 — Selective AVX-VNNI realtime sprint

M40 keeps the pure C/x86-64 ASM runtime and adds selective AVX-VNNI only for the real NSF-HiFiGAN C=128, T=3072, K7/K11 stage. FP32 weights remain embedded for exact fallback and A/B. `DSASM_VNNI=0|k11|k117` selects the runtime path. Activation quantization and im2col packing are parallelized on the existing persistent P-core pool; weights are quantized offline.

Run the packaged test with `bash scripts/run_m40.sh`.

# M39.1 — Pure-ASM Conv surgery + correct AVX-VNNI VEX encoding

M39 restores the M37 winning vocoder graph semantics by disabling M38 residual-store fusion by default. It then attacks Conv cost in two independent, measurable directions while keeping the production runtime pure CPU/C/ASM.

1. **Workspace single-pass padding.** The fused Leaky->Conv path no longer memset()s the entire padded workspace and then overwrites the center. `leaky_copy_nct_f32_avx2` now clears only the left/right halos in the same ASM channel pass that transforms/copies the center. `DSASM_FULL_MEMSET=1` restores the old path for same-binary A/B.
2. **Fixed K3/K7/K11 experiment.** Bit-exact fully-unrolled AVX2/FMA kernels are included behind `DSASM_KSPEC=1`. They are deliberately OFF by default because local synthetic A/B did not beat the generic K loop; the target machine can verify them on the real graph without contaminating the baseline.
3. **AVX-VNNI INT8 feasibility lab (M39.1 fixed).** A separate pure-ASM `vpdpbusd` kernel quantizes a captured real Conv activation to U8, uses per-output-channel S8 weights, and reports activation-pack cost, kernel cost, cosine/SNR and total speedup. This lab is not yet used by the production vocoder; it decides whether M40 should integrate INT8 into the full graph.

Important targets:

```bash
make m39-check
make m39-vnni-check
make m39-real-vocoder MODEL_DIR=/path/to/voicebank
make m39-vnni-real MODEL_DIR=/path/to/voicebank
make m39-real-e2e MODEL_DIR=/path/to/voicebank SPEAKER_EMB=/path/to/singer.emb
```

Final runtime executables remain free of ORT/oneDNN/OpenVINO/MKL/BLAS. ONNX Runtime is offline-only for fixed-shape packing and golden/reference extraction.

---

# M38 — Pure-ASM residual-store fusion

M38 fuses HiFi-GAN internal `Conv2 -> Add(residual)` into the packed8 AVX2/FMA Conv store path. The AOT compiler recognizes 45 residual-unit adds in the real NSF-HiFiGAN graph, removes those Add bytecode ops, keeps the residual tensor live through the Conv, and emits a residual-aware Conv flag. Both static-OC and M37 OC×time 2-D kernels have residual variants. Final runtime remains pure CPU/C/ASM; ONNX Runtime is offline-only for packing/golden parity.

# DiffSinger-ASM M37 — Pure-ASM vocoder Conv 2-D scheduling

M37 attacks the remaining full-native NSF-HiFiGAN Conv bottleneck without any
third-party runtime. All Conv layers divisible by eight now use packed-8, and
long low-channel layers (`C=32/16`) are dynamically split across both output
channel blocks and 512-sample time tiles so the whole P-core worker pool stays
busy. Large-channel layers keep M36's channel-owner execution for cache reuse.

The new `ds_conv1d_nct_f32_avx2_oc8_t8_range` kernel preserves each output
element's FP32 accumulation order; synthetic long-sequence 2-D/static A/B is
bit-exact. The AOT manifest reports the number of 8-worker 2-D candidates and
a real Conv shape histogram. Final execution is still strictly C + x86-64 ASM;
ORT is offline pack/golden only.

Useful targets:

```bash
make m37-check
make m37-real-vocoder MODEL_DIR=/path/to/voicebank
make m37-real-e2e MODEL_DIR=/path/to/voicebank SPEAKER_EMB=/path/to/singer.emb
```

See `docs/M37_VOCODER_CONV_2D.md`.

---

# DiffSinger-ASM M33 — Pure-ASM NSF-HiFiGAN structural blocks

M33 advances the vocoder from isolated Conv1d microbenchmarks to real HiFi-GAN
execution units while keeping the final runtime strictly **CPU + our own C/ASM**.
ORT/ONNX remains offline-only for importing real weights and golden tensors.

New native pieces:

- direct sparse AVX2/FMA `ConvTranspose1d` (no zero-insertion waste);
- AVX2 LeakyReLU and residual Add;
- a complete `LeakyReLU -> Conv -> LeakyReLU -> Conv -> Add` residual unit;
- persistent P-core-pool scheduling for ordinary Conv and ConvTranspose.

Useful targets:

```bash
make m33-check
make m33-real-blocks MODEL_DIR=/path/to/voicebank M33_ROUNDS=7
```

The real target benchmarks all three `resblocks.5` residual units plus all five
NSF-HiFiGAN upsamplers.  See `docs/M33_PURE_ASM_VOCODER_BLOCKS.md`.

---

# DiffSinger-ASM M31 — CPU Realtime Sprint 1

M31 turns CPU real-time synthesis into a hard acceptance target rather than a
future optimization wish.  On the target i5-13420H, end-to-end CPU
`conditioning -> WAV` must reach `RTF < 1.0`; the engineering target is
`RTF <= 0.8`.

M30 already established real-voicebank waveform parity, so M31 intentionally
does **not** change acoustic math.  It adds a bit-exact native stage profiler,
a deterministic `20/16/12/10/8/6/4` Rectified-Flow step Pareto sweep, an
interleaved NSF-HiFiGAN ORT `1/2/4/6/8` CPU-thread sweep, target P-core
affinity, per-step listening WAVs, and vocoder static/node profiling.

Important targets:

```bash
make m31-check
make m31-real-sprint MODEL_DIR=/path/voicebank SPEAKER_EMB=/path/singer.emb
```

The real sprint writes `build/m31_realtime/m31_realtime_report.json` and prints
FS2/Aux/RF timings, RF time per Euler step, vocoder best-thread latency,
end-to-end RTF, and whether `RTF<1` / `RTF<=0.8` has actually been crossed.
See `docs/M31_REALTIME_SPRINT.md`.

---

# DiffSinger-ASM M29 — Cross-lingual language-mask parity fix

M29 fixes the first real DongFangZhiZi FS2 divergence found by M28.  Current
DiffSinger deployment does **not** add `lang_embed(language_id)` to every token.
It first builds `lang_mask = any(tokens[...,None] == cross_lingual_token_idx)`
and evaluates `lang_embed(languages * lang_mask)`.  M25–M28 packed the language
embedding table but omitted this token mask, so monolingual/non-cross-lingual
tokens incorrectly received the selected language vector.

The ONNX importer now extracts `cross_lingual_token_idx` directly from the
constant input of `/fs2/Equal`, stores a `[vocab]` language-token mask in the
DSFS25 payload, and sets feature bit `0x100`.  The loader mmaps that section and
the native encoder forces the language row to zero for tokens outside the mask.
Older DSFS25 bundles without the new bit retain their previous behavior.

`tools/validate_real_onnx_m29.py` performs two checks on a real deployment model:
1. pre-Transformer parity for raw token embedding, duration log/linear, masked
   language IDs, language embedding, token scaling and final encoder input;
2. the full M27 FS2-stage + Aux + RF parity using the corrected bundle.

Useful targets:

```bash
make m29-check
make m29-real-parity MODEL_DIR=/path/voicebank SPEAKER_EMB=/path/singer.emb
```


## M25.2 importer hotfix

M25.2 fixes the real-ONNX RF importer control-flow bug discovered with
DongFangZhiZi Nectar.  The per-residual-block `key()` helper previously wrote
`if ...: raise ...; return ...` on one line, which made the `return` part of the
`if` suite.  A uniquely found initializer therefore returned `None`.  The
return is now outside the conditional.  Native kernels, bundle ABIs and model
math are unchanged from M25.1.

# DiffSinger-ASM M25

M25 promotes the runtime from the training-checkpoint/default-profile world to current DiffSinger **deployment ONNX** voicebanks. It adds an ONNX topology importer, `DSFS25`, multilingual/variance/gender/velocity/speaker conditioner branches, exported stretch-table support, and an extended native CLI. M21–M24 default-profile ABIs remain supported.

The target real-world graph that motivated M25 exposes `tokens, languages, durations, f0, breathiness, voicing, tension, gender, velocity, spk_embed, depth, steps`; M25 preserves these conditioner semantics while continuing to execute FS2/Aux/LYNXNet2 through the native kernels.

## Why M23 exists

M22's phase-ordered profiler could report a misleading result such as a ~430 ms
sum of individually repeated stages but a ~1.4 s full wrapper. A stricter local
experiment showed that the full wrapper was not secretly repeating work: when a
manual `FS2 -> Aux -> RF20` chain and the full wrapper are alternated under the
same sustained load, their medians match and their outputs are bit-exact.

M23 therefore replaces the important performance check with an ABBA-style paired
benchmark. Every measured manual-chain call times FS2, Aux and RF20 inside the
same call, and manual-chain/full-wrapper calls are interleaved. Per-call times,
p90/max values, >1.5x outlier counts and CPU frequency (when cpufreq sysfs is
available) are printed.

## Native packed-model loader

M23 promotes the three packed formats from test-only parsers to a runtime API:

- `DSFS21`  — FastSpeech2 acoustic conditioner
- `DSAUX20` — shallow ConvNeXt aux decoder
- `DSLYNX7` — LYNXNet2 Rectified-Flow velocity network

`include/dsasm_model.h` provides:

```c
DSAsmAcousticModel model;
ds_acoustic_model_load(&model,
    "fs2_acoustic.dsfs",
    "aux_convnext.dsa",
    "lynxnet2.dsn");

size_t n = ds_acoustic_model_workspace_floats(&model, text_tokens, mel_frames);

ds_acoustic_model_infer_f32_avx2(
    &model,
    token_ids, text_tokens,
    mel2ph, f0, mel_frames,
    noise,
    spec_min, spec_max, range_dims,
    0.4f, 1000.0f, 20,
    output_mel, workspace, pool);

ds_acoustic_model_unload(&model);
```

The bundle files are `mmap(MAP_PRIVATE)` and their 64-byte-aligned packed
sections are referenced directly; large weights are not copied into another
runtime allocation. The loader validates headers, section bounds and cross-file
dimensions before exposing the unified model.

## Important targets

```bash
make m23            # loader + M22/M21/M20/M19/DSLYNX7 correctness regression
make m23-bench      # paired ABBA official-shape benchmark
make m23-model-check
```

Expected paired-benchmark interpretation:

```text
manual-chain median ~= full-wrapper median
chain/full ~= 1.0x
old_vs_new max_abs=0
```

Large isolated spikes should be treated as system/frequency/thermal outliers,
not immediately attributed to hidden model work.

## M24 — real checkpoint pack-all + native CLI

M24 turns the M23 loader into a usable default-profile acoustic model directory.

```bash
python tools/pack_acoustic_model_m24.py model.ckpt --config acoustic.yaml --out packed/model
build/dsasm-acoustic inspect packed/model
build/dsasm-acoustic infer packed/model \
  --tokens tokens.txt --durations durations.txt --f0 f0.txt \
  --seed 1234 --out mel.f32
```

The pack-all step emits DSFS21 + DSAUX20 + DSLYNX7 together with `model.conf`.
With `--config`, unsupported optional acoustic branches are rejected rather than
silently omitted. The CLI generates 1-based `mel2ph` from durations, supports a
deterministic native Gaussian RNG or caller-supplied raw FP32 noise, and writes
raw `[T,mel_bins]` FP32 mel. See `docs/M24_REAL_MODEL_CLI.md`.

## M25 — ONNX deployment import + extended conditioner

Pack a current exported DiffSinger voicebank directly from `acoustic.onnx`:

```bash
python tools/pack_acoustic_onnx_m25.py /voicebank/acoustic.onnx \
  --model-dir /voicebank --out packed/voicebank
build/dsasm-acoustic inspect packed/voicebank
```

The importer walks ONNX `If` subgraphs recursively and recovers anonymous
`MatMul` weights from graph topology instead of depending on exporter-generated
names such as `onnx::MatMul_1651`. It emits `DSFS25 + DSAUX20 + DSLYNX7`,
`model.conf`, spec ranges, and copies deployment metadata/`.emb` files when a
model directory is supplied.

`DSFS25` extends the conditioner with language ID, breathiness, voicing,
tension, gender/key-shift, velocity/speed and external speaker embedding. It
also supports deployment graphs where the stretch MLP has been constant-folded
into the 1001-entry lookup table used by the ONNX graph.

Example Chinese/speaker inference after import:

```bash
build/dsasm-acoustic infer packed/voicebank \
  --tokens tokens.txt --durations durations.txt --f0 f0.txt \
  --language-id 4 --speaker-emb singer.emb \
  --depth 0.6 --steps 20 --seed 25 --out mel.f32
```

Optional frame curves can be supplied with `--breathiness`, `--voicing`,
`--tension`, `--gender`, and `--velocity`. Missing variance curves default to
neutral values (0; velocity defaults to 1), while language and speaker inputs
are required when the packed model declares those branches. `--depth` is a
normalized deployment convenience and maps to `t_start=max(1-depth,0)`.

Validation targets:

```bash
make m25
make m25-onnx-real MODEL_DIR=/path/to/voicebank
```

The older checkpoint-vs-PyTorch M25 prototype is retained as
`m25-ckpt-legacy`/`m25-real` for regression work, but it is no longer the main
M25 path. See `docs/M25_ONNX_DEPLOYMENT.md`.

## M25.1 hotfix

The ONNX deployment importer now resolves exported nodes by exact name before falling back to substring matching. This fixes real deployment graphs containing sibling nodes such as `/fs2/Clip` and `/fs2/Clip_1`, and applies the same rule to linear-node lookup. Runtime math, DSFS25/DSAUX20/DSLYNX7 formats, and native CLI ABI are unchanged from M25.

## M26 — real deployment ONNX parity

M26 adds `tools/validate_real_onnx_m26.py` and validation-only CLI dumps. It performs deterministic original-ONNX ↔ packed-native comparisons for `condition`, `aux_mel`, and final `mel` by replacing ONNX `RandomNormalLike` with the same external FP32 noise used by native inference. See `docs/M26_REAL_ONNX_PARITY.md`.

## M27 — real FS2 stage locator

M26 established that the first measurable DongFangZhiZi discrepancy is already
present in the final FS2 `condition` tensor (`max_abs ~= 8.2e-2`), before the
Aux ConvNeXt and Rectified Flow amplify it.  M27 intentionally does **not**
change inference math.  It adds a debug-only native API and CLI surface that
captures the same FS2 residual checkpoints exposed from the deployment ONNX:

- token-level encoder output (`/fs2/encoder/Mul_6_output_0`)
- mel2ph gather (`/fs2/GatherElements_output_0`)
- + stretch embedding (`/fs2/Add_2_output_0`)
- + stretch GRU (`/fs2/Add_3_output_0`)
- + pitch (`/fs2/Add_5_output_0`)
- + breathiness/voicing/tension (`/fs2/Add_6_output_0`)
- + gender/key-shift (`/fs2/Add_8_output_0`)
- + velocity/speed (`/fs2/Add_9_output_0`)
- + speaker embedding (`condition`)

`tools/validate_real_onnx_m27.py --fs2-stages` patches the original ONNX to
export those tensors, runs ORT and the native CLI on identical inputs/noise, and
prints `max_abs`, RMSE and cosine for every stage plus the first stage whose
`max_abs` exceeds the diagnostic threshold (default `5e-4`).  By default M27 is
a locator and exits successfully after producing a report even when parity is
not yet fixed; pass `--strict` to retain M26 pass/fail behavior.

The native CLI adds only the debug option:

```
--dump-fs2-stages DIR
```

It writes `encoder_txt.f32`, `gathered.f32`, `stretch.f32`, `gru.f32`,
`pitch.f32`, `variance.f32`, `key_shift.f32`, `speed.f32`, and `speaker.f32`.
The normal inference ABI and packed model formats are unchanged.

## M28 — real encoder accuracy A/B

M28 follows the first real DongFangZhiZi parity failure from M27. It adds an
accuracy-oriented FS2-only LayerNorm kernel using a centered two-pass variance
calculation and selects it with `DSASM_FS2_PRECISE_LN=1`; LYNXNet2 and Aux keep
the established fast LayerNorm path. M28 also fixes the deployment importer so
`speed_embed` takes its bias from the actual ONNX `Add` input. This matters for
exports that deduplicate the speed bias with `fs2.key_shift_embed.bias`.

`tools/validate_real_onnx_m28.py` runs the real voicebank twice (fast LN and
precise LN) and independently reconstructs the packed DSFS25 Transformer in
PyTorch. It reports packed-PyTorch vs ONNX at embedding/layer0..3/final and
native fast-vs-precise errors. This distinguishes importer/semantic errors from
native numerical errors without changing the production default silently.

## M29 — real cross-lingual deployment semantics

M29 fixes the deployment-only language masking used by current multilingual
DiffSinger exports. The ONNX graph applies `language_id` only to tokens in its
`cross_lingual_token_idx`; all other tokens use language row 0. The importer
stores a vocabulary-sized `language-token-mask` in DSFS25 (feature bit `0x100`)
and the native conditioner applies it before language embedding. Old DSFS25
bundles without this bit retain the earlier behavior.

On the real DongFangZhiZi model this restores the entire acoustic chain to
near-FP32 parity: front-end and FS2 are around 1e-6, Aux around a few e-5, and
final mel around a few e-5 with identical external RF noise.

## M30 — first real waveform path

M30 freezes the now-validated native acoustic path and connects it to the
voicebank's packaged `dsvocoder/nsf_hifigan.onnx` through ONNX Runtime. It
creates a deterministic audible Chinese-token smoke input, runs both native
acoustic and original ONNX acoustic with identical noise, passes both mel
outputs through the same vocoder, compares waveform parity, and writes standard
44.1-kHz PCM16 WAV files.

```bash
make m30-real-wave \
  MODEL_DIR=/path/to/voicebank \
  SPEAKER_EMB=/path/to/singer.emb
```

See `docs/M30_REAL_WAVEFORM.md`. The vocoder is deliberately not yet native in
M30; its ORT result becomes the golden reference for the next vocoder-porting
milestone.

## M32 — pure CPU/ASM vocoder Conv1d kernel lab

The final DiffSinger-ASM runtime is **CPU-only and self-hosted**: no oneDNN,
OpenVINO, BLAS, or ONNX Runtime backend is allowed in the production inference
path. M32 starts the NSF-HiFiGAN native port with a direct x86-64 AVX2/FMA
Conv1d kernel tiled as 4 output channels x 8 time samples, plus P-core
persistent-pool scheduling. ORT is used only offline to capture golden tensors
from the deployment vocoder; the timed kernel benchmark contains only our C
runtime and assembly.

```bash
make m32-check
make m32-real-kernel MODEL_DIR=/path/to/voicebank M32_ROUNDS=5
```

See `docs/M32_PURE_ASM_VOCODER.md`.

## M34 — stride2/K4 phase-vectorized ConvTranspose

M33 showed the late NSF-HiFiGAN upsamplers (`ups.2/3/4`) were the native
vocoder bottleneck: all are `stride=2, kernel=4, pad=1`, while the generic
M33 ConvTranspose vectorized over kernel taps. With only four taps it fell
entirely into the scalar FMA tail (~5 GFLOP/s on the target i5-13420H).

M34 keeps the generic direct-sparse kernel for `ups.0/1`, but automatically
selects a dedicated pure x86-64 AVX2/FMA phase kernel for `s2/k4/p1`.
For output pair n it uses:

```
y[2n]   = b + sum_ci(x[n]*w1 + x[n-1]*w3)
y[2n+1] = b + sum_ci(x[n]*w2 + x[n+1]*w0)
```

Eight time positions are accumulated at once into even/odd YMM vectors and
interleaved into 16 contiguous output samples. No inserted zeros, no ORT,
no oneDNN/OpenVINO/MKL/BLAS in the timed runtime.

## M35 — first complete pure CPU/C/ASM NSF-HiFiGAN executor

M35 adds a fixed-shape AOT compiler (`ONNX -> DSVOC35`) and a full native graph
executor. The timed runtime contains only our C scheduler/runtime plus x86-64
ASM kernels; ORT is pack-time/golden-only. The executor covers the exact 16-op
family in the deployed NSF-HiFiGAN and routes Conv/ConvTranspose/LeakyReLU/Add
to the M32–M34 AVX2/FMA kernels.

The first acceptance shape is 48 mel frames -> 24576 samples (44.1 kHz,
512-hop). `make m35-real-vocoder` validates full-vocoder parity and timing.
`make m35-real-e2e` additionally runs the 4-step native acoustic model and
prints the pure-native end-to-end RTF against the hard realtime target.


### M35.1
Real NSF-HiFiGAN AOT compatibility: adds `Unsqueeze` to the pure-native DSVOC35 executor and preflights unsupported ONNX operators before packing. No third-party inference backend is added to runtime.

## M36 — Realtime Sprint 2: oc8 Conv + fused Leaky + lifetime arena

M36 keeps the production path pure CPU/C/x86-64 ASM. ONNX Runtime remains
pack-time/golden-only.

Changes:
- new AVX2/FMA Conv1d microkernel `oc8 x time8` for vocoder convolutions with
  `Cout >= 64`; small layers retain the proven oc4 kernel;
- AOT peephole fusion of single-consumer `LeakyRelu -> Conv`: the Leaky
  transform is performed by an ASM NCT copy directly into the padded Conv
  workspace, eliminating the materialized activation and the second memcpy;
- linear-scan lifetime allocator for DSVOC35 work tensors instead of the M35
  append-only arena. The real 48-frame packer reports both `arena_naive_mib`
  and the planned `arena_mib`;
- real benchmark target sweeps 4/6/8 P-core workers from one packed bundle;
- 4-step acoustic + native M36 vocoder remains the end-to-end realtime gate.

Hard gate stays `CPU-only RTF < 1.0`; engineering target is `RTF <= 0.8`.

## M38 local verification

- residual-aware oc8 full and time-range ASM kernels compile and execute;
- synthetic long-sequence graph exercises fused Leaky+Conv+Residual under M37 2-D scheduling and static-OC fallback; outputs are bit-identical between the two schedulers;
- waveform parity remains within ~1.5e-7 max_abs on the synthetic graph;
- M29 acoustic deployment regression remains ~9.54e-7 max_abs;
- final runtime remains pure libc/libm/pthread + project C/ASM.

The real target packer reports `fused_residual_add`; the expected real NSF-HiFiGAN count is 45 internal residual-unit adds if the exported graph matches M37's topology. This count is verified on the user's machine by `m38-real-vocoder`, not assumed by the runtime.


## M39.1 hotfix
The original M39 source let GAS choose the encoding for `vpdpbusd ymm`, which produced EVEX/AVX-512-VNNI (`62...`) on the tested binutils. M39.1 explicitly prefixes every dot-product with `{vex}` so the object contains AVX-VNNI VEX encoding (`c4...`) suitable for AVX-VNNI CPUs without AVX-512. The test executable also checks CPU support and prints `SKIP` instead of raising SIGILL when AVX-VNNI is unavailable.
