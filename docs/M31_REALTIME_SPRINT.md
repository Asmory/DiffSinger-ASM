# M31 — CPU Realtime Sprint 1

The product requirement is now explicit: on the target Intel Core i5-13420H,
**CPU-only tokens/conditioning -> WAV must run at RTF < 1.0**.  The engineering
target is `RTF <= 0.8` so longer utterances and system jitter have headroom.

M30 established the correctness baseline on the real DongFangZhiZi Nectar
voicebank: the native acoustic mel and the original deployment ONNX produce
nearly identical waveforms through the bundled NSF-HiFiGAN.  M31 therefore does
not change model math.  It is a performance/quality Pareto milestone.

## Native stage timing

`dsasm-acoustic infer` adds `--profile-stages`.  With that flag the inference is
executed as the same normalized fast path but with explicit timers around:

1. FS2 conditioner
2. shallow Aux ConvNeXt
3. Rectified-Flow decode

The output is still the actual inference output.  The M31 smoke test compares a
profiled invocation against the old full wrapper and requires bit-exact output.

Example output:

```text
stages: fs2=... ms aux=... ms rf=... ms rf_per_step=... ms sum=... ms
```

## Real-time Pareto harness

`tools/realtime_sprint_m31.py` uses one deterministic real-voicebank input and
one external noise tensor.  It performs:

- native RF step sweep: `20,16,12,10,8,6,4`
- symmetric/reversed acoustic measurement order to reduce simple phase bias
- NSF-HiFiGAN ORT thread sweep: `1,2,4,6,8`
- process affinity to target P-core logical CPUs `0,2,4,6,1,3,5,7`
- 20-step native mel/waveform as the quality reference
- per-step mel and waveform max-abs/RMSE/cosine/SNR
- standard WAV output for every RF step count
- vocoder static ONNX operator/parameter summary
- ORT node-level profiling at the best measured vocoder thread count

The harness prints and stores end-to-end budget RTF for every RF step count.  A
configuration is *speed-real-time* only when `RTF < 1.0`; `RTF <= 0.8` is the
engineering goal.  Waveform similarity numbers are diagnostics, not a claim of
perceptual equivalence; the generated WAV files are retained for listening.

## Target

```bash
make m31-real-sprint \
  MODEL_DIR=/path/to/voicebank \
  SPEAKER_EMB=/path/to/singer.emb
```

Main output: `build/m31_realtime/m31_realtime_report.json`.
