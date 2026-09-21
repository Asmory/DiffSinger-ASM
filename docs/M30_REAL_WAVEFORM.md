# M30 — real voicebank waveform smoke

M29 establishes real `acoustic.onnx` parity for the DongFangZhiZi deployment
model: packed-native condition/aux/final-mel tensors match ONNX to roughly
FP32 accumulation noise.  M30 therefore freezes the acoustic math and extends
the acceptance path to the voicebank's packaged NSF-HiFiGAN vocoder.

The M30 boundary is intentional:

```
voicebank acoustic.onnx -> M29 importer -> native AVX2 acoustic -> mel
                                                       |
voicebank dsvocoder/nsf_hifigan.onnx <-----------------+
                  ORT CPU -> waveform -> WAV
```

The vocoder remains ONNX Runtime in M30.  This gives a waveform golden
reference before attempting a separate native/oneDNN/ASM vocoder port.

`tools/run_real_voicebank_m30.py` also patches the original acoustic ONNX in the
same way as M26-M29 so that its `RandomNormalLike` consumes the exact same
external noise as native inference.  Both mel tensors are then passed through
the **same vocoder session**.  It writes:

- `native_mel.f32`
- `onnx_mel.f32`
- `native_waveform.f32`
- `onnx_waveform.f32`
- `native_pipeline.wav`
- `onnx_reference.wav`
- `m30_report.json`

The standard OpenVPI NSF-HiFiGAN deployment interface is expected to contain
`mel`, `f0` inputs and a `waveform` output.  M30 inspects shapes at runtime and
supports `[B,T,M]` and `[B,M,T]` mel layouts plus ordinary 1D/2D/3D f0 layouts.
Unexpected extra vocoder inputs are rejected instead of silently fabricated.

Run:

```bash
make m30-real-wave \
  MODEL_DIR=/path/to/voicebank \
  SPEAKER_EMB=/path/to/singer.emb \
  LANGUAGE_ID=4 DEPTH=0.6 STEPS=20
```

The demo input is deterministic and prefers a simple Chinese sustained-vowel
sequence from `phonemes.json`, falling back to valid Chinese tokens when the
inventory uses different spellings.  It is an acoustic/vocoder engineering
smoke, not a frontend/lyrics-to-score implementation.
