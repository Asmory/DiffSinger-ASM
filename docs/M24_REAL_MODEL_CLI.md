# M24 — real checkpoint packer + native acoustic CLI

M24 promotes the M23 three-bundle loader into a usable model directory and CLI.
It targets the **current default DiffSinger acoustic profile** implemented by M21:

- B=1 inference;
- no language/speaker/variance/key-shift/speed optional embeddings;
- FastSpeech2 hidden 384, 2-head RoPE (`rope_interleaved=false`), FFN Conv1d k=3;
- stretch embedding + GRU and pitch embedding;
- shallow ConvNeXt aux decoder;
- Rectified Flow + LYNXNet2, Euler sampling.

## Pack one checkpoint

```bash
python tools/pack_acoustic_model_m24.py model.ckpt \
  --config path/to/acoustic.yaml \
  --out packed/model
```

Output:

```text
packed/model/
  fs2_acoustic.dsfs
  aux_convnext.dsa
  lynxnet2.dsn
  model.conf
  model.json
  fs2_acoustic.json
  aux_convnext.json
  lynxnet2.json
```

`--config` is strongly recommended. M24 reads the sampling/spec-range settings
and rejects optional branches that the current native default-profile runtime
cannot reproduce yet. Without a config, packing is allowed but this validation
cannot be performed.

## Native CLI

Inspect:

```bash
build/dsasm-acoustic inspect packed/model
```

Infer:

```bash
build/dsasm-acoustic infer packed/model \
  --tokens tokens.txt \
  --durations durations.txt \
  --f0 f0.txt \
  --seed 1234 \
  --out mel.f32
```

`tokens.txt` and `durations.txt` contain one integer per phoneme/token.
`sum(durations)` must equal the number of frame-level values in `f0.txt`.
The CLI expands durations into DiffSinger's 1-based `mel2ph` mapping.

For parity testing, supply exact raw FP32 Gaussian noise instead of the native RNG:

```bash
build/dsasm-acoustic infer packed/model ... --noise noise.f32 --out mel.f32
```

Output is raw contiguous little-endian FP32 `[frames, mel_bins]`.

This is a **numeric acoustic frontend**, not a `.ds`/phoneme-dictionary parser.
Phoneme text to token IDs and curve resampling remain a higher-level frontend task.

## M25 acceptance harness
`tools/validate_real_model_m25.py` packs one supported checkpoint, evaluates an independent checkpoint-tensor PyTorch reference, runs the native CLI with the exact same noise, and compares final raw mel.
