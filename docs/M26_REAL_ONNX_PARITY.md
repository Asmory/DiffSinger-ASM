# M26 — real ONNX vs native parity

M26 freezes the M25.2 importer and validates the packed runtime against the original deployment ONNX with identical inputs.

The validator patches a copy of `acoustic.onnx` only for validation:

1. expose internal `condition` and `aux_mel` as graph outputs;
2. replace the single top-level `RandomNormalLike` with an external `noise_external` graph input;
3. feed ONNX Runtime and `dsasm-acoustic` the same tokens, languages, durations, f0, variance curves, gender, velocity, speaker embedding, depth, steps, and FP32 noise;
4. compare `condition`, `aux_mel`, and final `mel` using max_abs / max_rel / RMSE / cosine.

The original model file is never modified. The patched graph is written under the M26 work directory.

The native CLI adds validation-only options:

- `--dump-condition FILE.f32`
- `--dump-aux-mel FILE.f32`

Normal runtime/model ABIs and packed bundle formats are unchanged.
