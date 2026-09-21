# M19 — Native acoustic decoder envelope

M19 wraps the M18 Rectified Flow Euler sampler at the same logical boundary as
current DiffSinger `DiffSingerAcoustic.forward()` after FastSpeech2 and the
shallow aux decoder, and before the vocoder.

## Upstream order reproduced

For shallow reflow inference the current model does:

1. FastSpeech2 emits `condition [B,T,H]`.
2. The aux decoder emits raw mel `aux_mel [B,T,128]`.
3. `aux_mel *= (mel2ph > 0)`.
4. RectifiedFlow normalizes `aux_mel` using `spec_min/spec_max`.
5. Shallow init: `x = T_start * src_norm + (1-T_start) * noise`.
6. Euler, 20 steps by the default acoustic config:
   `x += velocity_fn(x, 1000*t, condition) * dt`.
7. RectifiedFlow denormalizes the result.
8. `mel *= (mel2ph > 0)` again.

`ds_acoustic_reflow_decode_f32_avx2()` performs steps 3–8 natively.  Its
`condition_tc` input is frames-major `[T,Q]`, which corresponds directly to the
upstream FastSpeech2 output `[B,T,H]`; callers do not need to expose the
internal Python `condition.transpose(1,2)` layout.

## Current official defaults

- mel bins: 128
- condition hidden size: 384
- LYNXNet2 channels: 1024
- layers: 6
- `spec_min = [-12]`
- `spec_max = [0]`
- `sampling_algorithm = euler`
- `sampling_steps = 20`
- `T_start_infer = 0.4`
- `time_scale_factor = 1000`

The runtime supports scalar-broadcast or per-bin `spec_min/spec_max`.

## API boundary

Input:

- `condition_tc [T,Q]`
- `aux_mel_tc [T,D]` in raw mel domain
- `noise_tc [T,D]` in normalized RF domain
- `frame_mask_t [T]`, normally 0/1
- normalization range and RF schedule

Output:

- `output_mel_tc [T,D]`, raw/denormalized mel, padding frames exactly zero

FastSpeech2, the aux decoder, RNG choice, and the vocoder are deliberately not
implemented in this milestone.

## Correctness gates

`tools/validate_pytorch_acoustic.py` covers:

- full diffusion (`T_start=0`)
- shallow diffusion (`T_start=0.4`)
- bypass (`T_start=1`)
- scalar range `[-12,0]`
- per-mel-bin ranges
- frame-mask ordering and exact zero padding
- 6-layer / 128-bin medium shape
- official 64-frame / 1024-channel / 20-step benchmark
