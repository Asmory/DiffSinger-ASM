# M18 — Rectified Flow Euler sampler

M18 moves the project from a single LYNXNet2 velocity/denoiser forward into the
current OpenVPI DiffSinger acoustic sampler loop.

The implementation follows `modules/core/reflow.py` on current upstream:

- Euler: `x += velocity_fn(x, time_scale_factor * t, cond) * dt`
- `dt = (1 - T_start) / max(1, sampling_steps)`
- shallow init: `x = T_start * x_end + (1 - T_start) * noise`
- current acoustic defaults: `T_start=0.4`, `time_scale_factor=1000`,
  `sampling_algorithm=euler`, `sampling_steps=20`
- conditioner projection is cached once and reused across every solver step.

## Native API

`include/dsasm_reflow.h` adds:

- `ds_reflow_norm_spec_f32`
- `ds_reflow_denorm_spec_f32`
- `ds_reflow_euler_sample_cached_condition_f32_avx2`
- `ds_reflow_euler_sample_f32_avx2`

The sampler consumes caller-provided Gaussian noise. RNG is deliberately kept
outside the sampler so deployment may choose a native RNG while parity tests can
feed the exact same noise to PyTorch and C.

## CPU freeze

Repeated i5-13420H measurements showed M15-M17 K-block/N-owner experiments are
state-dependent. M18 therefore freezes the default CPU denoiser on:

- P-core-aware 8-worker pool
- 2-D adaptive scheduling
- tile-local ATan pipeline
- serial depthwise
- condvar dispatch
- dynamic full-K ATan Linear

K-block, N-owner, indexed Linear, spin dispatch, and parallel depthwise remain
available as explicit experimental controls.
