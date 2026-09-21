# M44 asymmetric VNNI + quality-gated stage admission

M43 established three target-machine facts:

- parallel residual Add is a regression and must stay off by default;
- broad `all-k711` AVX-VNNI is much faster but fails the waveform quality gate;
- selecting a wider vocoder worker count from one long sweep can be polluted by
  thermal/frequency state, so M44 keeps the search small and re-runs finalists.

The M40 VNNI activation quantizer used a symmetric U8 mapping with a fixed zero
point of 128.  Fused LeakyReLU activations are strongly asymmetric, so this
wastes code points on the small negative tail.  M44 adds
`DSASM_VNNI_ASYM=1`: the runtime measures the post-LeakyReLU min/max, maps the
full interval to U8 0..255, encodes padding with the dynamic zero point, and
recomputes the integer zero-point correction before the unchanged AVX-VNNI
kernel.  `DSASM_VNNI_ASYM=0` is bit-compatible with M40.1.

An `all-k711` bundle can now be restricted at runtime with
`DSASM_VNNI_CIN=128,64,...`.  M44 starts at the known-safe 128-channel stage,
tries adding 256/64/32/16 stages one at a time, and admits a stage only when the
whole real vocoder still passes cosine >= 0.999 and SNR >= 25 dB.  The winning
stage mask is then re-tested with k11/k117 and 8/10 workers before the real
acoustic + vocoder E2E run.
