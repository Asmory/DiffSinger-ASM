# M55 long-audio memory-pass sprint

Target: 4.458 s / 384-frame sustained vocoder.

M54 showed ~205.7 ms standalone Add and ~2.006 s Conv. Fused-leaky Conv still performs a serial leaky+pad copy before each convolution. M55 adds an opt-in channel-parallel leaky-copy job and re-tests parallel same-shape Add on long tensors.

Switches:
- `DSASM_PARALLEL_ADD=1`
- `DSASM_PARALLEL_LEAKY_COPY=1`
- `DSASM_PARALLEL_LEAKY_MIN=<floats>` (default 262144)

Release defaults remain OFF until target-machine A/B proves a win.
