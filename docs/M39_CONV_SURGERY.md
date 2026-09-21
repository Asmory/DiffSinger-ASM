# M39 Conv surgery

Target-machine facts entering M39:

- M37 4-worker full native vocoder: ~415.7 ms, RTF ~0.746.
- Ordinary Conv: ~344.8 ms, the dominant remaining cost.
- M38 fused residual-store reduced standalone Add work but regressed Conv throughput, so that fusion is no longer the default.

M39 therefore keeps M37's graph semantics and makes every new optimization independently switchable for A/B. `DSASM_FULL_MEMSET=1` restores the old workspace clear; `DSASM_KSPEC=1` enables fixed-K3/K7/K11 kernels. The VNNI path is a separate feasibility benchmark and does not alter normal vocoder output.

The AVX-VNNI lab packs four reduction elements into each dword lane. `vpdpbusd` accumulates U8 activations times S8 weights into int32, applies the U8 zero-point correction, converts to FP32 and applies activation_scale * per-output weight_scale plus the original bias. Real-layer testing must judge both total time including dynamic activation packing and waveform-relevant numerical quality.
