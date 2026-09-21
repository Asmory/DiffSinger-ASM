# M15 — K-blocked ATan Linear

M14 showed that reducing generic cache references did not materially reduce
P-core cache misses or cycles on the target i5-13420H. M15 therefore changes
*when* a packed16 weight block is reused rather than changing arithmetic.

For one 16-output block with K=1024, the legacy 4x16 kernel streams the full
64 KiB weight block for each four-row group. M15 splits K into 128/256/512
chunks. A chunk is consumed across every row in the task before advancing, so
a 512-wide chunk is 32 KiB of weights and is much more L1-friendly.

The partial-accumulate kernel preserves exact FMA order. The first K chunk
starts from bias; subsequent chunks reload the previous FP32 accumulators and
continue K in increasing order. Kernel and whole-forward A/B require max_abs=0
against the legacy path.

The feature is experimental and disabled by default. Use
`DSASM_KBLOCK_ATAN=1 DSASM_K_BLOCK=512` or the public thread-pool setters.
