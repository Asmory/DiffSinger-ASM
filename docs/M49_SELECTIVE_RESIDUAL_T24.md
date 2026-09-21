# M49 selective residual-store t24 fusion

M38 fused every eligible Conv->Add residual merge into the old oc8xt8 store
path and regressed whole-vocoder throughput. M49 revisits the idea only after
M45/M46/M48 established lower-instruction oc4xt24 kernels.

Production experiment `--residual-scope k7ge128` fuses only K=7 Conv->Add
pairs with Cout>=128 and T divisible by 24. These shapes stay on the full
channel-owner path at 8 workers and local A/B showed positive fused-store
microbench results. C64/C32/C16 and K3/K11 remain unfused.

DSVOC35 flag bit 8 means the residual tensor id is stored in p[8]. This leaves
`reserved` available for the M40 VNNI blob, so VNNI and graph residual fusion
can coexist. If a fused op executes through VNNI, runtime performs the residual
add after VNNI; FP32 t24 paths use the new fused residual ASM kernel.

Runtime switch: `DSASM_RESIDUAL_T24=1`.
