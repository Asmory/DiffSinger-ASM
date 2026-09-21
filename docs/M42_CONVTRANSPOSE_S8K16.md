# M42 — stride-8 / K16 ConvTranspose1d ASM

M41 measured the two early NSF-HiFiGAN upsamplers as the largest individual
vocoder operators at 8 workers: `256->128 K16 T384->3072 stride=8 pad=4` and
`512->256 K16 T48->384 stride=8 pad=4`.

The previous generic kernel loops over `(Cin,Tin)` and scatters each input sample
into 16 output samples. That repeatedly read/modify/writes the same output for
every input channel. M42 uses the exact stride-8/K16 geometry to invert that
loop: output samples stay in XMM registers across the full `Cin` reduction and
are stored once. Four adjacent 8-sample output blocks are computed together so
a 16-float weight row is loaded once for 32 outputs.

`DSASM_CONVT_S8K16=0` disables the new kernel for A/B testing. It defaults on.
`scripts/run_m42_s8k16.sh` performs direct-kernel parity, real-model generic vs
specialized profiling at 8 workers, and one native end-to-end run.
