# M53 true long-audio E2E

Moves the benchmark away from the 48-frame / 0.557 s micro-utterance.

Default lengths:
- 48 frames = ~0.557 s
- 192 frames = ~2.229 s
- 384 frames = ~4.458 s

The demo phoneme sequence itself is repeated, so both native acoustic `T` and native vocoder shape grow. Each measured vocoder request uses `--rounds 1`; ORT golden generation happens only during preparation. P4 physical cores are discovered dynamically because M52 identified P4 as the best topology.

Optional 8.9 s test:
```bash
M53_FRAMES='48 192 384 768' PROJECT=~/asm/diffsinger bash run_m53_long_e2e_allinone.sh
```
