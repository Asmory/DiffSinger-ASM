# ABI v3 stage-aware cancellation candidate

Date: 2026-09-22
Host: `kisaragi`, Intel Core i5-13420H
Gate: p99 cancel-to-return at most 371.52 ms
Result: PASS

Profile revision 2 checks the engine cancellation epoch after FS2, after the aux
decoder, after Rectified Flow conditioner projection, and before and after each
Euler denoiser step. Neural arithmetic and the successful-render numerical
path are unchanged.

Every measured cancellation returned `DSASM_E_CANCELLED` and emitted zero PCM
callbacks. The primary repeated runs used 5 warmups and 100 measured requests:

| Mode | Delay | Samples | p50 | p90 | p99 | Worst | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Realtime streaming, 4 workers, 32 frames | 10 ms | 100 | 3.535 ms | 12.253 ms | 19.007 ms | 20.381 ms | PASS |
| Block batch, 8 workers, 384 frames | 10 ms | 100 | 41.502 ms | 50.551 ms | 58.225 ms | 58.691 ms | PASS |

Additional batch phase samples exercised later cancellation points:

| Delay | Samples | p50 | p90 | p99 | Worst | Gate |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 100 ms | 25 | 131.063 ms | 166.953 ms | 176.767 ms | 178.101 ms | PASS |
| 250 ms | 25 | 29.536 ms | 131.201 ms | 210.443 ms | 214.869 ms | PASS |

The raw sample files contain the exact commands' results. `lscpu.txt`,
`uname.txt`, `source-head.txt`, and `sha256.txt` fingerprint the host, source
base, measurement binary, runtime library, and model artifacts.

Correctness validation:

- `make -j8 engine-check build/test_engine_stream` passed.
- `make engine-real-stream-check` passed on the archived real model with 17
  contiguous callbacks, 204800 samples, and one final callback.
- Current-object PyTorch parity passed for M18 reflow, M20 post-FS2, M21 full
  acoustic, and M22 normfast. M22 reported zero difference between its old and
  new successful-render paths.
- The repository's old milestone shared-library link recipes no longer include
  all objects required by the current threadpool. Parity was therefore run
  against an unversioned validation library linked from the current
  `PRODUCT_OBJS`; the shipped version-scripted library remains unchanged.

This evidence supports the provider's 371.52 ms offer for the fingerprinted
fixed-region strata. It is not an archived contract. Other devices and longer
requests must pass their applicable gate before relying on the same bound.
