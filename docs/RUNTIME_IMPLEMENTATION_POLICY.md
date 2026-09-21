# CPU runtime implementation policy

The online CPU inference runtime has a strict implementation boundary:

- Assembly implements neural-network arithmetic kernels.
- C loads validated packed data, selects shapes and kernels, schedules worker
  jobs, manages buffers, handles cancellation, and publishes PCM callbacks.
- Python is offline-only and may inspect or convert model containers, generate
  fixtures, compare against golden outputs, and analyze benchmark evidence.
- Python and other language runtimes must never be linked, embedded, or invoked
  by the online inference library.

New model features such as adaptive LayerNorm or energy conditioning require an
assembly kernel before runtime support can be declared. A C reference may exist
only in offline tests and correctness tooling; it is not an acceptable product
fallback. Unsupported model graphs must fail conversion with an explicit
diagnostic instead of silently dropping a network branch.

Performance specializations are separated by fixed shape. Small-T real-time
kernels and large-block batch kernels may use different assembly, packing, and
thread schedules. C dispatches between them using validated model metadata and
the explicit ABI bucket size.
