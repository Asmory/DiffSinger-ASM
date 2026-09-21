# M51 — clean E2E / cold-start diagnosis

M51 intentionally does not change the C/ASM kernels. It fixes the benchmark methodology first:

- ORT vocoder golden is generated once during fixture preparation.
- Every measured E2E run is strictly native acoustic -> native vocoder, with no ORT inference between them.
- The acoustic output is checked bit-for-bit against the prepared reference before reusing the golden waveform.
- Four forced-cold conditions are compared in balanced order: cold, page prefault, CPU ramp, prefault+ramp.
- File cache state is controlled with `posix_fadvise(POSIX_FADV_DONTNEED)` where supported.
- CPU ramp is a short AVX2/FMA helper restricted to the detected P-core logical CPUs.
- Startup prep cost is reported separately from native compute RTF.

Run via `run_m51_allinone.sh`.
