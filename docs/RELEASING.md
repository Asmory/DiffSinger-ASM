# CI and release process

## Pull requests and main

`.github/workflows/ci.yml` runs on pull requests, pushes to `main`, and manual
dispatches. GCC and Clang both build the stable engine ABI and execute the core
assembly tests plus deterministic M35, M38, and M58 synthetic parity gates.
The GCC job also uploads a Linux x86-64 archive for integration testing.

Cloud runners are not used for RTF promotion. Their CPU topology, contention,
and power policy are not controlled tightly enough for the performance policy
in `docs/PERFORMANCE.md`. Run the real-model persistent E2E gate on the target
machine before creating a release tag.

## Release

1. Confirm the target-machine quality and performance gates.
2. Update release notes and version-facing documentation.
3. Create and push an annotated `v*` tag.

```sh
git tag -a v0.2.0 -m "DiffSinger-ASM v0.2.0"
git push origin v0.2.0
```

The tag starts `.github/workflows/release.yml`. It reruns all deterministic CI
gates, creates a reproducible `diffsinger-asm-<tag>-linux-x86_64.tar.gz`, writes
its SHA-256 file, and publishes both files to a GitHub Release. The workflow
uses the repository `GITHUB_TOKEN` with only `contents: write` permission.
Release binaries are compiled with an explicit AVX2/FMA baseline rather than
the repository's local-development `-march=native` default, so a hosted
runner's additional ISA features cannot leak into the artifact. Optional
AVX-VNNI kernels remain protected by runtime feature selection.

The archive contains:

```text
bin/dsasm-acoustic
bin/dsasm-vocoder-m40
lib/libdsasm.so
include/dsasm_engine.h
config/best-inference.env
README.md
docs/ENGINE_ABI.md
docs/PERFORMANCE.md
```

Build the same archive locally with:

```sh
make package VERSION=v0.2.0
```
