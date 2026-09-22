# Manual validation and release process

## Validation

This repository does not run hosted CI or continuous deployment. Before a
push or release, run the stable engine ABI build, core assembly tests, and the
deterministic synthetic parity gates locally:

```sh
make -j"$(nproc)" engine-check
make -j"$(nproc)" test
make -j"$(nproc)" m35-check m38-check m58-check
```

RTF promotion must run on the fingerprinted target machine because hosted CPU
topology, contention, and power policy are not controlled tightly enough for
the performance policy in `docs/PERFORMANCE.md`.

## Release

1. Confirm the target-machine quality and performance gates.
2. Update release notes and version-facing documentation.
3. Build the archive and checksum locally.
4. Create and push an annotated `v*` tag and publish the archive manually.

```sh
git tag -a v0.2.0 -m "DiffSinger-ASM v0.2.0"
git push origin v0.2.0
```

Create the archive locally with:

```sh
make package VERSION=v0.2.0
```

Release builders that need a portable binary must pass an explicit AVX2/FMA
baseline instead of the local-development `-march=native` default. Optional
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
