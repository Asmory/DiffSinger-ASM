# DiffSinger-ASM v0.2.0

Version 0.2.0 turns the experimental persistent engine into a provider package
with explicit OpenUtau boundaries for online rendering and offline model
preparation.

## Highlights

- ABI v3 exposes separate real-time streaming and block batch engines. The
  queried profiles use 4 workers with a 32-frame bucket and 8-frame overlap for
  playback, and 8 workers with a 384-frame bucket for complete rendering.
- Profile revision 2 observes cancellation between acoustic stages, before and
  after every Rectified Flow Euler step, and between vocoder buckets.
- `dsasm-model-tool` implements protocol v1 `inspect`, `plan`, `convert`, and
  `validate` operations with JSON/JSONL results, stable reason codes, bounded
  space estimates, source fingerprints, and authoritative bundle manifests.
- The Linux x86-64 archive embeds CPython 3.12 and exact hash-locked offline
  dependencies. It does not require a source checkout or repository virtual
  environment.
- Offline validation is independent from native runtime availability and checks
  complete packed payload bounds, cross-file dimensions, both product buckets,
  paths, digests, format revisions, and bundle identity.

## Compatibility

ABI v3 is intentionally incompatible with the v0.1 ABI. Consumers must query
both mode configurations, initialize 200-byte requests with the selected mode,
and retain separate engines for streaming and batch work. Product bundles must
contain `vocoder/32.dsv35` and `vocoder/384.dsv35`.

The offline converter revision is `m25-m35-v1`, and the default ISA profile is
`avx2-fma`. Reusing PCM requires matching source and bundle fingerprints plus a
nonzero output compatibility revision.

## Verified Provider Evidence

- ABI v3 layout and runtime purity checks pass.
- Fixed-region cancellation passes the proposed 371.52 ms p99 gate on the
  fingerprinted Intel Core i5-13420H reference target.
- Yousa V1.56 converts into both product buckets and validates with source
  fingerprint
  `e456cf48bdfafee3b9568d79bdff3aba1fe5601bf113b545fff50704be8e2875`
  and bundle fingerprint
  `29bc1b9164c330fd8bf4c6ad28fa2ea97ea92224ef39c8fb383e65e62c3e96ed`.
- Both product modes create and destroy directly from the validated staging
  paths.
- SIGINT cancellation returns exit 7 without a valid staging manifest, and a
  stale expected source fingerprint returns exit 9 before work begins.

## Known Limits

- Real-time output compatibility revision remains zero. Streaming PCM is
  session-only and cannot populate the canonical cache.
- The fixed 32/384-frame cancellation evidence does not establish a complete
  long-phrase bound or transfer to an unmeasured target device.
- The active offline protocol remains an implementation draft until the
  OpenUtau consumer build, publication, recovery, cache invalidation, and real
  singer acceptance gates all pass. The provider package is usable without
  prematurely archiving that shared contract.
- The product target remains Linux x86-64 with AVX2 and FMA. Optional AVX-VNNI
  paths are runtime selected.
