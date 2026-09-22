# DiffSinger-ASM v0.2.2

Version 0.2.2 corrects the ABI v3 canonical cache declaration. It does not
change inference arithmetic, packed model formats, converter revision, or the
offline model protocol schema.

## Fix

- `DSASM_MODE_REALTIME_STREAMING` now reports
  `output_compatibility_revision = 1`, matching `DSASM_MODE_BLOCK_BATCH`.
- The revision is a cache protocol declaration rather than a quality grade.
  Both modes already pass the provider's quality gates.
- A complete batch pre-render can satisfy later playback, and a complete
  real-time render can satisfy later pre-render, mixdown, or export.
- Mode and profile revision remain provenance and performance diagnostics. They
  do not split a canonical PCM entry when the nonzero output compatibility
  revision and the rest of the canonical render identity match.

Partial, cancelled, stale, callback-aborted, non-finite, and incomplete output
remains ineligible for canonical cache commit.
