# DiffSinger-ASM v0.2.1

Version 0.2.1 hardens the protocol-v1 offline model tool without changing ABI
v3, packed model bytes, fingerprints, or the `m25-m35-v1` converter revision.

## Fixes

- Each acoustic or vocoder packer now runs in an isolated process group.
  SIGINT is forwarded to that group, which is reaped before the tool returns;
  an unresponsive group is killed after two seconds.
- Candidate manifest data passes the complete offline validator before
  `bundle.json` is atomically installed. A malformed internal candidate can no
  longer appear as a complete staging bundle, even briefly.
- Permission, read-only filesystem, disk-space, quota, and general I/O failures
  use the protocol's stable reason codes and exit 6 instead of falling through
  to `internal.error`.
- Validation rejects a non-object manifest, missing provider identity, malformed
  fingerprints, and a symlink used as the bundle root.
- Conversion rejects overlapping staging and work paths, ordinary files in
  place of either directory, and nonempty staging with `request.invalid`.
- Acoustic and vocoder parse failures retain their respective stable reason
  families. Plan output now derives identity and space estimates from the same
  source-closure scan.

## Verification

- All 26 offline protocol regression tests pass.
- Yousa V1.56 still produces source fingerprint
  `e456cf48bdfafee3b9568d79bdff3aba1fe5601bf113b545fff50704be8e2875`
  and bundle fingerprint
  `29bc1b9164c330fd8bf4c6ad28fa2ea97ea92224ef39c8fb383e65e62c3e96ed`.
- A real conversion interrupted while a packer process existed returned exit 7,
  left no `bundle.json`, and left neither wrapper nor child process running.
- A deterministic 60-second child-process test confirmed that signalling only
  the wrapper still reaps the isolated child process group.

## Known Limits

- Real-time output compatibility revision remains zero, so streaming PCM is
  session-only.
- Fixed-region cancellation evidence does not establish a complete long-phrase
  native render bound.
- The shared offline protocol remains an implementation draft until consumer
  acceptance is complete.
