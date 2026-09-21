# DiffSinger-ASM agent instructions

## Required project policy

Before changing runtime or benchmark code, read:

- `docs/DUAL_ARCHITECTURE_OPTIMIZATION.md`
- `docs/STREAMING_PERFORMANCE.md`
- `docs/BATCH_PERFORMANCE.md`
- `docs/RUNTIME_IMPLEMENTATION_POLICY.md`

Online CPU neural-network arithmetic must be assembly. C is limited to model
loading, validation, dispatch, scheduling, buffer ownership, cancellation, and
callbacks. Python is offline-only for conversion, golden/parity validation, and
benchmark analysis. Unsupported graph branches must fail explicitly.

## Skill routing

Skills are installed under `/home/fuurin/.agents/skills/`. Read the complete
`SKILL.md` for every applicable skill before acting.

- Performance profiling or optimization: first read
  `skills/performance-gradient-optimization/SKILL.md`, then
  `/home/fuurin/.agents/skills/profiling/SKILL.md`.
- Crashes, incorrect output, races, memory faults, or flaky behavior: read
  `/home/fuurin/.agents/skills/debugging/SKILL.md`.
- Test design or coverage work: read
  `/home/fuurin/.agents/skills/testing/SKILL.md`.
- Code review: read `/home/fuurin/.agents/skills/code-review/SKILL.md`.
- Commits, tags, branches, or recovery: read
  `/home/fuurin/.agents/skills/git-cli/SKILL.md`.
- Baseline archives and reproducibility: explicitly read
  `/home/fuurin/.agents/skills/versioning-reproducibility/SKILL.md`.

The repository copy of `performance-gradient-optimization` is authoritative for
this project. Keep the installed Claude copy synchronized when it changes.

## Performance invariants

Never guess hotspots. Select candidates from measurements on the exact target
workload, and use instrumented runs for localization plus separate paired
uninstrumented runs for promotion.

Real-time streaming and block batch rendering have independent best baselines,
gates, tags, and archives. Compare a candidate only with the historical best
for the same architecture and workload stratum. Promote and archive only after
the architecture's E2E gate passes.
