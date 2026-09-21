# DiffSinger-ASM agent instructions

## Required project policy

Before changing runtime or benchmark code, read:

- `docs/DUAL_ARCHITECTURE_OPTIMIZATION.md`
- `docs/STREAMING_PERFORMANCE.md`
- `docs/BATCH_PERFORMANCE.md`
- `docs/RUNTIME_IMPLEMENTATION_POLICY.md`

Hand-written CPU neural-network arithmetic should use x86-64 assembly for hot
kernels. oneDNN and GGML are also approved runtime backends, and GPU execution
frameworks may be introduced. C/C++ integration code may construct graphs,
invoke approved backends, and handle model loading, validation, dispatch,
scheduling, buffers, cancellation, and callbacks; do not add hand-written
scalar C/C++ neural kernels as a substitute for an approved backend. Python is
offline-only for conversion, golden/parity validation, and benchmark analysis.
Unsupported graph branches must fail explicitly. This backend permission is
the authoritative override when an older project document says ASM-only.

## Tooling and dependency policy

Provision what the task requires. If a correct, high-quality implementation or
validation needs a compiler, profiler, debugger, library, model, SDK, data set,
system package, service, or other component that is missing, request the needed
privilege or network access promptly, install or download it, and continue with
the intended method. Treat provisioning required dependencies and instruments
as part of completing the task. When the execution environment provides an
escalation mechanism, use it directly instead of merely suggesting that the
user run the privileged step.

Do not silently reduce scope, weaken validation, skip measurements, or replace
the intended implementation with an inferior workaround because the current
environment lacks a tool or permission. A sandbox, permission, or network
failure is a reason to request escalation, not evidence that a lower-quality
method is acceptable. If access is denied or the required component remains
unavailable after escalation, report the concrete blocker and its effect.

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

Before streaming reaches stable worst RTF below 1 with zero deadline misses,
promote only stable steps that reduce the largest worst RTF by at least 5%; do
not use median RTF. After crossing that target, switch the 5% staircase to the
largest CPU-RTF while preserving the service target.

ASM, oneDNN, GGML, and GPU results are separate workload strata whenever their
backend, device, precision, graph partition, or scheduling contract differs.
Record backend and device fingerprints, and never promote a candidate by
comparing measurements collected from unlike strata.

## Language rules

- Prefer ASM for hand-written CPU hot kernels.
- oneDNN and GGML are approved CPU/runtime backends.
- GPU frameworks are approved runtime backends; their required toolchains and
  kernel languages are allowed within that backend.
- Use C/C++ for OS integration, scheduling, graph/backend integration, and
  arranging ASM kernels, not for hand-written scalar neural arithmetic.
- Use Python only for offline model conversion, correctness validation, golden
  generation, benchmark analysis, and development scripts.
- Use shell only for small, convenient tooling scripts.
- Do not introduce another online inference language or backend without an
  explicit project decision.
