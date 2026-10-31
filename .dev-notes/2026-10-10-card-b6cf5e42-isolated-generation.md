---
id: b6cf5e42
title: "Execution: cut generation over to the isolated path (serve without ProcessGuard) — behind SLO_ISOLATED_INFERENCE, default off"
status: review
board_id: Main
date: 2026-10-10
branch: feat/isolated-generation
tags: [gene, execution, inference, pugqeep, apps-api, domain-inference]
---

# Card b6cf5e42 — isolated generation path (batch of 2026-10-10, story: c550503c)

Batch note: `2026-10-10-gene-seam-event-loop-starvation.md`.

## Status

**RESTARTED** by Gene batch 2026-10-14. Worktree `.wt-iso`, branch
`feat/isolated-generation` (base c6d95a9e2 = trunk-lane HEAD).

Restart facts (2026-10-14, coordinator-verified):

- The 2026-10-10 batch died mid-implementation: `.wt-seam` holds the stranded
  uncommitted diff (233 insertions: slonet_server.py dispatch + flag +
  `_generate_isolated`, engine.py `GuardPool.subprocess_for`, isolated.py
  `IsolatedCancelledError` + parent-side cancel watcher — the card's option
  (a)). Saved as
  `/home/mana/.cache/gene-batch-20261014/wt-seam-stranded-b6cf5e42.patch`
  and PORTED; `.wt-seam` itself is read-only for this batch.
- Isolation primitive (3e896b1cd = cherry-pick of ad3f2cb05) is NOT in trunk
  HEAD — ported first as its own commit.
- Stranded diff is UNTESTED and carries a verified wiring bug: the slonet
  call passes `cancel_event` positionally into `_generate_sync`, so
  `IsolatedExecutor.run`'s keyword-only `cancel_event` (the parent-side
  watcher) never receives it — mid-run cancel would not terminate the child.
  Fix = also pass `cancel_event=cancel_event` as a keyword, with a test.

Prior 2026-10-10 claim kept below for context.

## Status (2026-10-10 — superseded)

**Claimed** by Gene batch 2026-10-10. Worktree `.wt-seam`, branch
`fix/event-loop-seam`.

## Port-don't-rebuild

- Card c73aac23 (built, `.wt-gil`, `feat/gil-isolation`, commit **ad3f2cb05**):
  `packages/core-py/.../pugqeep/isolated.py` `IsolatedExecutor`,
  `InferencePool.run_isolated()`, `scripts/benchmark_isolated_execution.py`,
  14 tests. Benchmark: host stall 1007ms → 13ms (77.5x), task-time parity.
  NOT on main, NOT on our base branch → port `ad3f2cb05` first.
- Scope from the card: (1) route `SloNetServer.generate` through the isolated
  seam behind `SLO_ISOLATED_INFERENCE` **default off**; (2) cancel propagation
  — a parent `threading.Event` is invisible to the forked child; parent-side
  watcher terminating the child, or CANCEL frame over socketpair; (3)
  `generate_stream` stays thread-domain this pass — documented, streaming is
  the majority of chat traffic so the wedge risk is explicitly NOT closed.

## Must not break

Token-level stream semantics; circuit breaker + metrics in SloNetServer;
ProcessGuard autoload; startup budget; PGQ event-loop-free (zero asyncio
imports in pugqeep/); errors surfacing to the caller (no swallowed
exceptions, no silent fallback to inline execution).

## Coordination

- Prior deferral (.wt-guard 2026-10-10): slonet area contested —
  `tests/core-py/test_slonet_legacy.py` + `test_slonet_wave_a.py` dirty.
  Primary target `domain/infrastructure/_internal/slonet_server.py` CLEAN.
- Card df5affc7 (fork → forkserver) must agree before either lands —
  IsolatedExecutor depends on `os.fork()` memory inheritance.
- Card d3a4a079 (ProcessGuard heartbeat) must NOT touch isolation (this card).

## Done when

Flag off = byte-identical old path (existing tests green); flag on = generate
runs in child process, cancel terminates the child and surfaces to caller,
child errors raise; tests + benchmark green; ruff clean.

## LANDED (2026-10-15) — sha 93271ece6 on feat/isolated-generation

All Done-when gates green:

- **Tests**: 99 passed (18 new contract tests in
  `test_slonet_isolated_generate.py`: flag parsing, flag-off byte-identity
  via thread path, flag-on fork dispatch (pid proof), pre-start cancel,
  mid-run cancel terminating the child + surfacing CancelledError, child
  errors raising IsolatedExecutionError, guard bypass, stream stays
  thread-domain) + full `test_slonet_server.py` green.
- **Benchmark**: `benchmark_isolated_execution.py` — host stall
  1001.9ms → 13.5ms (74.2x), fork +15.2ms per trivial task, task-time
  parity. Matches the primitive's original numbers (77.5x, card c73aac23).
- **ruff** check + format clean.

Review found and fixed 4 defects in the stranded-diff reconstruction:

1. **Double-load identity split** — the legacy
   `domain.infrastructure._internal.pugqeep` package `__init__` shim
   copies `__path__` from the canonical package, so the legacy import
   path loads the SAME file as a SECOND module object: two class
   identities (pytest.raises could not catch) and two executor
   singletons (stats read from an empty twin). Cut-over now imports the
   canonical `domains.infrastructure.pugqeep.isolated`. **47 files still
   import the legacy path** — systemic pre-existing trap; only
   isolated-module consumers are affected today
   (`apps/api/server/infrastructure/inference_pool.py` is one). Separate
   follow-up card recommended.
2. Test-file mocks returned one decode string for prompt AND generation
   → the 30192039d prompt-echo strip ate entire results (7 of 18 red).
3. Pre-start cancel test expected RuntimeError; `generate()` has raised
   CancelledError since a170a7694 (BaseException — breaker/metrics
   correctly skip it).
4. Stream test lacked `generate_numpy_stream` on the model mock.

Also repaired 5 PRE-EXISTING red tests in `test_slonet_server.py` (red on
base c6d95a9e2 — stale mocks from the same 30192039d echo-strip commit).
They now document the strip contract; `_FakeGuard` returns a full decode
like the real worker.

Still open on this card (deliberately, per card scope): streaming wedge
risk (stream majority of chat traffic — needs its own isolation
decision), and the flag is a raw env var (`SLO_ISOLATED_INFERENCE`), not
a registry flag — reconciliation with `process_isolation` rides the
flags card memo (d94aea1c) decision.
