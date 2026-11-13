---
id: df5affc7-0d8b-4483-a46a-baf2df495766
title: "L2: SubprocessProcess start method fork -> forkserver (kill BLAS fork-deadlock class)"
status: done
board_id: gene-batch-20261010
date: 2026-10-10
branch: feat/pgq-codec-dispatch
tags: [testing, concurrency, multiprocessing, core-py, l2]
---

Card df5affc7 — batch gene-batch-20261010.

## Port-don't-rebuild (verified 2026-10-10)

main (62f34afec) already solved this card in `72e8fae2f` ("fix(pugqeep+vector):
fork -> forkserver start methods"). That commit is NOT an ancestor of our
shared branch. Our branch's engine is the NEWER owned-fork design
(18c720be5 — os.fork + socket.socketpair framed channel, no multiprocessing),
so 72e8fae2f cannot cherry-pick; its design ports:

- forkserver default via `multiprocessing.get_context("forkserver")` scoped to
  SubprocessProcess (spawn on win32); NO global set_start_method.
- fork stays as explicit escape hatch (config.start_method="fork" /
  SLO_PGQ_START_METHOD=fork) — closures OK, and IsolatedExecutor (card
  c73aac23, NOT on this branch) will need fork for model inheritance when it
  lands; the escape hatch is the coordination point.
- pre-flight ForkingPickler.dumps((fn,args,kwargs)) -> hard-fail with guidance
  before any fd/socket exists.
- forkserver preload of [__main__, engine, target-module] (72e8fae2f measured
  ~16ms/child steady vs ~523ms cold; scripts/benchmark_pugqeep_spawn.py exists
  on main — port it).

Keep: L1 dedupe/cancel defense (card 20260928_075), framed-channel protocol,
watchdog, PGQ event-loop-free (no asyncio in pugqeep/).

## LANDED 2026-10-11 — exec start method (trunk `54eb4a3a4`)

The design port above needs a forkserver RELAY (a clean single-threaded
process that does the forking) which does not exist on the owned engine —
building it is the "medium effort" option B and was not done here. What
landed is the doctrine-compliant subset that kills the hazard class for
picklable jobs, on the owned socketpair channel, with zero behavior or
latency change for existing callers:

- `SubprocessConfig.start_method`: `"fork"` (default — unpicklable
  workers keep working, unchanged behavior) | `"exec"` (fresh
  interpreter via `pugqeep.exec_worker`: the image is replaced before the
  worker runs its first line, so no lock held by this process's threads
  can be inherited — the BLAS/OpenMP futex-deadlock class cannot occur).
- `_ExecChild` mirrors `_ForkedChild` (reaping delegated to Popen);
  pickle pre-flight happens in the PARENT before any child exists —
  unpicklable fails loud via `proc.fail`, never a hang; channel fd via
  `pass_fds` (PGQ-owned), job bytes ride stdin (never mixed into frames).
- `Engine.run_subprocess(start_method=...)` for callers.
- Tests: `TestExecStartMethod` (picklable, capture, unpicklable-fails-
  loud, terminate escalation); targeted suite 272 passed.

REJECTED: porting `72e8fae2f`/`851b19fa5` (option A) — both build on the
pre-`18c720be5` multiprocessing engine and would revert the owned-fork
architecture (`851b19fa5` evaluated 2026-10-11, 5 engine conflicts each
reintroducing `multiprocessing.Pipe`/`ForkingPickler`; stranded with
reason in the commit message of `54eb4a3a4`).

FOLLOW-UP (unclaimed): option B — pugqeep-owned forkserver relay over the
existing socketpair channel — would extend prevention to unpicklable
(closure) workers; forkserver preload benchmark (`scripts/benchmark_
pugqeep_spawn.py` on main) is the reference. Prevention for those workers
today = L1 defenses + card d3a4a079 detection.

## GATE EVIDENCE (2026-10-12) — closed

- Full tests/core-py gate on `.wt-pugqeep` @ `54eb4a3a4`: 6/6 chunks
  EXIT, ~37k tests, 537 failed + 4 ERROR → comparator: 456 in-baseline/
  owned, 85 unexplained, **A/B vs `8c26ce22e` (= parent) proves all 85
  pre-existing**: 82 fail at the parent directly; the other 3 pass at
  mine isolated (chunk-order/load flakes) or fail 3/3 isolated at both
  (`model_health::detect_drift` — a pre-existing trunk red missing from
  the older baseline snapshot). Zero failures attributable to the
  commit. Verdict file: `~/.cache/opencode-gates/pugqeep_verdict_54eb4a3a4.txt`.
- Targeted: `test_pugqeep_{process_mgmt,config}.py` 272 passed / 0 failed
  incl. the 4 `TestExecStartMethod`.
- Errors: 2 comprehensive_training_journeys (journey/env family), 2
  compat_track (foreign dirty pair), `test_shell_tui_live.py` collection
  = committed test imports untracked `term_screen.py` — fixed by the
  pending TUI port landing.
