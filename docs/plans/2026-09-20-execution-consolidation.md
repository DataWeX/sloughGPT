# Execution consolidation — spec

Date: 2026-09-20. Status: stage 1 built (shared fire-and-forget pool +
`training/helpers._run_async` routed through it); stages 2–4 open.
Amended 2026-11-04: `process` kind added (pugqeep `Engine`); stage 3
split into A/B from a full call-site audit (see below).

## Problem

Four overlapping execution systems, picked per call site by habit:

| System                        | Location                                                            | Shape                                                                   | Callers                                                        |
| ----------------------------- | ------------------------------------------------------------------- | ----------------------------------------------------------------------- | -------------------------------------------------------------- |
| `TrainingExecutor`            | `domain/training/_internal/executor.py`                             | ThreadPool (1–2), job-tracked, cancel/finish, PointLibrary sink         | all 8 training start paths                                     |
| `TaskQueue` (infra)           | `domain/infrastructure/_internal/task_queue.py`                     | async, priority, pause/resume/cancel, handler-based, needs running loop | `training/sse_stream.py`, `startup.py`, rag/memory/maintenance |
| pugqeep executor + task queue | `packages/core-py/domains/infrastructure/pugqeep/`                  | batch compute (drain-blocking), model-tree weight loading               | `tree.py`; compressor used by `TrainingExecutor`               |
| raw threads                   | ~10 sites (`main.py`, `training/helpers.py`, feedback, collectors…) | unbounded, untracked                                                    | notifications, watchdogs, collectors                           |

Consequences already observed: turbo's `_run()` died silently because
`TrainingExecutor.submit` injects `job_id` positionally; notify fan-out
blocked a worker because `_run_async` awaited inline; the SSE training
path and the turbo path use different engines inside one feature.

## Proposal

One dispatch surface in `domain/infrastructure`, kind-routed:

- `tracked-job` → `TrainingExecutor` (training lifecycle, cancel, Points sink)
- `background-task` → infra `TaskQueue` (priority, pause/resume, needs loop)
- `compute-batch` → pugqeep `ParallelExecutor` (drain-blocking array work)
- `fire-and-forget` → bounded shared pool (fixed size, daemon, never blocks)
- `process` → pugqeep `Engine.spawn()` / `run_subprocess()` (Python fns —
  restart, watchdog, timeout, metrics come from
  `SubprocessProcess`/`GuardTree`, not per-site hand-rolls) and
  `Engine.spawn_command()` (external commands: eager `Popen` at call
  time, live stdout/stderr, raw-Popen `poll`/`terminate`, run-loop
  reaping + `timeout`; landed 2026-11-04, first consumer `cmd_dev`)

Call-site rule: pick by need (tracking? priority? drain result? nothing?),
never by import habit. `threading.Thread(` at a call site becomes a
review flag.

## Staging

1. Add the `fire-and-forget` pool + route `training/helpers._run_async`
   through it (benchmarked: 1.4ms dispatch, 5 peak threads vs 141).
2. Document the routing table; add a lint hint against raw
   `threading.Thread(target=` in `apps/` + `domain/` (allowlist file).
   **Done (2026-11-05)** — `scripts/check_raw_threads.py` +
   `scripts/raw_threads_allowlist.json` (54 sites across 46
   `path::qualname` entries with declared counts; a new site, a shrunk
   site, a schema-less entry, or a stale entry all fail; AST-based, so
   docstring mentions and `from threading import Thread` aliases are
   handled). Gate: `tests/core-py/test_raw_thread_gate.py`; hint:
   pre-commit `check-raw-threads`. Allowlist entries carry `kind` —
   `followup` marks stage-3/4 candidates the audit parked (grep
   `"kind": "followup"` for the backlog; 1 site today — the
   knowledge-store sync bridge, owned by stage 4).
3. Migrate raw-thread sites one by one, in two passes (audit2026-11-04, ~130 production call sites):
   - **3A — process kind (~30 sites)**: long-lived supervised work →
     `engine.spawn()` / `run_subprocess()` / `spawn_command()`. Priority
     order: `dev.py` server supervision (**done for `cmd_dev`** api+web —
     `spawn_command`; `_cmd_api_only`/`_cmd_api_and_*` migrate when next
     touched) → shell command execution (**done** — `shell/repl.py`
     background jobs via `engine.spawn` + `_BgJob` handle,
     `shell/runtime.py` + `shell/_internal/init.py` via
     `spawn_command(start_new_session=)`; `routers/shell.py:158` ruled
     not-an-isolation-case: drives the in-process REPL singleton,
     thread-mode by necessity) → `startup.py` engine-child (**done** —
     initial + restart launch sites via `spawn_command`).
   - **3B — thread/FAF kind (~25 sites)**: short-lived in-process work →
     FAF pool or engine thread mode (`main.py` rag-init, preloader,
     feedback loops, collectors, warmups, probes). **One-shot batch
     done (2026-11-05)** — must-run one-shots → `engine.spawn` (never
     drops: each has a one-shot/status flag a dropped FAF submit would
     strand — rag-init, feedback-workflow startup, preloader,
     slonet warmup, accel probe, context auto-ingest, multimodal
     autosave); drop-tolerant online-train update → FAF pool
     (buffer only drains inside the update, so a drop re-triggers);
     join-all collectors/validators → `ParallelExecutor.map`
     (compute-batch: drain result outranks the pass's FAF/engine
     default under the call-site rule). **Loop batch done
     (2026-11-05)** — `spawn_loop(fn, stop=Event|callable)` +
     `ThreadHandle` (`is_alive`/`join`/`daemon`/`name`) in
     `pugqeep/singleton.py`, on a dedicated `loops` tree
     (max_stems=32, pool_workers=16) so a forever-loop can't crowd
     short work off the default tree's 4 pool workers; 8 sites
     migrated (workflow scheduler/training, auto-trainer, feed
     poll, scheduler jobs, compress, soul warm, pretrain,
     continual — the last via a `stop=` callable since it polls a
     bool, not an Event). Exit ordering is load-bearing: the
     singleton shutdown registers via `threading._register_atexit`,
     not plain `atexit` — `concurrent.futures._python_exit` joins
     pool workers inside `threading._shutdown` _before_ plain
     atexit runs, so a plain atexit cancel deadlocks interpreter
     exit on a blocked forever-loop (gate:
     `tests/core-py/test_pugqeep_spawn_loop.py`, incl. a
     subprocess interpreter-exit test). The
     stack-dump dumper is ruled grandfathered (global handle +
     join-on-cancel + `.name` test assertion). **Followup batch done
     (2026-11-06)** — 6 of the 7 allowlist `kind: followup` sites
     migrated: idle-manager check loop → `spawn_loop(stop=)` (loop now
     honors the stop event, which the engine's cancel sets — the
     daemon-equivalent release), model-server warmups ×2 →
     `engine.spawn`, REPL progress-load → spawn + `is_done`/`wait_for`
     (the bar's contract), TUI restart → spawn, parent-preload inner
     thread → spawn (the `threading.Timer` stays as the delay
     scheduler — timers are not gated Thread sites). Allowlist shrank
     54 → 48; the gate's declared counts forced every removal to be
     explicit.
   - **Grandfathered (~75 sites)**: the execution infra itself (pugqeep,
     FAF, process_guard, watchdog, server accept/pump loops), tests,
     scripts/benchmarks, one-shot blocking `subprocess.run` (build,
     system, lscpu/git/gguf/shader one-shots), and I/O pumps bound to a
     live handle. No migration; the lint allowlist covers them.
4. Reconcile the two task queues (infra vs pugqeep) — **boundary, not
   convergence** (design approved 2026-11-06). Explicitly NOT in stages 1–3.
   - **The boundary**: infra `TaskQueue` stays the _async orchestration_
     layer (SSE fan-out, pause/resume/cancel — it never executes
     compute); pugqeep `TaskQueue` stays the _durable pull-store_
     (kg-pipeline); the **Engine is the one execution layer**. No queue
     convergence — the three solve different problems (in-flight
     orchestration vs durable claims vs execution).
   - **The seam**: `run_on_engine(fn, ...)` in
     `domain/infrastructure/_internal/engine_seam.py` — spawns on the
     shared engine and awaits loop-safely via `call_soon_threadsafe`
     (never blocks the loop); re-raises the _original_ exception (the
     engine stores only `str(e)`, so the callable's exception is captured
     and re-raised — `except InterruptedError` call sites keep working);
     CANCELLED resolves as `EngineCancelled`; `timeout=` cancels the
     process so compute never outlives the wait. Lives **outside**
     `pugqeep/` — PGQ stays event-loop-free; the seam is the async host
     of a sync core (`tests/core-py/test_engine_seam.py`).
   - **Duration rule** (what migrates): long/indeterminate compute —
     training, inference, captioning, model loads, shell — → engine
     (`run_on_engine` or spawn per kind); micro-I/O (`os.unlink`, config
     reads) stays on `asyncio.to_thread`. Gate: the stage-2 scanner
     extended to `asyncio.to_thread`/`run_in_executor` AST nodes, same
     allowlist schema + ratchet (grandfathered sites counted both ways).
   - **Engine fix landed with the seam**: terminal state is honored at
     dispatch — a proc cancelled while queued never runs its fn, and
     `branch()`/`_execute()` no longer clobber CANCELLED with
     READY/RUNNING (the subprocess path already had this guard;
     `test_cancelled_before_dispatch_never_runs` pins it).
   - **Migrations**: training handlers (`training_queue.py` ×2),
     `turbo_endpoints.py` (`start_turbo_training`), multimodal
     caption/ask `run_in_executor` sites, and the knowledge-store
     `_run_async` sync bridge (the last `kind:followup` allowlist entry).
   - **Deletions**: `packages/infra-lib/infra_lib/task_queue.py` (zero
     consumers outside its own tests) — its own commit, after the seam
     lands.

## Audit 2026-11-04 (call-site inventory)

Counts (main checkout; `.wt-*` worktrees excluded as copies):

| Area                        | `threading.Thread`         | `subprocess.*` |
| --------------------------- | -------------------------- | -------------- |
| `apps/`                     | 23                         | 41             |
| `domain/`                   | 48                         | 26             |
| `packages/` (src, non-test) | ~11                        | —              |
| `scripts/`                  | 11                         | —              |
| **Total**                   | **~160** (~130 production) |                |

Migration candidates ≈ 55 (bucket A ~30, bucket B ~25); the rest are
grandfathered per stage 3. The `process` kind is the pugqeep-engine half
of this spec's original four-kind table — the audit confirmed the
kind-routing design and quantified stage 3's scope.

## Stage 1 — built

- New module `domain/infrastructure/_internal/fire_and_forget.py`:
  `FireAndForgetPool` (fixed daemon workers, bounded non-blocking queue) +
  process-global `get_pool()` singleton.
- `apps/api/server/training/helpers.py::_run_async` now submits to the
  shared pool instead of spawning a raw `threading.Thread` per call.
- Overflow policy decision (open question): **drop + count + log**. Growing
  would unbounded the pool; blocking would re-introduce the worker stall
  this replaces. Losing a notification is accepted over stalling a worker.
  Overflow warning is de-duplicated per episode (drops within one saturated
  burst log once, not once-per-drop).
- Pool size 4, queue cap 256. Baseline sweep (12 layers, 1 token/layer,
  1024 steps) showed the KV-style raw-thread fan-out replaced with bounded
  workers; dispatch is a queue `put_nowait` (µs), never thread spawn.
- Benchmarked `scripts/benchmark_execution.py` (recorded under kind
  `execution` in `benchmark_results.py`): **4.31 µs dispatch, peak 5
  threads vs 133 raw** (plan target: 1.4ms / 141) — dispatch ~325x faster,
  peak bounded at pool size + 1.
- Tests: `packages/core-py/tests/test_fire_and_forget.py` (10 tests).

## Non-goals

- No change to `TrainingExecutor` semantics (job_id injection contract stays;
  it bit us once — document it instead).
- No change to pugqeep compute internals.
- No new thread pools per feature; one shared pool for fire-and-forget.

## Open questions

All three closed as of stage 4 (2026-11-07):

- Shared pool size / overflow policy — **answered (stage 1)**: drop +
  count + log, pool 4 / queue cap 256 (see Stage 1 above; growing would
  unbounded the pool, blocking would re-introduce the worker stall).
- Should `TaskQueue` gain a sync bridge so non-async callers stop
  hand-rolling threads — **answered (stage 4): no**. The boundary ruling
  gives each layer one job: async callers cross via the `run_on_engine`
  seam (`domain/infrastructure/_internal/engine_seam.py`), sync callers
  (scripts/CLI/tests) use the sync core directly — `Engine`/`Pool` are
  synchronous by design and need no bridge. Queues gain none.
- Who owns lifecycle (shutdown/flush) of the shared pool — **answered**:
  `startup.py` — `StartupOrchestrator._shutdown_pool` registered as
  `ShutdownHook("pool_shutdown", ...)` (not `lifespan`, not process
  guard).
