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
3. Migrate raw-thread sites one by one, in two passes (audit2026-11-04, ~130 production call sites):
   - **3A — process kind (~30 sites)**: long-lived supervised work →
     `engine.spawn()` / `run_subprocess()` / `spawn_command()`. Priority
     order: `dev.py` server supervision (**done for `cmd_dev`** api+web —
     `spawn_command`; `_cmd_api_only`/`_cmd_api_and_*` migrate when next
     touched) → shell command execution (`routers/shell.py`,
     `shell/repl.py` background jobs, `shell/runtime.py`, `init.py`) →
     `startup.py` engine-child (**done** — initial + restart launch sites
     via `spawn_command`).
   - **3B — thread/FAF kind (~25 sites)**: short-lived in-process work →
     FAF pool or engine thread mode (`main.py` rag-init, preloader,
     feedback loops, collectors, warmups, probes).
   - **Grandfathered (~75 sites)**: the execution infra itself (pugqeep,
     FAF, process_guard, watchdog, server accept/pump loops), tests,
     scripts/benchmarks, one-shot blocking `subprocess.run` (build,
     system, lscpu/git/gguf/shader one-shots), and I/O pumps bound to a
     live handle. No migration; the lint allowlist covers them.
4. Reconcile the two task queues (infra vs pugqeep) — converge or draw an
   explicit compute-vs-orchestration boundary. Explicitly NOT in stages 1–3.

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

- Shared pool size / overflow policy (drop? grow? block with timeout?).
- Should `TaskQueue` gain a sync bridge so non-async callers stop
  hand-rolling threads?
- Who owns lifecycle (shutdown/flush) of the shared pool — `lifespan`,
  `startup.py`, or process guard?
