# Execution consolidation — spec

Date: 2026-09-20. Status: stage 1 built (shared fire-and-forget pool +
`training/helpers._run_async` routed through it); stages 2–4 open.

## Problem

Four overlapping execution systems, picked per call site by habit:

| System | Location | Shape | Callers |
|---|---|---|---|
| `TrainingExecutor` | `domain/training/_internal/executor.py` | ThreadPool (1–2), job-tracked, cancel/finish, PointLibrary sink | all 8 training start paths |
| `TaskQueue` (infra) | `domain/infrastructure/_internal/task_queue.py` | async, priority, pause/resume/cancel, handler-based, needs running loop | `training/sse_stream.py`, `startup.py`, rag/memory/maintenance |
| pugqeep executor + task queue | `packages/core-py/domains/infrastructure/pugqeep/` | batch compute (drain-blocking), model-tree weight loading | `tree.py`; compressor used by `TrainingExecutor` |
| raw threads | ~10 sites (`main.py`, `training/helpers.py`, feedback, collectors…) | unbounded, untracked | notifications, watchdogs, collectors |

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

Call-site rule: pick by need (tracking? priority? drain result? nothing?),
never by import habit. `threading.Thread(` at a call site becomes a
review flag.

## Staging

1. Add the `fire-and-forget` pool + route `training/helpers._run_async`
   through it (benchmarked: 1.4ms dispatch, 5 peak threads vs 141).
2. Document the routing table; add a lint hint against raw
   `threading.Thread(target=` in `apps/` + `domain/` (allowlist file).
3. Migrate raw-thread sites one by one (startup, feedback, collectors).
4. Reconcile the two task queues (infra vs pugqeep) — converge or draw an
   explicit compute-vs-orchestration boundary. Explicitly NOT in stages 1–3.

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
