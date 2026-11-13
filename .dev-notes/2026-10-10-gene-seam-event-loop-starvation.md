---
id: c550503c
title: "COORD: live :8000 wedge — accept queue 0→191, faulthandler dumps starved (event-loop starvation story: c550503c + ff81b07a + b6cf5e42)"
status: in_progress
board_id: Main
date: 2026-10-10
branch: fix/event-loop-seam
tags: [gene, coordination, incident, seam, event-loop, server]
---

# Gene batch 2026-10-10 — event-loop starvation (3-card story)

Cards folded into this rollout as ONE story:

- `c550503c` (evidence card): live `:8000` wedged 2026-10-06 10:07–10:25,
  `ss LISTEN` Recv-Q 150→191 unaccepted, faulthandler dumps starved mid-write
  (GIL-held block on the main thread).
- `ff81b07a`: ~40 blocking calls run inline on the single uvicorn loop.
  Worst four: `routers/models.py:1151` (FENCED — card 3f749b34),
  `training/router.py:175` (FENCED), `startup.py:1135 proc.wait(10)`,
  `profiles.py:96 nvidia-smi(5)`.
- `b6cf5e42`: `SloNetServer.generate` still dispatches via
  `asyncio.to_thread` (slonet_server.py:519) — thread-domain, can wedge the
  accept loop (GIL-held numba kernels).

## Plan

Work in worktree `.wt-seam`, branch `fix/event-loop-seam`, off
`fix/db-pool-root-and-download-progress`. Main tree untouched (~216 dirty
entries, other sessions).

1. **Port, don't rebuild** the settled seam fixes from
   `feat/contract-ts-projection`: `15cfcdac7`, `ff85e6d02`, `74d0ecc67`,
   `e7a42cb81`, `5a92109a8` (source shas quoted in commits).
2. Port the isolation machinery `ad3f2cb05` (`pugqeep IsolatedExecutor` +
   `InferencePool.run_isolated`, host stall 1007ms → 13ms) and wire
   `SloNetServer.generate` behind `SLO_ISOLATED_INFERENCE` (DEFAULT OFF).
   Cancel propagation solved (parent-side terminate of forked child);
   `generate_stream` stays thread-domain, documented.
3. Reproduction for `c550503c`: accept-queue growth under blocking load
   BEFORE (inline) vs NO growth AFTER (`to_thread` / isolated) — real
   listening socket, own port, never `:8000`.

## Fences (do not touch)

`routers/models.py`, `training/router.py`,
`tests/server/test_models_router.py`, `tests/server/test_audit_instrumentation.py`,
`apps/web/**`, any file with another session's uncommitted hunks
(`tests/core-py/test_slonet_legacy.py`, `test_slonet_wave_a.py` are dirty).

Known pre-existing `tests/server` failures (NOT ours):
`test_audit_instrumentation.py::TestModelAudit::test_set_precision_logs_event`,
`test_models_router.py::TestPrecision::test_cpu_path_uses_suggestion`.

## Status log

- 2026-10-10: batch started; `.wt-seam` created clean at a5019d841;
  baselines for tests/api-server + tests/server running.
