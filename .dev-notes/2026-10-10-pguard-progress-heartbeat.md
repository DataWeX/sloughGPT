---
id: d3a4a079-0271-4403-b3be-4a052a545f30
title: "ProcessGuard: progress heartbeat + wire watchdog.set_recovery_fn (livelock is invisible today)"
status: wip
board_id: gene-batch-20261010
date: 2026-10-10
branch: fix/pguard-heartbeat-batch
tags: [infrastructure, process-guard, observability, concurrency]
---

Card d3a4a079 — batch gene-batch-20261010.

## Port-don't-rebuild (verified 2026-10-10)

Landed on `fix/processguard-progress-heartbeat` as `a5b85e2be` (2026-10-10
03:26, parallel session working duplicate card 461834b2). Our batch's job:
cherry-pick that commit onto the shared branch, resolving the test-root rename
(packages/core-py/tests/test_process_guard.py -> tests/core-py/test_process_guard.py,
commit dcb2a5eaf) and any process_guard.py drift (855f4621e startup-cause fix
landed on our side since merge-base).

## Scope corrections vs card text (both verified on HEAD)

- set_recovery_fn IS already wired: `apps/api/server/main.py:580`
  (`watchdog.set_recovery_fn(_recover)`). The "never wired" claim is stale —
  no new wiring needed; the commit documents this.
- Heartbeat = monotonic `_last_progress` recorded per completed request and per
  streamed token; health() exposes last_progress_age_s + stall reason, so a
  livelocked-but-alive worker (2026-10-05: 90-min wedge, alive=true) becomes
  visible. Predicate tested with fake clock — no real waits.
