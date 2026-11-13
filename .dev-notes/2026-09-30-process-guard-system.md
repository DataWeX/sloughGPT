---
id: 20260930_074_process_guard_system
status: done
title: "Card 074 — ProcessGuard as a whole system: one supervision policy"
tags: [core, process-guard, supervision, reliability, health]
sprint: current
gh: null
---

# Card 074 — ProcessGuard whole-system supervision

## Scope (user-directed)

- Review of `domain/infrastructure/_internal/process_guard.py` surfaced three
  systemic gaps (budget never resets, message-sniffing stall detection,
  alive-but-wedged after exhaustion).
- User: **"build it as a whole system, not a subset/subsystem"** → fix the
  policy core AND every consumer state (slonet dispatch, controller status,
  core /health, architecture doc), tested end-to-end.

## Design — one supervision policy (same shape as Rust supervisor + host loops)

- [x] Backoff: `restart_delay` = initial, doubles to `restart_delay_max`
      (default 30s), resets with budget.
- [x] `budget_reset_after` (default 60s, ≤0 off): worker alive that long →
      `restart_count` resets — a guard that crashed 3 times months apart is
      not permanently unguarded.
- [x] Typed stalls: `WorkerStreamStalledError` | builtin `TimeoutError`
      (model_worker raises both); substring sniffing ("timeout" in message)
      removed — it restarted healthy workers on unrelated messages AND
      missed real `timed out` timeouts.
- [x] Exhaustion = terminal + honest: wedged worker is STOPPED (`alive`
      False) → slonet `_use_guard()` falls back in-process per call;
      `health()["state"]` ∈ running/stopped/exhausted + crash_count,
      stall_count, restarts_left, budget_reset_after; generate error states
      exhaustion; crash callbacks fire on exhaustion stop (breaker sees it).
- [x] Monitor loop → `_monitor_tick()` (deterministic test seam).
- [x] Consumers verified: slonet per-call fallback (no change needed),
      `/models/process-guard` + core `/health` flow richer health
      automatically (health() additive).
- [x] Docs: `docs/process-guard-architecture.md` gains Supervision Policy
      section (no new doc files).

## Gates

- [x] RED first — 17/17 new tests failed on base for the right reasons
- [x] pytest green — full suite exit 0 (edge's stale copy had been used first by mistake;
      work re-based on main tip bbf46e7c6 in fresh worktree /home/mana/Documents/sloughgpt-pg)
- [x] ruff clean; py_compile ok; 12-file consumer set = only pre-existing
      test_provider_processors failure (A/B verified identical on pristine HEAD)
- [x] Commit `031d3c7b8` on feat/process-guard-system → land `08d25df2f`

## Results (2026-09-30)

- Branch: `feat/process-guard-system` (worktree `/home/mana/Documents/sloughgpt-pg`, base = main tip).
- Feature commit `031d3c7b8` → cherry-picked to `land/gateway` as `08d25df2f`.
- `land/gateway` also merged current main (8 new commits, zero file overlap, conflict-free) — merge `e45e081fd`; land = main + 19 commits, worktree clean.
- Gates on land: `test_process_guard.py` + `test_slonet_server.py` + `test_slonet_provider.py` exit 0; ruff clean on changed files.
- 17 new tests: typed stall (TimeoutError/WorkerStreamStalledError true; message-mention false),
  backoff [2,4,8,8] + cap, budget refill (window/zero/off), monitor tick restart+skip,
  exhaustion stops wedged worker + fires crash callbacks + generate hint + health state lifecycle.
- Discovered mid-task: main had advanced 8 commits past our fork base and edge-cascade's
  process_guard copy was stale — re-based the whole effort onto main tip instead.
- Landing to api-std still blocked (other session: 30 dirty files).
