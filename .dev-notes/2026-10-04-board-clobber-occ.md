---
id: 20261004_board_clobber_occ
title: "fix board.jsonl clobber: OCC on PlannerStore writes"
status: done
date: 2026-10-04
tags: [bug, concurrency, app-planner]
board_id: d7fc4b87-6077-435b-a83c-00e65d687f51
board: d7fc4b87-6077-435b-a83c-00e65d687f51
hash: 4f397ec5fa1e7496246af9b3b51d395718fb8fc2cf08b98d7b2f479769fcb2f2
---

FINDING (2026-10-04): `.kanban/board.jsonl` suffers LOST UPDATES — cards silently vanish. Two incidents: card 0d4ad67b (mine, 07:3x) and other session's meta-weights card ("lost to a concurrent board write"). ROOT CAUSE: every write funnel does read-whole-file -> rebuild -> atomic-replace with NO cross-process mutual exclusion and NO generation validation. Atomic replace prevents torn files but NOT stale-snapshot overwrites: process B rebuilding from a read taken before process A's append erases A's line. `flock`/locks exist nowhere in the live path (only gui.py's in-process threading.Lock on the legacy KanbanStore/board.json).

WRITE MAP (verified by reading): live CLI path = cli._board_* -> PlannerStore. Funnels: `_append_card` (full rebuild — its "append" is not an append), `_surgical_rewrite` (fresh read but validate->replace window), `_write_board` (no callers), `_write_notes` (non-atomic write_text — lost updates AND torn-file risk), notes mutators (read notes -> full rewrite), `sync()` delegates per-card via add_card/move_card (PLAN CORRECTION: no wholesale compute, so NO whole-body wrapper needed — funnel OCC covers sync; avoids the nested-guard reentrancy hazard entirely).

DECISION (user picked via question tool after expert review): OPTIMISTIC concurrency control — sha256 CONTENT tokens (not mtime: granularity/touch issues), validation ADJACENT to replace, non-blocking flock(LOCK_NB) as commit barrier only (never blocks, auto-releases on death, closes the validate->replace window), bounded retry with exponential backoff + jitter (30 attempts, 5ms->100ms cap — raised from the planned ~10 linear after 4/40 hammer runs exhausted under box load spikes), loud BoardWriteConflict after exhaustion. Reads stay 100% lock-free. No nesting exists anywhere in the write paths -> no reentrancy machinery.

SCOPE: funnels + notes mutators + atomic notes write; readers untouched (atomic replace already gives them consistent snapshots); legacy KanbanStore/board.json, migrate_boards.py, slot_history (derived, self-heals) OUT. Same-card last-writer-wins accepted (semantic, not a clobber). Sync-vs-sync duplicate-title race = pre-existing (dedupe card exists), out of scope.

EVIDENCE (2026-10-04): pre-fix hammers failed 4/40 — exhaustion causes: 'commit barrier busy' (lock-holder descheduled under external load spikes, load avg >9) and 'changed since read' (linear no-jitter backoff made retryers herd). Fix = exponential+jitter budget; post-fix 0/40 expected, 0/25 observed under load 6-9. BENCHMARK scripts/benchmark_board_write.py (SOP, no prior board-write benchmark): single add 1477 ops/s, notes 444 ops/s, 8-proc 726 ops/s aggregate, integrity 200/200.

TESTS: 8-process add hammer (all cards must survive), notes hammer, single-syncer + 6-adders (reproduces the real-world sync-eats-cards incident), forced-conflict test proving the retry loop re-reads.

ACCEPTANCE: new tests green + full app-planner suite green + ruff clean -> merge; card -> done.
