---
title: 2026-10-24 — training router delegation part 1 (card 50675952)
status: open
date: 2026-10-24
tags: [kanban, training, engine, delegation, boundary]
card: 50675952-8fad-42f6-92ad-7a2e27b7a72b
branch: feat/npu-caps
commit: 5be763e55
---

# Training router → TrainingEngine delegation, part 1

**Card**: `feat(api): wire training/router.py checkpoint/monitor/advisor endpoints through TrainingEngine (build order #4, part 1)` — moved wip→review, closes on reaching origin/main.

## What landed (commit 5be763e55)

**Engine** (`domain/training/engine.py`, +93 lines): new section *Router delegation
(raw passthrough)* — 14 methods that return raw values and let exceptions bubble
(HTTP semantics unchanged, per card). Lazy `_internal` imports inside methods,
matching the engine's existing style:

- async service×8: `get_log`, `list_stored_checkpoints`, `delete_checkpoint`,
  `load_stored_checkpoint`, `download_checkpoint_path`, `checkpoint_info`,
  `get_all_checkpoint_data`
- sync: `get_state`, `is_valid_ckpt_name` (VALID_CKPT_NAME regex), `load_outcomes`
  (TrainingOutcomeTracker), `get_training_monitor`, `alert_severity` (enum, ValueError bubbles)
- advisor×2: `get_training_tips`, `recommend_training_config` (kwargs passthrough)

`list_stored_checkpoints`/`load_stored_checkpoint` names avoid collision with the
engine's existing soul-checkpoint `list_checkpoints`/`load_checkpoint`
(TrainingResult-wrapped, different store).

**Router** (`apps/api/server/training/router.py`): all 17 part-1 `_internal`
import sites rewired to `get_training_engine()` (lazy per-handler import, the
file's existing style). Stale boundary comment (L506) rewritten. Only part-2
sites remain: L22 `get_training_executor`, L271 `CheckpointManager`,
L356 `SloughGPTTrainer` (+ the `executor.submit` call site).

**Tests** (new, `apps/api/server/tests/test_training_router_delegation.py`):
boundary invariant (router source contains no part-1 `_internal` module refs)
+ engine delegation surface + semantics pins (regex equivalence, monitor
singleton, enum error propagation, state `.config`). RED proven: 6 failed →
GREEN: 6 passed; full router suite (70 incl. 6 files: unified routes, feed,
finetuned, recovery, support, feed journey) green; ruff clean.

## Site accounting vs card (tree drift, noted)

Card says "19 of 23 sites … service×11 … cache_tags". My tree: 18 part-1
source lines (17 imports + 1 comment), service×9, **no cache_tags site exists**
(drift — card's tree had one); +2 state (get_state×2 counted separately) +
VALID_CKPT_NAME + advisor×2 + outcome×1 + monitor×5. Remaining = 4 part-2
sites as specced. Counting method matches the card's intent; no site skipped.

## Gate / evidence

- RED: `pytest test_training_router_delegation.py` → 6 failed (AttributeError ×5,
  boundary ×1) before implementation.
- GREEN: 70 passed (7 test files), ruff clean.
- Verified after commit: `git show --stat HEAD` = 3 files, +233/−41, no foreign hunks.

## Status

- Board: wip→review (2026-10-24). Closes when `feat/npu-caps` stack lands on origin/main.
- Part 2 (`c48aaaad`): remaining 4 sites — `resolve_resume_checkpoint`
  (ValueError→422), `run_recovery_training(config, on_progress, resume, events)->trainer`,
  `executor_submit(fn,*args)`; router keeps closure/state.
