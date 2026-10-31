---
title: 2026-10-24 — training router delegation part 2 (card c48aaaad)
status: open
date: 2026-10-24
tags: [kanban, training, engine, delegation, recovery, boundary]
card: c48aaaad-29e3-4167-b9a8-b70fcfa5bc73
branch: feat/npu-caps
commit: 118e11a23
---

# Training router → TrainingEngine delegation, part 2 (recovery internals)

**Card**: `refactor(api): training router recovery run via engine (build order #4, part 2/2)` — moved wip→review, closes on reaching origin/main. Part 1: `50675952` (same session, commit `5be763e55`).

## What landed (commit 118e11a23)

**Engine** (`domain/training/engine.py`, +73): three recovery methods completing
the router-delegation section:

- `resolve_resume_checkpoint(checkpoint_path, checkpoint_dir) -> (path, bundle)` —
  exact replication of the handler's old CheckpointManager block: recorded path
  missing/unsupported/unreadable → **ValueError** (the router maps it to 422 —
  the one sanctioned HTTP mapping in the whole delegation); no recorded path →
  newest loadable file via `load_latest_with_path()`, `""` when nothing resumable.
- `run_recovery_training(config, on_progress, resume, events) -> trainer` — builds
  `SloughGPTTrainer(**config)`, `train(on_progress, resume=True,
  resume_checkpoint=resume, cancel_event, pause_event)`, returns the trainer
  (closure reads `_last_checkpoint_path`). `resume` = pre-loaded bundle (None
  ok) — preserves "no second disk load" semantics. Signature extends the card's
  sketch with the explicit bundle; docstring records why.
- `executor_submit(fn, *args)` — raw passthrough to the training executor.

**Router** (`apps/api/server/training/router.py`): last 4 `_internal` sites gone
— module-level executor import removed, resolve block → engine + `except
ValueError → raise_error(422)` (identical messages/status), closure's
`SloughGPTTrainer` build+train → `run_recovery_training(...)` (closure keeps
ALL state: training_jobs records, store updates, controller, webhooks,
cancel/pause Event creation, CancelManager registration — untouched),
`executor.submit` → `executor_submit`. **Zero `domain.training._internal`
references remain in the router** — the build-order-#4 boundary is complete:
routers never import `_internal`; the engine is the single seam.

**Tests**: boundary test extended to the full invariant (single assertion: no
`domain.training._internal` in router source) +5 semantics pins (ValueError
contract ×3 incl. unreadable-carrying message, trainer build/train kwargs,
executor fn+args forwarding). RED: 7 failed/4 passed → GREEN: 11/11.
Recovery fixture retargeted: `patch.object(router_mod, "get_training_executor")`
→ string patch on `domain.training._internal.executor.get_training_executor`
(the name left router_mod; module-attr patch survives the engine's lazy import —
same style as the suite's existing `train_pipeline.SloughGPTTrainer` patch).

## Gate / evidence

- RED: 7 failed (boundary, surface, 5 new) before implementation.
- GREEN: 75 passed (7 files: delegation 11, unified routes, feed, finetuned,
  recovery 15, support, feed journey); `ruff check` + `ruff format --check`
  clean on all 4 touched files (format fixed the 104-char signature wrap).
- `git show --stat HEAD`: 4 files, +223/−76, owned paths only, no foreign hunks.

## Status

- Board: wip→review (2026-10-24). Closes when `feat/npu-caps` lands on origin/main.
- Build order #4 (training router → engine delegation) is now FULLY built:
  parts 1+2, 17 engine methods total, router boundary complete.
