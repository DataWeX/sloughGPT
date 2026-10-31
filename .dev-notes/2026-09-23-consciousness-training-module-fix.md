# Fix ModuleNotFoundError: domain.consciousness.training

**Status:** done
**Sprint:** —
**gh:** —

## Task
`GET /consciousness/status` 500s: `No module named 'domain.consciousness.training'`
at `apps/api/server/routers/consciousness.py:99 (_get_trainer)`.

## Body
1. Training-loop consolidation (commit 09b49eea) deleted
   `packages/core-py/domains/consciousness/training.py`; the new
   `domain/consciousness/` package was re-added without a `training.py`.
2. Router still lazy-imports `domain.consciousness.training` in
   `_get_trainer()`, so `/consciousness/status`, `/train/status`,
   `/train/start`, `/stats`, `/stream` all 500.
3. Fix: adapt the router to the consolidated training module
   (`domain/training/_internal` — HFLoraTrainer + TrainResult) with a
   router-local adapter. Do NOT recreate a new training loop.

## Constraints
- No new training loop / no new `domain/consciousness/training.py`.
- Delegate forward/loss/backward/optimize to shared infra.
- Preserve API contract: pairs_collected, min_pairs, should_train,
  is_training, training_runs, last_result; TrainResult.to_dict().