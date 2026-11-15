---
id: 20261023_af19fb40_config_backfill
title: "af19fb40: training register() backfills config on pre-existing job rows (job_1438982c root cause fixed)"
status: done
date: 2026-10-23
tags: [training, jobstore, runtime, bugfix, af19fb40]
board_id: af19fb40-1bfc-4706-8f29-a0e8c0624da3
---

# af19fb40 fix: config lost on persist path (job_1438982c)

## Bug

`TrainingRuntime._ensure_row` early-returned when the job's store row already
existed, silently dropping the `config` handed by `register()`. The in-memory
`training_jobs[job_id]` dict has NO config key (execution.py:288), so any row
persisted from it before execution.py:341's `register(..., req_snapshot)`
lands stays `config = {}`. Recovery then reads the empty column and rebuilds
trainer kwargs from defaults (`_sloughgpt_trainer_kwds({})` →
block_size=128/batch_size=32 → the incident's4096>399 failure).

Verified still live on origin/main `4a25b8385` before fixing (runtime.py
identical in the stack worktree — `git diff origin/main` empty on
runtime.py/execution.py/job_store.py, so the fix applies at landing).

## Fix (`11aa66fc2`, rides stack tail feat/npu-caps)

`_ensure_row`: an existing row with an EMPTY config is backfilled from
register()'s snapshot (`store.update(job_id, config=...)`, warning-logged on
failure, same style as the create path). Non-empty config is
first-registration-wins — later register() calls cannot clobber.

## Tests (RED-proven)

- `test_register_backfills_config_on_existing_row` — RED as
  `assert {} == {'block_size':512,...}` (the incident), GREEN after fix.
- `test_register_does_not_clobber_non_empty_config` — passes before AND after
  (guards the fix from overwriting).
- Surface: tests/server training/runtime/job_store selection = 68 passed;
  ruff clean. No benchmark applies (one conditional store write at
  registration — cannot affect training throughput).

## Board

af19fb40: wip → review (fix complete + tested; stays review until it reaches
origin/main with the stack landing).
