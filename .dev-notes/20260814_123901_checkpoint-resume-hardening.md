---
id: 20260814_123901_checkpoint-resume-hardening
title: Checkpoint resume hardening
status: done
tags: training,resume,bugfix,core-py
created: 2026-08-14T12:39:01.048963+00:00
---

Checkpoint resume hardening

Hardening checkpoint resume against corrupt/truncated .soul/.npz files.

Round 1 (band-aid):
- CheckpointManager.latest_valid_path() + load_latest(): skip unreadable checkpoints newest-first (warn + continue), fall back to previous good checkpoint. Crash mid-write of the newest file no longer blocks recovery.
- train() explicit resume_path: unreadable file now raises ValueError 'Cannot resume from <path>: checkpoint is unreadable (...)' instead of a raw traceback; missing/unsupported keeps its contextual ValueError.
- /training recovery endpoint: fallback uses latest_valid_path().
- CLI (cmd_train, cmd_train_native) + train_pipeline __main__: catch ValueError, print clean error, exit 2.

Round 2 (deep refactor):
- Single-load primitive CheckpointManager.load_latest_with_path() -> (path, bundle); latest_valid_path()/load_latest() now delegate (no duplicated skip loop). No more resolve-then-reload double loads.
- train(resume=True, resume_checkpoint=bundle): accepts a pre-loaded bundle, takes precedence over resume_path, bypasses all disk I/O. Docstring updated.
- /training recovery endpoint: resolves + loads the checkpoint exactly once in the request handler (recorded path -> strict, corrupt -> HTTPException 422; fallback -> load_latest_with_path skip-corrupt) and hands the bundle to train(). Eliminates the second load in the worker thread.
- Atomic writes (root-cause fix): save_soul writes both the .soul binary and the .meta.json sidecar via temp+rename (previously meta was written non-atomically first); save_checkpoint_npz writes .npz via temp+os.replace. A crash can no longer leave a truncated newest checkpoint.
- Candidate scan excludes in-progress temp artifacts (*.tmp / *.tmp.npz): an orphaned atomic-save temp (Path.glob('*.npz') matches it) can never surface as a 'latest' resume candidate or pollute the checkpoint window. Found during review: glob matches dotted .npz too, so exclusion by name is required.
- resume_checkpoint without resume=True now raises ValueError (previously silently discarded the pre-loaded bundle and started fresh) - fail-loudly, never silently drop a caller's checkpoint. All callers verified keyword-based; only recovery passes the bundle (always with resume=True); distill_gpt2/chat_trainer use a separate path-string config, unaffected.
- Tests (+13): load_latest_with_path (empty/newest/skips-corrupt/all-corrupt/agree-with-latest), candidate temp-artifact exclusion, resume_checkpoint bundle restore, bundle precedence over corrupt path, bundle-without-resume raises, npz atomic no-tmp + failure-cleanup, save leaves no tmp, save_soul meta failure leaves no partial artifact. Full test_train_pipeline.py 150/150 pass; test_slo_format + test_training_status + test_slonet_legacy 325/325.
- Docs: CONTRIBUTING.md resume semantics (bundle handoff, single load, atomic writes, temp exclusion) + regression list updated.

Round 3 (5th review pass):
- train() type validation: resume_checkpoint must be a dict bundle (ValueError 'resume_checkpoint must be a checkpoint bundle dict (as returned by load_from_path / load_latest_with_path), got <type>' otherwise), checked before the requires-resume check. A path string passed where a bundle is expected can no longer be silently misloaded.
- First end-to-end tests for the recovery endpoint: apps/api/server/tests/test_training_recovery_router.py (7 tests) covering 404 missing job, 400 wrong status, 422 corrupt recorded path + no job created + no submit, 422 missing recorded path, 200 valid recorded path -> bundle handed to train(resume=True, resume_checkpoint=...) with no resume_path and no second load, fallback no-checkpoint -> fresh start ('beginning'), fallback -> latest bundle. Run from apps/api/server with the repo venv python (PYTHONPATH=.:packages/core-py); server tests are outside pytest.ini testpaths.
- Test counts: test_train_pipeline.py 151/151 pass (incl. test_resume_checkpoint_non_dict_raises); server tests test_training_finetuned_router 13 + test_training_recovery_router 7 = 20/20 pass under venv; py_compile clean.
- Docs: CONTRIBUTING.md resume semantics paragraph now documents the dict-bundle type validation; regression list adds test_resume_checkpoint_non_dict_raises, test_resume_checkpoint_without_resume_raises, test_candidates_exclude_tmp_artifacts + notes the endpoint test file.

Round 4 (6th review pass):
- Recovery endpoint recorded-path branch was DEAD-CODE WRONG: 'if checkpoint_path and is_resumable(...)' fell through to load_latest_with_path() when a recorded path existed but its file was missing on disk - silently resuming from a DIFFERENT checkpoint (the documented contract says corrupt/missing recorded path -> 422 before any job is created). Restructured to 'if checkpoint_path:' strict handling: not is_resumable -> 422 'missing or unsupported'; load_from_path raises -> 422 'unreadable' (chained); bundle None -> 422 'missing or unsupported'. Only an EMPTY recorded path falls back to load_latest_with_path().
- checkpoint_dir resolution bug: store's checkpoint_dir column is NULL for interrupted jobs (only mark_completed writes it); endpoint read job.get('checkpoint_dir', 'checkpoints') so real jobs scanned the wrong directory. Now resolves as job['checkpoint_dir'] or config['checkpoint_dir'] or 'checkpoints' - the stored request config (job_store.create persists req.model_dump() into the config column) is the authoritative source.
- recovery_job dict spread **config AFTER explicit keys, so stored config could clobber id/status/checkpoint_dir/checkpoint_path. Moved **config first so explicit control fields always win.
- Regression tests (+2, endpoint file now 9): test_recover_recorded_path_missing_on_disk_422 (recorded path absent on disk, real is_resumable -> 422, no job, no submit; fails against pre-fix code which returned 200 'beginning'), test_recover_checkpoint_dir_from_job_config (job with no store column but config checkpoint_dir -> recovery job uses config value).
- Test counts: test_train_pipeline.py 151/151; server recovery 9 + finetuned 13 = 22/22 under venv; py_compile clean.
- Docs: CONTRIBUTING.md regression paragraph documents missing-on-disk 422, config-derived checkpoint_dir, and config-first spread.

Round 5 (7th review pass):
- run_recovery wrote persistent store mutations to the EPHEMERAL recovery job id (jid) - which has no store row - so update_progress/mark_completed/mark_failed were silent no-ops and the original job's row stayed 'recovering' (or got a bogus status='recovered' on cancel/success). All progress/terminal writes now target the ORIGINAL job id: on_progress -> store.update_progress(job_id, ...), success -> store.mark_completed(job_id, recovery_checkpoint or ''), failure -> store.mark_failed(job_id, str(e)), cancel -> store.update(job_id, status='interrupted') so the job stays recoverable.
- run_recovery built a FIXED SUBSET trainer_config (epochs/batch_size/lr/n_embed/n_layer/n_head/block_size/checkpoint_dir/checkpoint_interval), silently dropping the original job's LoRA, dropout, scheduler, warmup, weight_decay, device, etc. Now builds {'data_path': ..., **_sloughgpt_trainer_kwds(recovery_job)} - the SAME builder as /training/start - so the recovered run continues with full hyperparameter parity.
- on_progress now mirrors /training/start fields (total_steps, steps_per_sec, eta_s, elapsed_s, train_loss/eval_loss/loss) instead of only progress/epoch/step.
- _FakeJobStore in the endpoint test file now records all mutations (method, args, kwargs) so the store lifecycle is assertable; _recover attaches the store to the response for inspection.
- Regression tests (+4, endpoint file now 13): test_recover_success_records_completion_on_original_job (mark_completed('job-1', ckpt), never a 'recovery_' row), test_recover_failure_marks_original_job_failed (mark_failed('job-1', ...)), test_recover_cancel_restores_interrupted (update('job-1', status='interrupted'), no mark_completed), test_recover_reuses_original_hyperparameters (constructor kwargs carry use_lora/dropout/lora_rank/scheduler_type/lr from the job config).
- Test counts: test_train_pipeline.py 151/151; server recovery 13 + finetuned 13 = 26/26 under venv; py_compile clean.
- Docs: CONTRIBUTING.md regression paragraph documents original-row store lifecycle + full trainer-config parity.
Round 6 (8th pass): job-store lifecycle correctness.
- detect_crashed_jobs compared ISO heartbeats (datetime.now().isoformat(), 'T' separator) against SQLite datetime(?, 'unixepoch') space-format strings - 'T' > ' ', so the stale test silently NEVER fired (root-cause crash-detection bug). Now builds the cutoff with datetime.fromtimestamp(...).isoformat() in the SAME format; also flags stale 'recovering' rows (status IN ('running','recovering'), heartbeat IS NOT NULL).
- get_recoverable_jobs returned only status='interrupted' while the recover endpoint accepted 'failed' too, and stale 'recovering' rows were unreachable until restart. Now returns interrupted + failed + stale-recovering (heartbeat cutoff), never an actively-recovered row.
- Added JobStore.mark_recovering(job_id) (status='recovering', crashed=0, fresh last_heartbeat) and JobStore.is_stale_heartbeat(job, timeout=300). recover_job uses mark_recovering and accepts 'recovering' with a stale heartbeat (fresh 'recovering' -> 400 'stale heartbeat'); 400 detail updated.
- New tests/test_job_store.py (10 tests, real SQLite): heartbeat-format regression (test_detect_crashed_jobs_stale_running), stale running/recovering detection, fresh non-flagged, non-running ignored, mark_recovering fresh-heartbeat + crashed-clear, is_stale_heartbeat (fresh/old/None/garbage), recoverable list (interrupted+failed, stale recovering in, active recovering out).
- Recovery endpoint file now 15 tests (+2): test_recover_400_when_recovering_with_fresh_heartbeat, test_recover_allows_stale_recovering_job (asserts mark_recovering called on the original id).
- Test counts: job_store 10/10, recovery 15/15, finetuned 13/13 (38 total); core test_train_pipeline.py 151/151; py_compile clean.
- Docs: CONTRIBUTING.md regression paragraph documents ISO-cutoff contract + stale-recovering semantics + test_job_store.py coverage.
Round 6 follow-up (docstring honesty): check_crashed_jobs + recover_job docstrings now state 'running'/'recovering' and stale-'recovering' acceptance. Full pass re-verified: server 38/38, core train_pipeline 151/151, py_compile clean.