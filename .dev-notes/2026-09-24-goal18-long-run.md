# Goal 18: Track B long-run pretraining

**Status:** done
**Sprint:** 2026-09-W4
**gh:** —

## Task
ROADMAP goal 18 — unattended from-scratch run: checkpoint/resume on crash, durable streaming loss/monitor logs, benchmark hooks. Starts only after goal 16 (green). No fourth training loop.

## Result
- Branch: `feat/goal18-long-run`
- **Durable metrics:** `domain/training/_internal/long_run.py` `LongRunRecorder` — append-only JSONL under `checkpoint_dir/metrics.jsonl`; lazy-exported as `domain.training.LongRunRecorder`
- **Cancel fix:** `train_pipeline.py` `cancel_event` sets `_is_training = False` (stops the whole run, not just the current epoch) so a cancelled long-run still hits the final save and leaves a resumable `.soul`
- **Tests:** `tests/test_long_run_resume.py` — (1) recorder append/reopen durability, (2) crash→resume step continuity 12→24 with finite losses + JSONL growth, (3) cancel mid-run → checkpoint → resume continues
- **Supervisor:** `scripts/train_longrun.py` — resume-latest when a checkpoint exists, restart budget with backoff, `--bench-every` eval hooks into metrics.jsonl, `--json` emits `benchmark_results record --kind training` shape, `--smoke` hermetic cycle-corpus gate
- **Bench:** smoke recorded `--kind training` (longrun 24 steps, loss 3.20→1.39, 3 eval hooks)
- **Gates:** long-run + smoke + progress + status transitions + pipeline + journey + convergence + scatter gate green; ruff clean
- ROADMAP 18 marked Done; kanban `20260924_035` → done (034 was already taken by domain-port card)
