---
id: 20261102_001425_decide-boot-time-auto-resume-of-interrupted-training-jobs-ma
status: done
board: 20261102_001425_decide-boot-time-auto-resume-of-interrupted-training-jobs-ma
title: "Decide: boot-time auto-resume of interrupted training jobs (manual Recover is the status quo)"
tags: []
sprint: ""
gh: null
---

# Decide: boot-time auto-resume of interrupted training jobs (manual Recover is the status quo)

Gap found while verifying the 'we already have resumability' claim (user was right — everything below EXISTS): JobStore persists jobs across restarts; TrainingRuntime.shutdown gives trainers a cancel signal to save a final checkpoint; POST /recovery/recover/{job_id} resumes from the last checkpoint (recorded path authoritative, corrupt mid-write skipped, creates recovery_<id> job); GET /recovery/recoverable + StatusBar recovery panel expose it; periodic checkpoints every 500 steps bound hard-kill loss. MISSING: nothing auto-resumes on boot — restore() marks stale running rows interrupted and seeds the panel; a human clicks Recover. PRODUCT CALL NEEDED before building: (a) keep manual (safe: an expensive run restarts only when someone asks — no surprise GPU burn after an infra reboot, no resume storms when many jobs were interrupted), or (b) auto-resume on boot (hands-off continuity: an infra crash becomes a pause; needs a policy anyway — max concurrent resumes, skip if checkpoint missing/corrupt, maybe only jobs explicitly flagged auto_resume in their config). If (b): one startup hook iterating store.recoverable() through the existing recovery path + config flag + tests; endpoints/panel stay as-is. Precedent: the model-inference watchdog in main.py deliberately chose LOG-ONLY ('model reload from a watchdog thread is dangerous') — same class of decision, same posture likely applies.
