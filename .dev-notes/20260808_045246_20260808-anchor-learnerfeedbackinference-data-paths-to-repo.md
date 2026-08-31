---
id: 20260808_045246_20260808-anchor-learnerfeedbackinference-data-paths-to-repo
title: 20260808 Anchor learner/feedback/inference data paths to repo root
status: done
tags: 
created: 2026-08-08T04:52:46.495387+00:00
---

20260808 Anchor learner/feedback/inference data paths to repo root

Anchored 8 CWD-relative data-path modules to _REPO_ROOT (Path(__file__).resolve().parents[N]): data_filter FILTER_CONFIG_PATH, learner continual LEARNER_STATE_DIR, knowledge_weight_integrator _ADAPTER_DIR, slo_manager _preference_file, chat domain log_dir, feedback workflow adapter path, per_user_lora store_path default, slonet adapter load path. All 8 test files hermetic. Combined 9-file chunk: 356 passed / 1 skipped / 1 xpassed in 45.45s (tmpfs basetemp; HDD fsync stalls were misdiagnosed earlier as cross-test contamination — root cause was spinning-disk sqlite fsync under editor load, verified D-state on submit_bio_wait/jbd2_log_wait_commit).