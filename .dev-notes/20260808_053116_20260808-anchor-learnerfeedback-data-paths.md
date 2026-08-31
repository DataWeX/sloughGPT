---
id: 20260808_053116_20260808-anchor-learnerfeedback-data-paths
title: 20260808 Anchor learner/feedback data paths
status: done
tags: 
created: 2026-08-08T05:31:16.050048+00:00
---

20260808 Anchor learner/feedback data paths

Anchored workspace-relative data paths in core-py modules (data_filter, continual, knowledge_weight_integrator, slo_manager, chat/domain, feedback/workflow, feedback/per_user_lora, training/slonet) to the repo root via parents[4]; relevance gate in knowledge_augmenter passes 40 tests; per_user_lora suite 44 passed; 9-file chunk 356 passed/1 skipped/1 xpassed in 45.45s under /dev/shm. The earlier 'test hang' was NOT cross-test contamination: tests are hermetic. Root cause was fsync latency on the drive-managed SMR HDD (TOSHIBA MQ04ABF100); proven by tmpfs runs and SMART (media healthy, worn head). Fix is OS migration from HDD to NVMe (data-root LV) in progress.