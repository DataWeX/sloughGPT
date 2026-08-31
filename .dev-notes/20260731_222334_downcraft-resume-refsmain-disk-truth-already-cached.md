---
id: 20260731_222334_downcraft-resume-refsmain-disk-truth-already-cached
title: downcraft resume: refs/main + disk-truth already_cached
status: done
tags: downcraft,resume,infra
created: 2026-07-31T22:23:34.731323+00:00
---

downcraft resume: refs/main + disk-truth already_cached

Completed downcraft pause/resume validation.

Fixes:
1. download_hf_model now writes refs/main -> default after completion (and in the already_cached path), so is_download_complete recognizes the snapshots/default layout it produces.
2. already_cached shortcut now verifies disk truth via is_download_complete; stale state (complete in ~/.downcraft/state.json but files missing on disk) triggers a redownload instead of skipping.

Tests: test_resume.py 24 passed (new: refs/main written + completeness recognized end-to-end; stale state redownloads). downcraft core suite 64 passed; core-py regression 74 passed.

Live E2E demo (prajjwal1/bert-tiny, 17.8MB): process SIGKILLed at byte 8388608 of pytorch_model.bin -> inspect_incomplete identifies pytorch_model.bin, resume offset 8388608 of 17756393 -> resume in fresh process completes at exact offset -> is_download_complete=True. PASS.

Remaining integration scope: rewire packages/core-py/domains/infrastructure/download_manager.py to delegate to downcraft (register project cache root models/hf-cache/hub), and replace snapshot_download in apps/api/server/infrastructure/startup.py autoload with downcraft path.