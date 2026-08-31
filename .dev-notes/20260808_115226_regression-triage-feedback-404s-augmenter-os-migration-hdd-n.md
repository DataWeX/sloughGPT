---
id: 20260808_115226_regression-triage-feedback-404s-augmenter-os-migration-hdd-n
title: Regression triage: feedback 404s + augmenter + OS migration HDD->NVMe
status: done
tags: regression,testing,migration,feedback
created: 2026-08-08T11:52:26.565159+00:00
---

Regression triage: feedback 404s + augmenter + OS migration HDD->NVMe

Regression sweep: 15 failures -> 0.

ROOT CAUSE (12x test_feedback_system 404s): test fixture returned TestClient(app) WITHOUT entering the context manager, so lifespan never ran -> _phase6_routers never registered feature routers -> only the 6 pre-lifespan routes existed. Fixed fixture to 'with TestClient(app) as test_client'. Also unwrapped success_response envelope in 4 assertions (db_stats/adapters/status pruned/running+stats).

knowledge_augmenter (2x, source 'none'): tests mocked memory.search without 'score' and without topical overlap; current pipeline gates on score>=0.15 (MIN_RELEVANCE_SCORE) and _topically_related (content token overlap). Updated mock data to supply score:0.9 + overlapping content.

feedback_domain health_check_loop (1x, flaky): fixed 0.3s sleep -> poll-wait for workflow_runs>=1 (deadline 5s).

Bonus suite fixes:
- test_auto_train_integration: stale transformers mock (teacher is SloNet now; transformers not installed) -> skipif guard. 3 passed, 1 skipped.
- test_core.py TestAPI: latent lifespan-404 + pre-envelope assertion for /souls -> with TestClient(app) + response.json()['data'] list assert. Module skips in this venv (torch absent).
- Audited all 45 TestClient-without-'with' files: only 3 import full app; test_endpoint_registry already correct (client.__enter__()).

OS migration HDD->NVMe (fix SMR sda fsync stalls): script /var/tmp/migrate_to_nvme.sh. Current run: interrupted rsync at ~236M (SMR saturation) -> script step3 now cleans nested leftover mounts (findmnt -R reverse) and distinguishes stale install (hostname ...0b33055d) from partial copy of current system (...0f5ecb9e) to resume instead of abort. Re-run resumed; copy progressing (18G/42G at note time). Reuses HDD ESP sda1 (BootCurrent 0001), fresh data-root UUID minted by mkfs.ext4 + blkid readback, encrypted swap f4e72739 kept. After DONE: reboot, verify mount '/' = /dev/mapper/data-root, restart server.