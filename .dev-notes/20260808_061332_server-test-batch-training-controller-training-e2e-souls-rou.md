---
id: 20260808_061332_server-test-batch-training-controller-training-e2e-souls-rou
title: Server test batch: training_controller, training_e2e, souls_router
status: done
tags: tests,server
created: 2026-08-08T06:13:32.008910+00:00
---

Server test batch: training_controller, training_e2e, souls_router

Thickened 3 thin server router suites. training_controller 17->28 (filesystem checkpoints/datasets, jobs persistence, corrupt jobs, kwargs merge). training_e2e 17->30 (deterministic 422 validation for start bounds, status/log shape). souls_router 18->36 (save/load/delete weight snapshots, save_trait_weights flatten, checkpoint_name switch flow, error paths). FOUND+FIXED BUG: souls.py all 10 error handlers passed invalid code= kwarg to error_response -> every error path 500'd as 'Internal Server Error' instead of JSON; same pattern fixed in models.py:386. Full suite: 1244 passed, 68 deselected.