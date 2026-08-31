---
id: 20260808_023715_server-suite-flake-elimination-no-model-inference-tests
title: Server suite flake elimination: no-model inference tests
status: done
tags: server,tests,inference,flake
created: 2026-08-08T02:37:15.809552+00:00
---

Server suite flake elimination: no-model inference tests

Made 3 order-dependent no-model inference tests deterministic: added no_model_state fixture (snapshots real state.model/provider + startup_progress.STARTUP_PHASE, forces no-model condition, restores in finally). Full tests/server suite: 824 passed, 68 deselected across 4 consecutive runs.