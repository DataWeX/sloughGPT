---
id: 20260808_141656_fix-ui-not-found-race-during-startup
title: Fix UI Not Found race during startup
status: done
tags: web,health,startup
created: 2026-08-08T14:16:56.338480+00:00
---

Fix UI Not Found race during startup

Root cause: lazy-guard autoload flips model_loaded=true at startup step 4, routers register at step 8. Home Quick test card shows during steps 4-8, clicks hit unregistered routes -> 404. Fix: health._is_app_ready() gates model_loaded on lifecycle RUNNING (only after routers register). Added TestIsAppReady + TestGetModelInfoReadyGate (5 tests). tests/server: 1898 passed.