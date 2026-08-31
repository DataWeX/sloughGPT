---
id: 20260809_161932_vm-console-persist-train-config-reset-fix-streamsse-regressi
title: VM console: persist train config + reset; fix streamSSE regression
status: done
tags: vm,console,frontend
created: 2026-08-09T16:19:32.246703+00:00
---

VM console: persist train config + reset; fix streamSSE regression

Training launch config persisted to localStorage (vm-train-config), restored on mount with clampTrainConfig, Reset config button restores defaults + clears custom dataset. +2 page tests (48), +1 e2e spec (11), docs updated. Also fixed a regression from an external streamSSE change (http-client.ts/chat-controller.ts edited 16:37): streamSSE now prefixes PUBLIC_API_URL and includes statusText in HTTP error messages; __test-helper mock factory preserves the real streamSSE export; generateController.generateStream calls onDone on error (was 30s hang). Full suite 327 files / 3164 tests, tsc exit 0.