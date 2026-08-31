---
id: 20260809_193404_vm-console-training-config-fallback-hints
title: VM console: training config fallback hints
status: done
tags: vm,console,frontend
created: 2026-08-09T19:34:04.567409+00:00
---

VM console: training config fallback hints

Training launch card now lists inline 'using default N' hints for cleared/invalid config fields (dataset, epochs, lr, batch, layers, heads, embed). Numeric field onChange uses num() to store NaN for empty strings so clamping and hints align. +2 page tests (52), +1 e2e spec (13), docs updated. Full suite 328 files / 3172 tests, tsc exit 0.