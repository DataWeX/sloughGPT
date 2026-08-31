---
id: 20260809_145738_vm-console-training-launch-card
title: VM console training launch card
status: done
tags: vm,training,frontend
created: 2026-08-09T14:57:38.216739+00:00
---

VM console training launch card

INCREMENT 2 (continued): Training launch card now uses a real dataset dropdown. app/(app)/vm/page.tsx: datasetController.list() fetched on mount into datasetNames; dataset field renders a <select> (backend names + current value + 'Custom…' option revealing a free-text input) when datasets exist, falling back to the text input when the list is empty/unreachable. clampTrainConfig() clamps invalid/empty numeric fields at generate time (integers >= 1, lr > 0, defaults restored for empty) — used by both 'Launch training' (handleLaunchTraining) and 'Load sample'. Tests: +4 page tests (42 total): dropdown render from backend, selecting dataset used at launch, Custom… reveals input used at launch, launch clamps cleared config to defaults. E2E launch spec updated to reload with a /datasets override and select 'tinyshakespeare' from the dropdown. docs/VM_CONSOLE.md updated. Verified: tsc exit 0; vm page 42/42; full web suite 327 files / 3158 tests all pass (ModelDetailPage.test.tsx excluded per documented hang).