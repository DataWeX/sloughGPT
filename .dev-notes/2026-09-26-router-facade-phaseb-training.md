---
title: Fix router domain-facade reach-ins — phase B (training _internal)
status: done
date: 2026-09-26
---

# Router facade reach-ins — phase B (Build Order #4)

## Task
Route all `domain.training._internal` reach-ins in routers through the
public `domain.training` facade; keep guard-test ratchet moving down.

## Result (2026-09-26)
Commit `a9e475d9f` on `feat/router-facade-phaseb`, merged to main.
- 26 reach-in sites swapped to facade imports across settings (24),
  dashboard (2), models (3), model_stack (1), system (5) — wait, counts:
  settings 24, dashboard 2, models 3, model_stack 1, system 5 = 35 sites.
- New facade lazy exports: ExportConfig, export_model, list_export_formats,
  get_cache_root, list_presets, get_preset, apply_preset, get_auto_trainer,
  generate_model_card, get_turbo_lock, get_turbo_state (+phase-A names).
- system.py touched executor._instance (module private) in 5 endpoints —
  replaced with new non-creating peek_training_executor() so
  initialized=False / "not initialized" semantics preserved
  (get_training_executor() auto-creates, would have side-effected status
  endpoints into reporting initialized=True after first read).
- Guard baseline tightened 131 -> 105 reach-ins; all facades resolve.
- Gates: facade probe 14/14, guard 2/2, ruff clean, targeted router
  suites 198/198, full tests/server 1167 passed / 1 failed
  (test_knowledge_router::test_search_files = pre-existing, card 063).

## Next
Phase C: settings._internal x17, cognition.consciousness.personality x16,
shell/auth/memory/agents reach-ins (remaining 105 baseline).
