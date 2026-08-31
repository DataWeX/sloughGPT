---
id: 20260806_133359_fix-failed-model-load-no-longer-breaks-live-inference-defaul
title: Fix: failed model load no longer breaks live inference (default router clobber)
status: done
tags: models,provider,slonet,stability
created: 2026-08-06T13:33:59.611393+00:00
---

Fix: failed model load no longer breaks live inference (default router clobber)

Root cause: setup_providers() (packages/core-py/domains/models/provider.py) always rebuilt the 'default' ProviderRouter when the requested SloNet model failed to load (e.g. gpt2, no .slnc). The slonet-native branch fails -> text_provider_name=None -> router rebuilt with no text provider -> registered as 'default', clobbering the working Qwen router. Inference then returned 'No text model configured' even though server_state.model still held Qwen.

Fix (provider.py ~line 1065): only rebuild the default router when a text provider was successfully registered OR no default router exists. A failed load now preserves the existing working router:
  if not _is_slonet and (text_provider_name or existing is None):

Complements the controller-level guard from the prior session (_load_hf_model verifies provider._model_id matches before publishing), which prevented state corruption but could not stop the router clobber inside setup_providers.

Test: test_provider_processors.py::test_failed_load_preserves_existing_default_router (46 tests now pass).

Verification (all green):
- provider_processors (46) + slonet_integration (23) + server_integration slow (57) pass
- Full apps/api/server/tests suite: 295 passed
- Live E2E: load Qwen -> infer OK -> failed gpt2 load returns error envelope (model_id null, guard message) -> infer STILL WORKS (2+2等于4。) -> /health model_type=Qwen
- Stability benchmark (AGENTS.md): 100/100 GOLD (was FAIL: latency_degradation 3.15, overall 75 before fix)

Pre-existing failures found (unrelated, untouched): test_agent_core.py 6 failures - ToolCapability enum missing CITATION/FILE_SEARCH members (AttributeError). test_chat_trainer.py is slow (~2min) but passes; full core suite exceeds 10min so targeted verification used.