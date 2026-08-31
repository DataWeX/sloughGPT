---
id: 20260806_120900_manual-model-load-flow-fixed-slonet-publish-to-state-servers
title: Manual model load flow fixed — SloNet publish to state + ServerState
status: done
tags: backend,models,api,bugfix
created: 2026-08-06T12:09:00.016774+00:00
---

Manual model load flow fixed — SloNet publish to state + ServerState

Fixed the manual POST /models/load flow so inference works after loading a model from the UI.

ROOT CAUSES + FIXES
- A: routers/models.py declared response_model=LoadModelResponse but load_model returned the success envelope -> FastAPI coerced it, stripping data. Now StandardResponse[LoadModelResponse]; live body returns {status:success, data:{status:loaded, model_id, type, device, loaded_at}}.
- B: event recorder checked result.get('success') but controller emits 'status' ('loaded'/'error') -> every load logged as error. Now uses result.get('status')=='loaded'.
- C: controllers/_load_hf_model ran setup_providers but never published the provider to server_state -> inference guards (state.model is not None) returned 503 forever. Now publishes provider to state.model/provider/model_type after success, and raises if 'slonet-native' is not registered.
- D: two independent model slots existed: apps/api/server/state.py module AtomicRefs (inference guards, health model_loaded) and core ServerState singleton (get_health_score). Fixed by mirroring the publish to get_server_state().model/model_type; unload_model now clears both.
- E: unload event recorded ctrl._current_model AFTER unload reset it (always 'unknown'); capture model id before unload, fall back to registry.default_id.

LIVE VERIFICATION (Qwen/Qwen2.5-0.5B-Instruct, cached .slnc on Linux)
- gpt2 NOT cached (no .slnc) -> setup_providers correctly logs 'No .slnc file' and the publish block now surfaces the real error. Loading gpt2 would require ~500MB download (not done, bandwidth).
- /models/load -> loaded envelope; /health + /health/detailed both model_loaded=True, model_type set, health_score 'Qwen... loaded.', summary no longer 'No model loaded.'
- /inference/generate: 'The capital of France is Paris.' /chat: 'Hello' /chat/stream + /inference/generate/stream: SSE tokens stream.
- Unload -> model_loaded False (both slots cleared); reload -> works again; events: load/unload/load with correct model ids.

TESTS (all pass)
- apps/api/server/tests: 294 passed
- test_models_router.py 22 (5 new load envelope/event + 2 new unload event)
- test_models_controller_finetuned.py 13 (2 new: publish to both slots on success; no publish on setup_providers failure)
- core test_server_integration.py 57 slow-marked pass

SERVER: left running on :8000 with Qwen loaded (start via setsid nohup env SLO_AUTOLOAD_MODEL='' .venv/bin/python apps/api/server/main.py).