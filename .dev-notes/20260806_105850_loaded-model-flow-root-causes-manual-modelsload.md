---
id: 20260806_105850_loaded-model-flow-root-causes-manual-modelsload
title: Loaded-model flow root causes (manual /models/load)
status: done
tags: models,backend,root-cause
created: 2026-08-06T10:58:50.422037+00:00
---

Loaded-model flow root causes (manual /models/load)

Session: live-integration check of loaded-models UI vs real API exposed 4 backend bugs in the manual POST /models/load path. No code changes made (user opted to document only).

ROOT CAUSES
A) routers/models.py:59 declares response_model=LoadModelResponse but load_model returns success_response(data=result). FastAPI coerces the envelope into LoadModelResponse: status field-name collision keeps 'success', data dropped -> live response {status:success, model_id:null, device:null, parameters:0, error:null, type:null, loaded_at:null}. Real result invisible to client.
B) routers/models.py:223 uses result.get('success') but controller result key is 'status' ('loaded'/'error') -> always falsy -> every load records an error model event with detail 'unknown' (visible in /health/stream model_events).
C) controllers/models.py:107-154 _load_hf_model runs setup_providers(slonet_hf_id=...) (log confirmed 'SloNet provider registered: gpt2 (quant=int8)') but never sets server_state.model/tokenizer/provider. Autoload (startup.py:607-613) does. Inference guards (inference.py:378/431/583/925) require state.model is not None -> /inference/generate, /chat, /chat/stream 503 'Model still loading' forever. state.model is a real model object used by export (models.py:366), MPS monitor, multimodal, inference.py:498.
D) Health contradiction: top-level model_loaded/model_type from _get_model_info_with_registry (controller _current_model fallback, health.py:84), but health_score.summary from ServerState.get_health_score() using self.model.get() (server_state.py:306-307) -> live detailed health model_loaded=True, model_type=gpt2, summary='No model loaded.' StatusBar shows green dot + 'No model loaded.' text.

FRONTEND: no changes needed - all error paths graceful (ModelPlaygroundCard setTestOutput Error:, generateController throws on result.error, /models/quantize 400 caught). Frontend consumes /models/hf uniformly for catalog.

FIX PLAN (not applied): A) response_model=StandardResponse[LoadModelResponse]; B) result.get('status')=='loaded'; D) align health_score model_loaded source; C) populate state.model from SloNet provider or relax guards.