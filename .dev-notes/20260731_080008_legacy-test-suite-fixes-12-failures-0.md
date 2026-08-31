---
id: 20260731_080008_legacy-test-suite-fixes-12-failures-0
title: Legacy test suite fixes (12 failures -> 0)
status: done
tags: tests,legacy
created: 2026-07-31T08:00:08.886779+00:00
---

Legacy test suite fixes (12 failures -> 0)

Fixed all 12 legacy tests/ failures so make test-py (packages/core-py/tests + tests) runs clean.

- tests/test_inference_generate.py: fixture builds minimal FastAPI + include_router(inference.router); mock_provider also patches state.model (route guard requires STARTUP_PHASE=ready AND state.model). 6/6 pass.
- tests/test_api.py: REWROTE 3 failing tests against current APIs instead of deleting (per user instruction - fix before delete): test_health_check -> get_server_state().uptime_seconds; test_quantization_type_validation -> QuantMode/QuantDtype/QuantEngine from domains.infrastructure.quantization; test_endpoints_exist -> quantize_state_dict/quantized_linear/NativeEngine. 14/14 pass.
- tests/test_config.py: removed TestInferenceConfig (2 tests of deleted GenerationConfig).
- tests/test_chat_loop_e2e.py: feedback payload updated to current WorkflowFeedbackRequest schema (conversation_id/rating/assistant_response/user_message). 6/6 pass.
- Previously git rm'd tests/test_agents_router.py (dup of tests/server/) and tests/test_torch_runtime.py (dead module).
- pytest.ini: 4 ignores remain (morph_tokenizer, safetensors_loader, numpy_engine, point_library) - weight-dependent, bandwidth-gated.