---
id: 20260731_032515_sloengine-tests-soulpy
title: SloEngine tests (soul.py)
status: done
tags: core,soul,tests
created: 2026-07-31T03:25:15.514814+00:00
---

SloEngine tests (soul.py)

Objective: test domains/core/soul.py (SloEngine, GenerationContext), fix bugs found.

- Added packages/core-py/tests/test_soul_engine.py: 126 tests covering GenerationContext defaults/custom, engine init/setters, system-prompt trait blocks, reasoning chains, history caps, temp adjustments, Hebbian connections, tokenize/detokenize fallbacks, generate() sync/async + model path + return_reasoning, chat() semantics, apply_personality + integrity hash, stats/status, working memory (cap 7), HD memory, semantic cache, removed stubs, to(device), deep_reason/prove_syllogism, grounding, save_soul. All 126 pass.
- Fixed real bug in soul.py: latency_ms was read in semantic-cache put() metadata before assignment -> NameError silently killed every cache store. Moved latency_ms = (time.time()-start_time)*1000 before the cache block; removed later duplicate. Verified cache entries now persist (was 0).
- Full regression (module-per-timeout, 5 ignored modules): 3988 passed, 35 skipped. 19 failures + 64 errors + 10 collection-skips all env/pre-existing (missing pydantic/starlette/psutil, missing model weights for gpt2/qwen2/numpy-engine/safetensors/morph, CPU-count detection, boot string, shell command exit code, slonet mul-backward grad shape). None touch soul.py.