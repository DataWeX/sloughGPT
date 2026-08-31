---
id: 20260808_130552_server-test-thickening-inference-infer-health
title: server-test-thickening-inference-infer-health
status: done
tags: tests,server
created: 2026-08-08T13:05:52.089861+00:00
---

server-test-thickening-inference-infer-health

Thickened 3 thin server router suites. test_inference_router.py 27->58 (generate/chat validation 422s+405s, session list/search empty+archived, context store/reset/facts no-core fallbacks, voice non-audio/no-file/not-found/traversal-guard via direct coroutine, stream no-model/no-provider SSE). test_infer_router.py 29->46 (stream SSE error paths, embed ngram-tfidf fallback + ndarray/list embeddings, tokenize/detokenize, 422s, detokenize negative ids, 405s). test_health_router.py 29->34 (model health ok-with-stats merge, set_model-on-state-load assertion, controller-exception->500, /health/model + /health/stream 405s). Full server suite: 1733 passed, 68 deselected, 67.25s. pycache cleared.