---
id: 20260808_030528_20260808-knowledge-enrichment-relevance-gate
title: 20260808 Knowledge enrichment relevance gate
status: done
tags: 
created: 2026-08-08T03:05:28.058445+00:00
---

20260808 Knowledge enrichment relevance gate

Root cause of the Quick-test /chat garbage output isolated: knowledge enrichment injects facts into the system prompt with NO relevance threshold (knowledge_augmenter.py ignored search score). /chat Hello! injected all 3 facts (Paris/ML/Python) -> model echoed them. Not KV-cache/session reuse (fresh session, /inference/generate clean on same model). Fix: _is_casual_small_talk guard skips retrieval for greetings/small talk; MIN_RELEVANCE_SCORE=0.15 filters low-similarity facts (configurable min_score param, applied to memory + web paths). Verified live: /chat Hello! -> clean reply; 'Is Paris the capital of France?' -> correct answer. 29/29 augmenter tests, 242 knowledge tests, 27/27 inference router tests pass.