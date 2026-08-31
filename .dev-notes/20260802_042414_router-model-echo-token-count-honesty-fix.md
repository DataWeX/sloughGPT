---
id: 20260802_042414_router-model-echo-token-count-honesty-fix
title: Router model echo + token count honesty fix
status: done
tags: api,inference,observability
created: 2026-08-02T04:24:14.929827+00:00
---

Router model echo + token count honesty fix

Fixed /inference/generate, /inference/generate/stream, /chat, /chat/stream to report the actual loaded model (state.model_type) instead of echoing the request default 'gpt2'. Added _count_tokens() using the loaded tokenizer (fallback len(text.split())). Verified live: model=Qwen/Qwen2.5-0.5B-Instruct, tokens=32 real. Tests: 7 (test_inference_generate) + 17 (server inference router) pass. Stability benchmark 100/100 (4 runs).