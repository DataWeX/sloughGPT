---
id: 20260804_052219_kv-cache-wave-4-provider-wiring-session-level-state
title: KV Cache Wave 4: Provider wiring + session-level state
status: done
tags: slonet,kv-cache,inference,provider
created: 2026-08-04T05:22:19.366751+00:00
---

KV Cache Wave 4: Provider wiring + session-level state

Wired cross-turn KV cache into slonet_provider.py. Added _kv_states dict (per-session NumpyKVState) to SloNetChatProvider instance. Updated _generate_sync() and chat_stream()._stream_generate() to accept session_id, resolve/create kv_state, and pass it to generate_numpy/generate_numpy_stream. Updated chat() to forward session_id from kwargs. All 210 slonet/provider/quant/kv tests pass.