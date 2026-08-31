---
id: 20260804_090304_streaming-stack-kv-benchmark-stack-stream
title: Streaming stack KV benchmark (--stack --stream)
status: done
tags: inference,kv-cache,benchmark
created: 2026-08-04T09:03:04.583200+00:00
---

Streaming stack KV benchmark (--stack --stream)

Extended benchmark_kv_reuse.py with --stack --stream driving SloNetServer.generate_stream -> generate_numpy_stream(kv_state=...), the real /chat/stream SSE path. Two benchmark bugs found while validating: (1) streaming yields only new tokens while batch generate echoes the prompt, so history must retain the full prior turn (prompt+output) or prompts never accumulate; (2) stack reused_tokens was session state fill (kv_len) which overstates reuse -> now honest prefix_match(prompt_ids, state.prev_ids). All three modes (direct, stack batch, stack stream) now report identical reuse 0->12->23->34 and 100% consistency; streaming overall 1.45x. 5 new tests (20 total) incl stack reuse = prior turn cached prefix.