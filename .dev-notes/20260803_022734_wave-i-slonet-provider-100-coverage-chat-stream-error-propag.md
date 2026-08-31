---
id: 20260803_022734_wave-i-slonet-provider-100-coverage-chat-stream-error-propag
title: Wave I: slonet_provider 100% coverage + chat_stream error-propagation fix
status: done
tags: coverage,inference,slonet
created: 2026-08-03T02:27:34.296199+00:00
---

Wave I: slonet_provider 100% coverage + chat_stream error-propagation fix

Wave I complete: domains/inference/slonet_provider.py at 100% coverage (447 stmts, 0 missed).

## Production changes (slonet_provider.py)
1. Bug fix — chat_stream silently swallowed producer errors that occurred before the first token. The sentinel branch (line 849) broke without draining err_q, contradicting the documented 'producer thread exceptions surface to consumer' contract. Now yields '[Generation error: ...]' before breaking. Verified: streamers that raise before yielding any token now surface the error instead of an empty stream.
2. Dead code removed — the post-loop fused-QKV fallback (was lines 245-263) was provably unreachable: the canonical None-branch (qkv.weight/qkv.bias slo_target=None) already matches the identical W.get(canonical) concrete keys via _split_fused_qkv and sets mapped=True. -19 lines.
3. Timeout literals hoisted to module constants _STREAM_GET_TIMEOUT_S=30.0 and _STREAM_TOTAL_TIMEOUT_S=120.0 (replaces local _GENERATION_TIMEOUT_S) — config-over-magic-values; testable via monkeypatch.
4. pragma: no cover on the global-key transpose at line 240 — provably unreachable for supported archs (every global 2D key is either in NO_TRANSPOSE_KEYS or already mapped; GPT2/LLAMA weight maps verified).

## Tests (new: tests/test_slonet_provider_wave_i.py, 6 tests)
- Producer error after tokens surfaces via err_q (immediate error case exercises the new sentinel-error fix)
- Immediate producer error (no tokens) surfaces instead of empty stream
- Total-generation timeout: yields '[Generation timed out...]', sets cancel_event (monkeypatched _STREAM_TOTAL_TIMEOUT_S=0.5 + fake asyncio.wait_for; real to_thread timeouts leak executor threads that hang asyncio.run shutdown — fake avoids this)
- Dead-thread wait timeout: drains remaining queued tokens
- Alive-thread wait timeout with error in err_q: yields error (drain tail exercises generator return)
- Error injected between tokens: post-get err_q check (857-859)
- No-server delegation path (server None) covered

## Coverage path
91% (42 misses) -> 100% (0/447) via the 12-file family: test_slonet_provider_wave_i/provider/features/generate/integration/lstm/cnn/broadcast/bidirectional_dag/server/wave_f/kernels. Regression: test_provider_processors + test_npu pass. py_compile clean, pycache cleared.

## Commit
covers provider + wave_i test + journal/board. Excludes gpu/accelerator.py + test_gpu_accelerator.py (orphaned WIP).