---
id: 20261102_012822_fixguard-slonetservergenerate-drops-session-id-on-the-proces
status: done
board: 20261102_012822_fixguard-slonetservergenerate-drops-session-id-on-the-proces
title: "fix(guard): SloNetServer.generate() drops session_id on the process-guard branch — cross-turn KV reuse is a silent no-op for direct provider.chat() callers"
tags: ["inference", "gpu", "process-guard", "kv-cache"]
sprint: ""
gh: null
---

# fix(guard): SloNetServer.generate() drops session_id on the process-guard branch — cross-turn KV reuse is a silent no-op for direct provider.chat() callers

FOUND 2026-11-02 while verifying the /v1 session_id KV reuse (landed in 3dbcac51).

THE GAP: `SloNetServer.generate()` has two branches. When a ProcessGuard is attached AND alive (the default: config get_process_guard_enabled() is True, boot logs "Lazy-guard autoload: ProcessGuard started (THREAD mode, pre-loaded provider)"), it delegates:
    result = self._process_guard.generate(prompt, max_new_tokens=..., temperature=..., ...)
and **session_id is not forwarded**. Even if it were, the worker side cannot use it as a KV key:
- ProcessGuard.generate (domain/infrastructure/_internal/process_guard.py:648) -> ThreadWorker.generate:172 mints its OWN `session_id = _new_session_id()` for IPC reply correlation, unrelated to the caller's KV session; the payload dict carries only max_new_tokens/temperature/top_p/top_k/repetition_penalty.
- ThreadWorker._generate_fn (process_guard.py:367) calls `model.generate_numpy(input_ids, ...)` with **no kv_state** at all.

So a direct `SloNetChatProvider.chat(...)` with a live guard silently forfeits cross-turn reuse — the exact silent-no-op class this port just fixed at the /v1 boundary.

NOT BROKEN TODAY: /v1 is unaffected. ProviderRouter.chat delegates to chat_stream (router.py:175's comment pins that), so /v1 traffic goes provider.chat_stream -> SloNetServer.generate_stream -> _generate_stream_sync(pump) -> SloNetServer._resolve_kv_state, which DOES honour session_id. Measured proof: with 8 /v1 requests served, GET /models/process-guard reported health.requests_served == 1 (the boot warmup only) — /v1 never entered the guard branch. Live reuse confirmed: turn-2-extends-turn-1 on the same session = 1.96s vs 19.38s cold.

WHY CARD IT ANYWAY: /v1 non-stream only escapes the guard branch by accident of going through chat_stream. `GET /models/process-guard` says the guard is the active execution path, and any future caller that (a) calls SloNetChatProvider.chat directly or (b) starts sending session_id on /infer (routers/infer.py:244) or /generate (routers/inference.py:1186) gets a silent no-op instead of an error. Also means guard-mode and non-guard-mode generation are not the same code path, which is a merge trap.

FIX SKETCH (needs approval; it touches the IPC contract, so it is a brief not a one-liner):
1. Split the two ids explicitly: keep `_new_session_id()` as the IPC correlation id, add `kv_session_id = kwargs.get("session_id")` to the payload dict in ThreadWorker.generate/generate_stream and in the subprocess worker equivalent.
2. ThreadWorker._generate_fn: resolve `provider._resolve_session_kv(kv_session_id)` (mirroring SloNetServer._resolve_kv_state) when a provider is bound, and pass `kv_state=` + `quantize_kv=` into `model.generate_numpy` / `generate_numpy_stream`.
3. SloNetServer.generate guard branch: forward `session_id=session_id`.
4. Test at the guard seam with a fake worker: same session_id twice must resolve the SAME kv_state object; a None session_id must resolve None. Assert the object identity, not just that a call happened (the lesson from bbcfafba1's reset_accelerator bug: count calls and you miss which state was served).

CONSTRAINT: the worker is the crash-isolation boundary — KV state lives in the worker process, so per-session state must be owned there (not in the parent), and the TTL/LRU eviction currently in SessionKVManager has to exist on that side too. Do not simply re-send buffers over IPC.
