---
id: 20261102_021851_perfinference-v1-has-no-response-cache-a-repeated-determinis
status: done
board: 20261102_021851_perfinference-v1-has-no-response-cache-a-repeated-determinis
title: "perf(inference): /v1 has no response cache — a repeated deterministic prompt still runs the model"
tags: ["inference", "serving", "cache", "/v1"]
sprint: ""
gh: null
---

# perf(inference): /v1 has no response cache — a repeated deterministic prompt still runs the model

SCOPE: apps/api/server/routers/llm_v1.py + apps/api/server/routers/health.py + tests.

FOUND 2026-11-02 while closing 20261102_012822. ResponseCache (domain/inference/_internal/response_cache.py) exists, is tested, and is wired in exactly ONE caller: ChatDomain.respond. /v1 and /infer never touch it, so a repeated deterministic request on the external wire re-runs the model even though the answer is a pure function of the inputs. The KV work landed today makes a warm prompt 1.26s; a response-cache hit is milliseconds. Different tiers, both worth having.

THE LADDER (agreed with the user before implementation; total function, first match wins, every ambiguous case falls to "generate" because a wrong miss costs latency and a wrong hit costs correctness, invisibly):

  1. stream            -> generate (a hit is not a stream; replay is a separate contract)
  2. temperature > 0.2 -> generate (output is not a function of the inputs)
  3. tools/tool_choice -> generate (the real input is the tool result, not the prompt)
  4. hidden context not in the key -> generate
  5. exact key hit     -> serve, tagged cached=true
  7. otherwise         -> generate, then store

Rule 6 (semantic paraphrase tier) is DELIBERATELY not wired: similarity may propose a candidate but must never decide. A test asserts a paraphrase MISSES, so re-enabling it requires arguing for it rather than flipping a flag. No key laundering either - a replay is never re-stored under a new key.

KEY COMPLETENESS is the whole correctness argument, so the key must cover every input that can change the output: messages, model id, temperature, max_tokens, top_p, top_k AS SENT (top_k=0 maps to 40 in _sampling, so the effective value is keyed or two identical requests miss each other), stop (ignored today, keyed so honouring it later cannot collide), and session_id (it selects a KV state; a non-matching state falls back to full prefill, which the tests pin as output-identical). The generation token already covers model load/unload/swap.

OBSERVABILITY: an unobservable cache becomes an unexplained latency cliff. get_stats() goes on health.health_summary (the status-bar surface) behind a try/except that can never break health.

OUT OF SCOPE, card separately: /infer (different envelope, same seam), semantic tier enablement, intent->solver routing (Rung C), cross-session prefix table (Rung A).

── IMPLEMENTED + MEASURED 2026-11-02 (landed on main as ebb79bd83, docs a2d748c01) ──

WHAT LANDED: apps/api/server/routers/llm_v1.py now runs non-stream requests through the response-cache ladder before the engine. `_cache_key()` IS the ladder — a total function, first match wins, documented in its docstring so the next session reads the rule rather than reverse-engineering it:

    stream              -> generate (a hit is not a stream; replay is its own contract)
    temperature > 0.2   -> generate (output is not a function of the inputs)
    tools / tool_choice -> generate (the real input is the tool result, not the prompt)
    otherwise           -> exact key; hit -> replay tagged cached=true, miss -> generate then store

Every ambiguous case falls to "generate" on purpose: a wrong miss costs latency, a wrong hit costs correctness and is invisible (the replay is fluent either way).

RULE 6 (paraphrase/semantic tier) IS DELIBERATELY NOT WIRED. similarity may propose a candidate but must never decide one. test_paraphrase_is_not_promoted posts "2+3" then "4+12" and asserts BOTH reach the engine — so turning the semantic tier on for /v1 requires deleting an argument, not flipping a flag. No key laundering either: a replay is never re-stored under another prompt's key.

KEY COMPLETENESS (the whole correctness argument): messages, model id, temperature, max_tokens, top_p, top_k AS SENT (note: _sampling maps top_k=0 -> 40, so keying the raw 0 would make two identical requests miss each other), stop (phase 1 ignores it, keyed anyway so honouring it later cannot collide), session_id (selects a KV state; a non-matching state falls back to full prefill, pinned output-identical by test_slonet_kv_cache). The module's generation token covers model load/unload/swap.

NEW WIRE FIELDS: ChatBody.tools / ChatBody.tool_choice, declared so the ladder can SEE a tool turn instead of silently caching one. Same convention as the existing `stop` field ("Accepted; phase 1 ignores"). CHAT_SPEC.descriptor untouched -> gen_router_manifest --check and gen_ts_contracts --check both report up to date.

OBSERVABILITY: ResponseCache.get_stats() is exposed on health.health_summary (the status-bar surface) behind a try/except that can never break health. Cache hits record 0 model tokens in record_inference, so tokens/s is not inflated by work nobody did. Hits also log at INFO and write an audit row with cached=1.

MEASURED (scripts/benchmark_response_cache.py, extended rather than duplicated — the file already existed from commit 51d3a60ac and I initially CLOBBERED it, then restored it from HEAD and added a /v1 section):
    build_key[33 msgs]          88.26 us/op
    v1_gate_hit_check           32.49 us/op     (ladder + exact lookup)
    v1_replay_shape              5.72 us/op     (wire payload shaping)
    get[hit]                     0.74 us/op
    -> full /v1 replay ~82 us, verdict PASS (<1 ms budget)
For scale, the same 482-token prompt on the real model: 1.26 s from a warm KV state, 43.02 s from an empty one (card 20261102_012822). So a replay is ~15,000x a warm KV resume and ~520,000x a cold prefill.

TESTS (10 new, apps/api/server/tests/test_llm_v1_router.py::TestResponseCacheLadder): replay happens and is shaped identically to the generated response; replay is tagged and generation is not; stochastic always generates (+bypass counter); streaming neither reads nor fills the tier (+bypass counter); top_p is in the key; top_k keyed as sent (0 == 40, 40 != 20); session_id in the key; generation bump invalidates; tool-carrying request never replayed; paraphrase not promoted. Plus TestCacheObservability: hit_rate/generation present on the health snapshot, and the snapshot returns {} instead of raising when the cache is unavailable.

GATES: 247 passed (llm_v1 router, response_cache, semantic_cache, chat_domain, chat_domain2, health router, contracts); scripts/benchmark_slonet.py all 7 PASS (gradient_flow, training_convergence, export_import_roundtrip, accelerator_speed, lstm_forward_backward_speed, inference_consistency, transformer_vs_pytorch); ruff check + format clean; manifest + TS contracts up to date.

HONEST LIMITATION: /v1's default temperature is 0.7, which is > 0.2, so the cache only engages for clients that ask for determinism (temperature <= 0.2) and do not stream. That is the correct default — a stochastic request is not a function of its inputs — but it means this is a win for eval/agent/benchmark traffic, not for chat traffic. Widening it means changing the determinism threshold, which is a separate argument.

FOLLOW-UPS FILED: cross-session prefix table (same token prefix, different client); /infer parity; intent->solver routing for genuinely different prompts with the same answer.
