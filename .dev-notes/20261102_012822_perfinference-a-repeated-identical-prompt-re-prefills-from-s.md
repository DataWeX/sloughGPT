---
id: 20261102_012822_perfinference-a-repeated-identical-prompt-re-prefills-from-s
status: done
board: 20261102_012822_perfinference-a-repeated-identical-prompt-re-prefills-from-s
title: "perf(inference): a repeated identical prompt re-prefills from scratch — /v1 session_id gains nothing when the prompt is unchanged"
tags: ["inference", "kv-cache", "perf", "v1"]
sprint: ""
gh: null
---

# perf(inference): a repeated identical prompt re-prefills from scratch — /v1 session_id gains nothing when the prompt is unchanged

FOUND 2026-11-02 while verifying the /v1 session_id KV reuse that landed in bbcfafba1/3dbcac51. This is an OPTIMIZATION, not a correctness bug: the current behaviour is deliberate and pinned by tests/core-py/test_slonet_kv_cache.py::test_resolve_identical_prompt_falls_back ("same prompt -> degenerate case -> falls back to fresh").

MEASURED (live :8000, Qwen2.5-0.5B-Instruct, 482-token prompt, temperature=0):
- turn 1 (cold), session=ext-A .............. 16.48s
- turn 2 EXTENDS turn 1, same session ........ 1.96s   <- reuse works
- turn 2 same messages, COLD session ........ 19.38s   <- full prefill
- repeat the SAME 482-token prompt, warm ..... ~16s    <- header says: no gain

WHY: SloTransformer._resolve_kv_state (domain/training/_internal/slonet.py:6003) walks the shared prefix to `s`, then:
    if 0 < s < prompt_len: start = s
    else: start = 0
When the new prompt is IDENTICAL to the cached one, s walks all the way to prompt_len, the strict `<` fails, and start=0 — the entire 482-token prefill is recomputed even though the KV for nearly all of it is sitting in the session state. Verified at unit level (no model): prev=[1,2,3,4] -> identical prompt start=0, +1 token start=4, diverge at idx1 start=1, disjoint start=0.

WHY IT MATTERS: a retried request, a client resending the same context, or an agent loop that re-submits an unchanged prompt pays the full prefill every time (482 tokens ~= 16s on this box). Related: card 20261102_001621 (chat/generate repeats the completed block) shows repeat submissions are plausible in practice.

PROPOSED (one line, no new machinery): allow a full-prefix match to resume at prompt_len - 1, i.e. recompute ONLY the last prompt token, which is exactly what an incremental decode step does and is required anyway to produce logits for the next token:
    if 0 < s: start = min(s, prompt_len - 1)
    else: start = 0
The CV backends already handle arbitrary start>0 (the s<prompt_len path is in production); `state.backend.lens[:] = [start]` resets the fill length so the recomputed token's K/V overwrites the stale one. Expected: a repeat of a 482-token prompt drops from ~16s to roughly one decode step.

NOT YET IMPLEMENTED — needs approval + a benchmark (AGENTS.md: inference changes run scripts/benchmark_slonet.py / benchmark_latency.py) before it lands. The pinning test must be updated in the same commit, and its docstring should say WHICH card changed the contract.

ACCEPTANCE: replaying an identical prompt on a warm session is measurably faster than a cold session on the same prompt, with output byte-identical to the current full-recompute path (extend tests/core-py/test_slonet_kv_cache.py::TestCrossTurnGenerateNumpy::test_fresh_vs_reuse_same_input to cover the now-resuming case); no regression in benchmark_slonet.py.

── IMPLEMENTED + MEASURED 2026-11-02 (landed on main as 959484af8) ──

FIX: `_resolve_kv_state` had `if 0 < s < prompt_len: start = s else: start = 0`. A full-prompt match (s == prompt_len) therefore fell back to a full re-prefill. Now `if 0 < s: start = min(s, prompt_len - 1)` — exactly one token is recomputed, which is what an ordinary incremental decode step does and is required anyway because the final token's logits are never stored. Same function serves generate_numpy and generate_numpy_stream, so the stream path (what /v1 actually uses) is covered by the same change.

A/B ON THE REAL MODEL (Qwen2.5-0.5B .slnc, 482-token prompt, in-process, same weights, temperature=0, model load excluded; unchanged tree vs the branch):

                       before      after
  cold (empty state)   42.98s     49.69s
  repeat (warm state)  43.02s      1.26s   <- the path this card is about
  fresh (no state)     44.51s     43.62s

~34x on the repeat path. Output identity is asserted, not assumed: cold == repeat == fresh in both trees (and repeat == fresh is now a test for both the non-stream and the stream path).

TESTS: the degenerate case was pinned on purpose (`test_resolve_identical_prompt_falls_back`), so the pin was UPDATED, not deleted — renamed `test_resolve_identical_prompt_resumes_at_the_last_token`, asserting start == prompt_len - 1 and kv_len reset to it. Added `test_identical_prompt_resume_matches_fresh` and `test_identical_prompt_stream_resume_matches_fresh`, because nothing previously executed a full-prompt resume and /v1 is served by generate_numpy_stream.

GATES: 175 passed (test_slonet_kv_cache, test_slonet_generate, test_slonet_session_ttl, test_slonet_kv_benchmark, /v1 router, test_response_cache); scripts/benchmark_slonet.py all 7 benchmarks PASS (gradient_flow, training_convergence, export_import_roundtrip, accelerator_speed, lstm_forward_backward_speed, inference_consistency, transformer_vs_pytorch); ruff check + format clean.

NOTE ON A FLAKE SEEN ONCE: test_slonet_kv_benchmark.py::test_two_turn_speedup asserts wall-clock (`total_cached <= total_fresh * 1.5`) on a tiny model and failed once in a combined run (0.62x), then passed 3/3 alone and 2/2 in the same combined set, and also passed on the pre-change tree. It is timing noise, not this change — but a wall-clock assertion is the wrong gate for a cache. Deserves its own card: assert recomputed-token counts (deterministic) and report wall clock as output only.

COMPANION: this makes a warm PREFIX cheap (~1 token). It does not skip the model call — that is the response-cache layer (domain/inference/_internal/response_cache.py, exact sha256 tier, deterministic-only), which is wired in ChatDomain.respond but NOT in /v1 or /infer.
