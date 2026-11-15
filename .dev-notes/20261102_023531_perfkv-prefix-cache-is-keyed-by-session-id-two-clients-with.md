---
id: 20261102_023531_perfkv-prefix-cache-is-keyed-by-session-id-two-clients-with
status: done
board: 20261102_023531_perfkv-prefix-cache-is-keyed-by-session-id-two-clients-with
title: "perf(kv): prefix cache is keyed by session_id — two clients with the same token prefix both re-prefill"
tags: ["inference", "kv-cache", "serving"]
sprint: ""
gh: null
---

# perf(kv): prefix cache is keyed by session_id — two clients with the same token prefix both re-prefill

FOUND 2026-11-02 while landing 20261102_021851 (/v1 response cache).

THE GAP: cross-turn KV reuse is keyed by session_id (SessionKVManager -> KVState._sessions). Two clients — or one client reconnecting, or two routes — that send the SAME 400-token system prompt each pay full prefill for it, because their session ids differ. The prefix comparison itself already exists and returns the shared length (KVState.session_get returns (past_key_values, prefix_len), and prefix_match() walks the ids); only the LOOKUP KEY is per-session.

THE MECHANISM: a content-addressed prefix table. Hash block-aligned token prefixes (e.g. every 32/64 tokens), store the paged K/V blocks under the hash in a global LRU, and on a session miss fall back to a prefix-hash lookup before prefill. This is vLLM's automatic prefix caching, and it is the honest reading of "identical but a different route": the route differs, the tokens up to a point do not.

WHY IT IS SAFE (unlike the semantic tier): K/V is a pure function of the token ids and their positions, so a block-aligned prefix hit is token-exact by construction. No threshold, no similarity, no wrong-answer class — the only hazard is a hash collision, which a full-prefix comparison on hit removes.

MEASUREMENT THAT WOULD PROVE IT: two different session_ids, same 400-token prefix, second request timed against a cold prefill on the real model. Reference points already on the board: 43.02 s cold, 1.26 s warm for 482 tokens (card 20261102_012822).

OUT OF SCOPE: response-level replay (that is 20261102_021851, answer tier), /infer parity (separate card), and any cross-token reuse — K/V cannot be transplanted between different token sequences, and pretending otherwise produces fluent wrong output.
