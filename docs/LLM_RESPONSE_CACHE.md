# LLM Response Caching

Two response-level cache tiers over one key discipline, wired into the real
chat serving path (`domain/chat/_internal/domain.py::ChatDomain.respond`).

Module: `domain/inference/_internal/response_cache.py`

## The invariant

**A cache key is valid only if every input that can change the output is in
the key.** The `SessionKVCache` benchmark-poisoning incident (same prompt +
same session replayed stale tokens from the previous model at 0.017s) was a
key that omitted infrastructure state. `ResponseCache.build_key()` therefore
hashes a canonical JSON of:

- ordered messages (role + content each)
- system prompt
- model id
- temperature (rounded) + max_tokens
- **generation token** — a process-wide counter

Change any one field → different key → miss. Correctness by construction, not
by convention.

## Invalidation: generation bump

`bump_cache_generation()` advances the process-wide token and is called from
`register_provider()` (`domain/models/_internal/provider/registry.py`) — the
single choke point for every model load/unload/swap. Because the token is part
of every key, a bump invalidates both tiers coherently at once. No lazy
per-entry revalidation, no stale-model replay.

## Store gates (what may be cached)

A response is stored only when **all** hold:

| Gate          | Rule                     | Why                                                           |
| ------------- | ------------------------ | ------------------------------------------------------------- |
| Deterministic | `temperature <= 0.2`     | Replaying a stochastic sample is a correctness lie, not a hit |
| Non-streaming | `respond()` path only    | A cached response cannot serve a half-stream                  |
| Non-error     | not `[Error...` prefixed | Never cache failures                                          |

Stochastic requests are counted (`bypass_stochastic`) and always generate.

## Tiers

- **Exact tier** (always on): sha256 exact match, LRU + TTL, thread-safe
  (RLock + OrderedDict), stats surface (hits/misses/stores/evictions/
  expirations/bypasses/semantic_hits).
- **Semantic tier** (opt-in, off by default): paraphrase fallback reusing
  `SemanticCache`, double-gated behind the `semantic_cache` feature flag AND
  the deterministic-temperature rule, high similarity threshold (0.85).
  Semantic matching is unsafe as a primary lookup for general chat.

The session KV machinery in `slonet_provider.py` is a separate, lower-level
layer (token-prefix reuse within a session) and is untouched by this design.

## Benchmarks

`scripts/benchmark_response_cache.py` — microbenchmark, no model required:

```
PYTHONNOUSERSITE=1 PYTHONPATH="$PWD:$PWD/packages/core-py" \
  python scripts/benchmark_response_cache.py
```

Measured (2026-10-12, CPU box): build_key 15–70µs by history size, get hit
0.55µs, put 3.2µs, warm cached chat turn 0.111ms vs 0.114ms provider-mocked —
negligible against any model's token latency.
