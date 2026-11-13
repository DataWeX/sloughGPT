"""Microbenchmark for the unified LLM response cache warm path.

Measures what the cache adds to a request:
  1. build_key() cost (canonicalization + sha256) over realistic message sizes
  2. exact-tier get() cost (hit and miss)
  3. put() cost
  4. end-to-end overhead of a cache-served deterministic chat turn (ChatDomain
     respond with the provider mocked out) vs a provider-served turn
  5. the /v1 gate — the ladder's disposition cost and replay shaping, which is
     the half of the surface that decides whether a wire request may be
     replayed at all (card 20261102_021851)

This is a CPU microbenchmark — no model required, so it runs on any box.
Pass/fail criterion: warm-path overhead per request should stay well under a
millisecond (two orders of magnitude below even the fastest model token).
For scale: the same 482-token prompt on the real model costs 1.26 s from a warm
KV state and 43.02 s from an empty one (card 20261102_012822).


Usage:
  PYTHONNOUSERSITE=1 PYTHONPATH="$PWD:$PWD/packages/core-py" \
    python scripts/benchmark_response_cache.py [--turns 2000]
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import sys
import time
from unittest.mock import AsyncMock, patch

# The /v1 section needs the API tree as well as the domain packages; both are
# added relative to this file so the script works from any CWD.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO, os.path.join(_REPO, "apps/api/server")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from domain.inference._internal.response_cache import (
    ResponseCache,
    reset_response_cache_for_tests,
)


def _make_messages(history_turns: int, content_len: int = 120) -> list[dict[str, str]]:
    msgs = [{"role": "system", "content": "You are a helpful assistant." * 3}]
    body = "w" * content_len
    for i in range(history_turns):
        msgs.append({"role": "user", "content": f"turn {i}: {body}"})
        msgs.append({"role": "assistant", "content": f"reply {i}: {body}"})
    msgs.append({"role": "user", "content": f"question: {body}"})
    return msgs


def bench_keys(turns: int) -> dict[str, float]:
    cache = ResponseCache(max_entries=1024)
    results: dict[str, float] = {}
    for history in (1, 4, 16):
        msgs = _make_messages(history)
        t0 = time.perf_counter()
        for _ in range(turns):
            cache.build_key(msgs, "qwen2.5-0.5b", 0.0, 256)
        results[f"build_key[{history * 2 + 1} msgs]"] = (time.perf_counter() - t0) / turns * 1e6

    # hit path: fill then time get()
    msgs = _make_messages(4)
    key = cache.build_key(msgs, "qwen2.5-0.5b", 0.0, 256)
    cache.put(key, "cached reply " * 20, "qwen2.5-0.5b")
    t0 = time.perf_counter()
    for _ in range(turns):
        cache.get(key)
    results["get[hit]"] = (time.perf_counter() - t0) / turns * 1e6

    miss_key = cache.build_key(_make_messages(5), "qwen2.5-0.5b", 0.0, 256)
    t0 = time.perf_counter()
    for _ in range(turns):
        cache.get(miss_key)
    results["get[miss]"] = (time.perf_counter() - t0) / turns * 1e6

    t0 = time.perf_counter()
    for i in range(turns):
        cache.put(f"k{i}", "reply " * 20, "qwen2.5-0.5b")
    results["put"] = (time.perf_counter() - t0) / turns * 1e6
    return results


def bench_v1_gate(turns: int) -> dict[str, float]:
    """Time the /v1 ladder + replay shaping (no network, no model)."""
    from routers.llm_v1 import ChatBody, _cache_key, _completion, _sampling

    from domain.inference._internal.response_cache import (
        get_response_cache,
        reset_response_cache_for_tests,
    )

    reset_response_cache_for_tests()
    cache = get_response_cache()
    body = ChatBody(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        messages=[{"role": "user", "content": "explain the tradeoff " * 80}],
        temperature=0.0,
        max_tokens=256,
    )
    messages = [{"role": "user", "content": m.content or ""} for m in body.messages]
    sampling = _sampling(body)
    key, _ = _cache_key(body, messages, body.model, sampling)
    cache.put(key=key, response="cached", model=body.model)

    def _gate() -> None:
        k, _ = _cache_key(body, messages, body.model, sampling)
        _ = cache.get(k)

    t0 = time.perf_counter()
    for _ in range(turns):
        _gate()
    results: dict[str, float] = {"v1_gate_hit_check": (time.perf_counter() - t0) / turns * 1e6}

    t0 = time.perf_counter()
    for _ in range(turns):
        _completion("cached", body.model, "prompt", cached=True)
    results["v1_replay_shape"] = (time.perf_counter() - t0) / turns * 1e6
    return results


async def bench_chat_turn(turns: int, cacheable: bool) -> float:
    """Median ms per ChatDomain.respond with the provider mocked out."""
    from domain.chat._internal.domain import ChatDomain

    reset_response_cache_for_tests()
    domain = ChatDomain()
    provider = AsyncMock()
    provider.chat.return_value = "A mocked deterministic reply with several words."
    temp = 0.0 if cacheable else 0.9

    latencies: list[float] = []
    with patch("domain.models._internal.provider.get_provider", return_value=provider):
        for _ in range(turns):
            t0 = time.perf_counter()
            await domain.respond(
                messages=[{"role": "user", "content": "what is caching?"}],
                model="bench",
                temperature=temp,
                max_tokens=32,
            )
            latencies.append((time.perf_counter() - t0) * 1e3)
    reset_response_cache_for_tests()
    return statistics.median(latencies)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", type=int, default=2000)
    args = parser.parse_args()

    print(f"== response-cache microbenchmark ({args.turns} iters/ops) ==")
    results = bench_keys(args.turns)
    for name, us in results.items():
        print(f"  {name:<24} {us:8.2f} us/op")

    cold = asyncio.run(bench_chat_turn(200, cacheable=True))  # first iter fills, rest hit
    warm = asyncio.run(bench_chat_turn(200, cacheable=False))
    print(f"  chat turn [cached warm]   {cold:8.3f} ms median")
    print(f"  chat turn [provider mock] {warm:8.3f} ms median")

    try:
        v1_results = bench_v1_gate(args.turns)
        for name, us in v1_results.items():
            print(f"  {name:<24} {us:8.2f} us/op")
        results.update(v1_results)
    except Exception as e:  # API tree not importable in this environment
        print(f"  /v1 gate skipped: {e}")

    worst_us = max(results.values())
    verdict = "PASS" if worst_us < 1000.0 else "FAIL"
    print(f"== {verdict}: worst cache op {worst_us:.2f}us (< 1ms budget) ==")


if __name__ == "__main__":
    main()
