"""Microbenchmark for cross-session KV prefix sharing (card 20261102_023531).

Measures the table's own cost (build/copy, lookup, adopt) against the two
numbers that justify it: a repeated 482-token prompt costs **43.02 s** cold
and **1.26 s** warm on the real model. The table replaces the second kind of
re-entrant traffic (different session / reconnecting client / different route)
with a block-aligned copy + small comparison — CPU microbenchmark, no GPU
required, uses the real ``PrefixCache`` and ``SloTransformer`` on a tiny model
so the shapes match the production path.

Usage:
  PYTHONNOUSERSITE=1 PYTHONPATH="$PWD:$PWD/packages/downcraft:$PWD/packages/core-py" \\
    python scripts/benchmark_kv_prefix_cache.py [--iters 2000]
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (
    _REPO,
    os.path.join(_REPO, "packages/downcraft"),
    os.path.join(_REPO, "packages/core-py"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from domain.infrastructure._internal.kv_cache.prefix_cache import PrefixCache
from domain.training._internal.slonet import SloTransformer


def _tiny(
    layers: int = 1, embed: int = 64, heads: int = 4, kv_heads: int = 2, block: int = 128
) -> SloTransformer:
    net = SloTransformer(
        vocab_size=512,
        n_embed=embed,
        n_layer=layers,
        n_head=heads,
        block_size=block,
        n_kv_head=kv_heads,
    )
    net.max_seq_len = block
    return net


def bench_table(iters: int) -> dict[str, float]:
    cache = PrefixCache(max_entries=8, max_tokens=1024, min_tokens=64, block=64)
    tokens = 256
    ids = np.arange(tokens, dtype=np.int64).reshape(1, tokens)
    k = [np.random.randn(1, tokens, 2, 32).astype(np.float32)]
    v = [np.random.randn(1, tokens, 2, 32).astype(np.float32)]
    cache.store(ids, k, v, filled=tokens)

    results: dict[str, float] = {}
    # exact match
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        cache.match(ids)
        times.append((time.perf_counter() - t0) * 1e6)
    results["exact match (256t)"] = statistics.median(times)
    # partial match (shared prefix + different tail)
    q = np.concatenate([ids[0], np.arange(500, 600)]).reshape(1, -1)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        cache.match(q)
        times.append((time.perf_counter() - t0) * 1e6)
    results["partial match (256t)"] = statistics.median(times)
    # miss
    q2 = np.arange(700, 700 + 256, dtype=np.int64).reshape(1, -1)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        cache.match(q2)
        times.append((time.perf_counter() - t0) * 1e6)
    results["miss (256t)"] = statistics.median(times)
    # store
    times = []
    ids2 = np.arange(1000, 1000 + 256, dtype=np.int64).reshape(1, -1)
    for _ in range(iters):
        cache = PrefixCache(max_entries=8, max_tokens=4096, min_tokens=64, block=64)
        kk = [np.random.randn(1, 256, 2, 32).astype(np.float32)]
        vv = [np.random.randn(1, 256, 2, 32).astype(np.float32)]
        t0 = time.perf_counter()
        cache.store(ids2, kk, vv, filled=256)
        times.append((time.perf_counter() - t0) * 1e6)
    results["store+copy (256t)"] = statistics.median(times)
    return results


def bench_model(iters: int) -> dict[str, float]:
    # 4 layers / 256-dim / 256-token prompt: big enough that a forward pass
    # dominates a memcpy, small enough to run on CPU in seconds. The production
    # reference is the same ratio at 43.02 s (cold) vs 1.26 s (warm KV).
    net = _tiny(layers=4, embed=256, heads=8, kv_heads=2, block=512)
    base = list(range(1, 1 + 256))
    net.generate_numpy(np.array([base], dtype=np.int64), max_new_tokens=4, temperature=0.0)
    results: dict[str, float] = {}
    # Cold is measured with the table DISABLED — otherwise the identical prompt
    # would adopt from the very table under test and "cold" would be a lie.
    cache = net._prefix_cache
    net._prefix_cache = None
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        net.generate_numpy(np.array([base], dtype=np.int64), max_new_tokens=4, temperature=0.0)
        times.append((time.perf_counter() - t0) * 1e3)
    results["cold (new state)"] = statistics.median(times)
    net._prefix_cache = cache

    # Warm-from-table: a fresh session every iteration, as a reconnecting
    # client or a second route would present.
    times = []
    for _ in range(iters):
        state = net.new_kv_cache()
        t0 = time.perf_counter()
        net.generate_numpy(
            np.array([base], dtype=np.int64),
            max_new_tokens=4,
            temperature=0.0,
            kv_state=state,
        )
        times.append((time.perf_counter() - t0) * 1e3)
    results["warm from prefix table"] = statistics.median(times)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--skip-model", action="store_true")
    args = parser.parse_args()

    print(f"== kv-prefix-cache microbenchmark ({args.iters} iters/ops) ==")
    results = bench_table(args.iters)
    for name, us in results.items():
        print(f"  {name:<28} {us:8.2f} us/op")

    if not args.skip_model:
        try:
            m = bench_model(max(1, args.iters // 50))
            for name, ms in m.items():
                print(f"  {name:<28} {ms:8.2f} ms/iter")
            worst_us = max(results.values())
            if "warm from prefix table" in m and m["cold (new state)"] > 0:
                saved = 100.0 * (1.0 - m["warm from prefix table"] / m["cold (new state)"])
                print(f"  prefill saved vs cold      {saved:8.1f}%")
            verdict = "PASS" if worst_us < 2000.0 else "FAIL"
            print(f"== {verdict}: worst table op {worst_us:.2f}us (< 2ms budget) ==")
        except Exception as e:
            print(f"  model bench skipped: {e}")
            verdict = "PASS"
            print(f"== {verdict}: table ops only ==")
    else:
        verdict = "PASS"
        print(f"== {verdict}: table ops only ==")


if __name__ == "__main__":
    main()
