"""Benchmark unified KV cache base vs legacy implementations."""

import time

import numpy as np

from domain.infrastructure._internal.kv_cache.base import KVCacheBase
from domain.infrastructure._internal.session_cache import SessionKVCache


def bench_numpy_kv_cache(
    n_layers=12, n_heads=12, head_dim=64, seq_lens=(1, 64, 256, 1024), n_iters=1000
):
    """Benchmark KVCacheBase concat mode."""
    results = {}
    for seq_len in seq_lens:
        cache = KVCacheBase(n_layers)
        k = np.random.randn(1, seq_len, n_heads, head_dim).astype(np.float32)
        v = np.random.randn(1, seq_len, n_heads, head_dim).astype(np.float32)

        start = time.perf_counter()
        for _ in range(n_iters):
            for layer_idx in range(n_layers):
                cache.update(layer_idx, k, v)
            cache.reset()
        elapsed = time.perf_counter() - start
        ops = n_iters * n_layers
        results[seq_len] = {
            "total_s": round(elapsed, 4),
            "ops_per_sec": round(ops / elapsed, 1),
            "us_per_op": round(elapsed / ops * 1e6, 2),
        }
    return results


def bench_session_cache(n_sessions=100, prefix_len=100, new_tokens=10, n_iters=500):
    """Benchmark legacy SessionKVCache."""
    cache = SessionKVCache(max_sessions=n_sessions, ttl=3600.0)
    prefix = list(range(prefix_len))
    full_ids = prefix + list(range(prefix_len, prefix_len + new_tokens))

    for sid in range(n_sessions):
        dummy_pkv = tuple(
            (np.zeros((1, prefix_len, 12, 64)), np.zeros((1, prefix_len, 12, 64)))
            for _ in range(12)
        )
        cache.store(f"sess_{sid}", prefix, dummy_pkv)

    start = time.perf_counter()
    for i in range(n_iters):
        sid = f"sess_{i % n_sessions}"
        pkv, prefix_len_found = cache.get(sid, full_ids)
        dummy_new = tuple(
            (np.zeros((1, new_tokens, 12, 64)), np.zeros((1, new_tokens, 12, 64)))
            for _ in range(12)
        )
        cache.store(sid, full_ids, dummy_new)
    elapsed = time.perf_counter() - start
    return {
        "total_s": round(elapsed, 4),
        "ops_per_sec": round(n_iters / elapsed, 1),
        "us_per_op": round(elapsed / n_iters * 1e6, 2),
    }


def bench_inline_concat(
    n_layers=12, n_heads=12, head_dim=64, seq_lens=(1, 64, 256, 1024), n_iters=1000
):
    """Benchmark slonet-style inline concat (raw list of tuples)."""
    results = {}
    for seq_len in seq_lens:
        caches = [(None, None)] * n_layers
        k = np.random.randn(1, seq_len, n_heads, head_dim).astype(np.float32)
        v = np.random.randn(1, seq_len, n_heads, head_dim).astype(np.float32)

        start = time.perf_counter()
        for _ in range(n_iters):
            for layer_idx in range(n_layers):
                k_cache, v_cache = caches[layer_idx]
                if k_cache is None:
                    caches[layer_idx] = (k, v)
                else:
                    caches[layer_idx] = (
                        np.concatenate([k_cache, k], axis=1),
                        np.concatenate([v_cache, v], axis=1),
                    )
            caches = [(None, None)] * n_layers
        elapsed = time.perf_counter() - start
        ops = n_iters * n_layers
        results[seq_len] = {
            "total_s": round(elapsed, 4),
            "ops_per_sec": round(ops / elapsed, 1),
            "us_per_op": round(elapsed / ops * 1e6, 2),
        }
    return results


def bench_base_session(n_sessions=100, prefix_len=100, new_tokens=10, n_iters=500):
    """Benchmark KVCacheBase session mode."""
    cache = KVCacheBase(n_layers=0, max_sessions=n_sessions, ttl=3600.0)
    prefix = list(range(prefix_len))
    full_ids = prefix + list(range(prefix_len, prefix_len + new_tokens))

    for sid in range(n_sessions):
        dummy_pkv = tuple(
            (np.zeros((1, prefix_len, 12, 64)), np.zeros((1, prefix_len, 12, 64)))
            for _ in range(12)
        )
        cache.session_store(f"sess_{sid}", prefix, dummy_pkv)

    start = time.perf_counter()
    for i in range(n_iters):
        sid = f"sess_{i % n_sessions}"
        pkv, prefix_len_found = cache.session_get(sid, full_ids)
        dummy_new = tuple(
            (np.zeros((1, new_tokens, 12, 64)), np.zeros((1, new_tokens, 12, 64)))
            for _ in range(12)
        )
        cache.session_store(sid, full_ids, dummy_new)
    elapsed = time.perf_counter() - start
    return {
        "total_s": round(elapsed, 4),
        "ops_per_sec": round(n_iters / elapsed, 1),
        "us_per_op": round(elapsed / n_iters * 1e6, 2),
    }


def bench_base_paged(n_layers=12, n_heads=12, head_dim=64, n_tokens=256, n_iters=1000):
    """Benchmark KVCacheBase paged mode (get_blocks, no concat)."""
    cache = KVCacheBase(
        n_layers, n_heads=n_heads, head_dim=head_dim, max_seq_len=n_tokens * 2, block_size=64
    )
    k = np.random.randn(1, n_tokens, n_heads, head_dim).astype(np.float32)
    v = np.random.randn(1, n_tokens, n_heads, head_dim).astype(np.float32)

    for layer_idx in range(n_layers):
        cache.update(layer_idx, k, v)

    start = time.perf_counter()
    for _ in range(n_iters):
        for layer_idx in range(n_layers):
            cache.get_blocks(layer_idx)
    elapsed = time.perf_counter() - start
    ops = n_iters * n_layers
    return {
        "total_s": round(elapsed, 4),
        "ops_per_sec": round(ops / elapsed, 1),
        "us_per_op": round(elapsed / ops * 1e6, 2),
    }


if __name__ == "__main__":
    print("=" * 70)
    print("KV Cache Benchmark — Unified Base vs Legacy")
    print("=" * 70)

    print("\n--- Legacy SessionKVCache ---")
    r1 = bench_session_cache()
    print(f"  {r1['us_per_op']:.2f} us/op  ({r1['ops_per_sec']:.1f} ops/s)")

    print("\n--- KVCacheBase session mode ---")
    r2 = bench_base_session()
    print(f"  {r2['us_per_op']:.2f} us/op  ({r2['ops_per_sec']:.1f} ops/s)")

    print("\n--- Inline Concat (slonet raw tuples) ---")
    r3 = bench_inline_concat()
    for sl, m in r3.items():
        print(f"  seq_len={sl:>5}: {m['us_per_op']:>8.2f} us/op  ({m['ops_per_sec']:>10.1f} ops/s)")

    print("\n--- KVCacheBase concat mode ---")
    r4 = bench_numpy_kv_cache()
    for sl, m in r4.items():
        print(f"  seq_len={sl:>5}: {m['us_per_op']:>8.2f} us/op  ({m['ops_per_sec']:>10.1f} ops/s)")

    print("\n--- KVCacheBase paged mode (get_blocks, no alloc) ---")
    r5 = bench_base_paged()
    print(f"  {r5['us_per_op']:.2f} us/op  ({r5['ops_per_sec']:.1f} ops/s)")

    print("\n" + "=" * 70)
    print("All modes in one class. Remove paged.py, session.py.")
    print("=" * 70)
