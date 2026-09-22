"""Benchmark unified KV cache base vs legacy implementations."""

import time

import numpy as np

from domain.infrastructure._internal.kv_cache.kv_state import KVState
from domain.infrastructure._internal.session_cache import SessionKVState


def bench_session_cache(n_sessions=100, prefix_len=100, new_tokens=10, n_iters=500):
    """Benchmark legacy SessionKVState."""
    cache = SessionKVState(max_sessions=n_sessions, ttl=3600.0)
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


def bench_append_strategy(n_tokens=1024, n_heads=12, head_dim=64):
    """Append strategies for a full decode (1 token/step), copy cost only.

    Concatenating append re-allocates + copies the whole accumulated K+V each
    step (O(T^2) total over a decode); slice-write into a preallocated buffer
    copies only the new token (O(T) total). This is the cost the inline-concat
    benchmark used to hide by dividing over the whole loop.
    """

    def concat():
        k_cache = np.zeros((1, 0, n_heads, head_dim), np.float32)
        v_cache = np.zeros((1, 0, n_heads, head_dim), np.float32)
        t0 = time.perf_counter()
        for _ in range(n_tokens):
            k_new = np.random.randn(1, 1, n_heads, head_dim).astype(np.float32)
            v_new = np.random.randn(1, 1, n_heads, head_dim).astype(np.float32)
            k_cache = np.concatenate([k_cache, k_new], axis=1)
            v_cache = np.concatenate([v_cache, v_new], axis=1)
        return time.perf_counter() - t0

    def prealloc():
        cap, grow = 16, 2.0
        shape = (1, cap, n_heads, head_dim)
        k_buf = np.empty(shape, np.float32)
        v_buf = np.empty(shape, np.float32)
        base = 0
        t0 = time.perf_counter()
        for _ in range(n_tokens):
            if base == cap:
                cap = int(cap * grow)
                new = np.empty((1, cap, n_heads, head_dim), np.float32)
                new[:, :base] = k_buf[:, :base]
                k_buf = new
                v_buf = np.empty((1, cap, n_heads, head_dim), np.float32)
                v_buf[:, :base] = k_buf[:, :base]
            k_new = np.random.randn(1, 1, n_heads, head_dim).astype(np.float32)
            v_new = np.random.randn(1, 1, n_heads, head_dim).astype(np.float32)
            k_buf[:, base] = k_new[:, 0]
            v_buf[:, base] = v_new[:, 0]
            base += 1
        return time.perf_counter() - t0

    c = concat()
    p = prealloc()
    return {
        "n_tokens": n_tokens,
        "concat_ms": round(c * 1e3, 1),
        "prealloc_ms": round(p * 1e3, 1),
        "speedup": round(c / p, 1),
        "concat_us_per_step": round(c / n_tokens * 1e6, 0),
        "prealloc_us_per_step": round(p / n_tokens * 1e6, 1),
    }


def bench_native_numpy(n_layers=12, n_heads=12, head_dim=64, n_tokens=512, n_iters=5):
    """Benchmark NativeKVState's numpy path across a decode (append + read)."""
    from domain.infrastructure._internal.kv_cache.native import NativeKVState

    cache = NativeKVState(n_layers, n_kv_heads=n_heads, head_dim=head_dim, seq_capacity=n_tokens)
    start = time.perf_counter()
    for _ in range(n_iters):
        cache.reset()
        for _ in range(n_tokens):
            k = np.random.randn(1, 1, n_heads, head_dim).astype(np.float32)
            v = np.random.randn(1, 1, n_heads, head_dim).astype(np.float32)
            for layer_idx in range(n_layers):
                k_cat, v_cat = cache.update(layer_idx, k, v)
                _ = k_cat, v_cat
    elapsed = time.perf_counter() - start
    ops = n_iters * n_tokens
    return {
        "total_s": round(elapsed, 4),
        "layers": n_layers,
        "us_per_token_step": round(elapsed / ops * 1e6, 2),
    }


def bench_base_session(n_sessions=100, prefix_len=100, new_tokens=10, n_iters=500):
    """Benchmark KVState session mode."""
    cache = KVState(n_layers=0, max_sessions=n_sessions, ttl=3600.0)
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
    """Benchmark KVState paged mode (get_blocks, no concat)."""
    cache = KVState(
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

    print("\n--- Legacy SessionKVState ---")
    r1 = bench_session_cache()
    print(f"  {r1['us_per_op']:.2f} us/op  ({r1['ops_per_sec']:.1f} ops/s)")

    print("\n--- KVState session mode ---")
    r2 = bench_base_session()
    print(f"  {r2['us_per_op']:.2f} us/op  ({r2['ops_per_sec']:.1f} ops/s)")

    print("\n--- Append strategy (1-token decode steps, copy cost) ---")
    r3 = bench_append_strategy()
    print(
        f"  {r3['n_tokens']} tokens: concat {r3['concat_ms']} ms "
        f"({r3['concat_us_per_step']} us/step) vs prealloc {r3['prealloc_ms']} ms "
        f"({r3['prealloc_us_per_step']:.1f} us/step)  => {r3['speedup']}x"
    )

    print("\n--- NativeKVState numpy path (append + read) ---")
    r6 = bench_native_numpy()
    print(f"  {r6['us_per_token_step']} us/token-step (12 layers)")

    print("\n--- KVState paged mode (get_blocks, no alloc) ---")
    r5 = bench_base_paged()
    print(f"  {r5['us_per_op']:.2f} us/op  ({r5['ops_per_sec']:.1f} ops/s)")

    print("\n" + "=" * 70)
    print("base = KVState (state+paged+session). native = NativeKVState (numpy prealloc | C).")
    print("=" * 70)
