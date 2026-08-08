#!/usr/bin/env python3
"""
Benchmark cross-turn KV cache reuse on SloTransformer.

Measures, across a multi-turn conversation whose turns share a growing
prompt prefix:
  - Cached tokens reused per turn (from the previous turn's KV state)
  - Warm (KV-reused) vs cold (fresh state) per-turn latency
  - End-to-end speedup and output consistency between warm and cold

Usage:
  python scripts/benchmark_kv_reuse.py [--turns 4] [--max-tokens 8]
                                       [--steps 3] [--json]
  python scripts/benchmark_kv_reuse.py --stack   # full serving-stack path
  python scripts/benchmark_kv_reuse.py --stack --stream  # /chat/stream SSE path
  python scripts/benchmark_kv_reuse.py --quantize-kv  # int8 KV cache (4x smaller)
  python scripts/benchmark_kv_reuse.py --compare-kv  # float32-vs-int8 quality
  python scripts/benchmark_kv_reuse.py --sessions 4  # concurrent-session isolation
  python scripts/benchmark_kv_reuse.py --layers 8 --embed 256 --heads 8  # arch sweep
"""

import argparse
import json
import sys
import time

import numpy as np

sys.path.insert(0, "packages/core-py")

from domains.training.slonet import SloTransformer


def create_model(vocab=32000, embed=128, layers=4, heads=8, seq_len=128):
    """Create a SloTransformer model for benchmarking.

    Seeds numpy before construction so weight draws are reproducible — the
    float-vs-int8 agreement and per-turn timings depend on the random init,
    and a fixed seed keeps benchmark runs comparable.
    """
    np.random.seed(0)
    return SloTransformer(
        vocab_size=vocab,
        n_embed=embed,
        n_layer=layers,
        n_head=heads,
        intermediate_size=embed * 4,
        block_size=seq_len,
        max_seq_len=seq_len,
        use_rope=True,
        dropout=0.0,
        tie_weights=False,
        use_abs_pos_emb=False,
        norm_type="rms_norm",
    )


def prefix_match(a, b):
    """Length of the longest common prefix of token sequences a and b."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def kv_state_memory_kb(state):
    """Allocated KV memory held by a ``NumpyKVState``, in KiB.

    Sums ``nbytes`` over every per-block K/V buffer and, when the state is
    quantized (int8), the per-token-head float32 scale buffers. The buffers
    are pre-allocated to the state's capacity, so this is the peak KV memory
    committed for the current sequence length — not just the filled portion.

    Args:
        state: A ``NumpyKVState`` (possibly empty — returns 0).

    Returns:
        Total allocated bytes / 1024.
    """
    total = 0
    for buf_k, buf_v, scale_k, scale_v in zip(
            state.kv_buf_k, state.kv_buf_v, state.kv_scale_k, state.kv_scale_v):
        total += buf_k.nbytes + buf_v.nbytes
        if scale_k is not None and scale_v is not None:
            total += scale_k.nbytes + scale_v.nbytes
    return total / 1024.0


def benchmark(model, base_ids, step, turns, max_tokens, steps,
              quantize_kv=False):
    """Run warm (persistent state) and cold (fresh state) generation per turn.

    Each turn's prompt is the *real* previous warm output followed by a new
    batch of user ids — exactly how a chat session grows. Because the prior
    output is a strict prefix of the next prompt, the persistent state reuses
    its entire cached K/V and only computes the appended tokens.

    Args:
        model: SloTransformer instance.
        base_ids: Fixed opening token ids shared by every turn.
        step: Number of new user token ids appended per turn.
        turns: Number of turns.
        max_tokens: New tokens generated per turn.
        steps: Repeat count for timing each turn.
        quantize_kv: When True the KV cache is stored as int8 (4x memory
            reduction) — validates cross-turn reuse on the quantized path.

    Returns:
        Dict of per-turn and aggregate metrics.
    """
    warm_state = model.new_kv_state()

    # Warm-up to trigger any lazy compilation / allocation.
    _ = model.generate_numpy(
        np.array([base_ids], dtype=np.int64),
        max_new_tokens=2, temperature=0.0, kv_state=model.new_kv_state(),
        quantize_kv=quantize_kv,
    )

    history = list(base_ids)
    prev_warm_output = None
    rows = []

    for i in range(turns):
        user_ids = list(range(1000 + i * step, 1000 + i * step + step))
        prompt = history + user_ids
        ids = np.array([prompt], dtype=np.int64)
        reused = 0 if prev_warm_output is None else prefix_match(prompt, prev_warm_output)

        # Warm: reuse the persistent state.
        warm_ms = []
        for _ in range(steps):
            t0 = time.perf_counter()
            out_warm = model.generate_numpy(
                ids, max_new_tokens=max_tokens, temperature=0.0,
                kv_state=warm_state, quantize_kv=quantize_kv,
            )
            warm_ms.append((time.perf_counter() - t0) * 1000)

        # Cold: fresh state each step (no reuse).
        cold_ms = []
        for _ in range(steps):
            fresh = model.new_kv_state()
            t0 = time.perf_counter()
            out_cold = model.generate_numpy(
                ids, max_new_tokens=max_tokens, temperature=0.0,
                kv_state=fresh, quantize_kv=quantize_kv,
            )
            cold_ms.append((time.perf_counter() - t0) * 1000)

        warm_ms = np.array(warm_ms)
        cold_ms = np.array(cold_ms)

        min_len = min(len(out_warm), len(out_cold))
        match = float(np.mean(out_warm[:min_len] == out_cold[:min_len])) * 100

        rows.append({
            "turn": i,
            "prompt_len": len(prompt),
            "reused_tokens": reused,
            "kv_memory_kb": kv_state_memory_kb(warm_state),
            "warm_ms": float(warm_ms.mean()),
            "cold_ms": float(cold_ms.mean()),
            "speedup": float(cold_ms.mean() / max(warm_ms.mean(), 1e-9)),
            "consistency_pct": match,
        })

        history = out_warm.tolist()[0]
        prev_warm_output = out_warm.tolist()[0]

    total_warm = sum(r["warm_ms"] for r in rows)
    total_cold = sum(r["cold_ms"] for r in rows)
    total_reused = sum(r["reused_tokens"] for r in rows)

    return {
        "rows": rows,
        "total_warm_ms": total_warm,
        "total_cold_ms": total_cold,
        "overall_speedup": total_cold / max(total_warm, 1e-9),
        "total_reused_tokens": total_reused,
        "turns": turns,
        "max_tokens": max_tokens,
    }


class _CharTokenizer:
    """Deterministic char-level tokenizer for the tiny benchmark model."""

    def __init__(self):
        self.eos_token_id = 0

    def encode(self, text):
        return [ord(c) % 256 for c in text]

    def decode(self, ids):
        return "".join(chr(i % 256) for i in ids)

    @staticmethod
    def chat_stop_ids():
        return ()


class _StackProvider:
    """Minimal provider exposing the session KV map used by the server.

    Binds the real ``SloNetChatProvider`` session-resolution methods so the
    benchmark exercises the same create-or-reuse / TTL / LRU logic that
    production ``/chat`` runs.
    """

    def __init__(self, model):
        import threading
        self._model = model
        self._kv_states = {}
        self._kv_last_access = {}
        self._kv_ttl = 3600.0
        self._kv_max_sessions = 64
        self._kv_lock = threading.Lock()

    def _cached_tokens(self):
        return sum(s.kv_len[0] if s.kv_len else 0 for s in self._kv_states.values())

    def _get_model(self):
        return self._model


def benchmark_stack(model, base_ids, step, turns, max_tokens, steps,
                    session_id="bench", stream=False, quantize_kv=False):
    """Run the cross-turn benchmark through the full serving stack.

    Drives ``SloNetServer.generate(session_id=...)`` (or
    ``generate_stream`` when ``stream=True``) → provider
    ``_resolve_session_kv`` → ``generate_numpy(_stream)(kv_state=...)`` — the
    exact path ``/chat`` and ``/chat/stream`` hit at runtime. Each turn's
    prompt is the *real* previous warm output plus a new batch of user ids;
    the persistent session KV state reuses the entire shared prefix.

    Args:
        model: SloTransformer instance.
        base_ids: Fixed opening token ids shared by every turn.
        step: Number of new user token ids appended per turn.
        turns: Number of turns.
        max_tokens: New tokens generated per turn.
        steps: Repeat count for timing each turn.
        session_id: Session identity exercised through the provider KV map.
        stream: When True drive ``generate_stream`` (token-by-token SSE path)
            instead of the batched ``generate``.
        quantize_kv: When True the server stores KV as int8 (4x memory
            reduction), validating cross-turn reuse on the quantized path.

    Returns:
        Same-shaped metrics dict as ``benchmark()`` with ``mode="stack"``.
    """
    tokenizer, provider, server = _make_stack(
        model, model_id="bench-kv-reuse", quantize_kv=quantize_kv)

    # Warm-up through the stack (lazy imports / allocation).
    _ = _run_server(server, tokenizer.decode(base_ids), max_new_tokens=2,
                    with_session=False, session_id=session_id, stream=stream)

    history = tokenizer.decode(base_ids)
    rows = []

    for i in range(turns):
        user_ids = list(range(1000 + i * step, 1000 + i * step + step))
        prompt = history + "".join(chr(u % 256) for u in user_ids)
        prompt_ids = tokenizer.encode(prompt)
        prompt_len = len(prompt_ids)

        # Honest reuse: how many of the current prompt's ids are already
        # cached in the session state's previous full output (prompt + out).
        state = provider._kv_states.get(session_id)
        if state is not None and state.prev_ids is not None:
            reused = prefix_match(prompt_ids, state.prev_ids.tolist()[0])
        else:
            reused = 0

        warm_ms = []
        for _ in range(steps):
            t0 = time.perf_counter()
            out_warm = _run_server(server, prompt, max_new_tokens=max_tokens,
                                   with_session=True, session_id=session_id,
                                   stream=stream)
            warm_ms.append((time.perf_counter() - t0) * 1000)

        cold_ms = []
        for _ in range(steps):
            t0 = time.perf_counter()
            out_cold = _run_server(server, prompt, max_new_tokens=max_tokens,
                                   with_session=False, session_id=session_id,
                                   stream=stream)
            cold_ms.append((time.perf_counter() - t0) * 1000)

        warm_ms = np.array(warm_ms)
        cold_ms = np.array(cold_ms)

        warm_ids = tokenizer.encode(out_warm)
        cold_ids = tokenizer.encode(out_cold)
        min_len = min(len(warm_ids), len(cold_ids))
        match = float(np.mean(
            np.array(warm_ids[:min_len]) == np.array(cold_ids[:min_len]))) * 100

        state = provider._kv_states.get(session_id)
        kv_mem = 0.0 if state is None else kv_state_memory_kb(state)

        rows.append({
            "turn": i,
            "prompt_len": prompt_len,
            "reused_tokens": reused,
            "kv_memory_kb": kv_mem,
            "warm_ms": float(warm_ms.mean()),
            "cold_ms": float(cold_ms.mean()),
            "speedup": float(cold_ms.mean() / max(warm_ms.mean(), 1e-9)),
            "consistency_pct": match,
        })

        # Batch generate() echoes the prompt; streaming yields only new tokens.
        # Retain the full prior turn (prompt + output) so the next prompt is a
        # strict extension of the cached sequence and can reuse its K/V.
        history = (prompt + out_warm) if stream else out_warm

    total_warm = sum(r["warm_ms"] for r in rows)
    total_cold = sum(r["cold_ms"] for r in rows)
    total_reused = sum(r["reused_tokens"] for r in rows)

    return {
        "mode": "stack" if not stream else "stack-stream",
        "rows": rows,
        "total_warm_ms": total_warm,
        "total_cold_ms": total_cold,
        "overall_speedup": total_cold / max(total_warm, 1e-9),
        "total_reused_tokens": total_reused,
        "turns": turns,
        "max_tokens": max_tokens,
        "cached_tokens": provider._cached_tokens(),
    }


def compare_kv_quality(model, base_ids, step, turns, max_tokens):
    """Measure the float32-vs-int8 KV cache quality trade-off.

    Builds one shared conversation (its history follows the float32 output),
    then for each turn generates cold outputs with float32 and int8 KV on the
    exact same prompt. int8 rounding only flips near-tie argmax tokens, so the
    outputs agree fully on short generations and diverge rarely as contexts
    lengthen.

    Args:
        model: SloTransformer instance.
        base_ids: Fixed opening token ids shared by every turn.
        step: Number of new user token ids appended per turn.
        turns: Number of turns.
        max_tokens: New tokens generated per turn.

    Returns:
        Dict with per-turn rows (``identical_pct`` of all matching positions,
        ``prefix_agree`` leading identical tokens, ``prefix_pct`` of the
        generated length identical before the first divergence) and the
        overall identical fraction.
    """
    history = list(base_ids)
    rows = []
    for i in range(turns):
        user_ids = list(range(1000 + i * step, 1000 + i * step + step))
        prompt = history + user_ids
        ids = np.array([prompt], dtype=np.int64)
        out_f = model.generate_numpy(
            ids, max_new_tokens=max_tokens, temperature=0.0,
            kv_state=model.new_kv_state(), quantize_kv=False)
        out_q = model.generate_numpy(
            ids, max_new_tokens=max_tokens, temperature=0.0,
            kv_state=model.new_kv_state(), quantize_kv=True)
        fl, ql = out_f.tolist()[0], out_q.tolist()[0]
        n = min(len(fl), len(ql))
        pre = 0
        for j in range(n):
            if fl[j] != ql[j]:
                break
            pre += 1
        ident = sum(1 for a, b in zip(fl[:n], ql[:n]) if a == b) / max(n, 1)
        rows.append({
            "turn": i,
            "prompt_len": len(prompt),
            "generated": n,
            "identical_pct": ident * 100,
            "prefix_agree": pre,
            "prefix_pct": pre / max(n, 1) * 100,
        })
        history = fl

    return {
        "mode": "compare-kv",
        "rows": rows,
        "overall_identical_pct": float(np.mean(
            [r["identical_pct"] for r in rows])),
    }


def print_quality_report(metrics, title="KV Cache Quality — float32 vs int8"):
    print("=" * 76)
    print(title)
    print("=" * 76)
    print(f"{'Turn':>4} {'Prompt':>7} {'Gen':>5} {'Identical':>10} "
          f"{'Prefix':>8} {'Prefix%':>8}")
    for r in metrics["rows"]:
        print(f"{r['turn']:>4} {r['prompt_len']:>7} {r['generated']:>5} "
              f"{r['identical_pct']:>9.1f}% {r['prefix_agree']:>8} "
              f"{r['prefix_pct']:>7.1f}%")
    print("-" * 76)
    print(f"Overall identical: {metrics['overall_identical_pct']:.1f}%")
    print("=" * 76)


def _make_stack(model, model_id="bench", quantize_kv=False):
    """Build the serving-stack triple used by the stack benchmark modes.

    Wires the provider's session-KV methods (``_resolve_session_kv`` and the
    TTL/LRU evictors) onto a minimal ``_StackProvider`` so the server can keep
    per-session KV state in a thread-safe map, exactly like production.

    Args:
        model: SloTransformer instance.
        model_id: Server model identifier.
        quantize_kv: When True the server stores KV as int8 (4x memory
            reduction).

    Returns:
        Tuple of ``(tokenizer, provider, server)``.
    """
    from types import MethodType
    from domains.inference.slonet_provider import SloNetChatProvider
    from domains.infrastructure.slonet_server import SloNetServer

    tokenizer = _CharTokenizer()
    provider = _StackProvider(model)
    provider._resolve_session_kv = MethodType(
        SloNetChatProvider._resolve_session_kv, provider)
    provider._evict_stale_sessions = MethodType(
        SloNetChatProvider._evict_stale_sessions, provider)
    provider._evict_lru_session = MethodType(
        SloNetChatProvider._evict_lru_session, provider)
    server = SloNetServer(
        model=model, tokenizer=tokenizer, model_id=model_id,
        enable_warmup=False, provider=provider, quantize_kv=quantize_kv,
    )
    return tokenizer, provider, server


def benchmark_sessions(model, base_ids, step, turns, max_tokens, steps,
                       n_sessions=2, stream=False, quantize_kv=False):
    """Interleave multiple sessions through one server and check isolation.

    Each session keeps its own prompt history (distinct user id ranges), and
    requests are served in round-robin order across sessions. Cross-turn reuse
    must stay within each session's own KV state — an interleaved session must
    neither inherit another session's cached prefix nor corrupt its own.

    Args:
        model: SloTransformer instance.
        base_ids: Fixed opening token ids shared by every turn.
        step: Number of new user token ids appended per turn.
        turns: Number of turns.
        max_tokens: New tokens generated per turn.
        steps: Repeat count for timing each turn.
        n_sessions: Number of concurrent sessions to interleave.
        stream: When True drive ``generate_stream`` instead of ``generate``.
        quantize_kv: When True the server stores KV as int8.

    Returns:
        Metrics dict with a ``rows`` entry per (session, turn) and
        ``isolation_ok`` = True when every session's reuse grows monotonically
        despite the interleaving.
    """
    tokenizer, provider, server = _make_stack(
        model, model_id="bench-sessions", quantize_kv=quantize_kv)

    _ = _run_server(server, tokenizer.decode(base_ids), max_new_tokens=2,
                    with_session=False, session_id="warm", stream=stream)

    histories = [tokenizer.decode(base_ids)] * n_sessions
    rows = []
    per_session_reuse = [[] for _ in range(n_sessions)]

    for i in range(turns):
        for s in range(n_sessions):
            user_ids = list(range(1000 + s * 10000 + i * step,
                                  1000 + s * 10000 + i * step + step))
            prompt = histories[s] + "".join(chr(u % 256) for u in user_ids)
            prompt_ids = tokenizer.encode(prompt)
            prompt_len = len(prompt_ids)
            sid = f"bench-{s}"

            state = provider._kv_states.get(sid)
            if state is not None and state.prev_ids is not None:
                reused = prefix_match(prompt_ids, state.prev_ids.tolist()[0])
            else:
                reused = 0
            per_session_reuse[s].append(reused)

            warm_ms = []
            for _ in range(steps):
                t0 = time.perf_counter()
                out_warm = _run_server(
                    server, prompt, max_new_tokens=max_tokens,
                    with_session=True, session_id=sid, stream=stream)
                warm_ms.append((time.perf_counter() - t0) * 1000)

            cold_ms = []
            for _ in range(steps):
                t0 = time.perf_counter()
                out_cold = _run_server(
                    server, prompt, max_new_tokens=max_tokens,
                    with_session=False, session_id=sid, stream=stream)
                cold_ms.append((time.perf_counter() - t0) * 1000)

            warm_ms = np.array(warm_ms)
            cold_ms = np.array(cold_ms)

            warm_ids = tokenizer.encode(out_warm)
            cold_ids = tokenizer.encode(out_cold)
            min_len = min(len(warm_ids), len(cold_ids))
            match = float(np.mean(
                np.array(warm_ids[:min_len]) == np.array(cold_ids[:min_len]))) * 100

            state = provider._kv_states.get(sid)
            kv_mem = 0.0 if state is None else kv_state_memory_kb(state)

            rows.append({
                "session": s,
                "turn": i,
                "prompt_len": prompt_len,
                "reused_tokens": reused,
                "kv_memory_kb": kv_mem,
                "warm_ms": float(warm_ms.mean()),
                "cold_ms": float(cold_ms.mean()),
                "speedup": float(cold_ms.mean() / max(warm_ms.mean(), 1e-9)),
                "consistency_pct": match,
            })

            histories[s] = (prompt + out_warm) if stream else out_warm

    isolation_ok = all(
        reuse[0] == 0 and all(b > a for a, b in zip(reuse, reuse[1:]))
        for reuse in per_session_reuse)

    total_warm = sum(r["warm_ms"] for r in rows)
    total_cold = sum(r["cold_ms"] for r in rows)

    return {
        "mode": "stack-sessions",
        "n_sessions": n_sessions,
        "rows": rows,
        "per_session_reuse": per_session_reuse,
        "isolation_ok": bool(isolation_ok),
        "total_warm_ms": total_warm,
        "total_cold_ms": total_cold,
        "overall_speedup": total_cold / max(total_warm, 1e-9),
        "total_reused_tokens": sum(r["reused_tokens"] for r in rows),
        "turns": turns,
        "max_tokens": max_tokens,
        "cached_tokens": provider._cached_tokens(),
    }


def _run_server(server, prompt, max_new_tokens, with_session, session_id,
                stream=False):
    """Run one ``SloNetServer`` call and return the generated text.

    Args:
        server: SloNetServer instance.
        prompt: Prompt text.
        max_new_tokens: New tokens to generate.
        with_session: When True pass ``session_id`` so the provider reuses
            the session KV state; when False the call uses a fresh state.
        session_id: Session identity.
        stream: When True drive ``generate_stream`` (yield each token as
            produced) instead of the batched ``generate``.
    """
    import asyncio

    if stream:
        async def _call():
            out = []
            async for token in server.generate_stream(
                prompt, max_new_tokens=max_new_tokens, temperature=0.0,
                session_id=session_id if with_session else None,
            ):
                out.append(token)
            return "".join(out)
    else:
        async def _call():
            return await server.generate(
                prompt, max_new_tokens=max_new_tokens, temperature=0.0,
                session_id=session_id if with_session else None,
            )

    return asyncio.run(_call())


def print_report(metrics, title="Cross-Turn KV Cache Reuse Benchmark"):
    print("=" * 76)
    print(title)
    print("=" * 76)
    print(f"{'Turn':>4} {'Prompt':>7} {'Reused':>7} {'KV KiB':>8} "
          f"{'Warm ms':>9} {'Cold ms':>9} {'Speedup':>8} {'Match':>8}")
    for r in metrics["rows"]:
        print(f"{r['turn']:>4} {r['prompt_len']:>7} {r['reused_tokens']:>7} "
              f"{r['kv_memory_kb']:>8.1f} "
              f"{r['warm_ms']:>9.2f} {r['cold_ms']:>9.2f} {r['speedup']:>7.2f}x "
              f"{r['consistency_pct']:>7.1f}%")
    print("-" * 76)
    print(f"Total warm: {metrics['total_warm_ms']:.1f} ms   "
          f"Total cold: {metrics['total_cold_ms']:.1f} ms   "
          f"Overall speedup: {metrics['overall_speedup']:.2f}x")
    print(f"Total KV tokens reused: {metrics['total_reused_tokens']}")
    if metrics.get("cached_tokens") is not None:
        print(f"Cached tokens in session map: {metrics['cached_tokens']}")
    print("=" * 76)


def print_sessions_report(metrics, title="Cross-Turn KV Reuse — concurrent sessions"):
    print("=" * 76)
    print(title)
    print("=" * 76)
    print(f"Interleaving {metrics['n_sessions']} sessions across "
          f"{metrics['turns']} turns, {metrics['max_tokens']} tokens each.\n")
    for s in range(metrics["n_sessions"]):
        print(f"--- Session {s} ---")
        print(f"{'Turn':>4} {'Prompt':>7} {'Reused':>7} {'KV KiB':>8} "
              f"{'Warm ms':>9} {'Cold ms':>9} {'Speedup':>8} {'Match':>8}")
        for r in metrics["rows"]:
            if r["session"] != s:
                continue
            print(f"{r['turn']:>4} {r['prompt_len']:>7} {r['reused_tokens']:>7} "
                  f"{r['kv_memory_kb']:>8.1f} "
                  f"{r['warm_ms']:>9.2f} {r['cold_ms']:>9.2f} {r['speedup']:>7.2f}x "
                  f"{r['consistency_pct']:>7.1f}%")
        reuse = metrics["per_session_reuse"][s]
        print(f"  reuse = {reuse}")
    print("-" * 76)
    print(f"Total warm: {metrics['total_warm_ms']:.1f} ms   "
          f"Total cold: {metrics['total_cold_ms']:.1f} ms   "
          f"Overall speedup: {metrics['overall_speedup']:.2f}x")
    print(f"Total KV tokens reused: {metrics['total_reused_tokens']}   "
          f"Cached in session map: {metrics['cached_tokens']}")
    print(f"Session isolation: {'OK — reuse grew monotonically per session' if metrics['isolation_ok'] else 'BROKEN'}")
    print("=" * 76)


def main():
    parser = argparse.ArgumentParser(description="Benchmark cross-turn KV reuse")
    parser.add_argument("--turns", type=int, default=4, help="Number of conversation turns")
    parser.add_argument("--max-tokens", type=int, default=8, help="New tokens per turn")
    parser.add_argument("--steps", type=int, default=3, help="Timing repeats per turn")
    parser.add_argument("--stack", action="store_true",
                        help="Drive the full serving stack (SloNetServer + session KV map)")
    parser.add_argument("--stream", action="store_true",
                        help="Stream tokens one at a time (with --stack: the /chat/stream SSE path)")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable summary")
    parser.add_argument("--quantize-kv", action="store_true",
                        help="Store KV cache as int8 (4x memory reduction) and "
                             "validate cross-turn reuse on the quantized path")
    parser.add_argument("--compare-kv", action="store_true",
                        help="Measure float32-vs-int8 KV output quality on "
                             "identical prompts (near-tie argmax flips)")
    parser.add_argument("--embed", type=int, default=128,
                        help="Embedding dimension (model architecture sweep)")
    parser.add_argument("--layers", type=int, default=4,
                        help="Transformer blocks (model architecture sweep)")
    parser.add_argument("--heads", type=int, default=8,
                        help="Attention heads (model architecture sweep)")
    parser.add_argument("--step", type=int, default=3,
                        help="New user ids appended per turn (context growth)")
    parser.add_argument("--sessions", type=int, default=1,
                        help="Concurrent sessions interleaved through the "
                             "stack (implies --stack); verifies per-session "
                             "KV isolation")
    args = parser.parse_args()

    # Stack mode round-trips ids through the char tokenizer, so the model
    # vocab must stay within [0, 256) for lossless decode → encode history.
    vocab = 256 if (args.stack or args.sessions > 1) else 32000
    model = create_model(vocab=vocab, embed=args.embed, layers=args.layers,
                         heads=args.heads)
    arch = f" layers={args.layers} embed={args.embed} heads={args.heads}"

    if args.compare_kv:
        metrics = compare_kv_quality(model, [7], step=args.step, turns=args.turns,
                                     max_tokens=args.max_tokens)
        if args.json:
            print(json.dumps(metrics, indent=2))
        else:
            print_quality_report(metrics, title=(
                "KV Cache Quality — float32 vs int8" + arch))
        if metrics["overall_identical_pct"] < 50.0:
            print("WARNING: int8 KV agreement below 50% — quality trade-off "
                  "may be too costly", file=sys.stderr)
            return 1
        return 0

    if args.sessions > 1:
        metrics = benchmark_sessions(model, [7], step=args.step, turns=args.turns,
                                     max_tokens=args.max_tokens, steps=args.steps,
                                     n_sessions=args.sessions,
                                     stream=args.stream, quantize_kv=args.quantize_kv)
        title = (f"Cross-Turn KV Reuse — {args.sessions} concurrent sessions"
                 + arch)
        if args.json:
            print(json.dumps(metrics, indent=2))
        else:
            print_sessions_report(metrics, title=title)
        if not metrics["isolation_ok"]:
            print("WARNING: per-session reuse did not grow monotonically — "
                  "session KV isolation is broken", file=sys.stderr)
            return 1
        return 0

    if args.stack:
        metrics = benchmark_stack(model, [7], step=args.step, turns=args.turns,
                                  max_tokens=args.max_tokens, steps=args.steps,
                                  stream=args.stream, quantize_kv=args.quantize_kv)
        title = "Cross-Turn KV Reuse Benchmark (serving stack)" + (
            " — streaming" if args.stream else "") + arch
    else:
        metrics = benchmark(model, [7], step=args.step, turns=args.turns,
                            max_tokens=args.max_tokens, steps=args.steps,
                            quantize_kv=args.quantize_kv)
        title = "Cross-Turn KV Cache Reuse Benchmark" + arch

    if args.quantize_kv:
        title += " — int8 KV cache"

    if args.json:
        print(json.dumps(metrics, indent=2))
    else:
        print_report(metrics, title=title)

    # Reuse growth sanity: reused tokens must increase with each turn.
    reused = [r["reused_tokens"] for r in metrics["rows"]]
    if any(b <= a for a, b in zip(reused, reused[1:])):
        print("WARNING: reused tokens did not grow monotonically — reuse may be broken",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
