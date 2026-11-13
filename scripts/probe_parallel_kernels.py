"""Measure-first probe: does parallel=True + prange beat serial on the hot kernels?

Card 21135daa (follow-up to f30be275, which only added nogil=True — host
responsiveness, throughput intentionally flat). This script defines STANDALONE
TWINS of the three hottest kernels from
domain/training/_internal/slonet_kernels.py — the shipped serial version and
the candidate parallel=True + prange version — and times both at realistic
shapes. Nothing in production is touched; the twins exist so a losing variant
costs nothing but a failed measurement.

Why twins instead of editing slonet_kernels.py: the change under evaluation is
exactly the one most likely to *lose* (numba's parallel layer has per-call
fork/join overhead, and these kernels run once per token per layer). Measure
the loss in isolation first, land only a win.

Numeric contract (deliberately weaker than f30be275): parallel reductions
reorder floating-point adds, so bit-identity is NOT required here — only
agreement within a stated tolerance. Both are checked and reported so the
trade is visible, not assumed.

Usage:
    python -u scripts/probe_parallel_kernels.py            # all candidates
    python -u scripts/probe_parallel_kernels.py --repeat 20
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

# Match the repo default (benchmark_slonet.py): OpenBLAS is pinned to one
# thread so the only parallelism under test is numba's own threading layer.
os.environ.setdefault("OPENBLAS_NUM_THREADS", os.environ.get("SLO_OPENBLAS_NUM_THREADS", "1"))
os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("SLO_OMP_NUM_THREADS", "1"))

import numba
from numba import njit, prange

TOL_F32 = 1e-4  # relative tolerance for float32 reorderings


# ---------------------------------------------------------------------------
#  Candidate 1 — _fused_attention_multi (prompt prefill)
# ---------------------------------------------------------------------------


@njit(nogil=True, cache=False)
def serial_attn_multi(q, k_full, v_full, out, scale, H, E_head, seq_len, new_len):
    """Shipped serial shape: for h -> for s -> 3 passes. out[s,h,d]."""
    for h in range(H):
        for s in range(seq_len):
            scores_max = -1e30
            for n in range(new_len):
                dot = 0.0
                for d in range(E_head):
                    dot += q[s, h, d] * k_full[h, n, d]
                dot *= scale
                if n > s:
                    dot = -1e9
                if dot > scores_max:
                    scores_max = dot
            total = 0.0
            for n in range(new_len):
                dot = 0.0
                for d in range(E_head):
                    dot += q[s, h, d] * k_full[h, n, d]
                dot *= scale
                if n > s:
                    dot = -1e9
                total += np.exp(dot - scores_max)
            inv_total = 1.0 / total
            for d in range(E_head):
                acc = 0.0
                for n in range(new_len):
                    dot = 0.0
                    for dd in range(E_head):
                        dot += q[s, h, dd] * k_full[h, n, dd]
                    dot *= scale
                    if n > s:
                        dot = -1e9
                    acc += np.exp(dot - scores_max) * inv_total * v_full[h, n, d]
                out[s, h, d] = acc


@njit(parallel=True, nogil=True, cache=False)
def parallel_attn_multi(q, k_full, v_full, out, scale, H, E_head, seq_len, new_len):
    """prange over the flattened (h, s) space — each (h,s) owns out[:,h,:]."""
    for hs in prange(H * seq_len):
        h = hs // seq_len
        s = hs - h * seq_len
        scores_max = -1e30
        for n in range(new_len):
            dot = 0.0
            for d in range(E_head):
                dot += q[s, h, d] * k_full[h, n, d]
            dot *= scale
            if n > s:
                dot = -1e9
            if dot > scores_max:
                scores_max = dot
        total = 0.0
        for n in range(new_len):
            dot = 0.0
            for d in range(E_head):
                dot += q[s, h, d] * k_full[h, n, d]
            dot *= scale
            if n > s:
                dot = -1e9
            total += np.exp(dot - scores_max)
        inv_total = 1.0 / total
        for d in range(E_head):
            acc = 0.0
            for n in range(new_len):
                dot = 0.0
                for dd in range(E_head):
                    dot += q[s, h, dd] * k_full[h, n, dd]
                dot *= scale
                if n > s:
                    dot = -1e9
                acc += np.exp(dot - scores_max) * inv_total * v_full[h, n, d]
            out[s, h, d] = acc


# ---------------------------------------------------------------------------
#  Candidate 2 — _fused_attention_single (single-token decode)
# ---------------------------------------------------------------------------


@njit(nogil=True, cache=False)
def serial_attn_single(q, k_full, v_full, out, scale, H, E_head, seq_len, new_len):
    """Decode step: loop over heads only, KV cache is new_len long."""
    for h in range(H):
        scores_max = -1e30
        for n in range(new_len):
            dot = 0.0
            for d in range(E_head):
                dot += q[h, d] * k_full[h, n, d]
            dot *= scale
            if dot > scores_max:
                scores_max = dot
        total = 0.0
        for n in range(new_len):
            dot = 0.0
            for d in range(E_head):
                dot += q[h, d] * k_full[h, n, d]
            total += np.exp(dot * scale - scores_max)
        inv_total = 1.0 / total
        for d in range(E_head):
            acc = 0.0
            for n in range(new_len):
                dot = 0.0
                for dd in range(E_head):
                    dot += q[h, dd] * k_full[h, n, dd]
                acc += np.exp(dot * scale - scores_max) * inv_total * v_full[h, n, d]
            out[h, d] = acc


@njit(parallel=True, nogil=True, cache=False)
def parallel_attn_single(q, k_full, v_full, out, scale, H, E_head, seq_len, new_len):
    """prange over heads — H-way parallelism, the only independent axis here."""
    for h in prange(H):
        scores_max = -1e30
        for n in range(new_len):
            dot = 0.0
            for d in range(E_head):
                dot += q[h, d] * k_full[h, n, d]
            dot *= scale
            if dot > scores_max:
                scores_max = dot
        total = 0.0
        for n in range(new_len):
            dot = 0.0
            for d in range(E_head):
                dot += q[h, d] * k_full[h, n, d]
            total += np.exp(dot * scale - scores_max)
        inv_total = 1.0 / total
        for d in range(E_head):
            acc = 0.0
            for n in range(new_len):
                dot = 0.0
                for dd in range(E_head):
                    dot += q[h, dd] * k_full[h, n, dd]
                acc += np.exp(dot * scale - scores_max) * inv_total * v_full[h, n, d]
            out[h, d] = acc


# ---------------------------------------------------------------------------
#  Candidate 3 — _lm_head_argmax (token selection, every decode step)
# ---------------------------------------------------------------------------


@njit(nogil=True, cache=False)
def serial_lm_head_argmax(x, W):
    """Shipped: single serial pass over the vocab, argmax inline."""
    D = x.shape[1]
    V = W.shape[0]
    best_score = np.float32(-1e30)
    best_idx = np.int64(0)
    for v in range(V):
        score = np.float32(0.0)
        for d in range(D):
            score += x[0, d] * W[v, d]
        if score > best_score:
            best_score = score
            best_idx = v
    return best_idx


@njit(parallel=True, nogil=True, cache=False)
def parallel_lm_head_argmax(x, W):
    """prange over vocab into a scratch scores buffer, then serial argmax.

    numba's parallel reductions do not cover argmax-over-index, so the only
    way to prange this is to materialise scores — which is exactly what the
    shipped kernel exists to avoid. Kept here to measure whether the
    parallelism pays for the extra allocation.
    """
    D = x.shape[1]
    V = W.shape[0]
    scores = np.empty(V, dtype=np.float32)
    for v in prange(V):
        score = np.float32(0.0)
        for d in range(D):
            score += x[0, d] * W[v, d]
        scores[v] = score
    best_score = np.float32(-1e30)
    best_idx = np.int64(0)
    for v in range(V):
        if scores[v] > best_score:
            best_score = scores[v]
            best_idx = v
    return best_idx


# ---------------------------------------------------------------------------
#  Harness
# ---------------------------------------------------------------------------


def _time(fn, args, repeat):
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best * 1e3  # ms


def _max_abs_diff(a, b):
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def bench_case(
    name, serial, parallel, args_serial, args_parallel, out_serial, out_parallel, repeat, shape_note
):
    """Time both twins, check agreement, report the delta.

    The two arg tuples differ ONLY in which output buffer they carry — the
    serial twin must never write into the parallel twin's buffer or the
    correctness comparison would be against stale data.
    """
    s_ms = _time(serial, args_serial, repeat)
    p_ms = _time(parallel, args_parallel, repeat)

    out_serial.fill(0)
    out_parallel.fill(0)
    serial(*args_serial)
    parallel(*args_parallel)
    diff = _max_abs_diff(out_serial, out_parallel)
    scale = max(_max_abs_diff(out_serial, np.zeros_like(out_serial)), 1e-12)
    rel = diff / scale
    bit_identical = bool(np.array_equal(out_serial, out_parallel))

    speedup = s_ms / p_ms if p_ms > 0 else float("inf")
    return {
        "kernel": name,
        "shape": shape_note,
        "serial_ms": s_ms,
        "parallel_ms": p_ms,
        "speedup": speedup,
        "max_abs_diff": diff,
        "rel_diff": rel,
        "bit_identical": bit_identical,
        "within_tol": bool(rel < TOL_F32),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=20)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    repeat = args.repeat

    print(f"numba {numba.__version__}  threads={numba.config.NUMBA_NUM_THREADS}  repeat={repeat}")
    print(f"tolerance: relative < {TOL_F32:g} (bit-identity NOT required — parallel reorder)")
    print()

    rows = []

    # ---- candidate 1: prefill -------------------------------------------------
    # Realistic Qwen2.5-0.5B-Instruct attention: H=14 heads, E_head=64.
    # repeat shrinks as work grows — this kernel is O(seq^2 * E_head).
    for seq_len, note, rep in (
        (64, "prefill seq=64", repeat),
        (256, "prefill seq=256", max(3, repeat // 5)),
        (1024, "prefill seq=1024", 3),
    ):
        H, E = 14, 64
        scale = np.float32(1.0 / np.sqrt(E))
        q = rng.standard_normal((seq_len, H, E)).astype(np.float32)
        k = rng.standard_normal((H, seq_len, E)).astype(np.float32)
        v = rng.standard_normal((H, seq_len, E)).astype(np.float32)
        o_s = np.zeros((seq_len, H, E), dtype=np.float32)
        o_p = np.zeros((seq_len, H, E), dtype=np.float32)
        rows.append(
            bench_case(
                "fused_attention_multi",
                serial_attn_multi,
                parallel_attn_multi,
                (q, k, v, o_s, scale, H, E, seq_len, seq_len),
                (q, k, v, o_p, scale, H, E, seq_len, seq_len),
                o_s,
                o_p,
                rep,
                f"H={H} E={E} {note}",
            )
        )

    # ---- candidate 2: decode --------------------------------------------------
    for new_len, note in ((256, "decode kv=256"), (2048, "decode kv=2048")):
        H, E = 14, 64
        scale = np.float32(1.0 / np.sqrt(E))
        q = rng.standard_normal((H, E)).astype(np.float32)
        k = rng.standard_normal((H, new_len, E)).astype(np.float32)
        v = rng.standard_normal((H, new_len, E)).astype(np.float32)
        o_s = np.zeros((H, E), dtype=np.float32)
        o_p = np.zeros((H, E), dtype=np.float32)
        rows.append(
            bench_case(
                "fused_attention_single",
                serial_attn_single,
                parallel_attn_single,
                (q, k, v, o_s, scale, H, E, 1, new_len),
                (q, k, v, o_p, scale, H, E, 1, new_len),
                o_s,
                o_p,
                repeat,
                f"H={H} E={E} {note}",
            )
        )

    # ---- candidate 3: lm_head argmax -----------------------------------------
    D, V = 896, 151936  # Qwen2.5-0.5B: d_model=896, vocab=151936
    x = rng.standard_normal((1, D)).astype(np.float32)
    W = rng.standard_normal((V, D)).astype(np.float32)
    idx_s = serial_lm_head_argmax(x, W)
    idx_p = parallel_lm_head_argmax(x, W)
    s_ms = _time(serial_lm_head_argmax, (x, W), repeat)
    p_ms = _time(parallel_lm_head_argmax, (x, W), repeat)
    rows.append(
        {
            "kernel": "lm_head_argmax",
            "shape": f"D={D} V={V}",
            "serial_ms": s_ms,
            "parallel_ms": p_ms,
            "speedup": s_ms / p_ms if p_ms > 0 else float("inf"),
            "max_abs_diff": 0.0,
            "rel_diff": 0.0,
            "bit_identical": bool(idx_s == idx_p),
            "within_tol": bool(idx_s == idx_p),
        }
    )

    hdr = (
        f"{'kernel':24} {'shape':20} {'serial ms':>10} {'par ms':>9} "
        f"{'speedup':>8} {'max|d|':>10} {'bit-ident':>9} {'in-tol':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['kernel']:24} {r['shape']:20} {r['serial_ms']:10.3f} {r['parallel_ms']:9.3f} "
            f"{r['speedup']:7.2f}x {r['max_abs_diff']:10.2e} {str(r['bit_identical']):>9} {str(r['within_tol']):>7}"
        )

    wins = [r for r in rows if r["speedup"] > 1.05]
    losses = [r for r in rows if r["speedup"] < 0.95]
    print()
    print(
        f"wins (>5%): {len(wins)}   losses (>5% slower): {len(losses)}   flat: {len(rows) - len(wins) - len(losses)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
