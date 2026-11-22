"""Prefill latency benchmark — wall-clock vs prompt length (card 20261102_001622).

Investigation instrument for the prefill-latency-policy card: measures prefill
wall-clock as a function of prompt length on this box, to find where the
engine's 120 s request timeout (``DEFAULT_GENERATE_TIMEOUT``) actually bites.

Prefill only, per the card's scope correction — generation already streams and
must NOT be affected. Each sample runs ``max_new_tokens=1`` (greedy) and reads
``metrics.prefill_ms`` (``t_first_token - t_start``): the prompt forward plus
one last-position argmax, so the number is the prompt's cost, not decode's.

Usage:
    python scripts/benchmark_prefill.py                      # 1k,2k,4k,8k
    python scripts/benchmark_prefill.py --sizes 1024,2048    # subset
    python scripts/benchmark_prefill.py --timeout 30         # other budget
    python scripts/benchmark_prefill.py --json               # machine-readable
    SLO_USE_NUMBA=1 python scripts/benchmark_prefill.py      # compiled-kernel
                                                             # comparison run

Gates: none — this is a measurement, not a regression gate. The numbers feed
the truncation-semantics proposal on the card (a token clamp is a
context-window product decision, not a kernel flag).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def _resolve_slnc(model: str) -> Path:
    from domain.infrastructure._internal.model_resolver import get_model_dir

    slnc = get_model_dir(model) / "model.slnc"
    if not slnc.exists():
        raise SystemExit(
            f"no model.slnc at {slnc} — pass --slnc explicitly "
            "(model ids resolve via HF_HOME or repo models/hf-cache/)."
        )
    return slnc


def _load_model(slnc: Path, model_id: str):
    from domain.inference._internal.slonet_provider import SloNetChatProvider

    t0 = time.perf_counter()
    provider = SloNetChatProvider.from_slnc(str(slnc), model_id=model_id, quantize=True)
    return provider, time.perf_counter() - t0


def _prompt_ids(tokenizer, n: int) -> np.ndarray:
    """A real-text prompt tiled to exactly n tokens (deterministic content)."""
    base = tokenizer.encode("the quick brown fox jumps over the lazy dog ")
    if not base:
        base = [1, 2, 3]
    reps = (n // len(base)) + 1
    ids = (base * reps)[:n]
    return np.array([ids], dtype=np.int64)


def _prefill_once(provider, input_ids) -> tuple[float, float]:
    """One prefill measurement. Returns (prefill_s, wall_s).

    max_new_tokens=1: the first sampled token comes straight off the prompt's
    last-position logits, so wall ≈ prefill + one lm_head row.
    """
    model = provider._model
    t0 = time.perf_counter()
    res = model.generate_numpy(input_ids, max_new_tokens=1, temperature=0.0)
    wall = time.perf_counter() - t0
    prefill_ms = float(res.metrics.prefill_ms)
    return (prefill_ms / 1000.0) if prefill_ms > 0 else wall, wall


def _fit_quadratic(points: list[tuple[int, float]]) -> tuple[float, float, float]:
    """Least squares over basis [N², N]: time = a·N² + b·N (no intercept —
    prefill time is 0 at 0 tokens).

    Returns (a, b, r²). The a-term is the attention (O(N²)) share; a near-zero
    a with large b means the matmuls (O(N)) dominate and a linear cap policy
    would be enough.
    """
    # Solve normal equations for 2 params.
    s4 = sum(n**4 for n, _ in points)
    s3 = sum(n**3 for n, _ in points)
    s2 = sum(n**2 for n, _ in points)
    t_n2 = sum(t * n**2 for n, t in points)
    t_n = sum(t * n for n, t in points)
    det = s4 * s2 - s3 * s3
    if abs(det) < 1e-12:
        return 0.0, 0.0, 0.0
    a = (t_n2 * s2 - t_n * s3) / det
    b = (s4 * t_n - s3 * t_n2) / det
    # r² against the zero-mean baseline is ill-defined; report explained
    # variance vs the plain mean instead.
    ts = [t for _, t in points]
    mean_t = sum(ts) / len(ts)
    ss_tot = sum((t - mean_t) ** 2 for t in ts)
    ss_res = sum((t - (a * n**2 + b * n)) ** 2 for n, t in points)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 1.0
    return a, b, r2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--slnc", default=None, help="explicit model.slnc path")
    ap.add_argument("--sizes", default="1024,2048,4096,8192", help="prompt token counts")
    ap.add_argument("--reps", type=int, default=3, help="timed reps per size (interleaved)")
    ap.add_argument("--warmup", type=int, default=64, help="warmup prompt tokens (untimed)")
    ap.add_argument("--timeout", type=float, default=120.0, help="budget to project against (s)")
    ap.add_argument("--json", action="store_true", help="machine-readable summary")
    args = ap.parse_args()

    slnc = Path(args.slnc) if args.slnc else _resolve_slnc(args.model)
    provider, load_s = _load_model(slnc, args.model)
    tokenizer = provider._tokenizer
    model = provider._model

    from domain.training._internal import slonet_kernels as sk

    numba_on = sk._numba_opted_in()
    budget_mb = sk._prefill_fallback_budget() / 2**20

    first_block = next(l for l in model.layers if hasattr(l, "attn"))
    n_heads = first_block.attn.n_heads
    head_dim = first_block.attn.head_dim
    max_seq_len = getattr(model, "max_seq_len", 0)

    sizes = [int(s) for s in args.sizes.split(",")]
    over = [n for n in sizes if n > max_seq_len]
    if over:
        raise SystemExit(f"sizes {over} exceed model max_seq_len={max_seq_len}")

    # Warmup: prime kernel dispatch + page in weights.
    _prefill_once(provider, _prompt_ids(tokenizer, args.warmup))

    load_start = os.getloadavg()[0]
    t_run0 = time.perf_counter()

    # Interleaved round-robin (benchmark_decode_slope pattern): every size is
    # sampled across the whole run so box load drift averages out instead of
    # biasing one size.
    samples: dict[int, list[float]] = {n: [] for n in sizes}
    for _ in range(max(1, args.reps)):
        for n in sizes:
            prefill_s, _wall = _prefill_once(provider, _prompt_ids(tokenizer, n))
            samples[n].append(prefill_s)

    load_end = os.getloadavg()[0]
    run_s = time.perf_counter() - t_run0

    points: list[tuple[int, float]] = []
    rows = []
    for n in sizes:
        reps_list = samples[n]
        secs = min(reps_list)  # min kept: the box-fastest run is the true cost
        points.append((n, secs))
        rows.append((n, secs, secs / n * 1000.0, secs / args.timeout * 100.0))

    a, b, r2 = _fit_quadratic(points)
    if len(points) < 2:
        a = b = 0.0  # a single size cannot identify a curve — don't fake one

    # Solve a·N² + b·N = timeout for N (positive root) — where the budget bites.
    breach_n = None
    if a > 0 or b > 0:
        disc = b * b + 4 * a * args.timeout
        if disc >= 0:
            n_star = (-b + disc**0.5) / (2 * a) if a > 0 else args.timeout / b
            if n_star > 0:
                breach_n = int(n_star)

    if args.json:
        print(
            json.dumps(
                {
                    "model": args.model,
                    "load_s": round(load_s, 2),
                    "n_heads": n_heads,
                    "head_dim": head_dim,
                    "max_seq_len": max_seq_len,
                    "numba": numba_on,
                    "fallback_budget_mb": round(budget_mb, 1),
                    "reps": max(1, args.reps),
                    "run_s": round(run_s, 2),
                    "loadavg_start": round(load_start, 2),
                    "loadavg_end": round(load_end, 2),
                    "points": [
                        {
                            "n": n,
                            "prefill_s": round(s, 3),
                            "ms_per_tok": round(ms, 2),
                            "pct_budget": round(pct, 1),
                        }
                        for n, s, ms, pct in rows
                    ],
                    "prefill_s": {str(n): round(s, 3) for n, s, _, _ in rows},
                    "quad_a": a,
                    "quad_b": b,
                    "quad_r2": round(r2, 4),
                    "breach_tokens": breach_n,
                    "budget_s": args.timeout,
                }
            )
        )
    else:
        load1, load5, load15 = os.getloadavg()
        print(f"model: {args.model}  (H={n_heads}, E_head={head_dim}, max_seq={max_seq_len})")
        print(
            f"kernels: numba={'on' if numba_on else 'off (numpy fallback)'}  "
            f"fallback_budget={budget_mb:.0f} MB"
        )
        print(f"loadavg: {load1:.1f} {load5:.1f} {load15:.1f}  ({os.cpu_count()} cpus)")
        print(f"load:  {load_s:.1f}s   int8 symmetric")
        print(f"budget: {args.timeout:.0f}s request timeout")
        print()
        print(f"{'prompt':>8}  {'prefill':>9}  {'ms/tok':>8}  {'%budget':>8}")
        for n, secs, ms, pct in rows:
            print(f"{n:8d}  {secs:8.2f}s  {ms:7.1f}m  {pct:7.1f}%")
        print()
        print(f"fit: time = {a:.3e}·N² + {b:.3e}·N   r²={r2:.4f}  (a-term = attention share)")
        if breach_n:
            print(f"projected: the {args.timeout:.0f}s budget bites at ~{breach_n:,} tokens")
        else:
            print(f"projected: no breach within fit range (budget {args.timeout:.0f}s)")
        print(f"spread/min kept, run {run_s:.1f}s, load {load_start:.2f}→{load_end:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
