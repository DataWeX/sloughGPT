"""Prefill attention A/B — BLAS dispatch (``optimize=True``) vs the generic loop.

Why it exists (card 20261102_001616): a cProfile of a 681-token prefill put
**10.8s of 22.0s (49%) in ``c_einsum``** — the single largest lever on /chat
time-to-first-token. ``np.einsum`` defaults to ``optimize=False``, which takes
a generic contraction loop instead of dispatching to BLAS, so the shipped
prefill in ``fused_attention_multi`` now passes ``optimize=True`` on its two
einsums.

This benchmark is where the performance number lives, because a *test* must
not carry a stopwatch: the same shape has read 1.18x under contention and
5.9x when idle, so timing assertions are flake generators. The tests pin
*which path is requested*; this pins *what that path is worth*.

Method — both facts matter for the number being trustworthy:

  * **Interleaved arms.** Each repetition times generic then BLAS, alternating
    order every rep so CPU drift, thermal ramp and neighbouring sessions hit
    both arms equally. Timing one arm fully and then the other — the obvious
    way to write this — let drift through and disagreed by ~1.5x.
  * **Min-of-reps, not mean.** A single scheduler stall inflates one sample;
    the minimum is the standard estimator for "how fast can this run".

Outputs per size: generic ms, BLAS ms, the ratio, and an ``allclose`` check,
because a speedup that changes the numbers is not a speedup.

Usage:
    ./scripts/python scripts/benchmark_prefill_attention.py
    ./scripts/python scripts/benchmark_prefill_attention.py --reps 7
    ./scripts/python scripts/benchmark_prefill_attention.py --seqs 256,681,2048
    ./scripts/python scripts/benchmark_prefill_attention.py --json-out run.json
    ./scripts/python scripts/benchmark_prefill_attention.py --ci
        # fail unless allclose AND geo-mean speedup >= --min-ratio

``--min-ratio`` defaults to 1.5: a real regression back to ``optimize=False``
measures ~1.0x, so 1.5 separates cleanly while leaving headroom. It still
wants a quiet box — run ``--ci`` on an idle machine, not mid-suite.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Qwen2.5-0.5B prefill shapes: 14 heads, 2 kv heads (GQA 7:1), head_dim 128.
DEFAULT_SEQS = [256, 512, 681, 1024, 2048]
DEFAULT_HEADS = 14
DEFAULT_KV_HEADS = 2
DEFAULT_HEAD_DIM = 128
# Same tolerance the equivalence tests use.
RTOL, ATOL = 1e-5, 1e-6


def _force_generic(einsum):
    """Return a context manager pinning ``np.einsum`` to ``optimize=False``.

    The shipped code passes ``optimize=True`` explicitly, so simulating the
    pre-change arm means overriding the kwarg rather than omitting it.
    """
    import contextlib

    @contextlib.contextmanager
    def pinned():
        def forced(*args, **kwargs):
            kwargs["optimize"] = False
            return einsum(*args, **kwargs)

        import numpy as np

        np.einsum = forced
        try:
            yield
        finally:
            np.einsum = einsum

    return pinned


def _time_one(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return (time.perf_counter() - t0) * 1000.0


def bench_size(seq_len, heads, kv_heads, head_dim, reps) -> dict:
    import numpy as np

    from domain.training._internal import slonet_kernels as K

    scale = np.float32(1.0 / np.sqrt(head_dim))
    rng = np.random.default_rng(0)
    q = rng.standard_normal((seq_len, heads, head_dim), dtype=np.float32)
    k = rng.standard_normal((kv_heads, seq_len, head_dim), dtype=np.float32)
    v = rng.standard_normal((kv_heads, seq_len, head_dim), dtype=np.float32)

    call = lambda: K.fused_attention_multi(q, k, v, scale, heads, head_dim)  # noqa: E731
    generic = _force_generic(np.einsum)

    saved = K._nb_fused_attention_multi
    K._nb_fused_attention_multi = None  # force the numpy fallback branch
    try:
        with generic():
            call()  # warm both arms before any timing
            reference = call()
        with generic():
            call()

        generic_times: list[float] = []
        blas_times: list[float] = []
        for i in range(reps):
            # Alternate which arm runs first so drift is shared, not stacked.
            order = (False, True) if i % 2 == 0 else (True, False)
            for is_blas in order:
                if is_blas:
                    blas_times.append(_time_one(call))
                else:
                    with generic():
                        generic_times.append(_time_one(call))

        with generic():
            reference = call()
        blas_out = call()
    finally:
        K._nb_fused_attention_multi = saved

    gen_ms = min(generic_times)
    blas_ms = min(blas_times)
    return {
        "seq_len": seq_len,
        "generic_ms": round(gen_ms, 2),
        "blas_ms": round(blas_ms, 2),
        "ratio": round(gen_ms / blas_ms, 2),
        "allclose": bool(np.allclose(blas_out, reference, rtol=RTOL, atol=ATOL)),
        "reps": reps,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seqs", default=",".join(str(s) for s in DEFAULT_SEQS))
    ap.add_argument("--reps", type=int, default=7, help="timed reps per arm (min used)")
    ap.add_argument("--heads", type=int, default=DEFAULT_HEADS)
    ap.add_argument("--kv-heads", type=int, default=DEFAULT_KV_HEADS)
    ap.add_argument("--head-dim", type=int, default=DEFAULT_HEAD_DIM)
    ap.add_argument("--json-out", default=None, help="write full results JSON")
    ap.add_argument("--ci", action="store_true", help="exit non-zero on regression")
    ap.add_argument(
        "--min-ratio",
        type=float,
        default=1.5,
        help="with --ci: minimum geo-mean speedup (quiet box)",
    )
    args = ap.parse_args()

    try:
        seqs = [int(s) for s in args.seqs.split(",") if s.strip()]
    except ValueError:
        print(f"bad --seqs: {args.seqs!r}", file=sys.stderr)
        return 1
    if not seqs:
        print("--seqs is empty", file=sys.stderr)
        return 1

    print(
        f"prefill attention A/B — heads={args.heads} kv={args.kv_heads} "
        f"head_dim={args.head_dim} reps={args.reps} (min, interleaved)"
    )
    print(f"{'seq':>6} {'generic':>12} {'BLAS':>11} {'ratio':>8}  allclose")
    print("-" * 56)

    results = []
    for seq_len in seqs:
        r = bench_size(seq_len, args.heads, args.kv_heads, args.head_dim, args.reps)
        results.append(r)
        flag = "ok" if r["allclose"] else "MISMATCH"
        print(
            f"{r['seq_len']:>6} {r['generic_ms']:>10.2f}ms {r['blas_ms']:>9.2f}ms "
            f"{r['ratio']:>7.2f}x  {flag}"
        )
    print("-" * 56)

    ratios = [r["ratio"] for r in results]
    geo = math.exp(sum(math.log(x) for x in ratios) / len(ratios))
    print(f"  speedup  min {min(ratios):.2f}x  geo-mean {geo:.2f}x  max {max(ratios):.2f}x")

    if args.json_out:
        payload = {
            "kind": "prefill_attention",
            "heads": args.heads,
            "kv_heads": args.kv_heads,
            "head_dim": args.head_dim,
            "results": results,
            "geo_mean_ratio": round(geo, 3),
        }
        Path(args.json_out).write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {args.json_out}")

    if args.ci:
        bad = [r for r in results if not r["allclose"]]
        if bad:
            print(
                f"[FAIL] BLAS output diverged from generic at seq {[r['seq_len'] for r in bad]}",
                file=sys.stderr,
            )
            return 1
        if geo < args.min_ratio:
            print(
                f"[FAIL] geo-mean speedup {geo:.2f}x < {args.min_ratio}x — "
                "either optimize=True regressed or the box is contended "
                "(re-run idle before concluding)",
                file=sys.stderr,
            )
            return 1
        print(f"[OK] outputs identical within tolerance; speedup {geo:.2f}x >= {args.min_ratio}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
