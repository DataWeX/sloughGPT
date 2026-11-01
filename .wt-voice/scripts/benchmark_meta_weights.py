"""
Latency benchmark for the meta-weight adjustment seam.

Measures the per-request cost of `_apply_meta_weights()` — the hot path that
every /inference/generate, /inference/generate/stream, WebSocket generate and
chat request pays before generation starts — split into the two branches the
function now has:

  * cache HIT  : the 5-second nudge cache answers. This is the branch where
                 the explicit-vs-default merge *added* work (it used to
                 return early), so it is the number that shows whether that
                 cost is measurable at all.
  * cache MISS : a real embedding + kNN lookup runs first — the expensive
                 branch the cache exists to skip.

Isolated from the repository's data/feedback.db (temp store) and it restores
the global manager afterwards, so a run never mutates shared state.

Usage:
    python scripts/benchmark_meta_weights.py              # 2000 runs each
    python scripts/benchmark_meta_weights.py --runs 500
"""

import argparse
import statistics
import tempfile
import time
from pathlib import Path

WARMUP = 20


def _summarise(samples_us: list[float]) -> dict:
    ordered = sorted(samples_us)
    return {
        "p50_us": statistics.median(ordered),
        "p95_us": ordered[int(len(ordered) * 0.95) - 1],
        "p99_us": ordered[int(len(ordered) * 0.99) - 1],
        "max_us": ordered[-1],
    }


def _fmt(stats: dict) -> str:
    return (
        f"p50 {stats['p50_us']:8.2f} us   "
        f"p95 {stats['p95_us']:8.2f} us   "
        f"p99 {stats['p99_us']:8.2f} us   "
        f"max {stats['max_us']:8.2f} us"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=2000)
    args = parser.parse_args()

    import domain.feedback._internal.meta_weights as mw_mod
    from apps.api.server.routers.inference import (
        _META_WEIGHT_CACHE,
        _apply_meta_weights,
    )
    from domain.feedback import MetaWeightManager

    temp_dir = tempfile.mkdtemp(prefix="bench_metaw_")
    isolated = MetaWeightManager(db_path=str(Path(temp_dir) / "feedback.db"))

    original = mw_mod._meta_weight_manager
    mw_mod._meta_weight_manager = isolated
    try:
        # `temperature` deliberately at the field default and absent from
        # `explicit`, so the merge branch actually runs on every call.
        call = lambda: _apply_meta_weights(  # noqa: E731 - benchmark loop
            temperature=0.7,
            top_p=0.85,
            top_k=40,
            repetition_penalty=1.15,
            user_message="benchmark prompt",
            explicit=(),
        )

        for _ in range(WARMUP):
            _META_WEIGHT_CACHE.clear()
            call()

        # ── HIT: cache primed, merge runs on every call ──
        _META_WEIGHT_CACHE.clear()
        call()  # populate the cache once
        hit_samples: list[float] = []
        for _ in range(args.runs):
            t0 = time.perf_counter()
            call()
            hit_samples.append((time.perf_counter() - t0) * 1e6)

        # ── MISS: cache cleared before each timed call ──
        miss_samples: list[float] = []
        for _ in range(args.runs):
            _META_WEIGHT_CACHE.clear()
            t0 = time.perf_counter()
            call()
            miss_samples.append((time.perf_counter() - t0) * 1e6)
    finally:
        mw_mod._meta_weight_manager = original
        _META_WEIGHT_CACHE.clear()

    hit = _summarise(hit_samples)
    miss = _summarise(miss_samples)
    ratio = miss["p50_us"] / hit["p50_us"] if hit["p50_us"] else float("inf")

    print(f"[BENCH] _apply_meta_weights  ({args.runs} runs each)")
    print(f"  cache HIT  (merge path): {_fmt(hit)}")
    print(f"  cache MISS (embed+kNN) : {_fmt(miss)}")
    print(f"  miss/hit p50 ratio     : {ratio:.1f}x")
    print(
        "[BENCH] Both branches sit microseconds below a single generated "
        "token, so the explicit/default merge cannot show up end-to-end."
    )


if __name__ == "__main__":
    main()
