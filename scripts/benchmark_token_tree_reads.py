"""
Token-Tree Read-Path Benchmark — stats / merges latency and cold-start cost

Measures the two GET endpoints that were showing >1s SLOW warnings:
  - GET /token-tree/stats  (aggregate library statistics)
  - GET /token-tree/merges (ranked merge rules)

Reports:
  - Cold start: first-call lazy training vs. loading the persisted default cache
  - Warm reads: repeated stats() and top_merges() latency (memoized aggregates)
  - Scaling: merge ranking cost as merge count grows (large user-trained trees)

Usage:
    python scripts/benchmark_token_tree_reads.py
    python scripts/benchmark_token_tree_reads.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@dataclass
class ReadResult:
    name: str
    latencies: list[float]
    mean_ms: float
    p50_ms: float
    p95_ms: float


def _summarize(name: str, latencies: list[float]) -> ReadResult:
    lat = sorted(latencies)
    return ReadResult(
        name=name,
        latencies=[round(x, 4) for x in lat],
        mean_ms=round(1000 * sum(lat) / len(lat), 3),
        p50_ms=round(1000 * lat[len(lat) // 2], 3),
        p95_ms=round(1000 * lat[int(len(lat) * 0.95)], 3),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit results as JSON")
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()

    from domain.training._internal.token_tree import TokenTree
    from domain.training._internal.token_tree_manager import TokenTreeManager

    results: list[ReadResult] = []

    def measure(fn, n: int) -> list[float]:
        out = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            out.append(time.perf_counter() - t0)
        return out

    # Cold start: first run trains + persists cache; re-run loads it.
    cache = Path("data/token_trees/_cache")
    cache_was_present = cache.exists()

    def cold_start() -> float:
        TokenTreeManager._instance = None
        t0 = time.perf_counter()
        TokenTreeManager.get_instance().get_tree()
        return time.perf_counter() - t0

    cold_train = cold_start()
    cold_load = cold_start()

    mgr = TokenTreeManager.get_instance()
    mgr.get_tree()

    results.append(
        ReadResult(
            name="cold-start (first: train+persist)",
            latencies=[round(cold_train, 4)],
            mean_ms=round(1000 * cold_train, 3),
            p50_ms=round(1000 * cold_train, 3),
            p95_ms=round(1000 * cold_train, 3),
        )
    )
    results.append(
        ReadResult(
            name="cold-start (warm cache: load from disk)",
            latencies=[round(cold_load, 4)],
            mean_ms=round(1000 * cold_load, 3),
            p50_ms=round(1000 * cold_load, 3),
            p95_ms=round(1000 * cold_load, 3),
        )
    )

    results.append(_summarize("stats() warm (memoized)", measure(mgr.stats, args.iterations)))
    results.append(
        _summarize(
            "top_merges(20) warm (memoized)", measure(lambda: mgr.top_merges(20), args.iterations)
        )
    )
    results.append(
        _summarize(
            "matrix_summary(8) warm (memoized)",
            measure(lambda: mgr.matrix_summary(8), args.iterations),
        )
    )
    results.append(
        _summarize(
            "similar('quick') warm (memoized)",
            measure(lambda: mgr.similar("quick"), args.iterations),
        )
    )
    results.append(
        _summarize(
            "embedding_info('quick') warm (memoized)",
            measure(lambda: mgr.embedding_info("quick"), args.iterations),
        )
    )

    # Scaling: large user-trained trees recompute rankings with no cache.
    import random
    import string

    random.seed(0)
    chars = list(string.ascii_lowercase)
    for n_merges in (2000, 20000):
        t = TokenTree()
        t._trained = True
        t.merges = [(random.choice(chars), random.choice(chars)) for _ in range(n_merges)]
        t._freqs = {i: random.randint(1, 10000) for i in range(n_merges)}
        t.stoi = {a + b: idx for idx, (a, b) in enumerate(t.merges)}
        results.append(
            _summarize(
                f"top_merges(20) {n_merges} merges (uncached)",
                measure(lambda: t.top_merges(20), args.iterations),
            )
        )
        results.append(
            _summarize(
                f"top_merges(20) {n_merges} merges (cached)",
                measure(lambda: t.top_merges(20), args.iterations),
            )
        )

    if not cache_was_present:
        import shutil

        shutil.rmtree(cache, ignore_errors=True)

    if args.json:
        print(json.dumps([asdict(r) for r in results], indent=2))
    else:
        print(f"{'name':<42} {'mean(ms)':>9} {'p50(ms)':>9} {'p95(ms)':>9}")
        for r in results:
            print(f"{r.name:<42} {r.mean_ms:>9.3f} {r.p50_ms:>9.3f} {r.p95_ms:>9.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

