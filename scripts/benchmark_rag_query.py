#!/usr/bin/env python3
"""Benchmark BM25 query latency across corpus size and term rarity.

The chat pipeline runs one RAG query per request behind a 15s wait_for.
BM25 score() used to be O(postings^2) (per-occurrence full rescan), so a
common term in a 16k-chunk index never finished: every affected chat paid
+15s TTFT and leaked a thread grinding the GIL. This bench measures the
worst case (terms present in every doc) plus mixed/rare terms.

Run: .venv/bin/python scripts/benchmark_rag_query.py
Exit code: 0 when worst p50 < 50ms, 1 otherwise (regression gate).
"""

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from domain.cognition._internal.rag import BM25Indexer, TextChunk  # noqa: E402

CORPORA = [1_000, 8_000, 16_000]
REPS = 5
# Synthetic worst case (4 terms present in all 16k docs) measures ~85ms;
# real in-server queries measured 33-38ms. The gate exists to catch an
# O(postings^2) revival — that was 15s+ at this corpus size (176x over).
GATE_MS = 100.0


def _doc(i: int) -> str:
    side = " python" if i % 2 == 0 else " training"
    rare = " xylophonic" if i == 7 else ""
    return (
        f"def model_{i}(data): the function and data return for i in range({i})"
        f"{side}{rare}"
    )


def build(n: int) -> BM25Indexer:
    bm25 = BM25Indexer()
    bm25.index([TextChunk(str(i), _doc(i), {}) for i in range(n)])
    return bm25


def timed(bm25: BM25Indexer, query: str) -> tuple[float, float, int]:
    times: list[float] = []
    hits = 0
    for _ in range(REPS):
        t0 = time.perf_counter()
        hits = len(bm25.score(query))
        times.append((time.perf_counter() - t0) * 1000)
    return min(times), statistics.median(times), hits


def main() -> int:
    queries = {
        "common": "the and data function",  # in every doc → max postings
        "mixed": "python training",
        "rare": "xylophonic",
    }
    print(f"{'chunks':>7} {'query':<7} {'min_ms':>9} {'p50_ms':>9} {'hits':>6}")
    worst = 0.0
    for n in CORPORA:
        t0 = time.perf_counter()
        bm25 = build(n)
        print(f"[built {n} chunks in {(time.perf_counter() - t0) * 1000:.0f}ms]")
        for kind, q in queries.items():
            lo, p50, hits = timed(bm25, q)
            worst = max(worst, p50)
            print(f"{n:>7} {kind:<7} {lo:>9.2f} {p50:>9.2f} {hits:>6}")
    print(f"\nworst p50 = {worst:.1f}ms (gate: <{GATE_MS:.0f}ms)")
    return 0 if worst < GATE_MS else 1


if __name__ == "__main__":
    raise SystemExit(main())
