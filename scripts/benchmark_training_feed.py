"""
Training Feed Benchmark — measures the training-feed consumption path.

Three metrics, exercised against the live conversation corpus on disk:
  1. read_throughput  — records/sec the TrainingFeedClient can read (local disk).
  2. convert_throughput — query/assistant pairs per second through pairs_to_text.
  3. batch_latency    — p50/p95 of FeedBatchSampler.get_batch over N batches.

Usage:
    python scripts/benchmark_training_feed.py                  # whole pipeline
    python scripts/benchmark_training_feed.py --batches 200    # more batches
    python scripts/benchmark_training_feed.py --source feedback # different source

Gold Standard (pass/fail thresholds):
  - read_throughput:   >= 500 records/sec
  - convert_throughput: >= 2000 pairs/sec
  - batch_p95:          <= 50 ms
"""

import argparse
import statistics
import sys
import time

from domain.training._internal.training_feed import (
    FeedBatchSampler,
    TrainingFeedClient,
    pairs_to_text,
)
from domain.training._internal.training_handler import build_char_vocab

GOLD = {
    "min_read_records_per_sec": 500.0,
    "min_convert_pairs_per_sec": 2000.0,
    "max_batch_p95_ms": 50.0,
}


def _p95(samples):
    if not samples:
        return 0.0
    return statistics.quantiles(sorted(samples), n=20)[-1]


def benchmark(source: str, batches: int) -> dict:
    results = {}

    client = TrainingFeedClient(source=source)
    start = time.perf_counter()
    page = client.read(limit=0)
    read_s = time.perf_counter() - start
    results["source"] = source
    results["total_records"] = page.total
    results["read_records_per_sec"] = page.total / read_s if read_s else 0.0
    print(
        f"  read: {page.total} records in {read_s * 1000:.1f} ms "
        f"({results['read_records_per_sec']:.0f} rec/s)"
    )

    pairs = client.read_pairs(limit=0)
    start = time.perf_counter()
    pairs_to_text(pairs, use_prompt_engine=True)
    convert_s = time.perf_counter() - start
    results["pairs"] = len(pairs)
    results["convert_pairs_per_sec"] = len(pairs) / convert_s if convert_s else 0.0
    print(
        f"  convert: {len(pairs)} pairs -> text in {convert_s * 1000:.1f} ms "
        f"({results['convert_pairs_per_sec']:.0f} pairs/s)"
    )

    stoi, _ = build_char_vocab(pairs_to_text(pairs, use_prompt_engine=False))
    sampler = FeedBatchSampler(stoi, block_size=128, source=source, refresh_interval=0)
    latencies = []
    for _ in range(batches):
        start = time.perf_counter()
        sampler.get_batch(32)
        latencies.append((time.perf_counter() - start) * 1000.0)
    if latencies:
        results["batch_count"] = len(latencies)
        results["batch_p50_ms"] = statistics.median(latencies)
        results["batch_p95_ms"] = _p95(latencies)
        print(
            f"  batch: n={len(latencies)} p50={statistics.median(latencies):.2f} ms "
            f"p95={_p95(latencies):.2f} ms"
        )

    results["passed"] = all(
        [
            results["read_records_per_sec"] >= GOLD["min_read_records_per_sec"],
            results["convert_pairs_per_sec"] >= GOLD["min_convert_pairs_per_sec"],
            results.get("batch_p95_ms", 0.0) <= GOLD["max_batch_p95_ms"],
        ]
    )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="api-conversations")
    parser.add_argument("--batches", type=int, default=100)
    args = parser.parse_args()

    print(f"Training Feed Benchmark — source={args.source}")
    results = benchmark(args.source, args.batches)

    verdict = "PASS" if results["passed"] else "FAIL"
    print(f"\nVerdict: {verdict}")
    return 0 if results["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
