"""Cruncher benchmark — ingest throughput and dedup rate.

Measures the streaming pipeline (scrub → quality → MinHash dedup) over a
synthetic corpus with a controlled duplicate fraction, then seals a snapshot
and reports wall-clock numbers.

Usage:
    PYTHONPATH=. python scripts/benchmark_cruncher.py [--records N]
        [--dup-rate R] [--json-out PATH]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from domain.training._internal.cruncher import (  # noqa: E402
    Cruncher,
    SnapshotStore,
)

_TOPICS = [
    "kernel boot loader driver firmware bios uefi grub init systemd",
    "neural network gradient descent backpropagation loss optimizer adam",
    "photosynthesis chloroplast chlorophyll glucose oxygen carbon dioxide",
    "mitochondria atp oxidative phosphorylation cellular respiration energy",
    "photosynthesis light reactions calvin cycle rubisco stomata mesophyll",
]


def make_records(n: int, dup_rate: float, seed: int = 7) -> list[str]:
    """Synthetic corpus: unique docs plus planted near-copies at *dup_rate*."""
    rng = random.Random(seed)
    unique_n = max(1, int(n * (1 - dup_rate)))
    dup_n = n - unique_n
    records = []
    for i in range(unique_n):
        topic = _TOPICS[i % len(_TOPICS)]
        words = [rng.choice(topic.split()) for _ in range(rng.randint(40, 120))]
        words[0] = f"doc{i}"
        records.append(" ".join(words))
    for i in range(dup_n):
        # Near-copy of an earlier unique record (small tail edit).
        src = records[rng.randrange(unique_n)]
        records.append(src + f" additionally note {i}")
    rng.shuffle(records)
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark the cruncher pipeline")
    ap.add_argument("--records", type=int, default=2000, help="records to ingest")
    ap.add_argument("--dup-rate", type=float, default=0.3, help="planted duplicate fraction")
    ap.add_argument("--json-out", type=str, default=None, help="write results JSON here")
    args = ap.parse_args()

    records = make_records(args.records, args.dup_rate)
    total_bytes = sum(len(r.encode("utf-8")) for r in records)

    with tempfile.TemporaryDirectory(prefix="cruncher-bench-") as tmp:
        store = SnapshotStore(Path(tmp) / "crunched")
        cruncher = Cruncher("bench", store=store)

        t0 = time.perf_counter()
        result = cruncher.crunch_stream(records, source="synthetic")
        ingest_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        ref = cruncher.seal(sources=["synthetic"])
        seal_s = time.perf_counter() - t1

    mb = total_bytes / (1024 * 1024)
    report = {
        "records": args.records,
        "dup_rate_planted": args.dup_rate,
        "bytes_in_mb": round(mb, 2),
        "ingest_seconds": round(ingest_s, 3),
        "seal_seconds": round(seal_s, 3),
        "throughput_mb_s": round(mb / ingest_s, 2) if ingest_s else None,
        "records_per_s": round(args.records / ingest_s, 0) if ingest_s else None,
        "kept": result.records_kept,
        "dropped_duplicate": result.dropped_duplicate,
        "dropped_quality": result.dropped_quality,
        "dedup_rate": round(result.dedup_rate, 4),
        "snapshot_version": ref.version,
    }
    print(json.dumps(report, indent=2))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
