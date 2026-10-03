"""
MogDB SyncableCollection Benchmark Suite

Compares MogDB vs file-based storage performance across:
- Raw MogDB (no sync)
- MogDB + SyncableCollection (full sync)
- MogDB + SyncableCollection (lazy sync)
- MogDB + SyncableCollection (batch sync)
- Plain JSON file read/write
- Gzip JSON file read/write

Usage:
    python scripts/benchmark_mogdb_sync.py
    python scripts/benchmark_mogdb_sync.py --docs 5000 --ops 2000
"""

from __future__ import annotations

import gzip
import json
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, "packages/mogdb/src")

from mogdb import MogDB
from mogdb.json_sync import SyncableCollection


@dataclass
class BenchResult:
    name: str
    elapsed: float
    ops: int
    unit: str = "ops/sec"

    @property
    def rate(self) -> float:
        return self.ops / self.elapsed if self.elapsed > 0 else 0


@dataclass
class BenchSuite:
    results: list[BenchResult] = field(default_factory=list)

    def add(self, name: str, elapsed: float, ops: int):
        self.results.append(BenchResult(name, elapsed, ops))

    def report(self):
        print("\n" + "=" * 60)
        print("  MogDB SyncableCollection Benchmark Results")
        print("=" * 60)
        print(f"  {'Test':<40} {'Time':>8} {'Rate':>12}")
        print("-" * 60)
        for r in self.results:
            print(f"  {r.name:<40} {r.elapsed:>7.4f}s {r.rate:>10.0f} {r.unit}")
        print("=" * 60)


def make_docs(n: int) -> list[dict]:
    return [
        {
            "id": i,
            "name": f"model_{i}",
            "source": "huggingface" if i % 3 == 0 else "local",
            "status": "loaded" if i % 5 == 0 else "available",
            "parameters": 1_000_000 * (i % 10),
            "tags": ["small", "chat"] if i % 2 == 0 else ["large", "code"],
            "metadata": {"created": "2026-01-01", "version": i % 100},
        }
        for i in range(n)
    ]


def bench_json_write(docs: list[dict], path: Path) -> float:
    start = time.perf_counter()
    with open(path, "w") as f:
        json.dump(docs, f)
    return time.perf_counter() - start


def bench_json_read(path: Path) -> tuple[float, int]:
    start = time.perf_counter()
    with open(path) as f:
        data = json.load(f)
    return time.perf_counter() - start, len(data)


def bench_gzip_write(docs: list[dict], path: Path) -> float:
    start = time.perf_counter()
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(docs, f)
    return time.perf_counter() - start


def bench_gzip_read(path: Path) -> tuple[float, int]:
    start = time.perf_counter()
    with gzip.open(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    return time.perf_counter() - start, len(data)


def bench_mogdb_raw(docs: list[dict], db_path: Path) -> float:
    db = MogDB(str(db_path))
    col = db.collection("bench")
    start = time.perf_counter()
    col.insert_many(docs)
    elapsed = time.perf_counter() - start
    db.close()
    return elapsed


def bench_mogdb_sync_full(docs: list[dict], db_path: Path, sync_dir: Path) -> float:
    db = MogDB(str(db_path), sync_dir=str(sync_dir))
    col = db.collection("bench")
    start = time.perf_counter()
    col.insert_many(docs)
    elapsed = time.perf_counter() - start
    db.close()
    return elapsed


def bench_mogdb_sync_lazy(docs: list[dict], db_path: Path, sync_dir: Path) -> float:
    db = MogDB(str(db_path))
    raw = db.collection("bench")
    json_path = sync_dir / "bench.json"
    sync_col = SyncableCollection(raw, json_path, sync_mode="lazy", lazy_sync_interval=0.1)
    start = time.perf_counter()
    sync_col.insert_many(docs)
    elapsed = time.perf_counter() - start
    sync_col.close()
    db.close()
    return elapsed


def bench_mogdb_sync_batch(docs: list[dict], db_path: Path, sync_dir: Path) -> float:
    db = MogDB(str(db_path))
    raw = db.collection("bench")
    json_path = sync_dir / "bench.json"
    sync_col = SyncableCollection(raw, json_path, sync_mode="full")
    start = time.perf_counter()
    with sync_col.batch():
        for doc in docs:
            sync_col.insert_one(doc)
    elapsed = time.perf_counter() - start
    sync_col.close()
    db.close()
    return elapsed


def bench_single_insert_comparison(n_ops: int, tmp: Path) -> list[BenchResult]:
    """Compare single-insert performance across modes."""
    results = []
    doc_template = {"id": 0, "value": "x" * 100}

    # Raw MogDB
    db = MogDB(str(tmp / "raw"))
    col = db.collection("single")
    docs = [{**doc_template, "id": i} for i in range(n_ops)]
    start = time.perf_counter()
    for d in docs:
        col.insert_one(d)
    elapsed = time.perf_counter() - start
    results.append(BenchResult("MogDB single insert", elapsed, n_ops))
    db.close()

    # SyncableCollection full
    db = MogDB(str(tmp / "sync_full"))
    raw = db.collection("single")
    sc = SyncableCollection(raw, tmp / "sync_full.json")
    docs = [{**doc_template, "id": i} for i in range(n_ops)]
    start = time.perf_counter()
    for d in docs:
        sc.insert_one(d)
    elapsed = time.perf_counter() - start
    results.append(BenchResult("SyncCollection full (single)", elapsed, n_ops))
    sc.close()
    db.close()

    # SyncableCollection lazy
    db = MogDB(str(tmp / "sync_lazy"))
    raw = db.collection("single")
    sc = SyncableCollection(raw, tmp / "sync_lazy.json", sync_mode="lazy", lazy_sync_interval=0.05)
    docs = [{**doc_template, "id": i} for i in range(n_ops)]
    start = time.perf_counter()
    for d in docs:
        sc.insert_one(d)
    elapsed = time.perf_counter() - start
    results.append(BenchResult("SyncCollection lazy (single)", elapsed, n_ops))
    sc.close()
    db.close()

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="MogDB SyncableCollection Benchmark")
    parser.add_argument("--docs", type=int, default=1000, help="Number of documents")
    parser.add_argument("--ops", type=int, default=500, help="Number of single operations")
    args = parser.parse_args()

    n_docs = args.docs
    n_ops = args.ops
    docs = make_docs(n_docs)
    suite = BenchSuite()

    with tempfile.TemporaryDirectory(prefix="mogdb_bench_") as tmp:
        tmp_path = Path(tmp)

        # --- Bulk insert benchmarks ---
        t = bench_json_write(docs, tmp_path / "plain.json")
        suite.add(f"JSON write ({n_docs} docs)", t, n_docs)

        t, count = bench_json_read(tmp_path / "plain.json")
        suite.add(f"JSON read ({n_docs} docs)", t, count)

        t = bench_gzip_write(docs, tmp_path / "gz.json.gz")
        suite.add(f"Gzip write ({n_docs} docs)", t, n_docs)

        t, count = bench_gzip_read(tmp_path / "gz.json.gz")
        suite.add(f"Gzip read ({n_docs} docs)", t, count)

        t = bench_mogdb_raw(docs, tmp_path / "mogdb_raw")
        suite.add(f"MogDB raw insert ({n_docs} docs)", t, n_docs)

        t = bench_mogdb_sync_full(docs, tmp_path / "mogdb_sf", tmp_path / "sync_f")
        suite.add(f"MogDB+Sync full ({n_docs} docs)", t, n_docs)

        t = bench_mogdb_sync_lazy(docs, tmp_path / "mogdb_sl", tmp_path / "sync_l")
        suite.add(f"MogDB+Sync lazy ({n_docs} docs)", t, n_docs)

        t = bench_mogdb_sync_batch(docs, tmp_path / "mogdb_sb", tmp_path / "sync_b")
        suite.add(f"MogDB+Sync batch ({n_docs} docs)", t, n_docs)

        # --- Single insert benchmarks ---
        single_results = bench_single_insert_comparison(n_ops, tmp_path)
        for r in single_results:
            suite.add(r.name + f" ({n_ops} ops)", r.elapsed, r.ops)

        # --- File size comparison ---
        plain_size = (tmp_path / "plain.json").stat().st_size
        gz_size = (tmp_path / "gz.json.gz").stat().st_size
        print(f"\n  File sizes: plain={plain_size:,}B  gzip={gz_size:,}B  ratio={gz_size/plain_size:.1%}")

    suite.report()


if __name__ == "__main__":
    main()

