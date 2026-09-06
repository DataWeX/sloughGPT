"""Benchmark tests for MogDB vs file-based storage.

Measures read/write performance for:
1. MogDB (our embedded document DB)
2. Plain JSON files (baseline)
3. Compressed JSON files (gzip)
"""

import json
import gzip
import time
import tempfile
from pathlib import Path

import pytest


# ── Fixtures ─────────────────────────────────────────────────────────────────

NUM_DOCS = 1000
NUM_OPS = 500


@pytest.fixture
def sample_docs():
    """Generate sample documents for benchmarking."""
    return [
        {
            "id": i,
            "name": f"model_{i}",
            "source": "huggingface" if i % 3 == 0 else "local",
            "status": "loaded" if i % 5 == 0 else "available",
            "parameters": 1000000 * (i % 10),
            "tags": ["small", "chat"] if i % 2 == 0 else ["large", "code"],
        }
        for i in range(NUM_DOCS)
    ]


@pytest.fixture
def json_path(tmp_path):
    return tmp_path / "benchmark.json"


@pytest.fixture
def gz_path(tmp_path):
    return tmp_path / "benchmark.json.gz"


@pytest.fixture
def mogdb_path(tmp_path):
    return tmp_path / "mogdb_bench"


# ── JSON benchmarks ──────────────────────────────────────────────────────────

class TestJSONBenchmark:
    def test_write_json(self, sample_docs, json_path):
        """Benchmark: write docs to plain JSON."""
        start = time.perf_counter()
        with open(json_path, "w") as f:
            json.dump(sample_docs, f)
        elapsed = time.perf_counter() - start
        ops_per_sec = len(sample_docs) / elapsed
        assert json_path.exists()
        print(f"\n  JSON write: {elapsed:.4f}s ({ops_per_sec:.0f} docs/sec)")

    def test_read_json(self, sample_docs, json_path):
        """Benchmark: read docs from plain JSON."""
        with open(json_path, "w") as f:
            json.dump(sample_docs, f)

        start = time.perf_counter()
        with open(json_path) as f:
            data = json.load(f)
        elapsed = time.perf_counter() - start
        assert len(data) == len(sample_docs)
        print(f"\n  JSON read: {elapsed:.4f}s ({len(data)/elapsed:.0f} docs/sec)")

    def test_write_gzip(self, sample_docs, gz_path):
        """Benchmark: write docs to gzipped JSON."""
        start = time.perf_counter()
        with gzip.open(gz_path, "wt", encoding="utf-8") as f:
            json.dump(sample_docs, f)
        elapsed = time.perf_counter() - start
        ops_per_sec = len(sample_docs) / elapsed
        assert gz_path.exists()
        print(f"\n  Gzip write: {elapsed:.4f}s ({ops_per_sec:.0f} docs/sec)")

    def test_read_gzip(self, sample_docs, gz_path):
        """Benchmark: read docs from gzipped JSON."""
        with gzip.open(gz_path, "wt", encoding="utf-8") as f:
            json.dump(sample_docs, f)

        start = time.perf_counter()
        with gzip.open(gz_path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        elapsed = time.perf_counter() - start
        assert len(data) == len(sample_docs)
        print(f"\n  Gzip read: {elapsed:.4f}s ({len(data)/elapsed:.0f} docs/sec)")

    def test_file_size_comparison(self, sample_docs, json_path, gz_path):
        """Compare file sizes: plain JSON vs gzipped."""
        with open(json_path, "w") as f:
            json.dump(sample_docs, f)
        with gzip.open(gz_path, "wt", encoding="utf-8") as f:
            json.dump(sample_docs, f)

        plain_size = json_path.stat().st_size
        gz_size = gz_path.stat().st_size
        ratio = gz_size / plain_size if plain_size > 0 else 0
        print(f"\n  Plain JSON: {plain_size:,} bytes")
        print(f"  Gzipped:    {gz_size:,} bytes")
        print(f"  Ratio:      {ratio:.2%}")
        assert gz_size <= plain_size  # Gzip should be smaller or equal


# ── MogDB benchmarks ─────────────────────────────────────────────────────────

class TestMogDBBenchmark:
    def test_insert_many(self, sample_docs, mogdb_path):
        """Benchmark: bulk insert into MogDB."""
        from mogdb import MogDB
        db = MogDB(str(mogdb_path))
        col = db.collection("bench")

        start = time.perf_counter()
        col.insert_many(sample_docs)
        elapsed = time.perf_counter() - start
        ops_per_sec = len(sample_docs) / elapsed
        print(f"\n  MogDB insert_many: {elapsed:.4f}s ({ops_per_sec:.0f} docs/sec)")
        assert col.count() == len(sample_docs)

    def test_find_all(self, sample_docs, mogdb_path):
        """Benchmark: find all docs in MogDB."""
        from mogdb import MogDB
        db = MogDB(str(mogdb_path))
        col = db.collection("bench")
        col.insert_many(sample_docs)

        start = time.perf_counter()
        results = col.find()
        elapsed = time.perf_counter() - start
        print(f"\n  MogDB find all: {elapsed:.4f}s ({len(results)/elapsed:.0f} docs/sec)")
        assert len(results) == len(sample_docs)

    def test_find_one(self, sample_docs, mogdb_path):
        """Benchmark: find single doc by field."""
        from mogdb import MogDB
        db = MogDB(str(mogdb_path))
        col = db.collection("bench")
        col.insert_many(sample_docs)

        # Warm up
        col.find_one({"id": 500})

        start = time.perf_counter()
        for i in range(NUM_OPS):
            col.find_one({"id": i % NUM_DOCS})
        elapsed = time.perf_counter() - start
        print(f"\n  MogDB find_one: {elapsed:.4f}s ({NUM_OPS/elapsed:.0f} ops/sec)")

    def test_find_with_filter(self, sample_docs, mogdb_path):
        """Benchmark: find docs with filter."""
        from mogdb import MogDB
        db = MogDB(str(mogdb_path))
        col = db.collection("bench")
        col.insert_many(sample_docs)

        start = time.perf_counter()
        results = col.find({"status": "loaded"})
        elapsed = time.perf_counter() - start
        print(f"\n  MogDB filtered find: {elapsed:.4f}s ({len(results)/elapsed:.0f} docs/sec)")

    def test_update_one(self, sample_docs, mogdb_path):
        """Benchmark: update single doc."""
        from mogdb import MogDB
        db = MogDB(str(mogdb_path))
        col = db.collection("bench")
        col.insert_many(sample_docs)

        start = time.perf_counter()
        for i in range(NUM_OPS):
            col.update_one({"id": i % NUM_DOCS}, {"$set": {"status": "updated"}})
        elapsed = time.perf_counter() - start
        print(f"\n  MogDB update_one: {elapsed:.4f}s ({NUM_OPS/elapsed:.0f} ops/sec)")

    def test_aggregate(self, sample_docs, mogdb_path):
        """Benchmark: aggregation pipeline."""
        from mogdb import MogDB
        db = MogDB(str(mogdb_path))
        col = db.collection("bench")
        col.insert_many(sample_docs)

        start = time.perf_counter()
        results = col.aggregate([
            {"$group": {"_id": "$status", "count": {"$sum": 1}}},
        ])
        elapsed = time.perf_counter() - start
        print(f"\n  MogDB aggregate: {elapsed:.4f}s ({len(results)} groups in {elapsed:.4f}s)")

    def test_json_sync(self, sample_docs, mogdb_path):
        """Benchmark: MogDB with JSON sync."""
        from mogdb import MogDB
        sync_dir = str(mogdb_path.parent / "sync")
        db = MogDB(str(mogdb_path), sync_dir=sync_dir)
        col = db.collection("bench")

        start = time.perf_counter()
        col.insert_many(sample_docs)
        elapsed = time.perf_counter() - start
        print(f"\n  MogDB + sync insert: {elapsed:.4f}s ({len(sample_docs)/elapsed:.0f} docs/sec)")

        # Check sync file exists
        sync_file = Path(sync_dir) / "bench.json"
        assert sync_file.exists()


# ── Summary ──────────────────────────────────────────────────────────────────

class TestSummary:
    def test_summary(self):
        """Print a summary of what was tested."""
        print("\n  ╔══════════════════════════════════════════╗")
        print("  ║       MogDB vs File Benchmark Suite      ║")
        print("  ╠══════════════════════════════════════════╣")
        print("  ║ • JSON read/write (plain)                ║")
        print("  ║ • JSON read/write (gzip)                 ║")
        print("  ║ • File size comparison                   ║")
        print("  ║ • MogDB insert/find/update/aggregate     ║")
        print("  ║ • MogDB with JSON sync                   ║")
        print("  ╚══════════════════════════════════════════╝")
        assert True
