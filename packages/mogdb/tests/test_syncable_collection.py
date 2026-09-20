"""Tests for SyncableCollection — JSON sync layer for MogDB."""

import gzip
import json
import time

import pytest
from mogdb import MogDB
from mogdb.json_sync import SyncableCollection

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "mogdb_test"


@pytest.fixture
def sync_dir(tmp_path):
    return tmp_path / "sync"


@pytest.fixture
def collection(db_path):
    db = MogDB(str(db_path))
    return db.collection("test")


@pytest.fixture
def sync_col(collection, sync_dir):
    return SyncableCollection(collection, sync_dir / "test.json")


@pytest.fixture
def sample_docs():
    return [{"id": i, "name": f"doc_{i}", "value": i * 10} for i in range(50)]


# ── Basic CRUD with sync ─────────────────────────────────────────────────────


class TestSyncableCollectionCRUD:
    def test_insert_one_syncs(self, sync_col, sync_dir):
        sync_col.insert_one({"_id": "a", "x": 1})
        assert (sync_dir / "test.json").exists()
        data = json.loads((sync_dir / "test.json").read_text())
        assert len(data) == 1
        assert data[0]["x"] == 1

    def test_insert_many_syncs(self, sync_col, sync_dir):
        sync_col.insert_many([{"_id": f"doc_{i}", "v": i} for i in range(10)])
        data = json.loads((sync_dir / "test.json").read_text())
        assert len(data) == 10

    def test_update_one_syncs(self, sync_col, sync_dir):
        sync_col.insert_one({"_id": "u1", "status": "old"})
        sync_col.update_one({"_id": "u1"}, {"$set": {"status": "new"}})
        data = json.loads((sync_dir / "test.json").read_text())
        assert data[0]["status"] == "new"

    def test_update_many_syncs(self, sync_col, sync_dir):
        sync_col.insert_many([{"_id": f"d{i}", "status": "old"} for i in range(5)])
        count = sync_col.update_many({"status": "old"}, {"$set": {"status": "new"}})
        assert count == 5
        data = json.loads((sync_dir / "test.json").read_text())
        assert all(d["status"] == "new" for d in data)

    def test_delete_one_syncs(self, sync_col, sync_dir):
        sync_col.insert_many([{"_id": f"d{i}"} for i in range(3)])
        sync_col.delete_one({"_id": "d0"})
        data = json.loads((sync_dir / "test.json").read_text())
        assert len(data) == 2

    def test_delete_many_syncs(self, sync_col, sync_dir):
        sync_col.insert_many([{"_id": f"d{i}", "t": i % 2} for i in range(10)])
        count = sync_col.delete_many({"t": 0})
        assert count == 5
        data = json.loads((sync_dir / "test.json").read_text())
        assert len(data) == 5

    def test_drop_syncs(self, sync_col, sync_dir):
        sync_col.insert_one({"_id": "x"})
        sync_col.drop()
        data = json.loads((sync_dir / "test.json").read_text())
        assert len(data) == 0

    def test_find_one_and_update(self, sync_col, sync_dir):
        sync_col.insert_one({"_id": "fu1", "val": 1})
        result = sync_col.find_one_and_update({"_id": "fu1"}, {"$set": {"val": 2}})
        assert result["val"] == 1
        data = json.loads((sync_dir / "test.json").read_text())
        assert data[0]["val"] == 2

    def test_find_one_and_replace(self, sync_col, sync_dir):
        sync_col.insert_one({"_id": "fr1", "val": 1})
        result = sync_col.find_one_and_replace({"_id": "fr1"}, {"_id": "fr1", "val": 99})
        assert result["val"] == 1
        data = json.loads((sync_dir / "test.json").read_text())
        assert data[0]["val"] == 99

    def test_find_one_and_delete(self, sync_col, sync_dir):
        sync_col.insert_one({"_id": "fd1", "val": 1})
        result = sync_col.find_one_and_delete({"_id": "fd1"})
        assert result["val"] == 1
        data = json.loads((sync_dir / "test.json").read_text())
        assert len(data) == 0


# ── Read operations ──────────────────────────────────────────────────────────


class TestSyncableCollectionReads:
    def test_find(self, sync_col, sample_docs):
        sync_col.insert_many(sample_docs)
        results = sync_col.find()
        assert len(results) == len(sample_docs)

    def test_find_with_query(self, sync_col, sample_docs):
        sync_col.insert_many(sample_docs)
        results = sync_col.find({"id": 5})
        assert len(results) == 1
        assert results[0]["id"] == 5

    def test_find_one(self, sync_col, sample_docs):
        sync_col.insert_many(sample_docs)
        doc = sync_col.find_one({"id": 10})
        assert doc["id"] == 10

    def test_count(self, sync_col, sample_docs):
        sync_col.insert_many(sample_docs)
        assert sync_col.count() == len(sample_docs)

    def test_aggregate(self, sync_col, sample_docs):
        sync_col.insert_many(sample_docs)
        results = sync_col.aggregate([{"$group": {"_id": "$name", "count": {"$sum": 1}}}])
        assert len(results) == len(sample_docs)

    def test_find_with_sort(self, sync_col):
        sync_col.insert_many([{"_id": f"d{i}", "val": i} for i in range(5)])
        results = sync_col.find(sort=[("val", -1)])
        assert results[0]["val"] == 4

    def test_find_with_limit(self, sync_col):
        sync_col.insert_many([{"_id": f"d{i}", "val": i} for i in range(10)])
        results = sync_col.find(limit=3)
        assert len(results) == 3


# ── Batch writes ─────────────────────────────────────────────────────────────


class TestBatchWrites:
    def test_batch_defers_sync(self, sync_col, sync_dir):
        sync_col.insert_one({"_id": "initial"})
        with sync_col.batch():
            for i in range(10):
                sync_col.insert_one({"_id": f"b{i}", "v": i})
            # JSON should not be updated yet (still has old data)
            # The batch suppresses _on_write, so the file reflects state before batch
        # After batch exits, JSON is synced
        data = json.loads((sync_dir / "test.json").read_text())
        assert len(data) == 11  # initial + 10 batch

    def test_batch_single_write(self, sync_col, sync_dir):
        with sync_col.batch():
            sync_col.insert_one({"_id": "s1"})
        data = json.loads((sync_dir / "test.json").read_text())
        assert len(data) == 1

    def test_batch_no_writes_no_sync(self, sync_col, sync_dir):
        sync_col.insert_one({"_id": "before"})
        with sync_col.batch():
            pass
        data = json.loads((sync_dir / "test.json").read_text())
        assert len(data) == 1

    def test_nested_batches(self, sync_col, sync_dir):
        sync_col.insert_one({"_id": "start"})
        with sync_col.batch():
            sync_col.insert_one({"_id": "outer"})
            with sync_col.batch():
                sync_col.insert_one({"_id": "inner"})
            # Inner batch exit — no sync because outer batch still active
        # Outer batch exit — sync happens
        data = json.loads((sync_dir / "test.json").read_text())
        assert len(data) == 3

    def test_batch_exception_no_sync(self, sync_col, sync_dir):
        sync_col.insert_one({"_id": "pre"})
        with pytest.raises(ValueError):
            with sync_col.batch():
                sync_col.insert_one({"_id": "fail"})
                raise ValueError("test")
        # On exception, batch should NOT sync
        data = json.loads((sync_dir / "test.json").read_text())
        assert len(data) == 1  # only the pre-existing doc


# ── Lazy sync ────────────────────────────────────────────────────────────────


class TestLazySync:
    def test_lazy_sync_eventually_persists(self, collection, sync_dir):
        sc = SyncableCollection(
            collection,
            sync_dir / "lazy.json",
            sync_mode="lazy",
            lazy_sync_interval=0.05,
        )
        sc.insert_one({"_id": "lazy1", "v": 1})
        # Wait for background sync
        time.sleep(0.2)
        assert (sync_dir / "lazy.json").exists()
        data = json.loads((sync_dir / "lazy.json").read_text())
        assert len(data) == 1
        sc.close()

    def test_lazy_close_flushes(self, collection, sync_dir):
        sc = SyncableCollection(
            collection,
            sync_dir / "lazy_close.json",
            sync_mode="lazy",
            lazy_sync_interval=60,  # very long interval
        )
        sc.insert_one({"_id": "lc1"})
        # Before close, file may not exist
        sc.close()
        # After close, data is flushed
        data = json.loads((sync_dir / "lazy_close.json").read_text())
        assert len(data) == 1

    def test_lazy_batch_combined(self, collection, sync_dir):
        sc = SyncableCollection(
            collection,
            sync_dir / "lazy_batch.json",
            sync_mode="lazy",
            lazy_sync_interval=0.05,
        )
        with sc.batch():
            for i in range(20):
                sc.insert_one({"_id": f"lb{i}"})
        sc.close()
        data = json.loads((sync_dir / "lazy_batch.json").read_text())
        assert len(data) == 20


# ── Gzip compression ─────────────────────────────────────────────────────────


class TestGzipSync:
    def test_gzip_write_and_read(self, collection, sync_dir):
        gz_path = sync_dir / "compressed.json.gz"
        sc = SyncableCollection(collection, gz_path)
        sc.insert_many([{"_id": f"g{i}", "v": i} for i in range(20)])
        sc.close()

        assert gz_path.exists()
        with gzip.open(gz_path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 20

    def test_gzip_smaller_than_plain(self, collection, sync_dir):
        docs = [{"_id": f"d{i}", "data": "x" * 1000} for i in range(100)]

        plain_path = sync_dir / "plain.json"
        gz_path = sync_dir / "gz.json.gz"

        sc_plain = SyncableCollection(collection, plain_path)
        sc_plain.insert_many(docs)
        sc_plain.close()

        # Need a fresh collection for gz (can't reuse same DB cleanly)
        db2 = MogDB(str(sync_dir.parent / "mogdb_gz"))
        col2 = db2.collection("gz_test")
        sc_gz = SyncableCollection(col2, gz_path)
        sc_gz.insert_many(docs)
        sc_gz.close()

        plain_size = plain_path.stat().st_size
        gz_size = gz_path.stat().st_size
        assert gz_size < plain_size

    def test_gzip_bootstrap(self, tmp_path):
        gz_path = tmp_path / "bootstrap.json.gz"
        docs = [{"_id": f"b{i}", "v": i} for i in range(10)]

        # Create the gzip file manually
        with gzip.open(gz_path, "wt", encoding="utf-8") as f:
            json.dump(docs, f)

        # New collection should bootstrap from gzip
        db = MogDB(str(tmp_path / "mogdb_boot"))
        col = db.collection("boot")
        sc = SyncableCollection(col, gz_path)
        assert sc.count() == 10
        sc.close()


# ── Bootstrap ────────────────────────────────────────────────────────────────


class TestBootstrap:
    def test_bootstrap_from_json(self, tmp_path):
        json_path = tmp_path / "boot.json"
        docs = [{"_id": f"b{i}", "v": i} for i in range(15)]
        json_path.write_text(json.dumps(docs))

        db = MogDB(str(tmp_path / "mogdb_boot"))
        col = db.collection("boot")
        sc = SyncableCollection(col, json_path)
        assert sc.count() == 15
        sc.close()

    def test_no_bootstrap_if_collection_has_data(self, tmp_path):
        json_path = tmp_path / "noboot.json"
        json_path.write_text(json.dumps([{"_id": "old"}]))

        db = MogDB(str(tmp_path / "mogdb_noboot"))
        col = db.collection("noboot")
        col.insert_one({"_id": "existing"})
        sc = SyncableCollection(col, json_path)
        # Should not bootstrap; collection has existing data
        assert sc.count() == 1
        sc.close()

    def test_no_bootstrap_if_no_file(self, tmp_path):
        db = MogDB(str(tmp_path / "mogdb_nofile"))
        col = db.collection("nofile")
        sc = SyncableCollection(col, tmp_path / "nonexistent.json")
        assert sc.count() == 0
        sc.close()


# ── Sync modes ───────────────────────────────────────────────────────────────


class TestSyncModes:
    def test_full_mode_syncs_immediately(self, sync_col, sync_dir):
        sync_col.insert_one({"_id": "f1"})
        assert (sync_dir / "test.json").exists()

    def test_append_mode(self, collection, sync_dir):
        sc = SyncableCollection(collection, sync_dir / "append.json", sync_mode="append")
        sc.insert_one({"_id": "a1"})
        sc.insert_one({"_id": "a2"})
        data = json.loads((sync_dir / "append.json").read_text())
        assert len(data) == 2
        sc.close()

    def test_manual_sync(self, collection, sync_dir):
        sc = SyncableCollection(collection, sync_dir / "manual.json", sync_mode="lazy")
        sc.insert_one({"_id": "m1"})
        # File may not exist yet in lazy mode
        sc.sync()
        data = json.loads((sync_dir / "manual.json").read_text())
        assert len(data) == 1
        sc.close()

    def test_reload_from_json(self, sync_col, sync_dir):
        sync_col.insert_many([{"_id": f"r{i}"} for i in range(10)])
        # Modify the underlying collection directly
        sync_col.underlying.drop()
        assert sync_col.count() == 0
        # Reload from JSON
        sync_col.reload()
        assert sync_col.count() == 10


# ── Atomicity ────────────────────────────────────────────────────────────────


class TestAtomicity:
    def test_sync_creates_valid_json(self, sync_col, sync_dir):
        sync_col.insert_many([{"_id": f"a{i}", "val": i} for i in range(100)])
        path = sync_dir / "test.json"
        assert path.exists()
        # Verify valid JSON
        data = json.loads(path.read_text())
        assert len(data) == 100

    def test_no_tmp_files_left(self, sync_col, sync_dir):
        sync_col.insert_many([{"_id": f"t{i}"} for i in range(50)])
        files = list(sync_dir.iterdir())
        assert all(not f.suffix == ".tmp" for f in files)


# ── Concurrency ──────────────────────────────────────────────────────────────


class TestConcurrency:
    def test_concurrent_inserts(self, db_path, sync_dir):
        import threading

        db = MogDB(str(db_path))
        col = db.collection("conc")
        sc = SyncableCollection(col, sync_dir / "conc.json")

        def insert_batch(start):
            for i in range(start, start + 20):
                sc.insert_one({"_id": f"c{i}", "thread": start})

        threads = [threading.Thread(target=insert_batch, args=(i * 20,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert sc.count() == 100
        sc.close()


# ── Indexing ─────────────────────────────────────────────────────────────────


class TestIndexing:
    def test_create_index(self, sync_col):
        sync_col.create_index("name")
        sync_col.insert_one({"_id": "i1", "name": "test"})
        assert sync_col.find_one({"name": "test"}) is not None

    def test_drop_index(self, sync_col):
        sync_col.create_index("name")
        sync_col.drop_index("name")
        sync_col.insert_one({"_id": "i1", "name": "test"})
        assert sync_col.find_one({"name": "test"}) is not None


# ── Properties ───────────────────────────────────────────────────────────────


class TestProperties:
    def test_json_path(self, sync_col, sync_dir):
        assert sync_col.json_path == sync_dir / "test.json"

    def test_underlying(self, sync_col, collection):
        assert sync_col.underlying is collection


# ── Lazy async semantics (true background sync) ─────────────────────────────


class TestLazyAsync:
    def test_lazy_does_not_sync_inline(self, collection, sync_dir):
        sc = SyncableCollection(
            collection,
            sync_dir / "lazy_noinline.json",
            sync_mode="lazy",
            lazy_sync_interval=60,
        )
        try:
            sc.insert_one({"_id": "n1"})
            # Writer must return without touching disk
            assert not (sync_dir / "lazy_noinline.json").exists()
        finally:
            sc.close()
        # close() flushes pending writes
        data = json.loads((sync_dir / "lazy_noinline.json").read_text())
        assert len(data) == 1

    def test_lazy_background_sync_fires(self, collection, sync_dir):
        sc = SyncableCollection(
            collection,
            sync_dir / "lazy_bg.json",
            sync_mode="lazy",
            lazy_sync_interval=0.05,
        )
        try:
            sc.insert_one({"_id": "b1"})
            deadline = time.monotonic() + 5.0
            data = None
            while time.monotonic() < deadline:
                if (sync_dir / "lazy_bg.json").exists():
                    data = json.loads((sync_dir / "lazy_bg.json").read_text())
                    if len(data) == 1:
                        break
                time.sleep(0.02)
            assert data is not None and len(data) == 1
        finally:
            sc.close()

    def test_nested_batch_empty_inner_exception_preserves_outer(self, sync_col, sync_dir):
        sync_col.insert_one({"_id": "pre"})
        with sync_col.batch():
            sync_col.insert_one({"_id": "outer"})
            with pytest.raises(ValueError):
                with sync_col.batch():
                    raise ValueError("boom")
        data = json.loads((sync_dir / "test.json").read_text())
        assert {d["_id"] for d in data} == {"pre", "outer"}
