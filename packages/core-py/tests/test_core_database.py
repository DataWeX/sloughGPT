"""Tests for domains.core.database — in-memory document store."""

import threading
import pytest

from domains.core.database import _MemoryDB, get_db


# ---------------------------------------------------------------------------
# _MemoryDB unit tests
# ---------------------------------------------------------------------------

class TestMemoryDBInsert:
    def test_insert_single(self):
        db = _MemoryDB()
        db.insert("users", {"name": "Alice", "age": 30})
        assert db.count("users") == 1

    def test_insert_multiple(self):
        db = _MemoryDB()
        db.insert("c", {"v": 1})
        db.insert("c", {"v": 2})
        db.insert("c", {"v": 3})
        assert db.count("c") == 3

    def test_insert_stores_copy(self):
        db = _MemoryDB()
        data = {"k": "v"}
        db.insert("c", data)
        data["k"] = "changed"
        docs = db.find("c", {})
        assert docs[0]["k"] == "v"

    def test_insert_creates_collection(self):
        db = _MemoryDB()
        db.insert("brand_new", {"x": 1})
        assert db.count("brand_new") == 1


class TestMemoryDBFind:
    def setup_method(self):
        self.db = _MemoryDB()
        self.db.insert("users", {"name": "Alice", "role": "admin"})
        self.db.insert("users", {"name": "Bob", "role": "user"})
        self.db.insert("users", {"name": "Carol", "role": "admin"})

    def test_find_all(self):
        assert len(self.db.find("users", {})) == 3

    def test_find_by_single_field(self):
        results = self.db.find("users", {"role": "admin"})
        assert len(results) == 2
        names = {d["name"] for d in results}
        assert names == {"Alice", "Carol"}

    def test_find_by_multiple_fields(self):
        results = self.db.find("users", {"name": "Bob", "role": "user"})
        assert len(results) == 1
        assert results[0]["name"] == "Bob"

    def test_find_no_match(self):
        assert self.db.find("users", {"role": "superadmin"}) == []

    def test_find_empty_collection(self):
        assert self.db.find("nonexistent", {}) == []

    def test_find_returns_list(self):
        assert isinstance(self.db.find("users", {}), list)

    def test_find_missing_key_in_doc(self):
        db = _MemoryDB()
        db.insert("c", {"a": 1})
        results = db.find("c", {"b": 1})
        assert results == []


class TestMemoryDBUpsert:
    def setup_method(self):
        self.db = _MemoryDB()

    def test_upsert_inserts_when_no_match(self):
        self.db.upsert("c", {"id": 1}, {"value": "x"})
        assert self.db.count("c") == 1
        docs = self.db.find("c", {"id": 1})
        assert docs[0]["value"] == "x"

    def test_upsert_updates_existing(self):
        self.db.insert("c", {"id": 1, "value": "old"})
        self.db.upsert("c", {"id": 1}, {"value": "new"})
        assert self.db.count("c") == 1
        docs = self.db.find("c", {"id": 1})
        assert docs[0]["value"] == "new"

    def test_upsert_preserves_unmentioned_fields(self):
        self.db.insert("c", {"id": 1, "extra": "keep"})
        self.db.upsert("c", {"id": 1}, {"value": "updated"})
        docs = self.db.find("c", {"id": 1})
        assert docs[0]["extra"] == "keep"
        assert docs[0]["value"] == "updated"

    def test_upsert_adds_query_fields_on_insert(self):
        self.db.upsert("c", {"id": 5}, {"label": "new"})
        docs = self.db.find("c", {})
        assert len(docs) == 1
        assert docs[0]["id"] == 5
        assert docs[0]["label"] == "new"

    def test_upsert_multiple_queries(self):
        self.db.insert("c", {"id": 1, "val": "a"})
        self.db.insert("c", {"id": 2, "val": "b"})
        self.db.upsert("c", {"id": 2}, {"val": "updated"})
        assert self.db.count("c") == 2
        assert self.db.find("c", {"id": 2})[0]["val"] == "updated"
        assert self.db.find("c", {"id": 1})[0]["val"] == "a"


class TestMemoryDBCount:
    def test_count_empty(self):
        assert _MemoryDB().count("x") == 0

    def test_count_after_inserts(self):
        db = _MemoryDB()
        for i in range(5):
            db.insert("c", {"i": i})
        assert db.count("c") == 5


class TestMemoryDBThreadSafety:
    def test_concurrent_inserts(self):
        db = _MemoryDB()
        threads = []
        for _ in range(10):
            t = threading.Thread(target=lambda: [db.insert("c", {"v": i}) for i in range(50)])
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        assert db.count("c") == 500

    def test_concurrent_reads_writes(self):
        db = _MemoryDB()
        for i in range(100):
            db.insert("c", {"v": i})
        errors = []

        def reader():
            try:
                for _ in range(50):
                    db.find("c", {"v": 42})
            except Exception as e:
                errors.append(e)

        def writer():
            try:
                for i in range(50):
                    db.insert("c", {"v": 1000 + i})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(5)]
        threads += [threading.Thread(target=writer) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []


# ---------------------------------------------------------------------------
# Singleton get_db()
# ---------------------------------------------------------------------------

class TestGetDb:
    def test_returns_same_instance(self):
        a = get_db()
        b = get_db()
        assert a is b

    def test_is_memory_db(self):
        assert isinstance(get_db(), _MemoryDB)
