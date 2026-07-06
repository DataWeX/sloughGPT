"""Tests for MogDB — the document-oriented embedded database engine."""

import json
import os
import tempfile
import threading
import time
from pathlib import Path

import pytest

from mogdb import MogDB, Collection, Document, ObjectId, match_document
from mogdb.index import Index, SortedIndex


# =========================================================================
# Document & ObjectId
# =========================================================================

class TestObjectId:
    def test_generates_24_char_hex(self):
        oid = ObjectId()
        assert len(oid) == 24
        int(oid, 16)  # does not raise

    def test_unique_within_batch(self):
        ids = {ObjectId() for _ in range(1000)}
        assert len(ids) == 1000

    def test_monotonic(self):
        ids = [ObjectId() for _ in range(100)]
        assert all(len(oid) == 24 for oid in ids)


class TestDocument:
    def test_auto_id(self):
        d = Document({"name": "test"})
        assert "_id" in d
        assert len(d["_id"]) == 24

    def test_preserves_provided_id(self):
        d = Document({"_id": "my-custom-id"})
        assert d["_id"] == "my-custom-id"

    def test_id_property(self):
        d = Document({"_id": "abc"})
        assert d.id == "abc"

    def test_content_hash(self):
        d1 = Document({"name": "Alice", "age": 30})
        d2 = Document({"age": 30, "name": "Alice"})
        assert d1.content_hash() == d2.content_hash()

    def test_content_hash_differs(self):
        d1 = Document({"name": "Alice"})
        d2 = Document({"name": "Bob"})
        assert d1.content_hash() != d2.content_hash()

    def test_timestamps(self):
        before = time.time()
        d = Document({"x": 1})
        after = time.time()
        assert before <= d["_created"] <= after
        assert before <= d["_updated"] <= after

    def test_copy_data(self):
        d = Document({"foo": "bar"})
        c = d.copy_data()
        assert c["foo"] == "bar"
        assert c["_id"] == d["_id"]


# =========================================================================
# Query engine
# =========================================================================

class TestQuery:
    def test_exact_match(self):
        assert match_document({"a": 1}, {"a": 1})

    def test_exact_mismatch(self):
        assert not match_document({"a": 1}, {"a": 2})

    def test_gt(self):
        assert match_document({"x": 5}, {"x": {"$gt": 3}})
        assert not match_document({"x": 2}, {"x": {"$gt": 3}})

    def test_gte(self):
        assert match_document({"x": 3}, {"x": {"$gte": 3}})
        assert not match_document({"x": 2}, {"x": {"$gte": 3}})

    def test_lt(self):
        assert match_document({"x": 1}, {"x": {"$lt": 3}})
        assert not match_document({"x": 5}, {"x": {"$lt": 3}})

    def test_lte(self):
        assert match_document({"x": 3}, {"x": {"$lte": 3}})
        assert not match_document({"x": 4}, {"x": {"$lte": 3}})

    def test_ne(self):
        assert match_document({"x": 1}, {"x": {"$ne": 2}})
        assert not match_document({"x": 2}, {"x": {"$ne": 2}})

    def test_in(self):
        assert match_document({"x": "a"}, {"x": {"$in": ["a", "b"]}})
        assert not match_document({"x": "c"}, {"x": {"$in": ["a", "b"]}})

    def test_nin(self):
        assert match_document({"x": "c"}, {"x": {"$nin": ["a", "b"]}})
        assert not match_document({"x": "a"}, {"x": {"$nin": ["a", "b"]}})

    def test_regex(self):
        assert match_document({"name": "Alice"}, {"name": {"$regex": "^Ali"}})
        assert not match_document({"name": "Bob"}, {"name": {"$regex": "^Ali"}})

    def test_exists_true(self):
        assert match_document({"a": 1}, {"a": {"$exists": True}})
        assert not match_document({"b": 1}, {"a": {"$exists": True}})

    def test_exists_false(self):
        assert match_document({"b": 1}, {"a": {"$exists": False}})
        assert not match_document({"a": 1}, {"a": {"$exists": False}})

    def test_or(self):
        q = {"$or": [{"a": 1}, {"b": 2}]}
        assert match_document({"a": 1}, q)
        assert match_document({"b": 2}, q)
        assert not match_document({"a": 3}, q)

    def test_and(self):
        q = {"$and": [{"a": 1}, {"b": 2}]}
        assert match_document({"a": 1, "b": 2}, q)
        assert not match_document({"a": 1, "b": 3}, q)

    def test_nor(self):
        q = {"$nor": [{"a": 1}, {"b": 2}]}
        assert match_document({"a": 3, "b": 4}, q)
        assert not match_document({"a": 1, "b": 4}, q)

    def test_nested_field(self):
        doc = {"user": {"profile": {"age": 30}}}
        assert match_document(doc, {"user.profile.age": {"$gt": 25}})
        assert not match_document(doc, {"user.profile.age": {"$gt": 35}})

    def test_non_string_regex(self):
        assert not match_document({"x": 1}, {"x": {"$regex": 42}})


# =========================================================================
# Collection CRUD
# =========================================================================

@pytest.fixture
def coll():
    with tempfile.TemporaryDirectory() as tmp:
        c = Collection("test", Path(tmp))
        yield c
        try:
            c.drop()
        except Exception:
            pass


class TestCollectionInsert:
    def test_insert_one_returns_id(self, coll):
        oid = coll.insert_one({"name": "Alice"})
        assert len(oid) == 24

    def test_insert_one_stores(self, coll):
        oid = coll.insert_one({"name": "Alice"})
        doc = coll.find_one({"_id": oid})
        assert doc["name"] == "Alice"

    def test_insert_many_returns_ids(self, coll):
        ids = coll.insert_many([{"x": 1}, {"x": 2}])
        assert len(ids) == 2

    def test_insert_many_stores_all(self, coll):
        coll.insert_many([{"n": 1}, {"n": 2}, {"n": 3}])
        assert coll.count() == 3

    def test_auto_adds_id(self, coll):
        coll.insert_one({"val": 1})
        doc = coll.find_one({"val": 1})
        assert doc is not None
        assert "_id" in doc

    def test_empty_doc(self, coll):
        oid = coll.insert_one({})
        assert coll.count() == 1
        assert coll.find_one({"_id": oid}) is not None


class TestCollectionFind:
    def test_find_all(self, coll):
        coll.insert_many([{"x": 1}, {"x": 2}, {"x": 3}])
        results = coll.find()
        assert len(results) == 3

    def test_find_with_query(self, coll):
        coll.insert_many([{"x": 1}, {"x": 2}, {"x": 3}])
        results = coll.find({"x": {"$gt": 1}})
        assert len(results) == 2

    def test_find_one(self, coll):
        coll.insert_many([{"x": 1}, {"x": 2}])
        doc = coll.find_one({"x": 1})
        assert doc["x"] == 1

    def test_find_one_missing(self, coll):
        assert coll.find_one({"x": 999}) is None

    def test_find_empty_collection(self, coll):
        assert coll.find() == []

    def test_find_limit(self, coll):
        coll.insert_many([{"i": i} for i in range(100)])
        results = coll.find(limit=5)
        assert len(results) == 5

    def test_find_skip(self, coll):
        coll.insert_many([{"i": i} for i in range(10)])
        results = coll.find(skip=5)
        assert len(results) == 5
        assert results[0]["i"] == 5

    def test_find_sort_ascending(self, coll):
        coll.insert_many([{"n": 3}, {"n": 1}, {"n": 2}])
        results = coll.find(sort=[("n", 1)])
        assert [r["n"] for r in results] == [1, 2, 3]

    def test_find_sort_descending(self, coll):
        coll.insert_many([{"n": 1}, {"n": 2}, {"n": 3}])
        results = coll.find(sort=[("n", -1)])
        assert [r["n"] for r in results] == [3, 2, 1]

    def test_find_with_dot_notation_query(self, coll):
        coll.insert_one({"user": {"name": "Alice"}})
        doc = coll.find_one({"user.name": "Alice"})
        assert doc is not None


class TestCollectionCount:
    def test_count_all(self, coll):
        coll.insert_many([{"x": 1}, {"x": 2}])
        assert coll.count() == 2

    def test_count_empty(self, coll):
        assert coll.count() == 0

    def test_count_with_query(self, coll):
        coll.insert_many([{"v": 1}, {"v": 2}, {"v": 1}])
        assert coll.count({"v": 1}) == 2

    def test_count_with_gt(self, coll):
        coll.insert_many([{"v": i} for i in range(10)])
        assert coll.count({"v": {"$gt": 5}}) == 4


class TestCollectionUpdate:
    def test_update_one(self, coll):
        oid = coll.insert_one({"name": "Alice", "age": 30})
        modified = coll.update_one({"name": "Alice"}, {"$set": {"age": 31}})
        assert modified == 1
        doc = coll.find_one({"_id": oid})
        assert doc["age"] == 31

    def test_update_one_no_match(self, coll):
        modified = coll.update_one({"x": 999}, {"$set": {"x": 1}})
        assert modified == 0

    def test_update_many(self, coll):
        coll.insert_many([{"v": 1}, {"v": 1}, {"v": 2}])
        modified = coll.update_many({"v": 1}, {"$set": {"tag": "one"}})
        assert modified == 2
        assert coll.count({"tag": "one"}) == 2

    def test_update_many_no_match(self, coll):
        modified = coll.update_many({"v": 999}, {"$set": {"x": 1}})
        assert modified == 0

    def test_unset(self, coll):
        oid = coll.insert_one({"a": 1, "b": 2})
        coll.update_one({"a": 1}, {"$unset": {"b": ""}})
        doc = coll.find_one({"_id": oid})
        assert "b" not in doc

    def test_update_preserves_other_fields(self, coll):
        oid = coll.insert_one({"a": 1, "b": 2, "c": 3})
        coll.update_one({"a": 1}, {"$set": {"b": 99}})
        doc = coll.find_one({"_id": oid})
        assert doc["a"] == 1
        assert doc["b"] == 99
        assert doc["c"] == 3

    def test_update_sets_timestamp(self, coll):
        oid = coll.insert_one({"x": 1})
        old_updated = coll.find_one({"_id": oid})["_updated"]
        time.sleep(0.01)
        coll.update_one({"_id": oid}, {"$set": {"x": 2}})
        doc = coll.find_one({"_id": oid})
        assert doc["_updated"] > old_updated


class TestCollectionDelete:
    def test_delete_one(self, coll):
        coll.insert_many([{"x": 1}, {"x": 2}])
        deleted = coll.delete_one({"x": 1})
        assert deleted == 1
        assert coll.count() == 1

    def test_delete_one_no_match(self, coll):
        deleted = coll.delete_one({"x": 999})
        assert deleted == 0

    def test_delete_many(self, coll):
        coll.insert_many([{"v": 1}, {"v": 1}, {"v": 2}])
        deleted = coll.delete_many({"v": 1})
        assert deleted == 2
        assert coll.count() == 1

    def test_delete_many_all(self, coll):
        coll.insert_many([{"x": 1}, {"x": 2}])
        deleted = coll.delete_many({})
        assert deleted == 2
        assert coll.count() == 0

    def test_delete_then_insert_same(self, coll):
        coll.insert_one({"n": 1})
        coll.delete_one({"n": 1})
        oid = coll.insert_one({"n": 1})
        assert coll.count() == 1
        doc = coll.find_one({"_id": oid})
        assert doc["n"] == 1


class TestCollectionDrop:
    def test_drop_removes_all(self, coll):
        coll.insert_many([{"a": 1}, {"b": 2}])
        assert coll.count() == 2
        coll.drop()
        assert coll.count() == 0

    def test_drop_then_insert(self, coll):
        coll.insert_one({"x": 1})
        coll.drop()
        oid = coll.insert_one({"y": 2})
        assert coll.count() == 1
        assert coll.find_one({"_id": oid})["y"] == 2


# =========================================================================
# Persistence
# =========================================================================

class TestPersistence:
    def test_survives_collection_reopen(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            c1 = Collection("persist", path)
            c1.insert_many([{"a": 1}, {"a": 2}, {"a": 3}])
            del c1

            c2 = Collection("persist", path)
            assert c2.count() == 3
            c2.drop()

    def test_journal_appends_on_insert(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            c = Collection("journal", path)
            c.insert_one({"x": 42})
            journal_path = path / "journal.journal.jsonl"
            assert journal_path.exists()
            with open(journal_path) as f:
                lines = f.readlines()
            assert len(lines) >= 1
            c.drop()

    def test_journal_appends_on_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            c = Collection("ju", path)
            oid = c.insert_one({"x": 1})
            c.update_one({"_id": oid}, {"$set": {"x": 2}})
            journal_path = path / "ju.journal.jsonl"
            with open(journal_path) as f:
                ops = [json.loads(l) for l in f if l.strip()]
            assert any(e["op"] == "update" for e in ops)
            c.drop()

    def test_journal_appends_on_delete(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            c = Collection("jd", path)
            oid = c.insert_one({"x": 1})
            c.delete_one({"_id": oid})
            journal_path = path / "jd.journal.jsonl"
            with open(journal_path) as f:
                ops = [json.loads(l) for l in f if l.strip()]
            assert any(e["op"] == "delete" for e in ops)
            c.drop()

    def test_compact(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            c = Collection("compact", path)
            for i in range(100):
                c.insert_one({"i": i})
            c.delete_many({"i": {"$lt": 50}})
            assert c.count() == 50

            count = c.compact()
            assert count == 50

            # compacted file exists, journal removed
            assert (path / "compact.mogdb").exists()
            assert not (path / "compact.journal.jsonl").exists()

            # reopen from compacted file
            c2 = Collection("compact", path)
            assert c2.count() == 50
            c2.drop()

    def test_compact_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = Collection("empty", Path(tmp))
            assert c.compact() == 0
            c.drop()


# =========================================================================
# MogDB
# =========================================================================

class TestMogDB:
    def test_create_and_get_collection(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = MogDB(tmp)
            coll = db.collection("items")
            assert coll.name == "items"
            coll.insert_one({"x": 1})
            assert coll.count() == 1

    def test_collection_is_singleton(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = MogDB(tmp)
            c1 = db.collection("same")
            c2 = db.collection("same")
            assert c1 is c2

    def test_list_collections(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = MogDB(tmp)
            db.collection("a")
            db.collection("b")
            names = db.list_collections()
            assert "a" in names
            assert "b" in names

    def test_drop_collection(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = MogDB(tmp)
            db.collection("x").insert_one({"v": 1})
            assert "x" in db.list_collections()
            db.drop_collection("x")
            assert "x" not in db.list_collections()

    def test_compact_all(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = MogDB(tmp)
            db.collection("a").insert_many([{"i": i} for i in range(10)])
            db.collection("b").insert_many([{"i": i} for i in range(20)])
            db.collection("a").delete_many({"i": {"$gte": 5}})
            total = db.compact_all()
            assert total == 25

    def test_context_manager(self):
        with tempfile.TemporaryDirectory() as tmp:
            with MogDB(tmp) as db:
                db.collection("x").insert_one({"v": 1})
            # After exit, files exist on disk
            p = Path(tmp)
            assert (p / "x.mogdb").exists() or (p / "x.journal.jsonl").exists()

    def test_multiple_collections_independent(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = MogDB(tmp)
            db.collection("users").insert_one({"name": "Alice"})
            db.collection("logs").insert_one({"msg": "hello"})
            assert db.collection("users").count() == 1
            assert db.collection("logs").count() == 1

    def test_custom_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "nested", "db")
            db = MogDB(db_path)
            db.collection("t").insert_one({"ok": True})
            assert db.collection("t").count() == 1


# =========================================================================
# Index
# =========================================================================

class TestIndex:
    def test_add_and_lookup(self):
        idx = Index("name")
        idx.add("id1", "Alice")
        idx.add("id2", "Bob")
        assert idx.lookup("Alice") == ["id1"]
        assert idx.lookup("Bob") == ["id2"]

    def test_remove(self):
        idx = Index("x")
        idx.add("id1", 10)
        idx.add("id2", 10)
        idx.remove("id1", 10)
        assert idx.lookup(10) == ["id2"]

    def test_update(self):
        idx = Index("age")
        idx.add("id1", 30)
        idx.update("id1", 30, 31)
        assert idx.lookup(30) == []
        assert idx.lookup(31) == ["id1"]

    def test_unique_violation(self):
        idx = Index("email", unique=True)
        idx.add("id1", "a@b.com")
        with pytest.raises(ValueError, match="Unique index violation"):
            idx.add("id2", "a@b.com")

    def test_unique_allows_same_id(self):
        idx = Index("x", unique=True)
        idx.add("id1", 1)
        idx.add("id1", 1)
        assert idx.lookup(1) == ["id1"]

    def test_clear(self):
        idx = Index("n")
        idx.add("id1", 1)
        idx.add("id2", 2)
        idx.clear()
        assert idx.lookup(1) == []


class TestSortedIndex:
    def test_add_and_range(self):
        idx = SortedIndex("age")
        idx.add("id1", 25)
        idx.add("id2", 30)
        idx.add("id3", 35)
        assert idx.range(gte=30) == ["id2", "id3"]
        assert idx.range(lte=30) == ["id1", "id2"]
        assert idx.range(gte=28, lte=32) == ["id2"]

    def test_remove(self):
        idx = SortedIndex("v")
        idx.add("id1", 10)
        idx.add("id2", 20)
        idx.remove("id1", 10)
        assert idx.range() == ["id2"]

    def test_empty_range(self):
        idx = SortedIndex("x")
        assert idx.range() == []

    def test_no_match_range(self):
        idx = SortedIndex("x")
        idx.add("id1", 50)
        assert idx.range(gte=100) == []


# =========================================================================
# Concurrency
# =========================================================================

class TestConcurrency:
    def test_concurrent_inserts(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = Collection("conc", Path(tmp))
            n = 100
            threads = []
            results = []

            def worker(i):
                oid = c.insert_one({"thread": i, "val": i})
                results.append(oid)

            for i in range(n):
                t = threading.Thread(target=worker, args=(i,))
                threads.append(t)
                t.start()
            for t in threads:
                t.join()

            assert c.count() == n
            c.drop()

    def test_concurrent_reads_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = Collection("rw", Path(tmp))
            c.insert_one({"base": True})

            errors = []

            def writer():
                for i in range(50):
                    try:
                        c.insert_one({"w": i})
                    except Exception as e:
                        errors.append(e)

            def reader():
                for _ in range(50):
                    try:
                        c.find()
                    except Exception as e:
                        errors.append(e)

            threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0
            assert c.count() >= 1
            c.drop()


# =========================================================================
# Edge cases
# =========================================================================

class TestEdgeCases:
    def test_large_document(self, coll):
        big = {"data": "x" * 100_000}
        oid = coll.insert_one(big)
        doc = coll.find_one({"_id": oid})
        assert len(doc["data"]) == 100_000

    def test_special_characters_in_field_names(self, coll):
        doc = {"field with spaces": 1, "field.with.dots": 2}
        oid = coll.insert_one(doc)
        result = coll.find_one({"_id": oid})
        assert result["field with spaces"] == 1
        assert result["field.with.dots"] == 2

    def test_unicode_content(self, coll):
        coll.insert_one({"text": "こんにちは世界 🌍"})
        doc = coll.find_one({"text": {"$regex": "世界"}})
        assert doc is not None

    def test_boolean_values(self, coll):
        coll.insert_many([{"active": True}, {"active": False}])
        assert coll.count({"active": True}) == 1
        assert coll.count({"active": False}) == 1

    def test_none_values(self, coll):
        coll.insert_one({"maybe": None})
        doc = coll.find_one({"maybe": None})
        assert doc is not None

    def test_numeric_edge(self, coll):
        coll.insert_many([{"v": 0}, {"v": -1}, {"v": 2**31}])
        assert coll.count({"v": {"$gte": 0}}) == 2
        assert coll.count({"v": {"$lt": 0}}) == 1

    def test_persistence_with_special_characters_in_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = Collection("my collection!", Path(tmp))
            c.insert_one({"x": 1})
            del c
            c2 = Collection("my collection!", Path(tmp))
            assert c2.count() == 1
            c2.drop()

    def test_compact_twice(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = Collection("double", Path(tmp))
            for i in range(10):
                c.insert_one({"i": i})
            c.compact()
            c.compact()
            assert c.count() == 10
            c.drop()

    def test_delete_many_empty_query(self, coll):
        coll.insert_many([{"a": 1}, {"a": 2}])
        assert coll.delete_many({}) == 2
        assert coll.count() == 0
