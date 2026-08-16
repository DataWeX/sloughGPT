"""
Tests for Data Repository (repository.py).
"""

import json
import time
import pytest
from dataclasses import dataclass
from domains.infrastructure.repository import (
    FileRepository, MemoryRepository, CachedRepository,
    Migration, MigrationRunner, JsonSerializer,
)


@dataclass
class _RepoItem:
    id: str
    name: str
    value: int = 0


class TestJsonSerializer:
    def test_dataclass_serialize(self):
        item = _RepoItem(id="1", name="foo", value=42)
        s = JsonSerializer[_RepoItem](_RepoItem)
        d = s.serialize(item)
        assert d["id"] == "1"
        assert d["name"] == "foo"
        assert d["value"] == 42

    def test_dataclass_deserialize(self):
        s = JsonSerializer[_RepoItem](_RepoItem)
        item = s.deserialize({"id": "2", "name": "bar", "value": 99})
        assert item.id == "2"
        assert item.name == "bar"
        assert item.value == 99

    def test_dict_serialize(self):
        s = JsonSerializer[dict](dict)
        assert s.serialize({"a": 1}) == {"a": 1}

    def test_dict_deserialize(self):
        s = JsonSerializer[dict](dict)
        assert s.deserialize({"a": 1}) == {"a": 1}

    def test_model_dump_serialize(self):
        class _M:
            def model_dump(self):
                return {"a": 1}
        s = JsonSerializer[_M](_M)
        assert s.serialize(_M()) == {"a": 1}

    def test_namedtuple_serialize(self):
        from collections import namedtuple
        Point = namedtuple("Point", "x y")
        s = JsonSerializer[Point](Point)
        assert s.serialize(Point(1, 2)) == {"x": 1, "y": 2}

    def test_plain_object_serialize(self):
        s = JsonSerializer[object](object)
        d = s.serialize(object())
        assert set(d) == {"_value"}

    def test_model_validate_deserialize(self):
        class _M:
            def __init__(self, data):
                self.data = data

            @classmethod
            def model_validate(cls, data):
                return cls(data)
        s = JsonSerializer[_M](_M)
        m = s.deserialize({"a": 1})
        assert m.data == {"a": 1}


class TestMigration:
    def test_apply(self):
        m = Migration(1, "add version field", lambda d: {**d, "version": "1.0"})
        result = m.apply({"name": "test"})
        assert result["name"] == "test"
        assert result["version"] == "1.0"


class TestMigrationRunner:
    def test_no_migrations(self):
        r = MigrationRunner()
        assert r.latest_version == 0
        assert r.run({"key": "val"}) == {"key": "val"}

    def test_single_migration(self):
        r = MigrationRunner()
        r.add(Migration(1, "add version", lambda d: {**d, "v": 1}))
        data = r.run({"key": "val"})
        assert data["v"] == 1
        assert data["_schema_version"] == 1

    def test_sequential_migrations(self):
        r = MigrationRunner()
        r.add(Migration(1, "first", lambda d: {**d, "s": 1}))
        r.add(Migration(2, "second", lambda d: {**d, "s": 2}))
        data = r.run({"key": "val"})
        assert data["s"] == 2
        assert data["_schema_version"] == 2

    def test_skips_applied_migrations(self):
        r = MigrationRunner()
        r.add(Migration(1, "first", lambda d: {**d, "changed": True}))
        r.add(Migration(2, "second", lambda d: {**d, "changed": False}))
        data = r.run({"_schema_version": 1})
        assert data["changed"] is False  # only v2 ran
        assert data["_schema_version"] == 2

    def test_latest_version(self):
        r = MigrationRunner()
        r.add(Migration(1, "a", lambda d: d))
        r.add(Migration(3, "c", lambda d: d))
        assert r.latest_version == 3

    def test_failed_migration_raises(self):
        r = MigrationRunner()

        def boom(d):
            raise RuntimeError("migration failed")
        r.add(Migration(1, "boom", boom))
        with pytest.raises(RuntimeError, match="migration failed"):
            r.run({"key": "val"})


class TestFileRepository:
    def test_save_and_get(self, tmp_path):
        repo = FileRepository[_RepoItem](tmp_path, serializer=_RepoItem)
        repo.save("item1", _RepoItem(id="1", name="test", value=42))
        loaded = repo.get("item1")
        assert loaded is not None
        assert loaded.id == "1"
        assert loaded.name == "test"
        assert loaded.value == 42

    def test_get_nonexistent(self, tmp_path):
        repo = FileRepository[_RepoItem](tmp_path, serializer=_RepoItem)
        assert repo.get("nonexistent") is None

    def test_list(self, tmp_path):
        repo = FileRepository[_RepoItem](tmp_path, serializer=_RepoItem)
        repo.save("a", _RepoItem(id="a", name="A"))
        repo.save("b", _RepoItem(id="b", name="B"))
        items = repo.list()
        assert len(items) == 2

    def test_delete(self, tmp_path):
        repo = FileRepository[_RepoItem](tmp_path, serializer=_RepoItem)
        repo.save("x", _RepoItem(id="x", name="X"))
        assert repo.delete("x") is True
        assert repo.get("x") is None

    def test_delete_nonexistent(self, tmp_path):
        repo = FileRepository[_RepoItem](tmp_path, serializer=_RepoItem)
        assert repo.delete("nonexistent") is False

    def test_exists(self, tmp_path):
        repo = FileRepository[_RepoItem](tmp_path, serializer=_RepoItem)
        repo.save("e", _RepoItem(id="e", name="E"))
        assert repo.exists("e") is True
        assert repo.exists("no") is False

    def test_count(self, tmp_path):
        repo = FileRepository[_RepoItem](tmp_path, serializer=_RepoItem)
        repo.save("a", _RepoItem(id="a", name="a"))
        repo.save("b", _RepoItem(id="b", name="b"))
        assert repo.count() == 2

    def test_keys(self, tmp_path):
        repo = FileRepository[_RepoItem](tmp_path, serializer=_RepoItem)
        repo.save("k1", _RepoItem(id="k1", name="k1"))
        repo.save("k2", _RepoItem(id="k2", name="k2"))
        assert sorted(repo.keys()) == ["k1", "k2"]

    def test_search(self, tmp_path):
        repo = FileRepository[_RepoItem](tmp_path, serializer=_RepoItem)
        repo.save("a", _RepoItem(id="a", name="apple"))
        repo.save("b", _RepoItem(id="b", name="banana"))
        repo.save("c", _RepoItem(id="c", name="cherry"))
        results = repo.search("apple")
        assert len(results) == 1
        assert results[0].name == "apple"

    def test_search_by_field(self, tmp_path):
        repo = FileRepository[_RepoItem](tmp_path, serializer=_RepoItem)
        repo.save("a", _RepoItem(id="a", name="alpha", value=10))
        repo.save("b", _RepoItem(id="b", name="beta", value=20))
        results = repo.search("20", fields=["value"])
        assert len(results) == 1
        assert results[0].name == "beta"

    def test_cache_hit(self, tmp_path):
        repo = FileRepository[_RepoItem](tmp_path, serializer=_RepoItem)
        repo.enable_cache(ttl_seconds=60)
        repo.save("c", _RepoItem(id="c", name="cached"))
        # First call loads from disk
        assert repo.get("c") is not None
        # Delete file to prove cache works
        (tmp_path / "c.json").unlink()
        assert repo.get("c") is not None  # cache hit

    def test_cache_expiry(self, tmp_path):
        repo = FileRepository[_RepoItem](tmp_path, serializer=_RepoItem)
        repo.enable_cache(ttl_seconds=0.1)
        repo.save("e", _RepoItem(id="e", name="expiring"))
        repo.get("e")  # populate cache
        (tmp_path / "e.json").unlink()
        time.sleep(0.15)
        assert repo.get("e") is None  # cache expired

    def test_cache_invalidate_key(self, tmp_path):
        repo = FileRepository[_RepoItem](tmp_path, serializer=_RepoItem)
        repo.enable_cache(ttl_seconds=60)
        repo.save("i", _RepoItem(id="i", name="invalid"))
        repo.get("i")  # cache it
        repo.invalidate("i")
        (tmp_path / "i.json").unlink()
        assert repo.get("i") is None

    def test_cache_invalidate_all(self, tmp_path):
        repo = FileRepository[_RepoItem](tmp_path, serializer=_RepoItem)
        repo.enable_cache(ttl_seconds=60)
        repo.save("x", _RepoItem(id="x", name="x"))
        repo.save("y", _RepoItem(id="y", name="y"))
        repo.get("x")
        repo.get("y")
        repo.invalidate()
        (tmp_path / "x.json").unlink()
        (tmp_path / "y.json").unlink()
        assert repo.get("x") is None

    def test_migration_on_read(self, tmp_path):
        mr = MigrationRunner()
        mr.add(Migration(1, "increment value", lambda d: {**d, "value": d.get("value", 0) + 1}))
        repo = FileRepository[_RepoItem](tmp_path, serializer=_RepoItem, migration_runner=mr)
        # Write an older-format file (no value field)
        raw = {"id": "m", "name": "migrated"}
        with open(tmp_path / "m.json", "w") as f:
            json.dump(raw, f)
        loaded = repo.get("m")
        assert loaded is not None
        assert loaded.name == "migrated"
        assert loaded.value == 1

    def test_custom_serializer_instance(self, tmp_path):
        class _Ser:
            def serialize(self, obj):
                return {"value": obj}

            def deserialize(self, data):
                return data["value"]
        repo = FileRepository(tmp_path, serializer=_Ser())
        assert repo.save("k", 42) is True
        assert repo.get("k") == 42

    def test_default_dict_serializer(self, tmp_path):
        repo = FileRepository[dict](tmp_path)
        assert repo.save("d", {"a": 1}) is True
        assert repo.get("d") == {"a": 1}

    def test_disable_cache(self, tmp_path):
        repo = FileRepository[_RepoItem](tmp_path, serializer=_RepoItem)
        repo.enable_cache(ttl_seconds=60)
        repo.save("c", _RepoItem(id="c", name="cached"))
        repo.get("c")
        repo.disable_cache()
        (tmp_path / "c.json").unlink()
        assert repo.get("c") is None

    def test_save_atomic_no_tmp_residue(self, tmp_path):
        repo = FileRepository[_RepoItem](tmp_path, serializer=_RepoItem)
        repo.save("a", _RepoItem(id="a", name="atomic"))
        assert not (tmp_path / "a.json.tmp").exists()
        loaded = repo.get("a")
        assert loaded.name == "atomic"

    def test_save_overwrite_existing(self, tmp_path):
        repo = FileRepository[_RepoItem](tmp_path, serializer=_RepoItem)
        repo.save("x", _RepoItem(id="x", name="old"))
        repo.save("x", _RepoItem(id="x", name="new"))
        loaded = repo.get("x")
        assert loaded.name == "new"

    def test_get_empty_file_returns_none(self, tmp_path):
        repo = FileRepository[_RepoItem](tmp_path, serializer=_RepoItem)
        (tmp_path / "empty.json").write_text("")
        assert repo.get("empty") is None

    def test_get_whitespace_only_file_returns_none(self, tmp_path):
        repo = FileRepository[_RepoItem](tmp_path, serializer=_RepoItem)
        (tmp_path / "ws.json").write_text("   \n\t  ")
        assert repo.get("ws") is None

    def test_get_corrupt_file_returns_none(self, tmp_path):
        repo = FileRepository[_RepoItem](tmp_path, serializer=_RepoItem)
        (tmp_path / "bad.json").write_text("{not json")
        assert repo.get("bad") is None

    def test_save_error_returns_false(self, tmp_path):
        class _BoomSer:
            def serialize(self, obj):
                raise OSError("boom")

            def deserialize(self, data):
                return data
        repo = FileRepository(tmp_path, serializer=_BoomSer())
        assert repo.save("x", object()) is False

    def test_delete_error_returns_false(self, tmp_path):
        repo = FileRepository[_RepoItem](tmp_path, serializer=_RepoItem)
        (tmp_path / "x.json").mkdir()
        assert repo.delete("x") is False


class TestMemoryRepository:
    def test_crud(self):
        repo = MemoryRepository[_RepoItem]()
        repo.save("a", _RepoItem(id="a", name="A"))
        assert repo.get("a") is not None
        assert len(repo.list()) == 1
        assert repo.delete("a") is True
        assert repo.get("a") is None

    def test_search(self):
        repo = MemoryRepository[_RepoItem]()
        repo.save("a", _RepoItem(id="a", name="apple"))
        repo.save("b", _RepoItem(id="b", name="banana"))
        assert len(repo.search("apple")) == 1
        assert len(repo.search("z")) == 0

    def test_clear(self):
        repo = MemoryRepository[_RepoItem]()
        repo.save("a", _RepoItem(id="a", name="a"))
        repo.save("b", _RepoItem(id="b", name="b"))
        repo.clear()
        assert len(repo.list()) == 0


class TestCachedRepository:
    def test_caches_get(self):
        inner = MemoryRepository[_RepoItem]()
        cached = CachedRepository(inner, ttl=60)
        inner.save("x", _RepoItem(id="x", name="original"))
        assert cached.get("x").name == "original"
        inner.save("x", _RepoItem(id="x", name="updated"))
        # Cache returns stale
        assert cached.get("x").name == "original"

    def test_save_invalidates_cache(self):
        inner = MemoryRepository[_RepoItem]()
        cached = CachedRepository(inner, ttl=60)
        inner.save("x", _RepoItem(id="x", name="x"))
        cached.get("x")
        cached.save("x", _RepoItem(id="x", name="new"))
        assert cached.get("x").name == "new"

    def test_delete_invalidates_cache(self):
        inner = MemoryRepository[_RepoItem]()
        cached = CachedRepository(inner, ttl=60)
        inner.save("x", _RepoItem(id="x", name="x"))
        cached.get("x")
        cached.delete("x")
        assert cached.get("x") is None

    def test_list_caching(self):
        inner = MemoryRepository[_RepoItem]()
        cached = CachedRepository(inner, ttl=60)
        inner.save("a", _RepoItem(id="a", name="a"))
        inner.save("b", _RepoItem(id="b", name="b"))
        assert len(cached.list()) == 2
        inner.save("c", _RepoItem(id="c", name="c"))
        # list cache is stale
        assert len(cached.list()) == 2

    def test_invalidate(self):
        inner = MemoryRepository[_RepoItem]()
        cached = CachedRepository(inner, ttl=60)
        inner.save("x", _RepoItem(id="x", name="x"))
        cached.get("x")
        cached.invalidate()
        inner.delete("x")
        assert cached.get("x") is None

    def test_search_passthrough(self):
        inner = MemoryRepository[_RepoItem]()
        cached = CachedRepository(inner)
        inner.save("a", _RepoItem(id="a", name="apple"))
        assert len(cached.search("apple")) == 1
