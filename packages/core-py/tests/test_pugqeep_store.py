"""Tests for domains.infrastructure.pugqeep.store — MemoryStore, JSONStore, DirectoryStore."""

import tempfile
from pathlib import Path

import pytest
from domains.infrastructure.pugqeep.store import MemoryStore, JSONStore, DirectoryStore
from domains.infrastructure.pugqeep.point import Point


def _make_point(identity="p1", function_type="periodic", params=None):
    return Point(identity=identity, function_type=function_type, params=params or {"a": 1.0})


class TestMemoryStore:
    def test_save_and_load(self):
        s = MemoryStore()
        p = _make_point()
        s.save(p)
        assert s.load("p1") is p

    def test_load_missing(self):
        s = MemoryStore()
        assert s.load("nope") is None

    def test_remove(self):
        s = MemoryStore()
        s.save(_make_point())
        assert s.remove("p1") is True
        assert s.load("p1") is None

    def test_remove_missing(self):
        s = MemoryStore()
        assert s.remove("nope") is False

    def test_list_all(self):
        s = MemoryStore()
        s.save(_make_point("a"))
        s.save(_make_point("b"))
        assert len(s.list_all()) == 2

    def test_clear(self):
        s = MemoryStore()
        s.save(_make_point("a"))
        s.save(_make_point("b"))
        s.clear()
        assert s.count() == 0

    def test_count(self):
        s = MemoryStore()
        assert s.count() == 0
        s.save(_make_point("a"))
        assert s.count() == 1
        s.save(_make_point("b"))
        assert s.count() == 2


class TestJSONStore:
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            s = JSONStore(Path(tmpdir) / "test.json")
            s.save(_make_point())
            assert s.load("p1") is not None
            assert s.load("p1").identity == "p1"

    def test_load_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            s = JSONStore(Path(tmpdir) / "test.json")
            assert s.load("nope") is None

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            s1 = JSONStore(path)
            s1.save(_make_point("a"))
            s1.save(_make_point("b"))
            # Reload from disk
            s2 = JSONStore(path)
            assert s2.load("a") is not None
            assert s2.load("b") is not None
            assert s2.count() == 2

    def test_remove(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            s = JSONStore(Path(tmpdir) / "test.json")
            s.save(_make_point())
            assert s.remove("p1") is True
            assert s.load("p1") is None
            # Verify removed from disk
            s2 = JSONStore(Path(tmpdir) / "test.json")
            assert s2.load("p1") is None

    def test_remove_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            s = JSONStore(Path(tmpdir) / "test.json")
            assert s.remove("nope") is False

    def test_clear(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            s = JSONStore(Path(tmpdir) / "test.json")
            s.save(_make_point("a"))
            s.save(_make_point("b"))
            s.clear()
            assert s.count() == 0

    def test_list_all(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            s = JSONStore(Path(tmpdir) / "test.json")
            s.save(_make_point("a"))
            s.save(_make_point("b"))
            assert len(s.list_all()) == 2

    def test_creates_parent_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sub" / "test.json"
            s = JSONStore(path)
            s.save(_make_point())
            assert path.exists()


class TestDirectoryStore:
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            s = DirectoryStore(Path(tmpdir))
            s.save(_make_point("layer1"))
            assert s.load("layer1") is not None

    def test_load_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            s = DirectoryStore(Path(tmpdir))
            assert s.load("nope") is None

    def test_one_file_per_point(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            s = DirectoryStore(Path(tmpdir))
            s.save(_make_point("a"))
            s.save(_make_point("b"))
            files = list(Path(tmpdir).glob("*.point.json"))
            assert len(files) == 2

    def test_remove(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            s = DirectoryStore(Path(tmpdir))
            s.save(_make_point("x"))
            assert s.remove("x") is True
            assert s.load("x") is None

    def test_remove_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            s = DirectoryStore(Path(tmpdir))
            assert s.remove("nope") is False

    def test_clear(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            s = DirectoryStore(Path(tmpdir))
            s.save(_make_point("a"))
            s.save(_make_point("b"))
            s.clear()
            assert s.count() == 0

    def test_list_all(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            s = DirectoryStore(Path(tmpdir))
            s.save(_make_point("a"))
            s.save(_make_point("b"))
            assert len(s.list_all()) == 2

    def test_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            s = DirectoryStore(Path(tmpdir))
            assert s.count() == 0
            s.save(_make_point("a"))
            assert s.count() == 1

    def test_safe_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            s = DirectoryStore(Path(tmpdir))
            s.save(_make_point("a/b"))
            assert s.load("a/b") is not None
            # File uses underscore replacement
            files = list(Path(tmpdir).glob("*.point.json"))
            assert any("a_b" in f.name for f in files)
