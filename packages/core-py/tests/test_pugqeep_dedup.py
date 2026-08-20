"""Tests for domains.infrastructure.pugqeep.dedup — PointDeduplicator and PointLibrarySync."""

import numpy as np
import tempfile
from pathlib import Path

import pytest
from domains.infrastructure.pugqeep.dedup import PointDeduplicator, PointLibrarySync
from domains.infrastructure.pugqeep.library import PointLibrary
from domains.infrastructure.pugqeep.point import Point


class TestPointDeduplicator:
    def _make_point(self, identity, data):
        return Point(identity=identity, function_type="periodic", params={"data": data})

    def test_no_duplicates(self):
        lib = PointLibrary()
        lib.add(self._make_point("a", 1.0))
        lib.add(self._make_point("b", 2.0))
        dedup = PointDeduplicator()
        dedup.add_library(lib)
        groups = dedup.find_duplicates()
        assert groups == []

    def test_finds_duplicates(self):
        lib1 = PointLibrary()
        lib2 = PointLibrary()
        p = self._make_point("a", 1.0)
        lib1.add(p)
        lib2.add(self._make_point("a_copy", 1.0))
        dedup = PointDeduplicator()
        dedup.add_library(lib1)
        dedup.add_library(lib2)
        groups = dedup.find_duplicates()
        assert len(groups) >= 1

    def test_deduplicate_removes_duplicates(self):
        lib1 = PointLibrary()
        lib2 = PointLibrary()
        lib1.add(self._make_point("a", 1.0))
        lib2.add(self._make_point("a_dup", 1.0))
        dedup = PointDeduplicator()
        dedup.add_library(lib1)
        dedup.add_library(lib2)
        result = dedup.deduplicate()
        assert result["merged"] >= 1

    def test_cluster_fingerprint(self):
        lib = PointLibrary()
        weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        lib.compress_and_store(weights, identity="c1")
        dedup = PointDeduplicator()
        dedup.add_library(lib)
        groups = dedup.find_duplicates()
        assert groups == []

    def test_single_point_no_duplicates(self):
        lib = PointLibrary()
        lib.add(self._make_point("only", 42.0))
        dedup = PointDeduplicator()
        dedup.add_library(lib)
        assert dedup.find_duplicates() == []


class TestPointLibrarySync:
    def test_export_import_bytes(self):
        lib = PointLibrary(name="sync_test")
        lib.add(Point(identity="p1", function_type="periodic", params={"a": 1.0}))
        lib.add(Point(identity="p2", function_type="periodic", params={"b": 2.0}))

        sync = PointLibrarySync()
        data = sync.export_bytes(lib)
        assert isinstance(data, bytes)

        imported = sync.import_bytes(data)
        assert imported.name == "sync_test"
        assert imported.has("p1")
        assert imported.has("p2")

    def test_sync_to_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = PointLibrary(name="to_dir")
            lib.add(Point(identity="p1", function_type="periodic", params={}))

            sync = PointLibrarySync()
            path = sync.sync_to_directory(lib, Path(tmpdir))
            assert path.exists()

    def test_sync_from_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = PointLibrary(name="from_dir")
            lib.add(Point(identity="p1", function_type="periodic", params={}))
            lib.save(Path(tmpdir) / "from_dir.points.json")

            sync = PointLibrarySync()
            loaded = sync.sync_from_directory(Path(tmpdir), name="from_dir")
            assert loaded is not None
            assert loaded.has("p1")

    def test_sync_from_directory_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sync = PointLibrarySync()
            loaded = sync.sync_from_directory(Path(tmpdir), name="nonexistent")
            assert loaded is None

    def test_sync_from_directory_auto_find(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = PointLibrary(name="auto")
            lib.add(Point(identity="x", function_type="periodic", params={}))
            lib.save(Path(tmpdir) / "auto.points.json")

            sync = PointLibrarySync()
            loaded = sync.sync_from_directory(Path(tmpdir))
            assert loaded is not None
            assert loaded.has("x")

    def test_sync_from_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sync = PointLibrarySync()
            loaded = sync.sync_from_directory(Path(tmpdir))
            assert loaded is None

    def test_merge_libraries(self):
        lib1 = PointLibrary(name="l1")
        lib1.add(Point(identity="a", function_type="periodic", params={"data": 1.0}))
        lib2 = PointLibrary(name="l2")
        lib2.add(Point(identity="b", function_type="periodic", params={"data": 2.0}))

        sync = PointLibrarySync()
        merged = sync.merge([lib1, lib2])
        assert merged.has("a")
        assert merged.has("b")
