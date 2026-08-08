"""Tests for infrastructure/pugqeep/dedup.py."""

import numpy as np
from pathlib import Path

from domains.infrastructure.pugqeep.dedup import PointDeduplicator, PointLibrarySync
from domains.infrastructure.pugqeep.library import PointLibrary
from domains.infrastructure.pugqeep.point import Point


def make_cluster_point(identity, centroids, assignments, accuracy=0.9):
    return Point(
        identity=identity,
        function_type="cluster",
        params={"centroids": centroids.astype(np.float32),
                "assignments": assignments.astype(np.uint8)},
        accuracy=accuracy,
    )


def make_linear_point(identity, a=1.0, b=0.0):
    return Point(identity=identity, function_type="linear",
                 params={"a": a, "b": b})


def make_raw_point(identity, data_b64):
    return Point(identity=identity, function_type="raw",
                 params={"data_b64": data_b64})


class TestFingerprint:
    def test_cluster_points_identical_fingerprint(self):
        d = PointDeduplicator()
        centroids = np.array([0.1, 0.5, 0.9], dtype=np.float32)
        assignments = np.array([0, 1, 2, 0], dtype=np.uint8)
        p1 = make_cluster_point("a", centroids, assignments)
        p2 = make_cluster_point("b", centroids, assignments)
        assert d._fingerprint(p1) == d._fingerprint(p2)

    def test_cluster_points_different_fingerprint(self):
        d = PointDeduplicator()
        c1 = np.array([0.1, 0.5, 0.9], dtype=np.float32)
        c2 = np.array([0.2, 0.6, 1.0], dtype=np.float32)
        a = np.array([0, 1, 2, 0], dtype=np.uint8)
        assert d._fingerprint(make_cluster_point("a", c1, a)) != \
            d._fingerprint(make_cluster_point("b", c2, a))

    def test_linear_points_identical_fingerprint(self):
        d = PointDeduplicator()
        assert d._fingerprint(make_linear_point("a", 2.0, 1.0)) == \
            d._fingerprint(make_linear_point("b", 2.0, 1.0))

    def test_linear_points_different_params_differ(self):
        d = PointDeduplicator()
        assert d._fingerprint(make_linear_point("a", 2.0, 1.0)) != \
            d._fingerprint(make_linear_point("b", 3.0, 1.0))

    def test_raw_point_fingerprint(self):
        d = PointDeduplicator()
        import base64
        b64 = base64.b64encode(np.array([1, 2, 3], dtype=np.uint8).tobytes()).decode()
        assert d._fingerprint(make_raw_point("a", b64)) == \
            d._fingerprint(make_raw_point("b", b64))


class TestDeduplicator:
    def test_add_library_indexes_points(self):
        lib = PointLibrary(name="l1")
        lib.add(make_linear_point("p1", 1.0, 0.0))
        lib.add(make_linear_point("p2", 2.0, 0.0))
        d = PointDeduplicator()
        d.add_library(lib)
        assert len(d._fingerprints) == 2

    def test_find_duplicates_across_libraries(self):
        d = PointDeduplicator()
        lib1 = PointLibrary(name="l1")
        lib2 = PointLibrary(name="l2")
        lib1.add(make_linear_point("p1", 1.0, 0.0))
        lib2.add(make_linear_point("p2", 1.0, 0.0))
        d.add_library(lib1)
        d.add_library(lib2)
        groups = d.find_duplicates()
        assert len(groups) == 1
        assert set(groups[0]) == {"p1", "p2"}

    def test_no_duplicates_returns_empty(self):
        d = PointDeduplicator()
        lib = PointLibrary(name="l1")
        lib.add(make_linear_point("p1", 1.0, 0.0))
        lib.add(make_linear_point("p2", 2.0, 0.0))
        d.add_library(lib)
        assert d.find_duplicates() == []

    def test_deduplicate_merges_removing_duplicates(self):
        d = PointDeduplicator()
        lib1 = PointLibrary(name="l1")
        lib2 = PointLibrary(name="l2")
        lib1.add(make_linear_point("p1", 1.0, 0.0))
        lib2.add(make_linear_point("p2", 1.0, 0.0))
        d.add_library(lib1)
        d.add_library(lib2)
        result = d.deduplicate()
        assert result["merged"] == 1
        assert result["groups"] == 1
        assert not lib2.has("p2")
        assert lib1.has("p1")

    def test_deduplicate_counts_cluster_bytes(self):
        d = PointDeduplicator()
        lib1 = PointLibrary(name="l1")
        lib2 = PointLibrary(name="l2")
        centroids = np.array([0.1, 0.5, 0.9], dtype=np.float32)
        assignments = np.array([0, 1, 2, 0], dtype=np.uint8)
        lib1.add(make_cluster_point("c1", centroids, assignments))
        lib2.add(make_cluster_point("c2", centroids, assignments))
        d.add_library(lib1)
        d.add_library(lib2)
        result = d.deduplicate()
        assert result["merged"] == 1
        assert result["bytes_saved"] == centroids.nbytes + assignments.nbytes

    def test_deduplicate_noop_when_none_shared(self):
        d = PointDeduplicator()
        lib1 = PointLibrary(name="l1")
        lib2 = PointLibrary(name="l2")
        lib1.add(make_linear_point("p1", 1.0, 0.0))
        lib2.add(make_linear_point("p2", 9.0, 9.0))
        d.add_library(lib1)
        d.add_library(lib2)
        result = d.deduplicate()
        assert result["merged"] == 0
        assert lib2.has("p2")

    def test_deduplicate_skips_group_when_keep_missing(self):
        lib = PointLibrary(name="l1")
        lib.add(make_linear_point("p1", 1.0, 0.0))
        lib.add(make_linear_point("p2", 1.0, 0.0))
        d = PointDeduplicator()
        d.add_library(lib)
        lib.remove("p1")
        result = d.deduplicate()
        assert result["merged"] == 0
        assert lib.has("p2")


class TestPointLibrarySync:
    def test_export_import_roundtrip(self):
        lib = PointLibrary(name="src")
        lib.add(make_linear_point("p1", 2.0, 1.0))
        lib.add(make_raw_point("p2", "aGVsbG8="))
        sync = PointLibrarySync()
        data = sync.export_bytes(lib)
        restored = sync.import_bytes(data)
        assert restored.name == "src"
        assert set(p.identity for p in restored.list_all()) == {"p1", "p2"}

    def test_import_bytes_preserves_point_values(self):
        lib = PointLibrary(name="src")
        lib.add(make_linear_point("p1", 2.0, 1.0))
        sync = PointLibrarySync()
        restored = sync.import_bytes(sync.export_bytes(lib))
        out = restored.get("p1")
        assert np.isclose(out.params["a"], 2.0)
        assert np.isclose(out.params["b"], 1.0)

    def test_sync_to_directory_writes_file(self, tmp_path):
        lib = PointLibrary(name="mylib")
        lib.add(make_linear_point("p1", 1.0, 0.0))
        sync = PointLibrarySync()
        path = sync.sync_to_directory(lib, tmp_path)
        assert path == tmp_path / "mylib.points.json"
        assert path.exists()

    def test_sync_from_directory_roundtrip(self, tmp_path):
        lib = PointLibrary(name="mylib")
        lib.add(make_linear_point("p1", 3.0, 2.0))
        sync = PointLibrarySync()
        sync.sync_to_directory(lib, tmp_path)
        loaded = sync.sync_from_directory(tmp_path, name="mylib")
        assert loaded is not None
        assert loaded.get("p1").params["a"] == 3.0

    def test_sync_from_directory_missing_name_returns_none(self, tmp_path):
        sync = PointLibrarySync()
        assert sync.sync_from_directory(tmp_path, name="nope") is None

    def test_sync_from_directory_glob_finds_single(self, tmp_path):
        lib = PointLibrary(name="onlylib")
        lib.add(make_linear_point("p1", 1.0, 0.0))
        sync = PointLibrarySync()
        sync.sync_to_directory(lib, tmp_path)
        loaded = sync.sync_from_directory(tmp_path)
        assert loaded is not None
        assert loaded.name == "onlylib"

    def test_sync_from_directory_empty_dir_returns_none(self, tmp_path):
        sync = PointLibrarySync()
        assert sync.sync_from_directory(tmp_path) is None

    def test_merge_combines_points_and_deduplicates(self):
        lib1 = PointLibrary(name="l1")
        lib2 = PointLibrary(name="l2")
        lib1.add(make_linear_point("dup", 1.0, 0.0))
        lib2.add(make_linear_point("dup", 1.0, 0.0))
        lib2.add(make_linear_point("unique", 7.0, 0.0))
        sync = PointLibrarySync()
        merged = sync.merge([lib1, lib2])
        assert merged.name == "merged"
        assert merged.has("unique")
        assert merged.has("dup")
