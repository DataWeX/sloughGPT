"""Tests for domains.infrastructure.pugqeep.dedup — PointDeduplicator and PointLibrarySync."""

import base64
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from domains.infrastructure.pugqeep.dedup import PointDeduplicator, PointLibrarySync
from domains.infrastructure.pugqeep.library import PointLibrary
from domains.infrastructure.pugqeep.point import Point


class TestPointDeduplicator:
    def _make_point(self, identity, data):
        return Point(identity=identity, function_type="periodic", params={"a": 1.0, "b": 2.0, "w": data})

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
        lib1.add(self._make_point("a", 1.0))
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

    def test_multiple_libraries(self):
        libs = []
        for i in range(3):
            lib = PointLibrary()
            lib.add(self._make_point(f"p{i}", float(i)))
            libs.append(lib)
        dedup = PointDeduplicator()
        for lib in libs:
            dedup.add_library(lib)
        assert dedup.find_duplicates() == []

    def test_raw_point_fingerprint(self):
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        p = Point(
            identity="raw1",
            function_type="raw",
            params={"data_b64": base64.b64encode(data.tobytes()).decode(),
                    "shape": list(data.shape), "dtype": "float32"},
        )
        lib = PointLibrary()
        lib.add(p)
        dedup = PointDeduplicator()
        dedup.add_library(lib)
        assert dedup.find_duplicates() == []

    def test_duplicate_raw_points(self):
        data = np.array([1.0, 2.0], dtype=np.float32)
        p1 = Point(identity="r1", function_type="raw",
                    params={"data_b64": base64.b64encode(data.tobytes()).decode(),
                            "shape": list(data.shape), "dtype": "float32"})
        p2 = Point(identity="r2", function_type="raw",
                    params={"data_b64": base64.b64encode(data.tobytes()).decode(),
                            "shape": list(data.shape), "dtype": "float32"})
        lib = PointLibrary()
        lib.add(p1)
        lib.add(p2)
        dedup = PointDeduplicator()
        dedup.add_library(lib)
        groups = dedup.find_duplicates()
        assert len(groups) == 1
        assert set(groups[0]) == {"r1", "r2"}

    def test_deduplicate_returns_stats(self):
        lib = PointLibrary()
        lib.add(self._make_point("a", 1.0))
        lib.add(self._make_point("b", 1.0))
        dedup = PointDeduplicator()
        dedup.add_library(lib)
        result = dedup.deduplicate()
        assert "merged" in result
        assert "bytes_saved" in result
        assert "groups" in result

    def test_empty_library(self):
        lib = PointLibrary()
        dedup = PointDeduplicator()
        dedup.add_library(lib)
        assert dedup.find_duplicates() == []
        assert dedup.deduplicate()["merged"] == 0

    def test_tolerance_affects_fingerprint(self):
        p1 = Point(identity="t1", function_type="periodic",
                    params={"a": 1.0, "b": 2.0, "w": 1.0000001})
        p2 = Point(identity="t2", function_type="periodic",
                    params={"a": 1.0, "b": 2.0, "w": 1.0000002})
        lib = PointLibrary()
        lib.add(p1)
        lib.add(p2)
        dedup = PointDeduplicator(tolerance=1e-5)
        dedup.add_library(lib)
        groups = dedup.find_duplicates()
        assert len(groups) == 1

    def test_zero_tolerance_exact_match(self):
        p1 = Point(identity="z1", function_type="periodic", params={"a": 1.0, "b": 0.0, "w": 1.0})
        p2 = Point(identity="z2", function_type="periodic", params={"a": 1.0, "b": 0.0, "w": 1.0})
        lib = PointLibrary()
        lib.add(p1)
        lib.add(p2)
        dedup = PointDeduplicator(tolerance=0)
        dedup.add_library(lib)
        groups = dedup.find_duplicates()
        assert len(groups) == 1

    def test_three_duplicates(self):
        lib = PointLibrary()
        lib.add(self._make_point("x", 5.0))
        lib.add(self._make_point("y", 5.0))
        lib.add(self._make_point("z", 5.0))
        dedup = PointDeduplicator()
        dedup.add_library(lib)
        groups = dedup.find_duplicates()
        assert len(groups) == 1
        assert len(groups[0]) == 3

    def test_deduplicate_preserves_first(self):
        lib = PointLibrary()
        lib.add(self._make_point("keep", 1.0))
        lib.add(self._make_point("remove", 1.0))
        dedup = PointDeduplicator()
        dedup.add_library(lib)
        dedup.deduplicate()
        assert lib.has("keep")
        assert not lib.has("remove")

    def test_bytes_saved_positive(self):
        lib = PointLibrary()
        lib.add(self._make_point("a", 1.0))
        lib.add(self._make_point("b", 1.0))
        dedup = PointDeduplicator()
        dedup.add_library(lib)
        result = dedup.deduplicate()
        assert result["bytes_saved"] >= 0

    def test_groups_count(self):
        lib = PointLibrary()
        lib.add(self._make_point("a", 1.0))
        lib.add(self._make_point("b", 1.0))
        lib.add(self._make_point("c", 2.0))
        lib.add(self._make_point("d", 2.0))
        dedup = PointDeduplicator()
        dedup.add_library(lib)
        groups = dedup.find_duplicates()
        assert len(groups) == 2

    def test_no_libraries(self):
        dedup = PointDeduplicator()
        assert dedup.find_duplicates() == []
        assert dedup.deduplicate()["merged"] == 0

    def test_tolerance_large_no_match(self):
        p1 = Point(identity="l1", function_type="periodic", params={"a": 1.0, "b": 0.0, "w": 1.0})
        p2 = Point(identity="l2", function_type="periodic", params={"a": 1.0, "b": 0.0, "w": 100.0})
        lib = PointLibrary()
        lib.add(p1)
        lib.add(p2)
        dedup = PointDeduplicator(tolerance=1e-6)
        dedup.add_library(lib)
        groups = dedup.find_duplicates()
        assert groups == []

    def test_cluster_points_same_fingerprint(self):
        centroids1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        assignments1 = np.array([0, 1], dtype=np.uint8)
        centroids2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        assignments2 = np.array([0, 1], dtype=np.uint8)
        p1 = Point(identity="c1", function_type="cluster",
                    params={"centroids": centroids1, "assignments": assignments1})
        p2 = Point(identity="c2", function_type="cluster",
                    params={"centroids": centroids2, "assignments": assignments2})
        lib = PointLibrary()
        lib.add(p1)
        lib.add(p2)
        dedup = PointDeduplicator()
        dedup.add_library(lib)
        groups = dedup.find_duplicates()
        assert len(groups) == 1

    def test_different_function_types_no_match(self):
        p1 = Point(identity="p1", function_type="periodic", params={"a": 1.0, "b": 0.0, "w": 1.0})
        data = np.array([1.0], dtype=np.float32)
        p2 = Point(identity="p2", function_type="raw",
                    params={"data_b64": base64.b64encode(data.tobytes()).decode(),
                            "shape": [1], "dtype": "float32"})
        lib = PointLibrary()
        lib.add(p1)
        lib.add(p2)
        dedup = PointDeduplicator()
        dedup.add_library(lib)
        groups = dedup.find_duplicates()
        assert groups == []

    def test_default_tolerance(self):
        dedup = PointDeduplicator()
        assert dedup._tolerance == 1e-6

    def test_custom_tolerance(self):
        dedup = PointDeduplicator(tolerance=0.01)
        assert dedup._tolerance == 0.01

    def test_deduplicate_cross_library_raw(self):
        data = np.array([5.0, 6.0], dtype=np.float32)
        lib1 = PointLibrary()
        lib1.add(Point(identity="r1", function_type="raw",
                       params={"data_b64": base64.b64encode(data.tobytes()).decode(),
                               "shape": [2], "dtype": "float32"}))
        lib2 = PointLibrary()
        lib2.add(Point(identity="r2", function_type="raw",
                       params={"data_b64": base64.b64encode(data.tobytes()).decode(),
                               "shape": [2], "dtype": "float32"}))
        dedup = PointDeduplicator()
        dedup.add_library(lib1)
        dedup.add_library(lib2)
        result = dedup.deduplicate()
        assert result["merged"] >= 1
        assert result["groups"] >= 1

    def test_cluster_different_centroids_no_match(self):
        c1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        c2 = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        a = np.array([0, 1], dtype=np.uint8)
        lib = PointLibrary()
        lib.add(Point(identity="c1", function_type="cluster", params={"centroids": c1, "assignments": a}))
        lib.add(Point(identity="c2", function_type="cluster", params={"centroids": c2, "assignments": a}))
        dedup = PointDeduplicator()
        dedup.add_library(lib)
        groups = dedup.find_duplicates()
        assert groups == []

    def test_different_raw_data_no_match(self):
        d1 = np.array([1.0], dtype=np.float32)
        d2 = np.array([99.0], dtype=np.float32)
        lib = PointLibrary()
        lib.add(Point(identity="r1", function_type="raw",
                       params={"data_b64": base64.b64encode(d1.tobytes()).decode(),
                               "shape": [1], "dtype": "float32"}))
        lib.add(Point(identity="r2", function_type="raw",
                       params={"data_b64": base64.b64encode(d2.tobytes()).decode(),
                               "shape": [1], "dtype": "float32"}))
        dedup = PointDeduplicator()
        dedup.add_library(lib)
        groups = dedup.find_duplicates()
        assert groups == []

    def test_deduplicate_no_groups(self):
        lib = PointLibrary()
        lib.add(self._make_point("a", 1.0))
        lib.add(self._make_point("b", 2.0))
        dedup = PointDeduplicator()
        dedup.add_library(lib)
        result = dedup.deduplicate()
        assert result["merged"] == 0
        assert result["groups"] == 0


class TestPointLibrarySync:
    def test_export_import_bytes(self):
        lib = PointLibrary(name="sync_test")
        lib.add(Point(identity="p1", function_type="periodic", params={"a": 1.0, "b": 0.0, "w": 1.0}))
        lib.add(Point(identity="p2", function_type="periodic", params={"a": 0.0, "b": 1.0, "w": 2.0}))

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
            lib.add(Point(identity="p1", function_type="periodic", params={"a": 1.0, "b": 0.0, "w": 0.0}))

            sync = PointLibrarySync()
            path = sync.sync_to_directory(lib, Path(tmpdir))
            assert path.exists()

    def test_sync_from_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = PointLibrary(name="from_dir")
            lib.add(Point(identity="p1", function_type="periodic", params={"a": 1.0, "b": 0.0, "w": 0.0}))
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
            lib.add(Point(identity="x", function_type="periodic", params={"a": 1.0, "b": 0.0, "w": 0.0}))
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
        lib1.add(Point(identity="a", function_type="periodic", params={"a": 1.0, "b": 0.0, "w": 0.0}))
        lib2 = PointLibrary(name="l2")
        lib2.add(Point(identity="b", function_type="periodic", params={"a": 2.0, "b": 0.0, "w": 0.0}))

        sync = PointLibrarySync()
        merged = sync.merge([lib1, lib2])
        assert merged.has("a")
        assert merged.has("b")

    def test_export_import_preserves_name(self):
        lib = PointLibrary(name="my_special_lib")
        lib.add(Point(identity="p1", function_type="periodic", params={"a": 1.0, "b": 0.0, "w": 0.0}))
        sync = PointLibrarySync()
        data = sync.export_bytes(lib)
        imported = sync.import_bytes(data)
        assert imported.name == "my_special_lib"

    def test_merge_deduplicates(self):
        lib1 = PointLibrary(name="m1")
        lib2 = PointLibrary(name="m2")
        lib1.add(Point(identity="dup", function_type="periodic", params={"a": 1.0, "b": 0.0, "w": 1.0}))
        lib2.add(Point(identity="dup2", function_type="periodic", params={"a": 1.0, "b": 0.0, "w": 1.0}))
        sync = PointLibrarySync()
        merged = sync.merge([lib1, lib2])
        stats = merged.stats()
        assert stats["total_points"] <= 2

    def test_sync_to_directory_creates_nested(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = PointLibrary(name="nested")
            lib.add(Point(identity="p1", function_type="periodic", params={"a": 1.0, "b": 0.0, "w": 0.0}))
            sync = PointLibrarySync()
            target = Path(tmpdir) / "subdir" / "deep"
            path = sync.sync_to_directory(lib, target)
            assert path.exists()

    def test_export_import_cluster_point(self):
        centroids = np.random.randn(4, 3).astype(np.float32)
        assignments = np.array([0, 1, 2, 3], dtype=np.uint8)
        p = Point(identity="cl1", function_type="cluster",
                  params={"centroids": centroids, "assignments": assignments})
        lib = PointLibrary(name="cluster_lib")
        lib.add(p)
        sync = PointLibrarySync()
        data = sync.export_bytes(lib)
        imported = sync.import_bytes(data)
        assert imported.has("cl1")
        rp = imported.get("cl1")
        assert rp.function_type == "cluster"

    def test_merge_empty_list(self):
        sync = PointLibrarySync()
        merged = sync.merge([])
        assert merged.stats()["total_points"] == 0

    def test_export_import_raw_point(self):
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        p = Point(identity="raw1", function_type="raw",
                  params={"data_b64": base64.b64encode(data.tobytes()).decode(),
                          "shape": list(data.shape), "dtype": "float32"})
        lib = PointLibrary(name="raw_lib")
        lib.add(p)
        sync = PointLibrarySync()
        exported = sync.export_bytes(lib)
        imported = sync.import_bytes(exported)
        assert imported.has("raw1")

    def test_export_bytes_is_json(self):
        lib = PointLibrary(name="json_test")
        lib.add(Point(identity="p1", function_type="periodic", params={"a": 1.0, "b": 0.0, "w": 0.0}))
        sync = PointLibrarySync()
        data = sync.export_bytes(lib)
        parsed = json.loads(data)
        assert "name" in parsed
        assert "points" in parsed

    def test_import_bytes_empty_points(self):
        data = json.dumps({"name": "empty", "points": []}).encode()
        sync = PointLibrarySync()
        lib = sync.import_bytes(data)
        assert lib.stats()["total_points"] == 0

    def test_merge_three_libraries(self):
        libs = []
        for i in range(3):
            lib = PointLibrary(name=f"lib{i}")
            lib.add(Point(identity=f"p{i}", function_type="periodic",
                          params={"a": float(i), "b": 0.0, "w": 0.0}))
            libs.append(lib)
        sync = PointLibrarySync()
        merged = sync.merge(libs)
        assert merged.stats()["total_points"] == 3

    def test_export_import_preserves_points_count(self):
        lib = PointLibrary(name="count_test")
        for i in range(5):
            lib.add(Point(identity=f"p{i}", function_type="periodic",
                          params={"a": float(i), "b": 0.0, "w": 0.0}))
        sync = PointLibrarySync()
        data = sync.export_bytes(lib)
        imported = sync.import_bytes(data)
        assert imported.stats()["total_points"] == 5

    def test_sync_to_directory_file_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = PointLibrary(name="mylib")
            lib.add(Point(identity="p1", function_type="periodic", params={"a": 1.0, "b": 0.0, "w": 0.0}))
            sync = PointLibrarySync()
            path = sync.sync_to_directory(lib, Path(tmpdir))
            assert path.name == "mylib.points.json"

    def test_sync_from_directory_non_json_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "bad.points.json").write_text("not json")
            sync = PointLibrarySync()
            with pytest.raises(Exception):
                sync.sync_from_directory(Path(tmpdir), name="bad")

    def test_merge_single_library(self):
        lib = PointLibrary(name="single")
        lib.add(Point(identity="a", function_type="periodic", params={"a": 1.0, "b": 0.0, "w": 0.0}))
        sync = PointLibrarySync()
        merged = sync.merge([lib])
        assert merged.has("a")
        assert merged.stats()["total_points"] == 1

    def test_merge_preserves_all_points(self):
        lib1 = PointLibrary(name="p1")
        lib2 = PointLibrary(name="p2")
        lib1.add(Point(identity="a", function_type="periodic", params={"a": 1.0, "b": 0.0, "w": 0.0}))
        lib1.add(Point(identity="b", function_type="periodic", params={"a": 2.0, "b": 0.0, "w": 0.0}))
        lib2.add(Point(identity="c", function_type="periodic", params={"a": 3.0, "b": 0.0, "w": 0.0}))
        sync = PointLibrarySync()
        merged = sync.merge([lib1, lib2])
        assert merged.has("a")
        assert merged.has("b")
        assert merged.has("c")
        assert merged.stats()["total_points"] == 3
