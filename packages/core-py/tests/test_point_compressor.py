"""Tests for point_compressor — backward-compatible shim over pugqeep."""

import domains.infrastructure.pugqeep as pugqeep
from domains.infrastructure import point_compressor as pc


class TestReexports:
    def test_point_identity(self):
        assert pc.Point is pugqeep.Point

    def test_point_compressor_identity(self):
        assert pc.PointCompressor is pugqeep.PointCompressor

    def test_point_library_identity(self):
        assert pc.PointLibrary is pugqeep.PointLibrary

    def test_model_tree_identity(self):
        assert pc.ModelTree is pugqeep.ModelTree

    def test_point_deduplicator_identity(self):
        assert pc.PointDeduplicator is pugqeep.PointDeduplicator

    def test_point_library_sync_identity(self):
        assert pc.PointLibrarySync is pugqeep.PointLibrarySync

    def test_load_model_to_points_identity(self):
        assert pc.load_model_to_points is pugqeep.load_model_to_points

    def test_pgq_identity(self):
        assert pc.PGQ is pugqeep.PGQ

    def test_point_lib_alias(self):
        assert pc.PointLib is pugqeep.PGQ

    def test_load_library_alias(self):
        assert pc.load_library == pugqeep.PGQ.load

    def test_all_exports_importable(self):
        for name in pc.__all__:
            assert hasattr(pc, name)


class TestSaveLibrary:
    def test_delegates_to_library_save(self):
        calls = []

        class FakeLib:
            def save(self, path):
                calls.append(path)

        pc.save_library(FakeLib(), "/tmp/lib.json")
        assert calls == ["/tmp/lib.json"]
