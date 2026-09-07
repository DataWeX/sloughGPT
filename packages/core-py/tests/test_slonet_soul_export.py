"""Tests for training.slonet — SOU export/import and utility functions."""

from __future__ import annotations

import os
import json
import struct
import tempfile
import pytest
import numpy as np
from domains.training.slonet import (
    SloNet,
    SloTransformer,
    SloLinear,
    _sanitize,
    export_to_sou,
    import_from_sou,
    SOU_MAGIC,
    SOU_VERSION,
    create_scheduler,
)


# ── _sanitize ───────────────────────────────────────────────────────────────


class TestSanitize:

    def test_sanitize_dict(self):
        result = _sanitize({"a": float("nan"), "b": float("inf"), "c": 1.0})
        assert result["a"] is None
        assert result["b"] is None
        assert result["c"] == 1.0

    def test_sanitize_list(self):
        result = _sanitize([float("nan"), 1.0])
        assert result[0] is None
        assert result[1] == 1.0

    def test_sanitize_nested(self):
        result = _sanitize({"a": {"b": float("nan")}})
        assert result["a"]["b"] is None

    def test_sanitize_normal(self):
        result = _sanitize({"a": 1, "b": "hello"})
        assert result == {"a": 1, "b": "hello"}


# ── export_to_sou / import_from_sou ─────────────────────────────────────────


class TestSOUExportImport:

    def test_export_import_slo_net(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.soul")
            net = SloNet(
                layers=[SloLinear(10, 5)],
                soul_name="TestNet",
                soul_traits={"warmth": 0.8},
            )
            export_to_sou(net, path, include_weights=True)
            assert os.path.exists(path)
            assert os.path.exists(path + ".meta.json")

            loaded = import_from_sou(path)
            assert loaded.soul_name == "TestNet"

    def test_export_import_transformer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.soul")
            model = SloTransformer(
                vocab_size=100, n_embed=32, n_layer=1, n_head=2,
                soul_name="TestTransformer",
            )
            export_to_sou(model, path, include_weights=True)
            loaded = import_from_sou(path)
            assert loaded.soul_name == "TestTransformer"
            assert isinstance(loaded, SloTransformer)

    def test_export_no_weights(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.soul")
            net = SloNet(soul_name="NoWeights")
            export_to_sou(net, path, include_weights=False)
            loaded = import_from_sou(path)
            assert loaded.soul_name == "NoWeights"

    def test_invalid_magic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.soul")
            with open(path, "wb") as f:
                f.write(b"INVALID")
            with pytest.raises(ValueError, match="bad magic"):
                import_from_sou(path)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            import_from_sou("/nonexistent/path.soul")

    def test_meta_json_written(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.soul")
            net = SloNet(soul_name="MetaTest")
            export_to_sou(net, path)
            meta_path = path + ".meta.json"
            assert os.path.exists(meta_path)
            with open(meta_path) as f:
                meta = json.load(f)
            assert meta["soul_name"] == "MetaTest"


# ── SOU constants ───────────────────────────────────────────────────────────


class TestSOUConstants:

    def test_magic(self):
        assert SOU_MAGIC == b"SOUL"

    def test_version(self):
        assert SOU_VERSION == 1
