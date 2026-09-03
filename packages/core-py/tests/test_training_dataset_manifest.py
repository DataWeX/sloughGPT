"""Tests for dataset manifest — load and resolve training data paths."""
from __future__ import annotations

import json

import pytest

from domains.training.dataset_manifest import (
    ManifestError,
    _glob_train_files,
    load_manifest,
    resolve_training_data_path,
)


class TestLoadManifest:
    def test_valid(self, tmp_path):
        m = {
            "schema_version": "1.0",
            "dataset_id": "d1",
            "version": "1",
            "domain": "text",
            "pii_policy": "none",
            "sources": [{"type": "file", "path": "data.txt"}],
        }
        p = tmp_path / "manifest.json"
        p.write_text(json.dumps(m))
        data = load_manifest(p)
        assert data["dataset_id"] == "d1"

    def test_missing_file(self, tmp_path):
        with pytest.raises(ManifestError, match="not found"):
            load_manifest(tmp_path / "nope.json")

    def test_invalid_json(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{invalid")
        with pytest.raises(ManifestError, match="Invalid JSON"):
            load_manifest(p)

    def test_wrong_version(self, tmp_path):
        m = {"schema_version": "2.0", "dataset_id": "d1", "version": "1", "domain": "text", "pii_policy": "none", "sources": []}
        p = tmp_path / "m.json"
        p.write_text(json.dumps(m))
        with pytest.raises(ManifestError, match="Unsupported schema_version"):
            load_manifest(p)

    def test_missing_required_field(self, tmp_path):
        m = {"schema_version": "1.0"}
        p = tmp_path / "m.json"
        p.write_text(json.dumps(m))
        with pytest.raises(ManifestError, match="missing required field"):
            load_manifest(p)

    def test_empty_sources(self, tmp_path):
        m = {"schema_version": "1.0", "dataset_id": "d1", "version": "1", "domain": "text", "pii_policy": "none", "sources": []}
        p = tmp_path / "m.json"
        p.write_text(json.dumps(m))
        with pytest.raises(ManifestError, match="non-empty list"):
            load_manifest(p)


class TestGlobTrainFiles:
    def test_literal_path(self, tmp_path):
        result = _glob_train_files("data.txt", tmp_path)
        assert result == [tmp_path / "data.txt"]

    def test_glob_pattern(self, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        result = _glob_train_files("*.txt", tmp_path)
        assert len(result) == 2


class TestResolveTrainingDataPath:
    def test_splits_train(self, tmp_path):
        m = {"schema_version": "1.0", "dataset_id": "d1", "version": "1", "domain": "text", "pii_policy": "none", "sources": [{"type": "file", "path": "data.txt"}], "splits": {"train": "train.txt"}}
        (tmp_path / "manifest.json").write_text(json.dumps(m))
        (tmp_path / "train.txt").write_text("hello")
        path, data = resolve_training_data_path(tmp_path / "manifest.json")
        assert path.name == "train.txt"

    def test_fallback_to_input_txt(self, tmp_path):
        m = {"schema_version": "1.0", "dataset_id": "d1", "version": "1", "domain": "text", "pii_policy": "none", "sources": [{"type": "file", "path": "data.txt"}]}
        (tmp_path / "manifest.json").write_text(json.dumps(m))
        (tmp_path / "input.txt").write_text("hello")
        path, data = resolve_training_data_path(tmp_path / "manifest.json")
        assert path.name == "input.txt"

    def test_no_training_file(self, tmp_path):
        m = {"schema_version": "1.0", "dataset_id": "d1", "version": "1", "domain": "text", "pii_policy": "none", "sources": [{"type": "file", "path": "data.txt"}]}
        (tmp_path / "manifest.json").write_text(json.dumps(m))
        with pytest.raises(ManifestError, match="No training text file"):
            resolve_training_data_path(tmp_path / "manifest.json")
