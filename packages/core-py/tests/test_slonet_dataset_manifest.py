"""Tests for training.dataset_manifest — dataset manifest loading."""

from __future__ import annotations

import json
import os
import tempfile
import pytest
from pathlib import Path
from domains.training.dataset_manifest import (
    ManifestError,
    load_manifest,
    resolve_training_data_path,
    _glob_train_files,
)


# ── ManifestError ───────────────────────────────────────────────────────────


class TestManifestError:

    def test_is_value_error(self):
        assert issubclass(ManifestError, ValueError)


# ── load_manifest ───────────────────────────────────────────────────────────


class TestLoadManifest:

    def test_not_found(self):
        with pytest.raises(ManifestError, match="not found"):
            load_manifest("/nonexistent/path/manifest.json")

    def test_invalid_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not json")
            path = f.name
        try:
            with pytest.raises(ManifestError, match="Invalid JSON"):
                load_manifest(path)
        finally:
            os.unlink(path)

    def test_wrong_version(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"schema_version": "2.0"}, f)
            path = f.name
        try:
            with pytest.raises(ManifestError, match="Unsupported schema_version"):
                load_manifest(path)
        finally:
            os.unlink(path)

    def test_missing_fields(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"schema_version": "1.0"}, f)
            path = f.name
        try:
            with pytest.raises(ManifestError, match="missing required field"):
                load_manifest(path)
        finally:
            os.unlink(path)

    def test_empty_sources(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({
                "schema_version": "1.0",
                "dataset_id": "test",
                "version": "1.0",
                "domain": "test",
                "pii_policy": "none",
                "sources": [],
            }, f)
            path = f.name
        try:
            with pytest.raises(ManifestError, match="non-empty list"):
                load_manifest(path)
        finally:
            os.unlink(path)

    def test_valid_manifest(self):
        manifest = {
            "schema_version": "1.0",
            "dataset_id": "test",
            "version": "1.0",
            "domain": "test",
            "pii_policy": "none",
            "sources": [{"type": "file", "path": "data.txt"}],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(manifest, f)
            path = f.name
        try:
            result = load_manifest(path)
            assert result["dataset_id"] == "test"
        finally:
            os.unlink(path)


# ── _glob_train_files ──────────────────────────────────────────────────────


class TestGlobTrainFiles:

    def test_exact_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir)
            (p / "train.txt").write_text("hello")
            result = _glob_train_files("train.txt", p)
            assert len(result) == 1

    def test_glob_pattern(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir)
            (p / "train.txt").write_text("hello")
            result = _glob_train_files("*.txt", p)
            assert len(result) == 1


# ── resolve_training_data_path ─────────────────────────────────────────────


class TestResolveTrainingDataPath:

    def test_not_found(self):
        with pytest.raises(ManifestError, match="not a file"):
            resolve_training_data_path("/nonexistent/manifest.json")

    def test_with_splits_train(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir)
            manifest = {
                "schema_version": "1.0",
                "dataset_id": "test",
                "version": "1.0",
                "domain": "test",
                "pii_policy": "none",
                "sources": [{"type": "file", "path": "data.txt"}],
                "splits": {"train": "train.txt"},
            }
            (p / "manifest.json").write_text(json.dumps(manifest))
            (p / "train.txt").write_text("hello world")
            path, data = resolve_training_data_path(p / "manifest.json")
            assert path.name == "train.txt"

    def test_default_input_txt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir)
            manifest = {
                "schema_version": "1.0",
                "dataset_id": "test",
                "version": "1.0",
                "domain": "test",
                "pii_policy": "none",
                "sources": [{"type": "file", "path": "data.txt"}],
            }
            (p / "manifest.json").write_text(json.dumps(manifest))
            (p / "input.txt").write_text("hello world")
            path, data = resolve_training_data_path(p / "manifest.json")
            assert path.name == "input.txt"
