"""Tests for dataset manifest loading and training text path resolution."""

import json

import pytest

from domains.training.dataset_manifest import (
    ManifestError,
    _glob_train_files,
    load_manifest,
    resolve_training_data_path,
)


MINIMAL_MANIFEST = {
    "schema_version": "1.0",
    "dataset_id": "shakespeare",
    "version": "1",
    "domain": "text",
    "pii_policy": "none",
    "sources": [{"type": "local", "path": "shakespeare.txt"}],
}


class TestLoadManifest:
    def test_loads_valid_manifest(self, tmp_path):
        p = tmp_path / "manifest.json"
        p.write_text(json.dumps(MINIMAL_MANIFEST))
        data = load_manifest(p)
        assert data["dataset_id"] == "shakespeare"
        assert data["schema_version"] == "1.0"

    def test_accepts_str_path(self, tmp_path):
        p = tmp_path / "manifest.json"
        p.write_text(json.dumps(MINIMAL_MANIFEST))
        data = load_manifest(str(p))
        assert data["dataset_id"] == "shakespeare"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ManifestError, match="not found"):
            load_manifest(tmp_path / "nope.json")

    def test_invalid_json_raises(self, tmp_path):
        p = tmp_path / "manifest.json"
        p.write_text("{not json")
        with pytest.raises(ManifestError, match="Invalid JSON"):
            load_manifest(p)

    def test_wrong_schema_version_raises(self, tmp_path):
        p = tmp_path / "manifest.json"
        bad = dict(MINIMAL_MANIFEST, schema_version="0.9")
        p.write_text(json.dumps(bad))
        with pytest.raises(ManifestError, match="Unsupported schema_version"):
            load_manifest(p)

    def test_missing_required_field_raises(self, tmp_path):
        p = tmp_path / "manifest.json"
        bad = {k: v for k, v in MINIMAL_MANIFEST.items() if k != "pii_policy"}
        p.write_text(json.dumps(bad))
        with pytest.raises(ManifestError, match="pii_policy"):
            load_manifest(p)

    def test_empty_sources_raises(self, tmp_path):
        p = tmp_path / "manifest.json"
        bad = dict(MINIMAL_MANIFEST, sources=[])
        p.write_text(json.dumps(bad))
        with pytest.raises(ManifestError, match="non-empty list"):
            load_manifest(p)

    def test_non_list_sources_raises(self, tmp_path):
        p = tmp_path / "manifest.json"
        bad = dict(MINIMAL_MANIFEST, sources="txt")
        p.write_text(json.dumps(bad))
        with pytest.raises(ManifestError, match="non-empty list"):
            load_manifest(p)


class TestGlobTrainFiles:
    def test_literal_path(self, tmp_path):
        f = tmp_path / "input.txt"
        f.write_text("data")
        out = _glob_train_files("input.txt", tmp_path)
        assert out == [f]

    def test_single_star_glob(self, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        out = _glob_train_files("*.txt", tmp_path)
        assert len(out) == 2
        assert all(x.suffix == ".txt" for x in out)

    def test_recursive_glob(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "c.txt").write_text("c")
        out = _glob_train_files("**/*.txt", tmp_path)
        assert any(x.name == "c.txt" for x in out)

    def test_missing_literal_still_returned(self, tmp_path):
        out = _glob_train_files("missing.txt", tmp_path)
        assert out == [tmp_path / "missing.txt"]


class TestResolveTrainingDataPath:
    def _write_manifest(self, tmp_path, name="manifest.json", **overrides):
        data = dict(MINIMAL_MANIFEST, **overrides)
        p = tmp_path / name
        p.write_text(json.dumps(data))
        return p

    def test_uses_splits_train_single_file(self, tmp_path):
        (tmp_path / "train.txt").write_text("data")
        mp = self._write_manifest(tmp_path, splits={"train": "train.txt"})
        path, manifest = resolve_training_data_path(mp)
        assert path == tmp_path / "train.txt"
        assert manifest["dataset_id"] == "shakespeare"

    def test_uses_default_input_txt(self, tmp_path):
        (tmp_path / "input.txt").write_text("data")
        mp = self._write_manifest(tmp_path)
        path, manifest = resolve_training_data_path(mp)
        assert path == tmp_path / "input.txt"

    def test_glob_matching_one_txt(self, tmp_path):
        (tmp_path / "train_001.txt").write_text("a")
        (tmp_path / "notes.md").write_text("b")
        mp = self._write_manifest(tmp_path, splits={"train": "train_*.txt"})
        path, _ = resolve_training_data_path(mp)
        assert path == tmp_path / "train_001.txt"

    def test_multiple_matches_single_txt_resolves(self, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "a.md").write_text("md")
        mp = self._write_manifest(tmp_path, splits={"train": "a.*"})
        path, _ = resolve_training_data_path(mp)
        assert path == tmp_path / "a.txt"

    def test_multiple_txt_matches_raise(self, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        mp = self._write_manifest(tmp_path, splits={"train": "*.txt"})
        with pytest.raises(ManifestError, match="matched 2 files"):
            resolve_training_data_path(mp)

    def test_unresolvable_glob_raises(self, tmp_path):
        mp = self._write_manifest(tmp_path, splits={"train": "*.txt"})
        with pytest.raises(ManifestError, match="did not resolve"):
            resolve_training_data_path(mp)

    def test_no_train_no_input_raises(self, tmp_path):
        mp = self._write_manifest(tmp_path)
        with pytest.raises(ManifestError, match="No training text file"):
            resolve_training_data_path(mp)

    def test_missing_manifest_path_raises(self, tmp_path):
        with pytest.raises(ManifestError, match="not a file"):
            resolve_training_data_path(tmp_path / "ghost.json")

    def test_nested_directory_resolution(self, tmp_path):
        d = tmp_path / "data" / "nested"
        d.mkdir(parents=True)
        (d / "corpus.txt").write_text("data")
        mp = self._write_manifest(d, splits={"train": "corpus.txt"})
        path, _ = resolve_training_data_path(mp)
        assert path == d / "corpus.txt"
