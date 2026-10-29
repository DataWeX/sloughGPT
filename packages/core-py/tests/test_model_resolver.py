"""Tests for model_resolver — HuggingFace cache directory resolution."""

from __future__ import annotations

import json

import pytest

from domain.infrastructure._internal.model_resolver import (
    find_safetensors,
    get_model_dir,
    load_model_config,
)

# ── get_model_dir ─────────────────────────────────────────────────────────


class TestGetModelDir:
    def test_returns_first_candidate_when_none_exist(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
        monkeypatch.setattr(
            "domain.infrastructure._internal.model_resolver.find_repo_root",
            lambda _: tmp_path,
        )
        result = get_model_dir("gpt2")
        assert result.name == "models--gpt2"

    def test_returns_existing_hf_cache(self, tmp_path, monkeypatch):
        hf_home = tmp_path / "hf"
        model_dir = hf_home / "hub" / "models--gpt2"
        model_dir.mkdir(parents=True)
        monkeypatch.setenv("HF_HOME", str(hf_home))
        monkeypatch.setattr(
            "domain.infrastructure._internal.model_resolver.find_repo_root",
            lambda _: tmp_path,
        )
        result = get_model_dir("gpt2")
        assert result == model_dir

    def test_returns_existing_project_local_cache(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HF_HOME", str(tmp_path / "nonexistent"))
        project_cache = tmp_path / "models" / "hf-cache" / "hub" / "models--gpt2"
        project_cache.mkdir(parents=True)
        monkeypatch.setattr(
            "domain.infrastructure._internal.model_resolver.find_repo_root",
            lambda _: tmp_path,
        )
        result = get_model_dir("gpt2")
        assert result == project_cache

    def test_slashed_model_id_becomes_double_dash(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
        monkeypatch.setattr(
            "domain.infrastructure._internal.model_resolver.find_repo_root",
            lambda _: tmp_path,
        )
        result = get_model_dir("meta-llama/Llama-2-7b")
        assert result.name == "models--meta-llama--Llama-2-7b"


# ── weights preference (card c158a0e3) ────────────────────────────────────
#
# get_model_dir() returned the global HF cache dir whenever it existed —
# even when it was a refs-only stub (refs/main, zero blobs) shadowing the
# intact weights in the project-local cache. The candidate that actually
# holds weights must win wherever it sits in the search order; a stub-only
# result remains the fallback when no candidate has weights.


class TestGetModelDirWeightsPreference:
    @staticmethod
    def _make_stub(hf_home):
        """An HF-cache entry with refs/ only — no blobs, no snapshots."""
        stub = hf_home / "hub" / "models--gpt2"
        (stub / "refs").mkdir(parents=True)
        (stub / "refs" / "main").write_text("default")
        return stub

    @staticmethod
    def _make_local_with_safetensors(tmp_path):
        local = tmp_path / "models" / "hf-cache" / "hub" / "models--gpt2"
        snap = local / "snapshots" / "abc123"
        snap.mkdir(parents=True)
        (snap / "model.safetensors").write_text("fake-weights")
        return local

    def test_safetensors_beats_earlier_refs_only_stub(self, tmp_path, monkeypatch):
        hf_home = tmp_path / "hf"
        self._make_stub(hf_home)
        local = self._make_local_with_safetensors(tmp_path)
        monkeypatch.setenv("HF_HOME", str(hf_home))
        monkeypatch.setattr(
            "domain.infrastructure._internal.model_resolver.find_repo_root",
            lambda _: tmp_path,
        )
        assert get_model_dir("gpt2") == local

    def test_slnc_counts_as_weights(self, tmp_path, monkeypatch):
        hf_home = tmp_path / "hf"
        self._make_stub(hf_home)
        local = tmp_path / "models" / "hf-cache" / "hub" / "models--gpt2"
        local.mkdir(parents=True)
        (local / "model.slnc").write_text("fake")
        monkeypatch.setenv("HF_HOME", str(hf_home))
        monkeypatch.setattr(
            "domain.infrastructure._internal.model_resolver.find_repo_root",
            lambda _: tmp_path,
        )
        assert get_model_dir("gpt2") == local

    def test_soul_counts_as_weights(self, tmp_path, monkeypatch):
        hf_home = tmp_path / "hf"
        self._make_stub(hf_home)
        local = tmp_path / "models" / "hf-cache" / "hub" / "models--gpt2"
        local.mkdir(parents=True)
        (local / "model.soul").write_text("fake")
        monkeypatch.setenv("HF_HOME", str(hf_home))
        monkeypatch.setattr(
            "domain.infrastructure._internal.model_resolver.find_repo_root",
            lambda _: tmp_path,
        )
        assert get_model_dir("gpt2") == local

    def test_nonempty_blobs_counts_as_weights(self, tmp_path, monkeypatch):
        hf_home = tmp_path / "hf"
        weighted = hf_home / "hub" / "models--gpt2"
        blobs = weighted / "blobs"
        blobs.mkdir(parents=True)
        (blobs / "deadbeef").write_text("weight-bytes")
        monkeypatch.setenv("HF_HOME", str(hf_home))
        monkeypatch.setattr(
            "domain.infrastructure._internal.model_resolver.find_repo_root",
            lambda _: tmp_path,
        )
        assert get_model_dir("gpt2") == weighted

    def test_earlier_weighted_candidate_still_wins_over_later_one(self, tmp_path, monkeypatch):
        hf_home = tmp_path / "hf"
        weighted = hf_home / "hub" / "models--gpt2"
        snap = weighted / "snapshots" / "abc123"
        snap.mkdir(parents=True)
        (snap / "model.safetensors").write_text("fake-weights")
        self._make_local_with_safetensors(tmp_path)
        monkeypatch.setenv("HF_HOME", str(hf_home))
        monkeypatch.setattr(
            "domain.infrastructure._internal.model_resolver.find_repo_root",
            lambda _: tmp_path,
        )
        assert get_model_dir("gpt2") == weighted

    def test_empty_blobs_dir_does_not_count_as_weights(self, tmp_path, monkeypatch):
        hf_home = tmp_path / "hf"
        stub = hf_home / "hub" / "models--gpt2"
        (stub / "blobs").mkdir(parents=True)
        (stub / "refs").mkdir()
        (stub / "refs" / "main").write_text("default")
        local = self._make_local_with_safetensors(tmp_path)
        monkeypatch.setenv("HF_HOME", str(hf_home))
        monkeypatch.setattr(
            "domain.infrastructure._internal.model_resolver.find_repo_root",
            lambda _: tmp_path,
        )
        assert get_model_dir("gpt2") == local

    def test_all_stubs_falls_back_to_first_existing(self, tmp_path, monkeypatch):
        hf_home = tmp_path / "hf"
        stub = self._make_stub(hf_home)
        monkeypatch.setenv("HF_HOME", str(hf_home))
        monkeypatch.setattr(
            "domain.infrastructure._internal.model_resolver.find_repo_root",
            lambda _: tmp_path,
        )
        assert get_model_dir("gpt2") == stub


# ── find_safetensors ──────────────────────────────────────────────────────


class TestFindSafetensors:
    def test_finds_in_snapshots(self, tmp_path):
        snap = tmp_path / "snapshots" / "abc123"
        snap.mkdir(parents=True)
        st = snap / "model.safetensors"
        st.touch()
        assert find_safetensors(tmp_path) == st

    def test_finds_in_root(self, tmp_path):
        st = tmp_path / "model.safetensors"
        st.touch()
        assert find_safetensors(tmp_path) == st

    def test_returns_none_when_missing(self, tmp_path):
        assert find_safetensors(tmp_path) is None

    def test_prefers_snapshots_over_root(self, tmp_path):
        root_st = tmp_path / "model.safetensors"
        root_st.touch()
        snap = tmp_path / "snapshots" / "abc123"
        snap.mkdir(parents=True)
        snap_st = snap / "model.safetensors"
        snap_st.touch()
        assert find_safetensors(tmp_path) == snap_st


# ── load_model_config ─────────────────────────────────────────────────────


class TestLoadModelConfig:
    def test_loads_from_snapshots(self, tmp_path, monkeypatch):
        snap = tmp_path / "snapshots" / "abc123"
        snap.mkdir(parents=True)
        config = {"model_type": "llama", "hidden_size": 4096}
        (snap / "config.json").write_text(json.dumps(config))
        monkeypatch.setattr(
            "domain.infrastructure._internal.model_resolver.get_model_dir",
            lambda _: tmp_path,
        )
        result = load_model_config("test-model")
        assert result == config

    def test_loads_from_root(self, tmp_path, monkeypatch):
        config = {"model_type": "gpt2", "hidden_size": 768}
        (tmp_path / "config.json").write_text(json.dumps(config))
        monkeypatch.setattr(
            "domain.infrastructure._internal.model_resolver.get_model_dir",
            lambda _: tmp_path,
        )
        result = load_model_config("test-model")
        assert result == config

    def test_raises_when_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "domain.infrastructure._internal.model_resolver.get_model_dir",
            lambda _: tmp_path,
        )
        with pytest.raises(FileNotFoundError, match="No config.json"):
            load_model_config("test-model")
