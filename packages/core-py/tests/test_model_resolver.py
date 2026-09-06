"""Tests for model_resolver — HuggingFace cache directory resolution."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from domains.infrastructure.model_resolver import (
    get_model_dir,
    find_safetensors,
    load_model_config,
)


# ── get_model_dir ─────────────────────────────────────────────────────────


class TestGetModelDir:

    def test_returns_first_candidate_when_none_exist(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
        monkeypatch.setattr(
            "domains.infrastructure.model_resolver.find_repo_root",
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
            "domains.infrastructure.model_resolver.find_repo_root",
            lambda _: tmp_path,
        )
        result = get_model_dir("gpt2")
        assert result == model_dir

    def test_returns_existing_project_local_cache(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HF_HOME", str(tmp_path / "nonexistent"))
        project_cache = tmp_path / "models" / "hf-cache" / "hub" / "models--gpt2"
        project_cache.mkdir(parents=True)
        monkeypatch.setattr(
            "domains.infrastructure.model_resolver.find_repo_root",
            lambda _: tmp_path,
        )
        result = get_model_dir("gpt2")
        assert result == project_cache

    def test_slashed_model_id_becomes_double_dash(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
        monkeypatch.setattr(
            "domains.infrastructure.model_resolver.find_repo_root",
            lambda _: tmp_path,
        )
        result = get_model_dir("meta-llama/Llama-2-7b")
        assert result.name == "models--meta-llama--Llama-2-7b"


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
            "domains.infrastructure.model_resolver.get_model_dir",
            lambda _: tmp_path,
        )
        result = load_model_config("test-model")
        assert result == config

    def test_loads_from_root(self, tmp_path, monkeypatch):
        config = {"model_type": "gpt2", "hidden_size": 768}
        (tmp_path / "config.json").write_text(json.dumps(config))
        monkeypatch.setattr(
            "domains.infrastructure.model_resolver.get_model_dir",
            lambda _: tmp_path,
        )
        result = load_model_config("test-model")
        assert result == config

    def test_raises_when_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "domains.infrastructure.model_resolver.get_model_dir",
            lambda _: tmp_path,
        )
        with pytest.raises(FileNotFoundError, match="No config.json"):
            load_model_config("test-model")
