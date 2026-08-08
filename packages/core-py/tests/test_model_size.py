"""Tests for the shared model size calculator."""

import time

import numpy as np
import pytest

from domains.infrastructure import model_size as ms


@pytest.fixture
def clear_caches():
    ms._size_cache.clear()
    ms._cached_check_cache.clear()
    yield
    ms._size_cache.clear()
    ms._cached_check_cache.clear()


@pytest.fixture
def fake_clock(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(ms.time, "monotonic", lambda: now[0])
    return now


def write_weight_file(path, size):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.truncate(size)


class TestSumWeightFiles:
    def test_sums_safetensors_and_bin(self, tmp_path):
        write_weight_file(tmp_path / "model.safetensors", 500 * 1024**2)
        write_weight_file(tmp_path / "model.bin", 700 * 1024**2)
        assert ms._sum_weight_files(tmp_path) == round(1200 * 1024**2 / 1024**3, 2)

    def test_recurses_subdirs(self, tmp_path):
        write_weight_file(tmp_path / "sub" / "w.bin", 500 * 1024**2)
        assert ms._sum_weight_files(tmp_path) == round(500 * 1024**2 / 1024**3, 2)

    def test_ignores_small_files(self, tmp_path):
        write_weight_file(tmp_path / "tiny.bin", 500)
        assert ms._sum_weight_files(tmp_path) is None

    def test_ignores_other_extensions(self, tmp_path):
        write_weight_file(tmp_path / "config.json", 5000)
        assert ms._sum_weight_files(tmp_path) is None

    def test_rounds_to_two_decimals(self, tmp_path):
        write_weight_file(tmp_path / "w.safetensors", 1500 * 1024 * 1024)
        assert ms._sum_weight_files(tmp_path) == 1.46

    def test_empty_dir_returns_none(self, tmp_path):
        assert ms._sum_weight_files(tmp_path) is None

    def test_stat_error_ignored(self, tmp_path, monkeypatch):
        class _StatBoom:
            def stat(self):
                raise OSError("boom")

        monkeypatch.setattr(ms.Path, "rglob", lambda self, pattern: iter([_StatBoom()]))
        assert ms._sum_weight_files(tmp_path) is None


class _Sib:
    def __init__(self, rfilename, size=None):
        self.rfilename = rfilename
        self.size = size

    def to_dict(self):
        return {"rfilename": self.rfilename, "size": self.size}


def _model_info(*siblings):
    return {"siblings": [s.to_dict() for s in siblings]}


def _patch_fetch(monkeypatch, info):
    import domains.infrastructure.hf_hub as hub
    monkeypatch.setattr(hub, "fetch_model_info", lambda model_id: info)


class TestGetHubFileSize:
    def test_sums_weight_siblings(self, monkeypatch):
        _patch_fetch(monkeypatch, _model_info(
            _Sib("model.safetensors", 1000 * 1024 * 1024),
            _Sib("model-00001-of-00002.safetensors", 500 * 1024 * 1024),
            _Sib("config.json", 2048),
            _Sib("tokenizer.bin", 512 * 1024 * 1024),
        ))
        result = ms._get_hub_file_size_gb("org/model")
        assert result == round((1000 + 500 + 512) * 1024 * 1024 / 1024**3, 2)

    def test_ignores_none_size(self, monkeypatch):
        _patch_fetch(monkeypatch, _model_info(_Sib("model.safetensors", None)))
        assert ms._get_hub_file_size_gb("org/model") is None

    def test_no_siblings_returns_none(self, monkeypatch):
        _patch_fetch(monkeypatch, _model_info())
        assert ms._get_hub_file_size_gb("org/model") is None

    def test_hub_error_returns_none(self, monkeypatch):
        import domains.infrastructure.hf_hub as hub

        def boom(model_id):
            raise RuntimeError("network down")

        monkeypatch.setattr(hub, "fetch_model_info", boom)
        assert ms._get_hub_file_size_gb("org/model") is None

    def test_ignores_non_dict_siblings(self, monkeypatch):
        _patch_fetch(monkeypatch, {"siblings": [
            "not-a-dict",
            {"rfilename": "model.safetensors", "size": 500 * 1024**2},
        ]})
        assert ms._get_hub_file_size_gb("org/model") == round(500 * 1024**2 / 1024**3, 2)


class TestComputeModelSize:
    def test_uses_local_cache_when_complete(self, monkeypatch, tmp_path, clear_caches):
        write_weight_file(tmp_path / "model.safetensors", 500 * 1024**2)
        monkeypatch.setattr(ms, "is_download_complete", lambda model_id: True)
        monkeypatch.setattr(ms, "get_cache_dir", lambda model_id: tmp_path)
        assert ms.compute_model_size_gb("org/model") == round(500 * 1024**2 / 1024**3, 2)

    def test_uses_flat_project_cache_dir(self, monkeypatch, tmp_path, clear_caches):
        write_weight_file(tmp_path / "model.safetensors", 500 * 1024**2)
        monkeypatch.setattr(ms, "is_download_complete", lambda model_id: True)
        monkeypatch.setattr(ms, "find_cached_model_dir", lambda model_id: tmp_path)
        assert ms.compute_model_size_gb("org/model") == round(500 * 1024**2 / 1024**3, 2)

    def test_falls_back_to_hub_when_not_cached(self, monkeypatch, clear_caches):
        monkeypatch.setattr(ms, "is_download_complete", lambda model_id: False)
        monkeypatch.setattr(ms, "_get_hub_file_size_gb", lambda model_id: 1.5)
        assert ms.compute_model_size_gb("org/model") == 1.5

    def test_caches_result(self, monkeypatch, clear_caches, fake_clock):
        calls = []
        monkeypatch.setattr(ms, "is_download_complete", lambda model_id: (calls.append(1) or False))
        monkeypatch.setattr(ms, "_get_hub_file_size_gb", lambda model_id: 2.0)
        assert ms.compute_model_size_gb("org/model") == 2.0
        assert ms.compute_model_size_gb("org/model") == 2.0
        assert len(calls) == 1

    def test_cache_expires_after_ttl(self, monkeypatch, clear_caches, fake_clock):
        calls = []
        monkeypatch.setattr(ms, "is_download_complete", lambda model_id: (calls.append(1) or False))
        monkeypatch.setattr(ms, "_get_hub_file_size_gb", lambda model_id: 2.0)
        ms.compute_model_size_gb("org/model")
        fake_clock[0] += ms._SIZE_CACHE_TTL + 1
        ms.compute_model_size_gb("org/model")
        assert len(calls) == 2

    def test_none_result_cached(self, monkeypatch, clear_caches, fake_clock):
        calls = []
        monkeypatch.setattr(ms, "is_download_complete", lambda model_id: (calls.append(1) or False))
        monkeypatch.setattr(ms, "_get_hub_file_size_gb", lambda model_id: None)
        assert ms.compute_model_size_gb("org/model") is None
        assert ms.compute_model_size_gb("org/model") is None
        assert len(calls) == 1


class TestIsModelCached:
    def test_shallow_check_delegates(self, monkeypatch, clear_caches):
        monkeypatch.setattr(ms, "is_download_complete", lambda model_id: True)
        assert ms.is_model_cached("org/model") is True

    def test_deep_check_passes_flag(self, monkeypatch, clear_caches):
        seen = {}
        monkeypatch.setattr(
            ms, "is_download_complete",
            lambda model_id, deep_check=False: seen.setdefault("deep", deep_check) or True,
        )
        assert ms.is_model_cached("org/model", deep_check=True) is True
        assert seen["deep"] is True

    def test_result_cached(self, monkeypatch, clear_caches, fake_clock):
        calls = []
        monkeypatch.setattr(ms, "is_download_complete", lambda model_id: (calls.append(1) or False))
        ms.is_model_cached("org/model")
        ms.is_model_cached("org/model")
        assert len(calls) == 1

    def test_deep_check_not_cached(self, monkeypatch, clear_caches):
        calls = []
        monkeypatch.setattr(
            ms, "is_download_complete",
            lambda model_id, deep_check=False: (calls.append(1) or False),
        )
        ms.is_model_cached("org/model", deep_check=True)
        ms.is_model_cached("org/model", deep_check=True)
        assert len(calls) == 2


class TestImportFallback:
    def test_hf_hub_import_error_uses_fallback(self, monkeypatch):
        import importlib
        import sys

        monkeypatch.setitem(sys.modules, "domains.infrastructure.hf_hub", None)
        importlib.reload(ms)
        assert ms.is_download_complete("org/model") is False
        assert ms.get_cache_dir("org/model") == "~/.cache/huggingface/hub/models--org--model/"
        assert ms.find_cached_model_dir("org/model") is None


class TestFormatting:
    def test_format_size_gb_value(self):
        assert ms.format_size_gb(1.5) == "1.50 GB"

    def test_format_size_gb_decimals(self):
        assert ms.format_size_gb(1.555, decimals=1) == "1.6 GB"

    def test_format_size_gb_none(self):
        assert ms.format_size_gb(None) == "\u2014"

    def test_format_size_mb(self):
        assert ms.format_size_mb(0.5) == 512.0

    def test_format_size_mb_none(self):
        assert ms.format_size_mb(None) is None
