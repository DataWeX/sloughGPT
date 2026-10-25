"""compute_model_size_gb — online/offline resolution rules.

``offline=True`` is the first-paint contract: resolve from the local HF
cache only, never fall through to the Hub API, and never poison the shared
5-minute size cache with a miss (a later online call must still fill in
the real size).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from domain.infrastructure._internal import model_size


@pytest.fixture(autouse=True)
def _reset_size_caches():
    model_size._size_cache.clear()
    yield
    model_size._size_cache.clear()


def test_offline_miss_never_calls_hub(monkeypatch):
    def _boom(model_id: str):
        raise AssertionError("offline mode must not reach the Hub API")

    monkeypatch.setattr(model_size, "is_download_complete", lambda mid: False)
    monkeypatch.setattr(model_size, "_get_hub_file_size_gb", _boom)

    assert model_size.compute_model_size_gb("acme/remote", offline=True) is None
    assert "acme/remote" not in model_size._size_cache


def test_offline_hit_returns_local_size(monkeypatch, tmp_path: Path):
    (tmp_path / "model.safetensors").write_bytes(b"x" * 4096)
    monkeypatch.setattr(model_size, "is_download_complete", lambda mid: True)
    monkeypatch.setattr(model_size, "find_cached_model_dir", lambda mid: str(tmp_path))

    size = model_size.compute_model_size_gb("acme/local", offline=True)
    assert size is not None
    cached_ts, cached_val = model_size._size_cache["acme/local"]
    assert cached_val == size
    assert cached_ts > 0


def test_offline_miss_does_not_block_later_online_fill(monkeypatch):
    monkeypatch.setattr(model_size, "is_download_complete", lambda mid: False)
    monkeypatch.setattr(model_size, "_get_hub_file_size_gb", lambda mid: 1.5)

    assert model_size.compute_model_size_gb("acme/remote", offline=True) is None
    # online call after the offline miss still resolves through the Hub
    assert model_size.compute_model_size_gb("acme/remote") == 1.5


class TestHubSizeEstimateMatchesDownloadTruth:
    """_get_hub_file_size_gb mirrors list_model_files (card f74361ea).

    Dual-format repos ship the same weights as safetensors AND .bin; the
    downloader fetches only safetensors when present, so the listing must
    count only bytes that would actually be downloaded.
    """

    GIB = 1024**3

    @staticmethod
    def _estimate(monkeypatch, siblings):
        from domain.infrastructure._internal import hf_hub

        monkeypatch.setattr(hf_hub, "fetch_model_info", lambda mid: {"siblings": siblings})
        return model_size._get_hub_file_size_gb("org/model")

    def test_dual_format_counts_only_safetensors(self, monkeypatch):
        assert (
            self._estimate(
                monkeypatch,
                [
                    {"rfilename": "model.safetensors", "size": 5 * self.GIB},
                    {"rfilename": "pytorch_model.bin", "size": 3 * self.GIB},
                    {"rfilename": "config.json", "size": 10},
                ],
            )
            == 5.0
        )

    def test_bin_only_repo_still_counts_bin(self, monkeypatch):
        assert (
            self._estimate(
                monkeypatch,
                [
                    {"rfilename": "config.json", "size": 10},
                    {"rfilename": "pytorch_model.bin", "size": 3 * self.GIB},
                ],
            )
            == 3.0
        )

    def test_repo_without_weights_returns_none(self, monkeypatch):
        assert (
            self._estimate(
                monkeypatch,
                [{"rfilename": "config.json", "size": 10}],
            )
            is None
        )
