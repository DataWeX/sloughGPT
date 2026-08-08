"""Tests for domains.infrastructure.hf_hub — HuggingFace Hub layer.

The HuggingFace-specific workflows (``list_model_files``,
``is_download_complete``, ``download_hf_model``, ``resume_*``,
``verify_model``, ``list_missing_files``) moved here from the old
``downcraft.hf_hub`` / ``downcraft.resume`` modules when downcraft became
a HuggingFace-agnostic generic downloader.  Resume tests use a real local
HTTP server with ``Range`` support (no mocks).
"""

import hashlib
import os
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

import pytest
import requests

from domains.infrastructure.hf_hub import (
    HFFile,
    ResumeInfo,
    _deep_check,
    _derive_model_id,
    _ensure_sgpart,
    _find_snapshot_dir,
    _has_weight_files,
    _hf_api_get,
    _make_hf_chunk_cb,
    _matches_ignore,
    _match_repo_file,
    _snapshot_dir,
    _strip_incomplete_suffix,
    download_hf_model,
    fetch_dataset_search,
    fetch_model_info,
    find_cached_model_dir,
    get_cache_dir,
    inspect_incomplete,
    is_download_complete,
    list_missing_files,
    list_model_files,
    resolve_cached_path,
    resume_download,
    resume_model,
    resume_plan,
    verify_model,
)


class TestMatchesIgnore:
    def test_h5_ignored(self):
        assert _matches_ignore("model.h5") is True

    def test_onnx_ignored(self):
        assert _matches_ignore("model.onnx") is True

    def test_gguf_ignored(self):
        assert _matches_ignore("model.gguf") is True

    def test_msgpack_ignored(self):
        assert _matches_ignore("model.msgpack") is True

    def test_tflite_ignored(self):
        assert _matches_ignore("model.tflite") is True

    def test_ot_ignored(self):
        assert _matches_ignore("model.ot") is True

    def test_safetensors_not_ignored(self):
        assert _matches_ignore("model.safetensors") is False

    def test_bin_not_ignored(self):
        assert _matches_ignore("pytorch_model.bin") is False

    def test_json_not_ignored(self):
        assert _matches_ignore("config.json") is False

    def test_onnx_subdirectory_ignored(self):
        assert _matches_ignore("onnx/model.onnx") is True

    def test_tf_subdirectory_ignored(self):
        assert _matches_ignore("tf/variables") is True

    def test_regular_subdirectory_not_ignored(self):
        assert _matches_ignore("not-onnx/file.bin") is False

    def test_case_sensitive(self):
        assert _matches_ignore("Model.ONNX") is False  # fnmatch is case-sensitive on Linux/Mac


class TestGetCacheDir:
    def test_default_path(self):
        cache = get_cache_dir("gpt2")
        assert "models--gpt2" in str(cache)

    def test_with_org(self):
        cache = get_cache_dir("Qwen/Qwen2.5-0.5B-Instruct")
        assert "models--Qwen--Qwen2.5-0.5B-Instruct" in str(cache)

    def test_respects_hf_home(self):
        cache = get_cache_dir("gpt2", hf_home="/custom/hf")
        # When hf_home is provided, it is used directly as the base
        assert str(cache) == os.path.join("/custom/hf", "models--gpt2")

    def test_respects_hf_home_no_trailing(self):
        cache = get_cache_dir("gpt2", hf_home="/custom")
        assert str(cache) == os.path.join("/custom", "models--gpt2")


class TestIsDownloadComplete:
    def test_nonexistent_cache(self):
        assert is_download_complete("fake-model-nonexistent", hf_home="/tmp/nonexistent_hf_cache_xyz") is False

    def test_empty_cache_dir(self):
        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td) / "hub" / "models--test-model"
            cache_dir.mkdir(parents=True)
            assert is_download_complete("test-model", hf_home=str(Path(td) / "hub")) is False

    def test_no_refs_main(self):
        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td) / "hub" / "models--test-model"
            cache_dir.mkdir(parents=True)
            (cache_dir / "snapshots" / "abc123").mkdir(parents=True)
            assert is_download_complete("test-model", hf_home=str(Path(td) / "hub")) is False

    def test_no_snapshot(self):
        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td) / "hub" / "models--test-model"
            cache_dir.mkdir(parents=True)
            refs = cache_dir / "refs"
            refs.mkdir()
            (refs / "main").write_text("abc123")
            assert is_download_complete("test-model", hf_home=str(Path(td) / "hub")) is False

    def test_complete_with_safetensors(self):
        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td) / "hub" / "models--test-model"
            cache_dir.mkdir(parents=True)
            refs = cache_dir / "refs"
            refs.mkdir()
            (refs / "main").write_text("abc123")
            snap = cache_dir / "snapshots" / "abc123"
            snap.mkdir(parents=True)
            (snap / "model.safetensors").write_bytes(b"x" * 2000)
            assert is_download_complete("test-model", hf_home=str(Path(td) / "hub")) is True

    def test_complete_with_bin(self):
        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td) / "hub" / "models--test-model"
            cache_dir.mkdir(parents=True)
            refs = cache_dir / "refs"
            refs.mkdir()
            (refs / "main").write_text("abc123")
            snap = cache_dir / "snapshots" / "abc123"
            snap.mkdir(parents=True)
            (snap / "pytorch_model.bin").write_bytes(b"x" * 2000)
            assert is_download_complete("test-model", hf_home=str(Path(td) / "hub")) is True

    def test_incomplete_marker(self):
        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td) / "hub" / "models--test-model"
            cache_dir.mkdir(parents=True)
            refs = cache_dir / "refs"
            refs.mkdir()
            (refs / "main").write_text("abc123")
            snap = cache_dir / "snapshots" / "abc123"
            snap.mkdir(parents=True)
            (snap / "model.safetensors").write_bytes(b"x" * 2000)
            (cache_dir / ".incomplete").write_text("incomplete")
            assert is_download_complete("test-model", hf_home=str(Path(td) / "hub")) is False

    def test_sgpart_temp_file_blocks_snapshot(self):
        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td) / "hub" / "models--test-model"
            cache_dir.mkdir(parents=True)
            refs = cache_dir / "refs"
            refs.mkdir()
            (refs / "main").write_text("abc123")
            snap = cache_dir / "snapshots" / "abc123"
            snap.mkdir(parents=True)
            (snap / "model.safetensors").write_bytes(b"x" * 2000)
            (snap / "model.safetensors.sgpart").write_bytes(b"x" * 500)
            assert is_download_complete("test-model", hf_home=str(Path(td) / "hub")) is False

    def test_lock_file_returns_false(self):
        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td) / "hub" / "models--test-model"
            cache_dir.mkdir(parents=True)
            refs = cache_dir / "refs"
            refs.mkdir()
            (refs / "main").write_text("abc123")
            snap = cache_dir / "snapshots" / "abc123"
            snap.mkdir(parents=True)
            (snap / "model.safetensors").write_bytes(b"x" * 2000)
            (cache_dir / "some.lock").write_text("locked")
            assert is_download_complete("test-model", hf_home=str(Path(td) / "hub")) is False

    def test_small_file_less_than_1kb_not_counted(self):
        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td) / "hub" / "models--test-model"
            cache_dir.mkdir(parents=True)
            refs = cache_dir / "refs"
            refs.mkdir()
            (refs / "main").write_text("abc123")
            snap = cache_dir / "snapshots" / "abc123"
            snap.mkdir(parents=True)
            (snap / "model.safetensors").write_bytes(b"x" * 500)  # < 1KB
            assert is_download_complete("test-model", hf_home=str(Path(td) / "hub")) is False

    def test_flat_layout_complete(self):
        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td) / "hub" / "models--test-model"
            cache_dir.mkdir(parents=True)
            (cache_dir / "config.json").write_text("{}")
            (cache_dir / "model.safetensors").write_bytes(b"x" * 2000)
            assert is_download_complete("test-model", hf_home=str(Path(td) / "hub")) is True

    def test_flat_layout_with_refs_and_stale_locks(self):
        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td) / "hub" / "models--test-model"
            cache_dir.mkdir(parents=True)
            refs = cache_dir / "refs"
            refs.mkdir()
            (refs / "main").write_text("abc123")
            (cache_dir / "config.json").write_text("{}")
            (cache_dir / "model.safetensors").write_bytes(b"x" * 2000)
            dl = cache_dir / ".cache" / "huggingface" / "download"
            dl.mkdir(parents=True)
            (dl / "model.safetensors.lock").touch()
            assert is_download_complete("test-model", hf_home=str(Path(td) / "hub")) is True

    def test_flat_layout_active_lock_blocks(self):
        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td) / "hub" / "models--test-model"
            cache_dir.mkdir(parents=True)
            (cache_dir / "config.json").write_text("{}")
            (cache_dir / "model.safetensors").write_bytes(b"x" * 2000)
            dl = cache_dir / ".cache" / "huggingface" / "download"
            dl.mkdir(parents=True)
            (dl / "extra.bin.lock").touch()  # target missing -> active download
            assert is_download_complete("test-model", hf_home=str(Path(td) / "hub")) is False

    def test_flat_layout_missing_config_incomplete(self):
        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td) / "hub" / "models--test-model"
            cache_dir.mkdir(parents=True)
            (cache_dir / "model.safetensors").write_bytes(b"x" * 2000)
            assert is_download_complete("test-model", hf_home=str(Path(td) / "hub")) is False

    def test_flat_layout_incomplete_marker_blocks(self):
        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td) / "hub" / "models--test-model"
            cache_dir.mkdir(parents=True)
            (cache_dir / "config.json").write_text("{}")
            (cache_dir / "model.safetensors").write_bytes(b"x" * 2000)
            (cache_dir / "model.safetensors.incomplete").write_text("partial")
            assert is_download_complete("test-model", hf_home=str(Path(td) / "hub")) is False

    def test_flat_layout_sgpart_temp_file_blocks(self):
        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td) / "hub" / "models--test-model"
            cache_dir.mkdir(parents=True)
            (cache_dir / "config.json").write_text("{}")
            (cache_dir / "model.safetensors").write_bytes(b"x" * 2000)
            (cache_dir / "model.safetensors.sgpart").write_bytes(b"x" * 500)
            assert is_download_complete("test-model", hf_home=str(Path(td) / "hub")) is False

    def test_flat_layout_no_snapshot_but_refs(self):
        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td) / "hub" / "models--test-model"
            cache_dir.mkdir(parents=True)
            refs = cache_dir / "refs"
            refs.mkdir()
            (refs / "main").write_text("abc123")
            (cache_dir / "config.json").write_text("{}")
            (cache_dir / "model.safetensors").write_bytes(b"x" * 2000)
            assert is_download_complete("test-model", hf_home=str(Path(td) / "hub")) is True


class TestFindCachedModelDir:
    def test_returns_standard_dir_when_exists(self, tmp_path):
        cache_dir = tmp_path / "hub" / "models--org--model"
        cache_dir.mkdir(parents=True)
        result = find_cached_model_dir("org/model", hf_home=str(tmp_path / "hub"))
        assert result == cache_dir

    def test_returns_none_when_absent(self, tmp_path):
        assert find_cached_model_dir("org/model", hf_home=str(tmp_path / "hub")) is None


class TestListModelFiles:
    def test_real_model_returns_files(self):
        files = list_model_files("gpt2")
        assert len(files) > 0

        # Should include config, tokenizer, and model weights
        paths = [f.path for f in files]
        assert "config.json" in paths
        assert any("model" in p for p in paths)

        # Should exclude ignored formats
        assert not any(f.is_ignored for f in files if "model.safetensors" in f.path)

    def test_real_model_has_sizes(self):
        files = list_model_files("gpt2")
        safetensors = [f for f in files if f.path.endswith(".safetensors")]
        if safetensors:
            assert safetensors[0].size > 0
            assert safetensors[0].checksum != ""

    def test_real_model_has_download_urls(self):
        files = list_model_files("gpt2")
        non_ignored = [f for f in files if not f.is_ignored]
        assert len(non_ignored) > 0
        for f in non_ignored:
            assert f.download_url.startswith("https://")
            assert "gpt2" in f.download_url

    def test_ignored_files_have_empty_url(self):
        files = list_model_files("gpt2")
        ignored = [f for f in files if f.is_ignored]
        for f in ignored:
            assert f.download_url == ""


# ---------------------------------------------------------------------------
# Resume helpers (moved from downcraft.resume)
# ---------------------------------------------------------------------------

def _payload(size: int) -> bytes:
    pattern = b"0123456789abcdef"
    return (pattern * (size // len(pattern) + 1))[:size]


def _file(path: str, size: int, url: str = "http://x") -> HFFile:
    return HFFile(
        path=path,
        size=size,
        checksum=hashlib.sha256(_payload(size)).hexdigest(),
        download_url=url,
    )


# ---------------------------------------------------------------------------
# Real local HTTP server with Range support
# ---------------------------------------------------------------------------

class _RangeHandler(BaseHTTPRequestHandler):
    payload = b""

    def do_GET(self):
        start = 0
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            spec = rng[len("bytes="):].split("-")[0]
            if spec.isdigit():
                start = int(spec)
        data = self.payload[start:]
        if start > 0:
            self.send_response(206)
            self.send_header(
                "Content-Range",
                f"bytes {start}-{len(self.payload) - 1}/{len(self.payload)}",
            )
        else:
            self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("ETag", '"static"')
        self.end_headers()
        for i in range(0, len(data), 2048):
            self.wfile.write(data[i:i + 2048])

    def log_message(self, *args):
        pass


@pytest.fixture
def range_server():
    _RangeHandler.payload = _payload(12 * 1024 * 1024)  # 12 MB, > 1 chunk
    server = HTTPServer(("127.0.0.1", 0), _RangeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=5)


class TestStripIncompleteSuffix:
    def test_sgpart(self):
        assert _strip_incomplete_suffix("model.safetensors.sgpart") == "model.safetensors"

    def test_incomplete(self):
        assert _strip_incomplete_suffix("model.safetensors.incomplete") == "model.safetensors"

    def test_plain_name_unchanged(self):
        assert _strip_incomplete_suffix("model.safetensors") == "model.safetensors"


class TestDeriveModelId:
    def test_from_snapshot_path(self, tmp_path):
        partial = (
            tmp_path
            / "hub" / "models--Qwen--Qwen2.5-0.5B-Instruct"
            / "snapshots" / "default" / "model.safetensors.sgpart"
        )
        partial.parent.mkdir(parents=True)
        partial.write_bytes(b"x" * 100)
        assert _derive_model_id(partial) == "Qwen/Qwen2.5-0.5B-Instruct"

    def test_from_flat_cache_path(self, tmp_path):
        partial = tmp_path / "models--org--model" / "config.json.incomplete"
        partial.parent.mkdir(parents=True)
        partial.write_bytes(b"{}")
        assert _derive_model_id(partial) == "org/model"

    def test_none_outside_cache(self, tmp_path):
        partial = tmp_path / "downloads" / "file.sgpart"
        partial.parent.mkdir(parents=True)
        partial.write_bytes(b"x")
        assert _derive_model_id(partial) is None


class TestInspectIncomplete:
    def test_matches_shard_and_reports_offset(self, tmp_path):
        files = [_file("model-00001-of-00003.safetensors", 1_000_000, "http://x/model-1")]
        partial = tmp_path / "model-00001-of-00003.safetensors.sgpart"
        partial.write_bytes(b"y" * 400_000)

        info = inspect_incomplete(partial, model_id="org/model", files=files)

        assert info.model_id == "org/model"
        assert info.repo_path == "model-00001-of-00003.safetensors"
        assert info.resume_offset == 400_000
        assert info.total_bytes == 1_000_000
        assert info.download_url == "http://x/model-1"
        assert info.complete is False

    def test_incomplete_suffix(self, tmp_path):
        files = [_file("model.safetensors", 1000, "http://x/model")]
        partial = tmp_path / "model.safetensors.incomplete"
        partial.write_bytes(b"z" * 250)

        info = inspect_incomplete(partial, model_id="org/model", files=files)

        assert info.repo_path == "model.safetensors"
        assert info.resume_offset == 250
        assert info.final_path == tmp_path / "model.safetensors"

    def test_model_id_derived_from_path(self, tmp_path):
        files = [_file("model.safetensors", 1000)]
        partial = tmp_path / "models--org--model" / "model.safetensors.sgpart"
        partial.parent.mkdir(parents=True)
        partial.write_bytes(b"x" * 500)

        info = inspect_incomplete(partial, files=files)

        assert info.model_id == "org/model"

    def test_complete_partial(self, tmp_path):
        files = [_file("model.safetensors", 1000)]
        partial = tmp_path / "model.safetensors.sgpart"
        partial.write_bytes(b"x" * 1000)

        info = inspect_incomplete(partial, model_id="org/model", files=files)

        assert info.complete is True
        assert info.resume_offset == 1000

    def test_no_match_raises(self, tmp_path):
        files = [_file("model.safetensors", 1000)]
        partial = tmp_path / "other.bin.sgpart"
        partial.write_bytes(b"x")

        with pytest.raises(ValueError, match="does not match any file"):
            inspect_incomplete(partial, model_id="org/model", files=files)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            inspect_incomplete(tmp_path / "absent.sgpart", model_id="org/model")

    def test_requires_model_id_outside_cache(self, tmp_path):
        partial = tmp_path / "model.safetensors.sgpart"
        partial.write_bytes(b"x")
        with pytest.raises(ValueError, match="model_id is required"):
            inspect_incomplete(partial, files=[_file("model.safetensors", 1000)])


class TestResumePlan:
    def _cache(self, tmp_path):
        return tmp_path / "hub" / "models--org--model"

    def test_scans_all_marker_types(self, tmp_path):
        cache = self._cache(tmp_path)
        cache.mkdir(parents=True)
        (cache / "model.safetensors.sgpart").write_bytes(b"a" * 5_000_000)
        (cache / "config.json.incomplete").write_bytes(b"{}")
        (cache / "shard.bin").write_bytes(b"b" * 100)  # under-sized final

        files = [
            _file("model.safetensors", 12_000_000),
            _file("config.json", 500),
            _file("shard.bin", 1000),
        ]

        plan = resume_plan("org/model", hf_home=str(tmp_path / "hub"), files=files)

        by_path = {i.repo_path: i for i in plan}
        assert set(by_path) == {"config.json", "model.safetensors", "shard.bin"}
        assert by_path["model.safetensors"].resume_offset == 5_000_000
        assert by_path["config.json"].resume_offset == 2
        assert by_path["shard.bin"].resume_offset == 100
        assert by_path["shard.bin"].complete is False

    def test_sorted_by_repo_path(self, tmp_path):
        cache = self._cache(tmp_path)
        cache.mkdir(parents=True)
        (cache / "zz.bin.sgpart").write_bytes(b"x" * 10)
        (cache / "aa.bin.sgpart").write_bytes(b"y" * 20)
        files = [_file("aa.bin", 100), _file("zz.bin", 100)]

        plan = resume_plan("org/model", hf_home=str(tmp_path / "hub"), files=files)

        assert [i.repo_path for i in plan] == ["aa.bin", "zz.bin"]

    def test_empty_when_not_cached(self, tmp_path):
        assert resume_plan("org/model", hf_home=str(tmp_path / "hub")) == []

    def test_ignores_complete_marker(self, tmp_path):
        cache = self._cache(tmp_path)
        cache.mkdir(parents=True)
        (cache / "model.safetensors").write_bytes(b"x" * 1000)  # at expected size
        files = [_file("model.safetensors", 1000)]

        assert resume_plan("org/model", hf_home=str(tmp_path / "hub"), files=files) == []


class TestResumeDownload:
    def _server_files(self, server, name="model.safetensors"):
        payload = _RangeHandler.payload
        url = f"http://127.0.0.1:{server.server_port}/{name}"
        return [
            HFFile(
                path=name,
                size=len(payload),
                checksum=hashlib.sha256(payload).hexdigest(),
                download_url=url,
            )
        ]

    def test_resumes_sgpart_at_exact_offset(self, tmp_path, range_server):
        payload = _RangeHandler.payload
        files = self._server_files(range_server)

        partial = tmp_path / "model.safetensors.sgpart"
        partial.write_bytes(payload[: 5 * 1024 * 1024])  # 5 MB of 12 MB

        final = resume_download(partial, model_id="org/model", files=files)

        assert final == tmp_path / "model.safetensors"
        assert final.read_bytes() == payload
        assert not partial.exists()

    def test_resumes_incomplete_suffix(self, tmp_path, range_server):
        payload = _RangeHandler.payload
        files = self._server_files(range_server)

        partial = tmp_path / "model.safetensors.incomplete"
        partial.write_bytes(payload[: 3 * 1024 * 1024])

        resume_download(partial, model_id="org/model", files=files)

        assert (tmp_path / "model.safetensors").read_bytes() == payload
        assert not (tmp_path / "model.safetensors.incomplete").exists()

    def test_complete_partial_renamed_without_network(self, tmp_path):
        payload = _payload(1024)
        files = [
            HFFile(
                path="model.safetensors",
                size=len(payload),
                checksum=hashlib.sha256(payload).hexdigest(),
                download_url="http://127.0.0.1:1/unreachable",
            )
        ]
        partial = tmp_path / "model.safetensors.sgpart"
        partial.write_bytes(payload)

        resume_download(partial, model_id="org/model", files=files)

        assert (tmp_path / "model.safetensors").read_bytes() == payload

    def test_final_name_partial_resumed(self, tmp_path, range_server):
        payload = _RangeHandler.payload
        files = self._server_files(range_server)

        partial = tmp_path / "model.safetensors"  # under-sized final name
        partial.write_bytes(payload[: 1024 * 1024])

        resume_download(partial, model_id="org/model", files=files)

        assert partial.read_bytes() == payload

    def test_reports_resume_in_on_chunk(self, tmp_path, range_server):
        payload = _RangeHandler.payload
        files = self._server_files(range_server)

        partial = tmp_path / "model.safetensors.sgpart"
        partial.write_bytes(payload[: 4 * 1024 * 1024])

        seen = []
        resume_download(
            partial,
            model_id="org/model",
            files=files,
            on_chunk=lambda b, t: seen.append((b, t)),
        )

        assert seen
        assert all(b >= 4 * 1024 * 1024 for b, _ in seen)
        assert all(t == len(payload) for _, t in seen)


class TestDownloadHfModelRef:
    def test_writes_refs_main_and_reports_complete(self, tmp_path, range_server, monkeypatch):
        import domains.infrastructure.hf_hub as hub_mod
        from downcraft import state as state_mod

        payload = _RangeHandler.payload
        url = f"http://127.0.0.1:{range_server.server_port}/model.safetensors"
        model_id = "org/model"
        hf_home = str(tmp_path / "hub")

        monkeypatch.setattr(
            hub_mod, "list_model_files",
            lambda mid: [
                HFFile(
                    path="model.safetensors",
                    size=len(payload),
                    checksum=hashlib.sha256(payload).hexdigest(),
                    download_url=url,
                )
            ],
        )
        monkeypatch.setattr(
            state_mod, "get_state",
            lambda: state_mod.PersistentState(state_dir=tmp_path / "state"),
        )

        result = download_hf_model(model_id, hf_home=hf_home)

        assert result["status"] == "complete"
        cache = hub_mod.get_cache_dir(model_id, hf_home)
        assert (cache / "refs" / "main").read_text() == "default"
        assert (cache / "snapshots" / "default" / "model.safetensors").read_bytes() == payload
        assert is_download_complete(model_id, hf_home=hf_home) is True

    def test_stale_state_redownloads(self, tmp_path, range_server, monkeypatch):
        import domains.infrastructure.hf_hub as hub_mod
        from downcraft import state as state_mod

        payload = _RangeHandler.payload
        url = f"http://127.0.0.1:{range_server.server_port}/model.safetensors"
        model_id = "org/model"
        hf_home = str(tmp_path / "hub")

        monkeypatch.setattr(
            hub_mod, "list_model_files",
            lambda mid: [
                HFFile(
                    path="model.safetensors",
                    size=len(payload),
                    checksum=hashlib.sha256(payload).hexdigest(),
                    download_url=url,
                )
            ],
        )
        st = state_mod.PersistentState(state_dir=tmp_path / "state")
        stale = st.create(model_id, str(hub_mod.get_cache_dir(model_id, hf_home)))
        stale.status = "complete"
        monkeypatch.setattr(state_mod, "get_state", lambda: st)

        result = download_hf_model(model_id, hf_home=hf_home)

        assert result["status"] == "complete"
        assert (
            hub_mod.get_cache_dir(model_id, hf_home)
            / "snapshots" / "default" / "model.safetensors"
        ).read_bytes() == payload


class TestListMissingFiles:
    def test_nonexistent_model_returns_all_files(self):
        with patch("domains.infrastructure.hf_hub.list_model_files") as mock_list:
            mock_list.return_value = [
                HFFile(path="config.json", size=100, checksum="abc", download_url="https://x.com/cfg"),
                HFFile(path="model.safetensors", size=1000, checksum="def", download_url="https://x.com/model"),
            ]
            missing = list_missing_files("test-model", hf_home="/tmp/fake_hf_home")
            assert len(missing) == 2
            assert "config.json" in missing

    def test_missing_files_when_snap_missing(self):
        """When snapshot doesn't exist, all files should be reported missing."""
        with patch("domains.infrastructure.hf_hub.list_model_files") as mock_list:
            mock_list.return_value = [
                HFFile(path="weights.bin", size=500, checksum="abc", download_url="https://x.com/w"),
            ]
            missing = list_missing_files("no-snap-model", hf_home="/tmp/fake_hf_home")
            assert len(missing) == 1
            assert "weights.bin" in missing


def _resume_info(tmp_path, **overrides):
    fields = dict(
        model_id="org/model",
        repo_path="model.safetensors",
        partial_path=tmp_path / "model.safetensors.sgpart",
        final_path=tmp_path / "model.safetensors",
        resume_offset=0,
        total_bytes=1000,
        download_url="http://x/model",
        checksum="c",
        complete=False,
    )
    fields.update(overrides)
    return ResumeInfo(**fields)


class TestHfApiGet:
    def test_request_exception_returns_none(self, monkeypatch):
        import domains.infrastructure.hf_hub as hub_mod

        def boom(*args, **kwargs):
            raise requests.ConnectionError("network down")

        monkeypatch.setattr(hub_mod.requests, "get", boom)
        assert _hf_api_get("models/gpt2") is None

    def test_success_returns_json(self, monkeypatch):
        import domains.infrastructure.hf_hub as hub_mod

        class _Resp:
            def raise_for_status(self):
                return None

            def json(self):
                return {"id": "gpt2", "siblings": []}

        monkeypatch.setattr(hub_mod.requests, "get", lambda *a, **k: _Resp())
        assert _hf_api_get("models/gpt2") == {"id": "gpt2", "siblings": []}

    def test_fetch_model_info_non_dict_returns_none(self, monkeypatch):
        import domains.infrastructure.hf_hub as hub_mod

        monkeypatch.setattr(hub_mod, "_hf_api_get", lambda *a, **k: ["not", "a", "dict"])
        assert fetch_model_info("gpt2") is None

    def test_fetch_model_info_dict_returns_data(self, monkeypatch):
        import domains.infrastructure.hf_hub as hub_mod

        monkeypatch.setattr(hub_mod, "_hf_api_get", lambda *a, **k: {"id": "gpt2"})
        assert fetch_model_info("gpt2") == {"id": "gpt2"}


class TestFetchDatasetSearch:
    def test_non_list_returns_empty(self, monkeypatch):
        import domains.infrastructure.hf_hub as hub_mod

        monkeypatch.setattr(hub_mod, "_hf_api_get", lambda *a, **k: {"id": "x"})
        assert fetch_dataset_search("cats") == []

    def test_builds_results_skipping_non_dicts(self, monkeypatch):
        import domains.infrastructure.hf_hub as hub_mod

        monkeypatch.setattr(
            hub_mod,
            "_hf_api_get",
            lambda *a, **k: [
                {"id": "a", "downloads": 5},
                "junk",
                {"id": "b", "downloads": None},
                {},
            ],
        )
        assert fetch_dataset_search("cats") == [
            {"id": "a", "downloads": 5},
            {"id": "b", "downloads": 0},
            {"id": "", "downloads": 0},
        ]


class TestListModelFilesEdges:
    def test_no_info_returns_empty(self, monkeypatch):
        import domains.infrastructure.hf_hub as hub_mod

        monkeypatch.setattr(hub_mod, "fetch_model_info", lambda mid: None)
        assert list_model_files("gpt2") == []

    def test_skips_non_dict_siblings_and_dotfiles(self, monkeypatch):
        import domains.infrastructure.hf_hub as hub_mod

        monkeypatch.setattr(
            hub_mod,
            "fetch_model_info",
            lambda mid: {
                "siblings": [
                    "junk",
                    {"rfilename": ".hidden", "size": 1},
                    {"rfilename": "model.safetensors", "size": 10, "lfs": {"sha256": "abc"}},
                ]
            },
        )
        files = list_model_files("gpt2")
        assert [f.path for f in files] == ["model.safetensors"]
        assert files[0].checksum == "abc"


class TestProjectCacheRoots:
    def test_break_at_filesystem_root(self, monkeypatch):
        import domains.infrastructure.hf_hub as hub_mod

        monkeypatch.setattr(
            hub_mod.Path, "cwd", staticmethod(lambda: Path("/"))
        )
        assert hub_mod._project_cache_roots() == []


class TestSnapshotDirEdges:
    def test_read_error_returns_none(self, tmp_path, monkeypatch):
        import domains.infrastructure.hf_hub as hub_mod

        cache_dir = tmp_path / "models--org--model"
        refs_main = cache_dir / "refs" / "main"
        refs_main.parent.mkdir(parents=True)
        refs_main.write_text("abc")
        orig = Path.read_text

        def fake_read_text(self, *a, **k):
            if str(self) == str(refs_main):
                raise OSError("boom")
            return orig(self, *a, **k)

        monkeypatch.setattr(hub_mod.Path, "read_text", fake_read_text)
        assert _snapshot_dir(cache_dir) is None

    def test_empty_commit_returns_none(self, tmp_path):
        cache_dir = tmp_path / "models--org--model"
        refs_main = cache_dir / "refs" / "main"
        refs_main.parent.mkdir(parents=True)
        refs_main.write_text("   ")
        assert _snapshot_dir(cache_dir) is None

    def test_missing_snapshot_dir_returns_none(self, tmp_path):
        cache_dir = tmp_path / "models--org--model"
        refs_main = cache_dir / "refs" / "main"
        refs_main.parent.mkdir(parents=True)
        refs_main.write_text("abc")
        assert _snapshot_dir(cache_dir) is None


class TestHasWeightFiles:
    def test_stat_error_skipped(self, tmp_path, monkeypatch):
        import domains.infrastructure.hf_hub as hub_mod

        cache_dir = tmp_path / "models--org--model"
        target = cache_dir / "model.safetensors"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"x" * 2000)
        orig_stat = Path.stat

        def fake_stat(self, follow_symlinks=True):
            if str(self) == str(target):
                raise OSError("boom")
            return orig_stat(self, follow_symlinks=follow_symlinks)

        monkeypatch.setattr(hub_mod.Path, "stat", fake_stat)
        assert _has_weight_files(cache_dir) is False

    def test_returns_false_without_weights(self, tmp_path):
        cache_dir = tmp_path / "models--org--model"
        (cache_dir / "config.json").mkdir(parents=True)
        assert _has_weight_files(cache_dir) is False


class TestDeepCheck:
    def test_no_info_returns_true(self, monkeypatch):
        import domains.infrastructure.hf_hub as hub_mod

        monkeypatch.setattr(hub_mod, "fetch_model_info", lambda mid: None)
        assert _deep_check("org/model", Path("/nope")) is True

    def test_missing_local_file_returns_false(self, tmp_path, monkeypatch):
        import domains.infrastructure.hf_hub as hub_mod

        monkeypatch.setattr(
            hub_mod,
            "fetch_model_info",
            lambda mid: {"siblings": [{"rfilename": "model.safetensors", "size": 2000}]},
        )
        assert _deep_check("org/model", tmp_path) is False

    def test_size_mismatch_returns_false(self, tmp_path, monkeypatch):
        import domains.infrastructure.hf_hub as hub_mod

        (tmp_path / "model.safetensors").write_bytes(b"x" * 1000)
        monkeypatch.setattr(
            hub_mod,
            "fetch_model_info",
            lambda mid: {"siblings": [{"rfilename": "model.safetensors", "size": 2000}]},
        )
        assert _deep_check("org/model", tmp_path) is False

    def test_size_match_returns_true(self, tmp_path, monkeypatch):
        import domains.infrastructure.hf_hub as hub_mod

        (tmp_path / "model.safetensors").write_bytes(b"x" * 2000)
        monkeypatch.setattr(
            hub_mod,
            "fetch_model_info",
            lambda mid: {"siblings": [{"rfilename": "model.safetensors", "size": 2000}]},
        )
        assert _deep_check("org/model", tmp_path) is True

    def test_stat_error_returns_false(self, tmp_path, monkeypatch):
        import domains.infrastructure.hf_hub as hub_mod

        target = tmp_path / "model.safetensors"
        target.write_bytes(b"x" * 2000)
        orig_stat = Path.stat
        calls = {"n": 0}

        def fake_stat(self, follow_symlinks=True):
            if str(self) == str(target):
                calls["n"] += 1
                if calls["n"] > 1:
                    raise OSError("boom")
            return orig_stat(self, follow_symlinks=follow_symlinks)

        monkeypatch.setattr(hub_mod.Path, "stat", fake_stat)
        monkeypatch.setattr(
            hub_mod,
            "fetch_model_info",
            lambda mid: {"siblings": [{"rfilename": "model.safetensors", "size": 2000}]},
        )
        assert _deep_check("org/model", tmp_path) is False

    def test_skips_non_weight_siblings(self, tmp_path, monkeypatch):
        import domains.infrastructure.hf_hub as hub_mod

        monkeypatch.setattr(
            hub_mod,
            "fetch_model_info",
            lambda mid: {
                "siblings": [
                    "junk",
                    {"rfilename": ".hidden", "size": 10},
                    {"rfilename": "config.json", "size": 10},
                    {"rfilename": "onnx/model.onnx", "size": 10},
                ]
            },
        )
        assert _deep_check("org/model", tmp_path) is True

    def test_info_exception_returns_true(self, tmp_path, monkeypatch):
        import domains.infrastructure.hf_hub as hub_mod

        def boom(mid):
            raise RuntimeError("api down")

        monkeypatch.setattr(hub_mod, "fetch_model_info", boom)
        assert _deep_check("org/model", tmp_path) is True


class TestDeepCheckComplete:
    def test_snapshot_layout_fails_deep_check(self, tmp_path, monkeypatch):
        import domains.infrastructure.hf_hub as hub_mod

        cache_dir = hub_mod.get_cache_dir("org/model", str(tmp_path / "hub"))
        (cache_dir / "refs").mkdir(parents=True)
        (cache_dir / "refs" / "main").write_text("abc123")
        snap = cache_dir / "snapshots" / "abc123"
        snap.mkdir(parents=True)
        (snap / "model.safetensors").write_bytes(b"x" * 2000)
        monkeypatch.setattr(hub_mod, "_deep_check", lambda *a, **k: False)
        assert is_download_complete("org/model", hf_home=str(tmp_path / "hub"), deep_check=True) is False

    def test_flat_layout_fails_deep_check(self, tmp_path, monkeypatch):
        import domains.infrastructure.hf_hub as hub_mod

        cache_dir = hub_mod.get_cache_dir("org/model", str(tmp_path / "hub"))
        cache_dir.mkdir(parents=True)
        (cache_dir / "config.json").write_text("{}")
        (cache_dir / "model.safetensors").write_bytes(b"x" * 2000)
        monkeypatch.setattr(hub_mod, "_deep_check", lambda *a, **k: False)
        assert is_download_complete("org/model", hf_home=str(tmp_path / "hub"), deep_check=True) is False

    def test_snapshot_layout_passes_deep_check(self, tmp_path, monkeypatch):
        import domains.infrastructure.hf_hub as hub_mod

        cache_dir = hub_mod.get_cache_dir("org/model", str(tmp_path / "hub"))
        (cache_dir / "refs").mkdir(parents=True)
        (cache_dir / "refs" / "main").write_text("abc123")
        snap = cache_dir / "snapshots" / "abc123"
        snap.mkdir(parents=True)
        (snap / "model.safetensors").write_bytes(b"x" * 2000)
        monkeypatch.setattr(hub_mod, "_deep_check", lambda *a, **k: True)
        assert is_download_complete("org/model", hf_home=str(tmp_path / "hub"), deep_check=True) is True


class TestResolveCachedPath:
    def test_snapshot_layout(self, tmp_path):
        model_id = "org/model"
        hf_home = str(tmp_path / "hub")
        cache_dir = get_cache_dir(model_id, hf_home)
        (cache_dir / "refs").mkdir(parents=True)
        (cache_dir / "refs" / "main").write_text("abc")
        snap = cache_dir / "snapshots" / "abc"
        snap.mkdir(parents=True)
        (snap / "model.safetensors").write_bytes(b"x" * 2000)
        assert resolve_cached_path(model_id, "model.safetensors", hf_home=hf_home) == snap / "model.safetensors"

    def test_flat_layout(self, tmp_path):
        model_id = "org/model"
        hf_home = str(tmp_path / "hub")
        cache_dir = get_cache_dir(model_id, hf_home)
        cache_dir.mkdir(parents=True)
        (cache_dir / "model.safetensors").write_bytes(b"x" * 2000)
        assert resolve_cached_path(model_id, "model.safetensors", hf_home=hf_home) == cache_dir / "model.safetensors"

    def test_prefers_snapshot_over_flat(self, tmp_path):
        model_id = "org/model"
        hf_home = str(tmp_path / "hub")
        cache_dir = get_cache_dir(model_id, hf_home)
        (cache_dir / "refs").mkdir(parents=True)
        (cache_dir / "refs" / "main").write_text("abc")
        snap = cache_dir / "snapshots" / "abc"
        snap.mkdir(parents=True)
        (snap / "model.safetensors").write_bytes(b"x" * 2000)
        (cache_dir / "model.safetensors").write_bytes(b"y" * 2000)
        assert resolve_cached_path(model_id, "model.safetensors", hf_home=hf_home) == snap / "model.safetensors"

    def test_small_file_ignored(self, tmp_path):
        model_id = "org/model"
        hf_home = str(tmp_path / "hub")
        cache_dir = get_cache_dir(model_id, hf_home)
        cache_dir.mkdir(parents=True)
        (cache_dir / "model.safetensors").write_bytes(b"x" * 500)
        assert resolve_cached_path(model_id, "model.safetensors", hf_home=hf_home) is None

    def test_none_when_absent(self, tmp_path):
        assert resolve_cached_path("org/model", "model.safetensors", hf_home=str(tmp_path / "hub")) is None

    def test_stat_error_passes(self, tmp_path, monkeypatch):
        import domains.infrastructure.hf_hub as hub_mod

        model_id = "org/model"
        hf_home = str(tmp_path / "hub")
        cache_dir = get_cache_dir(model_id, hf_home)
        cache_dir.mkdir(parents=True)
        target = cache_dir / "model.safetensors"
        target.write_bytes(b"x" * 2000)
        orig_stat = Path.stat
        calls = {"n": 0}

        def fake_stat(self, follow_symlinks=True):
            if str(self) == str(target):
                calls["n"] += 1
                if calls["n"] > 1:
                    raise OSError("boom")
            return orig_stat(self, follow_symlinks=follow_symlinks)

        monkeypatch.setattr(hub_mod.Path, "stat", fake_stat)
        assert resolve_cached_path(model_id, "model.safetensors", hf_home=hf_home) is None


class TestDownloadHfModelEdges:
    def test_already_cached(self, tmp_path, monkeypatch):
        import domains.infrastructure.hf_hub as hub_mod
        from downcraft import state as state_mod

        model_id = "org/model"
        hf_home = str(tmp_path / "hub")
        cache_dir = hub_mod.get_cache_dir(model_id, hf_home)
        (cache_dir / "refs").mkdir(parents=True)
        (cache_dir / "refs" / "main").write_text("default")
        snap = cache_dir / "snapshots" / "default"
        snap.mkdir(parents=True)
        (snap / "model.safetensors").write_bytes(b"x" * 2000)
        st = state_mod.PersistentState(state_dir=tmp_path / "state")
        ms = st.create(model_id, str(cache_dir))
        ms.status = "complete"
        monkeypatch.setattr(state_mod, "get_state", lambda: st)

        result = download_hf_model(model_id, hf_home=hf_home)

        assert result["status"] == "already_cached"
        assert result["cache_dir"] == ms.cache_dir

    def test_stale_complete_state_redownloads(self, tmp_path, monkeypatch):
        import domains.infrastructure.hf_hub as hub_mod
        from downcraft import state as state_mod

        model_id = "org/model"
        hf_home = str(tmp_path / "hub")
        st = state_mod.PersistentState(state_dir=tmp_path / "state")
        ms = st.create(model_id, str(hub_mod.get_cache_dir(model_id, hf_home)))
        ms.status = "complete"
        monkeypatch.setattr(state_mod, "get_state", lambda: st)
        monkeypatch.setattr(hub_mod, "list_model_files", lambda mid: [])
        with pytest.raises(RuntimeError, match="No downloadable files"):
            download_hf_model(model_id, hf_home=hf_home)

    def test_no_files_raises(self, tmp_path, monkeypatch):
        import domains.infrastructure.hf_hub as hub_mod
        from downcraft import state as state_mod

        monkeypatch.setattr(hub_mod, "list_model_files", lambda mid: [])
        monkeypatch.setattr(
            state_mod, "get_state",
            lambda: state_mod.PersistentState(state_dir=tmp_path / "state"),
        )
        with pytest.raises(RuntimeError, match="No downloadable files"):
            download_hf_model("org/model", hf_home=str(tmp_path / "hub"))

    def test_disk_truth_skips_download(self, tmp_path, range_server, monkeypatch):
        import domains.infrastructure.hf_hub as hub_mod
        from downcraft import state as state_mod

        payload = _RangeHandler.payload
        url = f"http://127.0.0.1:{range_server.server_port}/model.safetensors"
        model_id = "org/model"
        hf_home = str(tmp_path / "hub")
        cache_dir = hub_mod.get_cache_dir(model_id, hf_home)
        dest = cache_dir / "snapshots" / "default" / "model.safetensors"
        dest.parent.mkdir(parents=True)
        dest.write_bytes(payload)
        monkeypatch.setattr(
            hub_mod, "list_model_files",
            lambda mid: [HFFile(
                path="model.safetensors",
                size=len(payload),
                checksum=hashlib.sha256(payload).hexdigest(),
                download_url=url,
            )],
        )
        monkeypatch.setattr(
            state_mod, "get_state",
            lambda: state_mod.PersistentState(state_dir=tmp_path / "state"),
        )

        result = download_hf_model(model_id, hf_home=hf_home)

        assert result["status"] == "complete"
        assert dest.read_bytes() == payload

    def test_existing_complete_in_state_skips(self, tmp_path, range_server, monkeypatch):
        import domains.infrastructure.hf_hub as hub_mod
        from downcraft import state as state_mod

        payload = _RangeHandler.payload
        url = f"http://127.0.0.1:{range_server.server_port}/model.safetensors"
        model_id = "org/model"
        hf_home = str(tmp_path / "hub")
        st = state_mod.PersistentState(state_dir=tmp_path / "state")
        st.create(model_id, str(hub_mod.get_cache_dir(model_id, hf_home)))
        st.update_file_progress(
            model_id, "model.safetensors", url, len(payload), len(payload),
            checksum="c", complete=True,
        )
        monkeypatch.setattr(
            hub_mod, "list_model_files",
            lambda mid: [HFFile(
                path="model.safetensors",
                size=len(payload),
                checksum=hashlib.sha256(payload).hexdigest(),
                download_url=url,
            )],
        )
        monkeypatch.setattr(state_mod, "get_state", lambda: st)

        result = download_hf_model(model_id, hf_home=hf_home)

        assert result["status"] == "complete"
        dest = hub_mod.get_cache_dir(model_id, hf_home) / "snapshots" / "default" / "model.safetensors"
        assert not dest.exists()

    def test_incomplete_marker_resumed_via_sgpart(self, tmp_path, range_server, monkeypatch):
        import domains.infrastructure.hf_hub as hub_mod
        from downcraft import state as state_mod

        payload = _RangeHandler.payload
        url = f"http://127.0.0.1:{range_server.server_port}/model.safetensors"
        model_id = "org/model"
        hf_home = str(tmp_path / "hub")
        cache_dir = hub_mod.get_cache_dir(model_id, hf_home)
        snap_dir = cache_dir / "snapshots" / "default"
        snap_dir.mkdir(parents=True)
        (snap_dir / "model.safetensors.incomplete").write_bytes(payload[: 2 * 1024 * 1024])
        monkeypatch.setattr(
            hub_mod, "list_model_files",
            lambda mid: [HFFile(
                path="model.safetensors",
                size=len(payload),
                checksum=hashlib.sha256(payload).hexdigest(),
                download_url=url,
            )],
        )
        monkeypatch.setattr(
            state_mod, "get_state",
            lambda: state_mod.PersistentState(state_dir=tmp_path / "state"),
        )

        result = download_hf_model(model_id, hf_home=hf_home)

        assert result["status"] == "complete"
        assert (snap_dir / "model.safetensors").read_bytes() == payload
        assert not (snap_dir / "model.safetensors.incomplete").exists()

    def test_download_error_marks_failed(self, tmp_path, monkeypatch):
        import domains.infrastructure.hf_hub as hub_mod
        from downcraft import state as state_mod
        from downcraft.downloader import DownloadError

        model_id = "org/model"
        hf_home = str(tmp_path / "hub")
        st = state_mod.PersistentState(state_dir=tmp_path / "state")
        monkeypatch.setattr(
            hub_mod, "list_model_files",
            lambda mid: [HFFile(path="model.safetensors", size=100, checksum="abc", download_url="http://x/model")],
        )
        monkeypatch.setattr(state_mod, "get_state", lambda: st)

        def boom(**kwargs):
            raise DownloadError("boom")

        monkeypatch.setattr(hub_mod, "download_file", boom)

        with pytest.raises(DownloadError):
            download_hf_model(model_id, hf_home=hf_home)

        assert st.get(model_id).status == "failed"


class TestMakeHfChunkCb:
    def test_on_progress_called_with_totals(self, monkeypatch):
        import domains.infrastructure.hf_hub as hub_mod

        class FakeState:
            def __init__(self):
                self.updates = []

            def update_file_progress(self, *a, **k):
                self.updates.append((a, k))

            def get(self, model_id):
                return None

        monkeypatch.setattr(hub_mod.time, "time", lambda: 100.0)
        st = FakeState()
        calls = []
        cb = _make_hf_chunk_cb(
            st, "org/model", "model.safetensors", "http://x",
            1000, "abc", 0.0, 4000,
            lambda m, b, t, s: calls.append((m, b, t, s)),
        )
        cb(500, 1000)
        assert calls == [("org/model", 500, 4000, 5.0)]
        assert st.updates

    def test_uses_state_total_when_available(self, monkeypatch):
        import domains.infrastructure.hf_hub as hub_mod
        from downcraft.state import FileProgress, ModelState

        monkeypatch.setattr(hub_mod.time, "time", lambda: 100.0)

        class FakeState:
            def __init__(self):
                self.ms = ModelState(model_id="org/model")
                self.ms.files["model.safetensors"] = FileProgress(
                    path="model.safetensors", url="u", bytes_downloaded=200, total_bytes=1000,
                )

            def update_file_progress(self, *a, **k):
                pass

            def get(self, model_id):
                return self.ms

        st = FakeState()
        calls = []
        cb = _make_hf_chunk_cb(
            st, "org/model", "other.bin", "http://x",
            1000, "abc", 0.0, 4000,
            lambda m, b, t, s: calls.append((m, b, t, s)),
        )
        cb(500, 1000)
        assert calls[0][:3] == ("org/model", 200, 4000)


class TestMatchRepoFile:
    def test_exact_path_match(self):
        f = HFFile(path="model.safetensors", size=10, checksum="c", download_url="u")
        assert _match_repo_file("model.safetensors", [f]) == f

    def test_basename_match(self):
        f = HFFile(path="subdir/model.safetensors", size=10, checksum="c", download_url="u")
        assert _match_repo_file("model.safetensors", [f]) == f

    def test_no_match(self):
        f = HFFile(path="model.safetensors", size=10, checksum="c", download_url="u")
        assert _match_repo_file("other.bin", [f]) is None


class TestEnsureSgpart:
    def test_already_sgpart(self, tmp_path):
        p = tmp_path / "model.safetensors.sgpart"
        p.write_bytes(b"x" * 10)
        info = _resume_info(tmp_path, partial_path=p)
        assert _ensure_sgpart(info) == p
        assert p.read_bytes() == b"x" * 10

    def test_sgpart_exists_partial_smaller(self, tmp_path):
        sgpart = tmp_path / "model.safetensors.sgpart"
        partial = tmp_path / "model.safetensors.incomplete"
        sgpart.write_bytes(b"y" * 500)
        partial.write_bytes(b"z" * 100)
        info = _resume_info(tmp_path, partial_path=partial)
        assert _ensure_sgpart(info) == sgpart
        assert not partial.exists()
        assert sgpart.read_bytes() == b"y" * 500

    def test_sgpart_exists_partial_larger(self, tmp_path):
        sgpart = tmp_path / "model.safetensors.sgpart"
        partial = tmp_path / "model.safetensors.incomplete"
        sgpart.write_bytes(b"y" * 100)
        partial.write_bytes(b"z" * 500)
        info = _resume_info(tmp_path, partial_path=partial)
        assert _ensure_sgpart(info) == sgpart
        assert not partial.exists()
        assert sgpart.read_bytes() == b"z" * 500

    def test_no_sgpart_renames_partial(self, tmp_path):
        partial = tmp_path / "model.safetensors.incomplete"
        partial.write_bytes(b"z" * 100)
        info = _resume_info(tmp_path, partial_path=partial)
        result = _ensure_sgpart(info)
        assert result == tmp_path / "model.safetensors.sgpart"
        assert not partial.exists()


class TestInspectIncompleteFetchesFiles:
    def test_fetches_files_when_none(self, tmp_path, monkeypatch):
        import domains.infrastructure.hf_hub as hub_mod

        files = [_file("model.safetensors", 1000)]
        monkeypatch.setattr(hub_mod, "list_model_files", lambda mid: files)
        partial = tmp_path / "model.safetensors.sgpart"
        partial.write_bytes(b"x" * 100)
        info = inspect_incomplete(partial, model_id="org/model")
        assert info.repo_path == "model.safetensors"
        assert info.resume_offset == 100


class TestResumePlanEdges:
    def test_cache_dir_not_a_dir(self, tmp_path, monkeypatch):
        import domains.infrastructure.hf_hub as hub_mod

        f = tmp_path / "models--org--model"
        f.write_bytes(b"x")
        monkeypatch.setattr(hub_mod, "find_cached_model_dir", lambda *a, **k: f)
        assert resume_plan("org/model") == []

    def test_fetches_files_when_omitted(self, tmp_path, monkeypatch):
        import domains.infrastructure.hf_hub as hub_mod

        cache = tmp_path / "hub" / "models--org--model"
        cache.mkdir(parents=True)
        (cache / "model.safetensors.sgpart").write_bytes(b"x" * 5_000_000)
        files = [_file("model.safetensors", 12_000_000)]
        monkeypatch.setattr(hub_mod, "list_model_files", lambda mid: files)
        plan = resume_plan("org/model", hf_home=str(tmp_path / "hub"))
        assert [i.repo_path for i in plan] == ["model.safetensors"]
        assert plan[0].resume_offset == 5_000_000

    def test_skips_non_file_marker(self, tmp_path):
        cache = tmp_path / "hub" / "models--org--model"
        cache.mkdir(parents=True)
        d = cache / "config.json.sgpart"
        d.mkdir()
        files = [_file("model.safetensors", 1000)]
        assert resume_plan("org/model", hf_home=str(tmp_path / "hub"), files=files) == []

    def test_skips_zero_size_repo_file(self, tmp_path):
        cache = tmp_path / "hub" / "models--org--model"
        cache.mkdir(parents=True)
        files = [HFFile(path="config.json", size=0, checksum="", download_url="u")]
        assert resume_plan("org/model", hf_home=str(tmp_path / "hub"), files=files) == []


class TestResumeDownloadEdges:
    def test_final_already_complete_drops_partial(self, tmp_path):
        payload = _payload(1024)
        files = [HFFile(
            path="model.safetensors", size=len(payload),
            checksum=hashlib.sha256(payload).hexdigest(),
            download_url="http://127.0.0.1:1/x",
        )]
        partial = tmp_path / "model.safetensors.sgpart"
        partial.write_bytes(b"partial")
        final = tmp_path / "model.safetensors"
        final.write_bytes(payload)
        done = []
        result = resume_download(
            partial, model_id="org/model", files=files,
            on_complete=lambda p: done.append(p),
        )
        assert result == final
        assert not partial.exists()
        assert done == [final]

    def test_final_stat_oserror_falls_through_to_download(self, tmp_path, monkeypatch):
        import domains.infrastructure.hf_hub as hub_mod

        payload = _payload(1024)
        files = [HFFile(
            path="model.safetensors", size=len(payload),
            checksum="c", download_url="u",
        )]
        partial = tmp_path / "model.safetensors.sgpart"
        partial.write_bytes(b"partial")
        final = tmp_path / "model.safetensors"
        final.write_bytes(b"x" * 2000)
        orig_stat = Path.stat
        calls = {"n": 0}

        def fake_stat(self, follow_symlinks=True):
            if str(self) == str(final):
                calls["n"] += 1
                if calls["n"] > 1:
                    raise OSError("boom")
            return orig_stat(self, follow_symlinks=follow_symlinks)

        monkeypatch.setattr(hub_mod.Path, "stat", fake_stat)
        called = {}
        monkeypatch.setattr(
            hub_mod, "download_file",
            lambda **kw: called.update(kw) or Path("/fake"),
        )
        result = resume_download(partial, model_id="org/model", files=files)
        assert result == final
        assert "url" in called

    def test_complete_partial_on_complete_callback(self, tmp_path):
        payload = _payload(1024)
        files = [HFFile(
            path="model.safetensors", size=len(payload),
            checksum=hashlib.sha256(payload).hexdigest(),
            download_url="http://127.0.0.1:1/x",
        )]
        partial = tmp_path / "model.safetensors.sgpart"
        partial.write_bytes(payload)
        done = []
        result = resume_download(
            partial, model_id="org/model", files=files,
            on_complete=lambda p: done.append(p),
        )
        assert result == tmp_path / "model.safetensors"
        assert done == [tmp_path / "model.safetensors"]


class TestResumeModel:
    def test_resumes_plan_then_downloads(self, tmp_path, monkeypatch):
        import domains.infrastructure.hf_hub as hub_mod

        partial = tmp_path / "model.safetensors.sgpart"
        partial.write_bytes(b"x" * 100)
        info = _resume_info(tmp_path, partial_path=partial, resume_offset=100, complete=False)
        monkeypatch.setattr(hub_mod, "resume_plan", lambda mid, hf_home=None: [info])
        resumed = []
        monkeypatch.setattr(
            hub_mod, "resume_download",
            lambda pp, model_id=None, hf_home=None, on_complete=None: resumed.append((pp, model_id, on_complete)),
        )
        base = {"status": "complete", "model_id": "org/model", "cache_dir": "/x", "elapsed": 1.0, "total_bytes": 1000}
        monkeypatch.setattr(hub_mod, "download_hf_model", lambda *a, **k: dict(base))

        result = resume_model("org/model", hf_home=str(tmp_path / "hub"))

        assert resumed[0][:2] == (partial, "org/model")
        assert callable(resumed[0][2])
        assert result["resumed_files"] == ["model.safetensors"]
        assert result["status"] == "complete"

    def test_empty_plan_just_downloads(self, tmp_path, monkeypatch):
        import domains.infrastructure.hf_hub as hub_mod

        monkeypatch.setattr(hub_mod, "resume_plan", lambda mid, hf_home=None: [])
        monkeypatch.setattr(
            hub_mod, "download_hf_model",
            lambda *a, **k: {"status": "complete", "resumed_files": []},
        )
        result = resume_model("org/model", hf_home=str(tmp_path / "hub"))
        assert result["resumed_files"] == []
        assert result["status"] == "complete"


class TestFindSnapshotDir:
    def test_none_when_cache_missing(self, tmp_path):
        assert _find_snapshot_dir("org/model", hf_home=str(tmp_path / "hub")) is None

    def test_none_when_refs_missing(self, tmp_path):
        cache = tmp_path / "hub" / "models--org--model"
        cache.mkdir(parents=True)
        assert _find_snapshot_dir("org/model", hf_home=str(tmp_path / "hub")) is None

    def test_none_when_commit_empty(self, tmp_path):
        cache = tmp_path / "hub" / "models--org--model"
        (cache / "refs").mkdir(parents=True)
        (cache / "refs" / "main").write_text("   ")
        assert _find_snapshot_dir("org/model", hf_home=str(tmp_path / "hub")) is None

    def test_none_when_snapshot_missing(self, tmp_path):
        cache = tmp_path / "hub" / "models--org--model"
        (cache / "refs").mkdir(parents=True)
        (cache / "refs" / "main").write_text("abc")
        assert _find_snapshot_dir("org/model", hf_home=str(tmp_path / "hub")) is None

    def test_returns_snapshot(self, tmp_path):
        cache = tmp_path / "hub" / "models--org--model"
        (cache / "refs").mkdir(parents=True)
        (cache / "refs" / "main").write_text("abc")
        snap = cache / "snapshots" / "abc"
        snap.mkdir(parents=True)
        assert _find_snapshot_dir("org/model", hf_home=str(tmp_path / "hub")) == snap


class TestVerifyModel:
    def _snap(self, tmp_path):
        import domains.infrastructure.hf_hub as hub_mod

        cache = hub_mod.get_cache_dir("org/model", str(tmp_path / "hub"))
        (cache / "refs").mkdir(parents=True)
        (cache / "refs" / "main").write_text("abc")
        snap = cache / "snapshots" / "abc"
        snap.mkdir(parents=True)
        return snap

    def test_not_found_returns_false(self, tmp_path, monkeypatch):
        import domains.infrastructure.hf_hub as hub_mod

        monkeypatch.setattr(hub_mod, "_find_snapshot_dir", lambda *a, **k: None)
        assert verify_model("org/model") is False

    def test_no_checksums_returns_false(self, tmp_path, monkeypatch):
        import domains.infrastructure.hf_hub as hub_mod

        snap = self._snap(tmp_path)
        monkeypatch.setattr(hub_mod, "_find_snapshot_dir", lambda *a, **k: snap)
        monkeypatch.setattr(
            hub_mod, "list_model_files",
            lambda mid: [HFFile(path="config.json", size=10, checksum="", download_url="u")],
        )
        assert verify_model("org/model") is False

    def test_missing_file_returns_false(self, tmp_path, monkeypatch):
        import domains.infrastructure.hf_hub as hub_mod

        snap = self._snap(tmp_path)
        monkeypatch.setattr(hub_mod, "_find_snapshot_dir", lambda *a, **k: snap)
        monkeypatch.setattr(
            hub_mod, "list_model_files",
            lambda mid: [HFFile(path="model.safetensors", size=10, checksum="abc", download_url="u")],
        )
        assert verify_model("org/model") is False

    def test_checksum_mismatch_returns_false(self, tmp_path, monkeypatch):
        import domains.infrastructure.hf_hub as hub_mod

        snap = self._snap(tmp_path)
        (snap / "model.safetensors").write_bytes(b"x" * 10)
        monkeypatch.setattr(hub_mod, "_find_snapshot_dir", lambda *a, **k: snap)
        monkeypatch.setattr(
            hub_mod, "list_model_files",
            lambda mid: [HFFile(path="model.safetensors", size=10, checksum="abc", download_url="u")],
        )
        monkeypatch.setattr(hub_mod, "_sha256_of", lambda p: "def")
        assert verify_model("org/model") is False

    def test_read_error_returns_false(self, tmp_path, monkeypatch):
        import domains.infrastructure.hf_hub as hub_mod

        snap = self._snap(tmp_path)
        (snap / "model.safetensors").write_bytes(b"x" * 10)
        monkeypatch.setattr(hub_mod, "_find_snapshot_dir", lambda *a, **k: snap)
        monkeypatch.setattr(
            hub_mod, "list_model_files",
            lambda mid: [HFFile(path="model.safetensors", size=10, checksum="abc", download_url="u")],
        )
        monkeypatch.setattr(hub_mod, "_sha256_of", lambda p: (_ for _ in ()).throw(OSError("boom")))
        assert verify_model("org/model") is False

    def test_all_ok_returns_true(self, tmp_path, monkeypatch):
        import domains.infrastructure.hf_hub as hub_mod

        snap = self._snap(tmp_path)
        (snap / "model.safetensors").write_bytes(b"x" * 10)
        monkeypatch.setattr(hub_mod, "_find_snapshot_dir", lambda *a, **k: snap)
        monkeypatch.setattr(
            hub_mod, "list_model_files",
            lambda mid: [HFFile(path="model.safetensors", size=10, checksum="abc", download_url="u")],
        )
        monkeypatch.setattr(hub_mod, "_sha256_of", lambda p: "abc")
        assert verify_model("org/model") is True


class TestListMissingFilesWithSnapshot:
    def test_reports_missing_and_bad_checksums(self, tmp_path, monkeypatch):
        import domains.infrastructure.hf_hub as hub_mod
        from downcraft.verify import _sha256_of

        model_id = "org/model"
        hf_home = str(tmp_path / "hub")
        cache = hub_mod.get_cache_dir(model_id, hf_home)
        (cache / "refs").mkdir(parents=True)
        (cache / "refs" / "main").write_text("abc")
        snap = cache / "snapshots" / "abc"
        snap.mkdir(parents=True)
        good = snap / "config.json"
        good.write_text("{}")
        bad = snap / "model.safetensors"
        bad.write_bytes(b"x" * 100)
        files = [
            HFFile(path="config.json", size=10, checksum=_sha256_of(good), download_url="u"),
            HFFile(path="model.safetensors", size=1000, checksum="mismatch", download_url="u"),
            HFFile(path="missing.bin", size=100, checksum="m", download_url="u"),
            HFFile(path="tf/model.onnx", size=10, checksum="", download_url="", is_ignored=True),
        ]
        monkeypatch.setattr(hub_mod, "list_model_files", lambda mid: files)
        assert list_missing_files(model_id, hf_home=hf_home) == ["model.safetensors", "missing.bin"]
