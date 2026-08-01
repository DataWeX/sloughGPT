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

from domains.infrastructure.hf_hub import (
    HFFile,
    _derive_model_id,
    _matches_ignore,
    _strip_incomplete_suffix,
    download_hf_model,
    find_cached_model_dir,
    get_cache_dir,
    inspect_incomplete,
    is_download_complete,
    list_missing_files,
    list_model_files,
    resume_download,
    resume_plan,
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
