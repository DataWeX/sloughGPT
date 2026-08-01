"""Tests for downcraft.resume — incomplete-download inspection and resume.

Uses a real local HTTP server with ``Range`` support (no mocks) to prove
that a partial file is resumed at its exact byte offset.
"""

import hashlib
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from downcraft import download_hf_model
from downcraft.hf_hub import HFFile, is_download_complete
from downcraft.resume import (
    _derive_model_id,
    _strip_incomplete_suffix,
    inspect_incomplete,
    resume_download,
    resume_plan,
)


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


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# inspect_incomplete
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# resume_plan
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# resume_download (real HTTP, real resume)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# download_hf_model end-to-end: refs/main + completeness
# ---------------------------------------------------------------------------

class TestDownloadHfModelRef:
    def test_writes_refs_main_and_reports_complete(self, tmp_path, range_server, monkeypatch):
        import downcraft.hf_hub as hub_mod
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
        import downcraft.hf_hub as hub_mod
        from downcraft import state as state_mod
        from downcraft.state import ModelState

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
