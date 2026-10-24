"""Tests for domain.files.FileHub — token auth, pull-through, sha256 dedup."""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from domain.files import FileHub, FileStore

PAYLOAD = b"weights-placeholder-" + bytes(range(256)) * 128  # incompressible


class _Handler(BaseHTTPRequestHandler):
    payload = PAYLOAD

    def do_GET(self):  # noqa: N802 - http.server naming
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.payload)))
        self.end_headers()
        self.wfile.write(self.payload)

    def log_message(self, *args):  # silence test output
        pass


@pytest.fixture
def upstream():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/model.gguf"
    server.shutdown()
    thread.join(timeout=5)


@pytest.fixture
def hub(tmp_path):
    return FileHub(store=FileStore(root=tmp_path / "uploads"), token="s3cret")


class TestTokenAuth:
    def test_empty_token_is_open(self, tmp_path):
        h = FileHub(store=FileStore(root=tmp_path / "u"), token="")
        assert h.check_token(None) and h.check_token("anything")

    def test_token_match(self, hub):
        assert hub.check_token("s3cret")

    def test_token_mismatch(self, hub):
        assert not hub.check_token("wrong")
        assert not hub.check_token(None)
        assert not hub.check_token("")


class TestPullThrough:
    def test_pull_stores_blob(self, hub, upstream):
        meta = hub.pull(upstream, "model.gguf")
        assert meta["deduped"] is False
        assert meta["size_bytes"] == len(PAYLOAD)
        assert meta["codec"] == "raw"  # model class — proven incompressible
        data, _ = hub.store.get(meta["file_id"])
        assert data == PAYLOAD

    def test_pull_derives_filename_from_url(self, hub, upstream):
        meta = hub.pull(upstream)
        assert meta["original_name"] == "model.gguf"

    def test_pull_requires_extension(self, hub, upstream):
        with pytest.raises(ValueError, match="extension"):
            hub.pull(upstream, "noext")

    def test_pull_dedup_by_sha(self, hub, upstream):
        first = hub.pull(upstream, "model.gguf")
        second = hub.pull(upstream, "model.gguf")
        assert second["deduped"] is True
        assert second["file_id"] == first["file_id"]
        # one blob on disk, one metadata entry
        assert len(hub.store.list()) == 1
