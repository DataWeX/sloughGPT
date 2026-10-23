"""Tests for the compressed files router — upload/download codec paths.

Exercises contracts: files.upload_file, files.download_file, files.get_file,
files.list_files, files.search_files, files.delete_file.
"""

from __future__ import annotations

import os

import pytest
from routers.files import FilesRouter, _get_db

from apps.api.server.tests.test_support import get_test_client

client = get_test_client()
_router = FilesRouter()
_upload_dir = _router.UPLOADS_DIR

CODE = b"def handler(event):\n    return {'ok': True}\n" * 200
MODEL = os.urandom(64 * 1024)  # genuinely incompressible


def _cleanup():
    db = _get_db()
    db.collection("files").delete_many({})
    for f in _upload_dir.iterdir():
        if f.is_file():
            f.unlink(missing_ok=True)


@pytest.fixture(autouse=True)
def _isolate():
    """Clean before AND after — the old suite only cleans on teardown."""
    _cleanup()
    yield
    _cleanup()


class TestModelFileUpload:
    def test_gguf_upload_accepted(self):
        _cleanup()
        resp = client.post(
            "/files/upload", files={"file": ("tiny.gguf", MODEL, "application/octet-stream")}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["size_bytes"] == len(MODEL)

    def test_gguf_download_roundtrip_raw(self):
        _cleanup()
        fid = client.post(
            "/files/upload", files={"file": ("tiny.gguf", MODEL, "application/octet-stream")}
        ).json()["id"]
        resp = client.get(f"/files/{fid}/download")
        assert resp.status_code == 200
        assert resp.content == MODEL
        assert "content-encoding" not in resp.headers


class TestCompressedDownload:
    def _upload_code(self) -> str:
        resp = client.post(
            "/files/upload", files={"file": ("handler.py", CODE, "text/x-python")}
        )
        assert resp.status_code == 200
        return resp.json()["id"]

    def test_download_decompresses_when_no_zstd_advertised(self):
        _cleanup()
        fid = self._upload_code()
        resp = client.get(f"/files/{fid}/download", headers={"Accept-Encoding": "identity"})
        assert resp.status_code == 200
        assert resp.content == CODE
        assert "content-encoding" not in resp.headers
        assert resp.headers.get("x-slo-codec") == "zstd"

    def test_download_zstd_passthrough_on_accept(self):
        _cleanup()
        fid = self._upload_code()
        resp = client.get(f"/files/{fid}/download", headers={"Accept-Encoding": "zstd"})
        assert resp.status_code == 200
        assert resp.headers.get("content-encoding") == "zstd"
        assert resp.headers.get("vary") == "Accept-Encoding"
        # The stored frame goes on the wire untouched (zero transcoding);
        # httpx (0.28+, like a browser) auto-decodes a valid zstd frame, so
        # an intact round-trip is itself the integrity proof.
        assert resp.content == CODE

    def test_get_file_returns_decompressed_text(self):
        _cleanup()
        fid = self._upload_code()
        resp = client.get(f"/files/{fid}")
        assert resp.status_code == 200
        assert resp.json()["text"] == CODE.decode()

    def test_stored_blob_is_zstd_on_disk(self):
        _cleanup()
        fid = self._upload_code()
        files = [f.name for f in _upload_dir.iterdir() if f.is_file()]
        assert any(name.startswith(fid) and name.endswith(".zst") for name in files)


class TestLifecycle:
    def test_delete_after_download(self):
        _cleanup()
        fid = client.post(
            "/files/upload", files={"file": ("handler.py", CODE, "text/x-python")}
        ).json()["id"]
        assert client.delete(f"/files/{fid}").status_code == 200
        assert client.get(f"/files/{fid}/download").status_code == 404

    def test_search_and_list(self):
        _cleanup()
        client.post(
            "/files/upload", files={"file": ("handler.py", CODE, "text/x-python")}
        )
        assert client.get("/files").json()["total"] == 1
        found = client.get("/files/search", params={"q": "handler"}).json()
        assert found["total"] == 1
