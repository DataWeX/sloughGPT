"""Wire-level tests for the FileHub standalone app (WireServer + ASGI).

Boots the real hub on an ephemeral port and speaks HTTP to it — auth,
listing, zstd passthrough, Range on raw blobs, pull-through e2e.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx
import pytest
from wire import WireServer

from apps.filehub.app import make_app
from domain.files import FileHub, FileStore

CODE = b"def hub_handler(event):\n    return {'ok': True}\n" * 300
MODEL = bytes(range(256)) * 256  # incompressible-ish model blob


@pytest.fixture
def hub(tmp_path):
    return FileHub(store=FileStore(root=tmp_path / "uploads"), token="tok123")


@pytest.fixture
def base_url(hub):
    server = WireServer(asgi_app=make_app(hub), host="127.0.0.1", port=0, lifespan="off")
    server.start()
    host, port = getattr(server, "_bound", ("127.0.0.1", 0))
    try:
        yield f"http://{host}:{port}"
    finally:
        server.stop()


@pytest.fixture
def upstream():
    payload = MODEL

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - http.server naming
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/pullme.gguf"
    server.shutdown()


def _upload(base_url: str, name: str, data: bytes, token: str = "tok123") -> dict:
    resp = httpx.post(
        f"{base_url}/files?name={name}",
        content=data,
        headers={"X-FileHub-Token": token},
        timeout=30,
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


class TestAuth:
    def test_health_open_or_tokened(self, base_url):
        # health sits behind auth like everything else on a LAN-facing hub
        assert httpx.get(f"{base_url}/health").status_code == 401
        resp = httpx.get(f"{base_url}/health", headers={"X-FileHub-Token": "tok123"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_token_via_query_param(self, base_url):
        resp = httpx.get(f"{base_url}/health?token=tok123")
        assert resp.status_code == 200


class TestServing:
    def test_upload_list_download_code(self, base_url):
        meta = _upload(base_url, "handler.py", CODE)
        assert meta["codec"] == "zstd"
        listed = httpx.get(
            f"{base_url}/files", headers={"X-FileHub-Token": "tok123"}
        ).json()["files"]
        assert [f["id"] for f in listed] == [meta["id"]]
        # identity client gets decompressed bytes
        resp = httpx.get(
            f"{base_url}/files/{meta['id']}/download",
            headers={"X-FileHub-Token": "tok123", "Accept-Encoding": "identity"},
        )
        assert resp.content == CODE

    def test_zstd_passthrough_saves_bandwidth(self, base_url):
        meta = _upload(base_url, "handler.py", CODE)
        resp = httpx.get(
            f"{base_url}/files/{meta['id']}/download",
            headers={"X-FileHub-Token": "tok123", "Accept-Encoding": "zstd"},
        )
        # httpx 0.28+ auto-decodes a valid zstd frame; the headers are the
        # wire-contract proof that the stored frame went out untouched.
        assert resp.headers.get("content-encoding") == "zstd"
        assert resp.content == CODE
        # content-length is the compressed length actually sent on the wire
        # (decoders don't rewrite headers) — strictly smaller than raw size.
        assert int(resp.headers["content-length"]) < int(resp.headers["x-slo-raw-size"])

    def test_model_blob_roundtrip_and_range(self, base_url):
        meta = _upload(base_url, "weights.gguf", MODEL)
        assert meta["codec"] == "raw"
        full = httpx.get(
            f"{base_url}/files/{meta['id']}/download",
            headers={"X-FileHub-Token": "tok123"},
        )
        assert full.content == MODEL
        # resumable pull: a client that already has the first 1000 bytes
        part = httpx.get(
            f"{base_url}/files/{meta['id']}/download",
            headers={"X-FileHub-Token": "tok123", "Range": "bytes=1000-"},
        )
        assert part.status_code == 206
        assert part.content == MODEL[1000:]
        assert part.headers["content-range"] == f"bytes 1000-{len(MODEL) - 1}/{len(MODEL)}"

    def test_range_open_ended_and_bounded(self, base_url):
        meta = _upload(base_url, "weights.gguf", MODEL)
        part = httpx.get(
            f"{base_url}/files/{meta['id']}/download",
            headers={"X-FileHub-Token": "tok123", "Range": "bytes=0-99"},
        )
        assert part.status_code == 206
        assert part.content == MODEL[:100]

    def test_unknown_file_404(self, base_url):
        resp = httpx.get(
            f"{base_url}/files/nope/download", headers={"X-FileHub-Token": "tok123"}
        )
        assert resp.status_code == 404

    def test_large_blob_streams_multi_chunk(self, base_url):
        # >1 MiB so the download crosses multiple read chunks — the case
        # where streaming framing (not a lucky single frame) is exercised.
        big = bytes(range(256)) * ((3 * 1024 * 1024) // 256)  # 3 MiB
        meta = _upload(base_url, "big.gguf", big)
        with httpx.stream(
            "GET",
            f"{base_url}/files/{meta['id']}/download",
            headers={"X-FileHub-Token": "tok123"},
        ) as r:
            assert r.status_code == 200
            r.read()
            assert r.content == big
        part = httpx.get(
            f"{base_url}/files/{meta['id']}/download",
            headers={"X-FileHub-Token": "tok123", "Range": "bytes=1048576-2097151"},
        )
        assert part.status_code == 206
        assert part.content == big[1048576 : 2097152]


class TestPullThrough:
    def test_pull_lands_and_dedups(self, base_url, upstream):
        hdr = {"X-FileHub-Token": "tok123"}
        first = httpx.post(f"{base_url}/pull", json={"url": upstream}, headers=hdr, timeout=60)
        assert first.status_code == 200, first.text
        body = first.json()
        assert body["deduped"] is False
        assert body["codec"] == "raw"
        second = httpx.post(f"{base_url}/pull", json={"url": upstream}, headers=hdr, timeout=60)
        assert second.json()["deduped"] is True
        assert second.json()["id"] == body["id"]
        # the pulled blob is servable immediately
        got = httpx.get(f"{base_url}/files/{body['id']}/download", headers=hdr)
        assert got.content == MODEL

    def test_pull_bad_body_400(self, base_url):
        resp = httpx.post(
            f"{base_url}/pull",
            content=b"not json",
            headers={"X-FileHub-Token": "tok123"},
        )
        assert resp.status_code == 400
