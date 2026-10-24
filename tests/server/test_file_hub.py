"""Wire-level tests for the FileHub standalone app (WireServer + ASGI).

Boots the real hub on an ephemeral port and speaks HTTP to it — auth,
listing, zstd passthrough, Range on raw blobs, pull-through e2e.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx
import numpy as np
import pytest
from wire import WireServer

from apps.filehub.app import make_app
from domain.files import FileHub, FileStore

CODE = b"def hub_handler(event):\n    return {'ok': True}\n" * 300
MODEL = bytes(range(256)) * 256  # incompressible-ish model blob


def _fp32_safetensors(seed: int = 3) -> bytes:
    """Small fp32 safetensors blob.

    Mirrors the builder in packages/core-py/tests/test_file_hub.py — the two
    test trees have separate conftest boundaries, so cross-importing a fixture
    helper is more fragile than the ~15 duplicated lines.
    """
    rng = np.random.default_rng(seed)
    tensors = {
        "blocks.0.weight": rng.normal(0, 0.05, (64, 128)).astype(np.float32),
        "blocks.0.bias": rng.normal(0, 0.05, (128,)).astype(np.float32),
    }
    header: dict = {}
    blobs: list[bytes] = []
    off = 0
    for name in sorted(tensors):
        data = tensors[name].astype("<f4").tobytes()
        header[name] = {
            "dtype": "F32",
            "shape": list(tensors[name].shape),
            "data_offsets": [off, off + len(data)],
        }
        blobs.append(data)
        off += len(data)
    hjson = json.dumps(header).encode()
    return struct.pack("<Q", len(hjson)) + hjson + b"".join(blobs)


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


@pytest.fixture
def float_upstream():
    """Upstream serving a fp32 safetensors (quantize-on-ingest target)."""
    payload = _fp32_safetensors()

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
    yield f"http://127.0.0.1:{server.server_address[1]}/float.safetensors"
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


class TestResumeJourney:
    """End-to-end kill-and-resume: the real proof of 'no wasted bytes'.

    The Range tests above assert a single slice against an intact buffer.
    These drive the actual client journey: start a download, kill it after a
    partial read, resume with ``Range: bytes=N-``, and prove (a) the resume
    returns 206 + only the missing suffix, and (b) the reassembled bytes are
    byte-identical to the original (sha256), so nothing was lost or wasted.
    """

    # 5 MiB so the first read lands mid-file and the resume spans chunks.
    SIZE = 5 * 1024 * 1024

    @pytest.fixture
    def big_blob(self):
        return os.urandom(self.SIZE)

    def test_kill_midflight_then_resume_sends_only_missing_suffix(self, base_url, big_blob):
        meta = _upload(base_url, "resume.gguf", big_blob)
        assert meta["codec"] == "raw"
        url = f"{base_url}/files/{meta['id']}/download"
        auth = {"X-FileHub-Token": "tok123"}

        # Start a full download, read ONE chunk, then kill it (close the
        # stream early — the connection drops with bytes still unsent).
        got = bytearray()
        with httpx.stream("GET", url, headers=auth, timeout=30) as r:
            assert r.status_code == 200
            for chunk in r.iter_bytes(chunk_size=65536):
                got.extend(chunk)
                break  # <- the kill: client aborts after the first chunk
        killed_at = len(got)
        assert 0 < killed_at < self.SIZE  # genuinely partial, not whole file

        # Resume exactly where the client died. Only the suffix is owed.
        resume = httpx.get(
            url, headers={**auth, "Range": f"bytes={killed_at}-"}, timeout=30
        )
        assert resume.status_code == 206
        expected_range = f"bytes {killed_at}-{self.SIZE - 1}/{self.SIZE}"
        assert resume.headers["content-range"] == expected_range
        # No wasted bytes: the server sent ONLY what was missing.
        assert len(resume.content) == self.SIZE - killed_at
        assert resume.content == big_blob[killed_at:]

        # Reassembled bytes are byte-identical to the original.
        reassembled = bytes(got) + resume.content
        assert reassembled == big_blob
        assert hashlib.sha256(reassembled).hexdigest() == hashlib.sha256(big_blob).hexdigest()

    def test_resume_across_multiple_kills_stays_lossless(self, base_url, big_blob):
        # A flaky link kills the transfer twice; the client resumes after
        # each kill and still ends with the exact original bytes.
        meta = _upload(base_url, "flaky.gguf", big_blob)
        url = f"{base_url}/files/{meta['id']}/download"
        auth = {"X-FileHub-Token": "tok123"}
        chunks: list[bytes] = []
        offset = 0
        kills = 0

        while offset < self.SIZE:
            hdr = {**auth, "Range": f"bytes={offset}-"}
            with httpx.stream("GET", url, headers=hdr, timeout=30) as r:
                assert r.status_code == 206
                for chunk in r.iter_bytes(chunk_size=32768):
                    chunks.append(chunk)
                    offset += len(chunk)
                    if kills < 2 and offset >= (kills + 1) * (self.SIZE // 3):
                        kills += 1
                        break  # kill mid-slice; next loop resumes via Range
            if kills >= 2 and offset >= self.SIZE:
                break

        assert kills == 2  # the journey actually got interrupted twice
        reassembled = b"".join(chunks)
        assert reassembled == big_blob


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


class TestQuantizeHTTP:
    def test_pull_quantize_via_wire(self, base_url, float_upstream):
        hdr = {"X-FileHub-Token": "tok123"}
        first = httpx.post(
            f"{base_url}/pull",
            json={"url": float_upstream, "quantize": True},
            headers=hdr,
            timeout=60,
        )
        assert first.status_code == 200, first.text
        body = first.json()
        assert body["quantization"]["bits"] == 8
        assert body["quantization"]["ratio"] >= 3.0
        assert body["deduped"] is False
        second = httpx.post(
            f"{base_url}/pull",
            json={"url": float_upstream, "quantize": True},
            headers=hdr,
            timeout=60,
        )
        assert second.status_code == 200
        assert second.json()["deduped"] is True
        assert second.json()["id"] == body["id"]

    def test_upload_quantize_flag(self, base_url):
        blob = _fp32_safetensors()
        resp = httpx.post(
            f"{base_url}/files?name=model.safetensors&quantize=1",
            content=blob,
            headers={"X-FileHub-Token": "tok123"},
            timeout=60,
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["quantization"]["bits"] == 8
        assert body["stored"] < len(blob)  # int8 blob smaller than fp32 source

    def test_upload_quantize_requires_safetensors(self, base_url):
        resp = httpx.post(
            f"{base_url}/files?name=weights.bin&quantize=1",
            content=b"\x00" * 64,
            headers={"X-FileHub-Token": "tok123"},
        )
        assert resp.status_code == 400
        assert "safetensors" in resp.json()["error"]

    def test_quantize_refusal_is_400(self, base_url, upstream):
        # .gguf upstream under opt-in quantize -> 400, never a silent rewrite
        resp = httpx.post(
            f"{base_url}/pull",
            json={"url": upstream, "quantize": True},
            headers={"X-FileHub-Token": "tok123"},
            timeout=60,
        )
        assert resp.status_code == 400
        assert "safetensors" in resp.json()["error"]
