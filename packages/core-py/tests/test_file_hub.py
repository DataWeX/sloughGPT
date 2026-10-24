"""Tests for domain.files.FileHub — token auth, pull-through, sha256 dedup,
opt-in quantize-on-ingest (pull with quantize=True, card 019464d8)."""

from __future__ import annotations

import json
import struct
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from domain.files import (
    FileHub,
    FileStore,
    UnsupportedForQuantization,
    load_quantized,
)

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


def _fp32_safetensors(seed: int = 3) -> tuple[bytes, dict[str, np.ndarray]]:
    """A small deterministic fp32 safetensors blob (2D weight + 1D bias)."""
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
    return struct.pack("<Q", len(hjson)) + hjson + b"".join(blobs), tensors


@pytest.fixture
def float_upstream():
    """Upstream serving a fp32 safetensors; yields (url, reference tensors)."""
    payload, tensors = _fp32_safetensors()

    class FloatHandler(_Handler):
        pass

    FloatHandler.payload = payload
    server = HTTPServer(("127.0.0.1", 0), FloatHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/float.safetensors", tensors
    server.shutdown()
    thread.join(timeout=5)


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


class TestQuantizePull:
    def test_pull_quantize_stores_int8(self, hub, float_upstream):
        url, _ = float_upstream
        meta = hub.pull(url, quantize=True)
        assert meta["deduped"] is False
        q = meta["quantization"]
        assert q["bits"] == 8
        assert q["ratio"] >= 3.0
        assert q["worst_cosine"] >= 0.99
        assert len(q["source_sha256"]) == 64  # pre-quant digest rides along
        assert meta["stored_size"] < q["orig_bytes"]

    def test_pull_quantize_roundtrips(self, hub, float_upstream):
        url, tensors = float_upstream
        meta = hub.pull(url, quantize=True)
        restored = load_quantized(hub.store.root / meta["filename"])
        assert set(restored) == set(tensors)
        for name, orig in tensors.items():
            got = restored[name]
            assert got.shape == orig.shape
            assert got.dtype == np.float32
            cos = float(
                np.dot(orig.ravel(), got.ravel())
                / (np.linalg.norm(orig) * np.linalg.norm(got) + 1e-12)
            )
            assert cos >= 0.99, f"{name} cosine {cos} below tolerance"

    def test_pull_quantize_dedups(self, hub, float_upstream):
        url, _ = float_upstream
        first = hub.pull(url, quantize=True)
        second = hub.pull(url, quantize=True)
        # quantization is deterministic -> identical int8 bytes -> dedup
        assert second["deduped"] is True
        assert second["file_id"] == first["file_id"]
        assert len(hub.store.list()) == 1

    def test_pull_quantize_refuses_ineligible(self, hub, upstream):
        # .gguf extension refuses before any byte is read (loss-on-loss guard)
        with pytest.raises(UnsupportedForQuantization, match="safetensors"):
            hub.pull(upstream, quantize=True)
