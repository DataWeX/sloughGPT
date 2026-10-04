"""Loopback HTTP wire tests for domain.infrastructure._internal.compressed_transfer.

Serves real bytes over a live 127.0.0.1 socket (no mocking of the wire layer)
and exercises the API that actually shipped: ``CompressedDownloader.download_from_url``
across a full-body download and a 404, plus the header and range contracts of
``peek_compressed_header`` / ``CompressedFileServer``.

These assertions follow the shipped contracts rather than an earlier draft's:

* ``download_from_url`` catches transport errors and returns
  ``DownloadResult(success=False, error=...)`` — it does not raise.
* It only decompresses when ``Content-Type: application/x-lz4`` or
  ``Content-Encoding: lz4`` is present. The loopback server sends neither, so
  the payload is written through verbatim, which is what lets us assert exact
  bytes and ``bytes_written``.
* ``peek_compressed_header`` takes a binary stream and returns a **dict** with
  ``uncompressed_size``; there is no ``compressed_size`` field.
* ``CompressedFileServer.serve_range`` takes a **file path** and returns a dict
  with a ``"data"`` key plus a ``Content-Range`` header — it is server-side,
  not a client-side range fetch.
* No shipped code sends an HTTP ``Range`` request today, so there is no
  range-client test here; the loopback server keeps its Range branch so that
  contract stays exercisable if a client lands.
"""

from __future__ import annotations

import hashlib
import http.server
import io
import threading
import urllib.parse

import pytest

from domain.infrastructure._internal.compressed_transfer import (
    MAGIC,
    CompressedDownloader,
    CompressedFileServer,
    compress_file,
    compress_stream,
    decompress_stream,
    peek_compressed_header,
)


class LoopbackServer(http.server.BaseHTTPRequestHandler):
    """Serves a single in-memory payload, with optional Range support."""

    payload: bytes = b""

    def do_GET(self) -> None:
        parsed = urllib.parse.urlsplit(self.path)
        if parsed.path != "/wire.bin":
            self.send_response(404)
            self.end_headers()
            return
        body = type(self).payload
        total = len(body)
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            spec = rng[len("bytes=") :]
            start_s, _, end_s = spec.partition("-")
            start = int(start_s)
            end = int(end_s) if end_s else total - 1
            if start > end or start >= total:
                self.send_response(416)
                self.end_headers()
                return
            chunk = body[start : end + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{total}")
            self.send_header("Content-Length", str(len(chunk)))
            self.end_headers()
            self.wfile.write(chunk)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(total))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: object) -> None:
        pass


@pytest.fixture(scope="module")
def wire_url():
    httpd = http.server.HTTPServer(("127.0.0.1", 0), LoopbackServer)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}/wire.bin"
    httpd.shutdown()


def test_wire_full_download_roundtrip(wire_url, tmp_path):
    payload = (b"wire-" * 2048) + b"\x00\x01" * 8
    LoopbackServer.payload = payload

    dest = tmp_path / "out.bin"
    result = CompressedDownloader().download_from_url(wire_url, dest)

    assert result.success is True, result.error
    assert dest.read_bytes() == payload
    assert result.bytes_written == len(payload)
    # expected_sha256 omitted -> nothing to compare against, reported as a match
    assert result.sha256_match is True
    assert result.sha256 == hashlib.sha256(payload).hexdigest()


def test_wire_404_returns_failure_not_exception(wire_url, tmp_path):
    """A 404 surfaces as success=False + error, since transport errors are caught."""
    dest = tmp_path / "missing.bin"
    result = CompressedDownloader().download_from_url(
        wire_url.replace("wire.bin", "missing.bin"), dest
    )

    assert result.success is False
    assert result.error
    assert "404" in result.error
    assert not dest.exists()


def test_wire_peek_header():
    """Header is parsed from a byte stream and reports uncompressed_size."""
    plain = b"peek" * 512
    src, dst = io.BytesIO(plain), io.BytesIO()
    compress_stream(src, dst, include_header=True)
    compressed = dst.getvalue()

    header = peek_compressed_header(io.BytesIO(compressed))

    assert header is not None
    assert header["magic"] == MAGIC
    assert header["uncompressed_size"] == len(plain)
    assert header["sha256"] == hashlib.sha256(plain).hexdigest()


def test_wire_serve_range_contract(tmp_path):
    """serve_range is file-based and returns data + a Content-Range header."""
    plain = (b"serve-" * 256) + b"\xab" * 3
    src, dest_file = tmp_path / "plain.bin", tmp_path / "plain.lz4"
    src.write_bytes(plain)
    compress_file(src, dest_file)

    start, end = 0, 127
    served = CompressedFileServer().serve_range(dest_file, start, end)

    assert served["data"] == plain[start : end + 1]
    assert served["size"] == end - start + 1
    assert served["headers"]["Content-Range"] == (f"bytes {start}-{end}/{len(plain)}")


def test_wire_serve_headers_roundtrip(tmp_path):
    """serve() takes the UNCOMPRESSED source and compresses on the fly.

    (Deliberate asymmetry with serve_range above, which decompresses a .lz4
    file first.) So X-Uncompressed-Size is the source size, and the iterator
    yields compressed bytes that decompress back to the original.
    """
    plain = (b"round-" * 512) + b"\x00" * 5
    src = tmp_path / "plain.bin"
    src.write_bytes(plain)

    meta = CompressedFileServer().serve(src)

    assert meta["content_type"] == "application/x-lz4"
    assert meta["headers"]["Content-Encoding"] == "lz4"
    assert meta["headers"]["X-Uncompressed-Size"] == str(len(plain))
    assert meta["size"] == len(plain)

    body = b"".join(meta["iterator"])
    out = io.BytesIO()
    decompress_stream(io.BytesIO(body), out, verify_header=True)
    assert out.getvalue() == plain


def test_wire_serve_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        CompressedFileServer().serve(tmp_path / "nope.lz4")
