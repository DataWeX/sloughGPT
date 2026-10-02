"""Shared test helpers for downcraft tests.

``RangeHandler`` / ``_range_url`` live here (not in ``conftest.py``) on
purpose: the repo ``pytest.ini`` uses ``--import-mode=importlib``, under
which pytest's import of ``tests/conftest.py`` (fixture plugin) and each
test module's ``sys.path.insert(...)`` + ``from conftest import ...`` can
produce TWO module instances with TWO distinct ``RangeHandler`` classes.
Payloads registered on the test-side class then never reach the server,
and every fixture-backed test 404s.

This module is never auto-imported by pytest — only via the explicit
``from helpers import ...`` in ``conftest.py`` and the test modules — so
there is exactly one ``RangeHandler`` class object per session.
"""

import hashlib
import io
from http.server import BaseHTTPRequestHandler

from downcraft.download.compress import compress_stream


def slz4_wire(payload: bytes, *, corrupt_header: bool = False) -> bytes:
    """Build an SLZ4 wire body (44-byte header + LZ4 frame) for *payload*.

    Mirrors what a downcraft compression-aware server sends for a full
    ``Content-Type: application/x-lz4`` response.  ``corrupt_header``
    flips a byte inside the stored SHA-256 to simulate a bad header.
    """
    buf = io.BytesIO()
    compress_stream(io.BytesIO(payload), buf, compression_level=6)
    wire = bytearray(buf.getvalue())
    if corrupt_header:
        wire[12] ^= 0xFF  # flip a bit inside the 32-byte header SHA-256
    return bytes(wire)


class RangeHandler(BaseHTTPRequestHandler):
    """Serves per-path payloads with HTTP Range + HEAD support.

    Class attributes (set per test):
        payloads: ``{path: bytes}`` — any other path returns 404.
        content_types: ``{path: str}`` — per-path Content-Type override.
        head_responses: ``{path: dict}`` — per-path HEAD response headers.
        encodings: ``{path: str}`` — ``gzip`` | ``zstd`` → edge-compress GET.
        lz4_paths: ``{path: True}`` → full GET = SLZ4 wire (application/x-lz4),
            Range GET = identity slice with ``X-SLZ4-SHA256`` relayed from
            the SLZ4 header (the downcraft compression protocol).
        lz4_naive: ``{path: True}`` → Range slices the WIRE bytes instead
            (byte-space trap: offsets are compressed-representation bytes).
        lz4_bad_header: ``{path: True}`` → corrupt SLZ4 header SHA-256.
        lz4_bad_resume_sha: ``{path: True}`` → 206 relays a WRONG
            ``X-SLZ4-SHA256`` (simulates version skew / corruption).
        truncate_once: ``{path: n}`` → first GET promises the full body
            (Content-Length) but sends only *n* bytes, then closes —
            a mid-stream network crash.
        requests_log: per-request dicts (path, range, accept_encoding,
            status, slz4_sha) for byte-space assertions in tests.
    """

    payloads: dict = {}
    content_types: dict = {}
    head_responses: dict = {}
    encodings: dict = {}
    lz4_paths: dict = {}
    lz4_naive: dict = {}
    lz4_bad_header: dict = {}
    lz4_bad_resume_sha: dict = {}
    truncate_once: dict = {}
    requests_log: list = []

    def _path(self):
        return self.path.split("?")[0]

    def _payload_for(self):
        return self.payloads.get(self._path())

    def _content_type_for(self):
        return self.content_types.get(self._path(), "application/octet-stream")

    def _encode(self, data: bytes) -> tuple[bytes, dict]:
        """Return (wire_bytes, extra_headers) for the path's edge encoding."""
        enc = self.encodings.get(self._path(), "")
        if enc == "gzip":
            import gzip

            wire = gzip.compress(data)
            return wire, {
                "Content-Encoding": "gzip",
                "X-Uncompressed-Content-Length": str(len(data)),
            }
        if enc == "zstd":
            import zstandard as zstd

            wire = zstd.ZstdCompressor(level=3).compress(data)
            return wire, {
                "Content-Encoding": "zstd",
                "X-Uncompressed-Content-Length": str(len(data)),
            }
        return data, {}

    def do_GET(self):
        path = self._path()
        payload = self._payload_for()

        rng = self.headers.get("Range")
        start = None
        if rng and rng.startswith("bytes="):
            spec = rng[len("bytes=") :].split("-")[0]
            if spec.isdigit():
                start = int(spec)

        entry = {
            "path": path,
            "range": rng,
            "start": start,
            "accept_encoding": self.headers.get("Accept-Encoding", ""),
            "status": 0,
            "slz4_sha": "",
        }
        RangeHandler.requests_log.append(entry)

        if payload is None:
            entry["status"] = 404
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        # ── SLZ4 compression protocol ──────────────────────────────────
        # Full GET: SLZ4 wire bytes (header + LZ4 frame).
        # Range GET: IDENTITY slice — Range byte-space == decoded bytes
        # (unless the path is marked lz4_naive, which slices the wire to
        # reproduce the byte-space trap).
        if path in self.lz4_paths:
            self._send_lz4(entry, path, payload, start)
            return

        # Edge-compressed full body: Range must NOT be used with encoding
        # (gateway returns identity for Range — mirror that here).
        if path in self.encodings and not rng:
            wire, extra = self._encode(payload)
            entry["status"] = 200
            self._respond(
                200,
                {
                    "Content-Length": str(len(wire)),
                    "Content-Type": self._content_type_for(),
                    **extra,
                },
                wire,
                path,
                entry,
            )
            return

        if start is not None and start > 0:
            body = payload[start:]
            entry["status"] = 206
            self._respond(
                206,
                {
                    "Content-Range": f"bytes {start}-{len(payload) - 1}/{len(payload)}",
                    "Content-Length": str(len(body)),
                    "Content-Type": self._content_type_for(),
                },
                body,
                path,
                entry,
            )
        else:
            entry["status"] = 200
            self._respond(
                200,
                {
                    "Content-Length": str(len(payload)),
                    "Content-Type": self._content_type_for(),
                },
                payload,
                path,
                entry,
            )

    def _send_lz4(self, entry: dict, path: str, payload: bytes, start: int | None) -> None:
        """Serve one SLZ4-protocol response (full wire or identity/wire range)."""
        corrupt = path in self.lz4_bad_header
        if start is None:
            body = slz4_wire(payload, corrupt_header=corrupt)
            entry["status"] = 200
            self._respond(
                200,
                {
                    "Content-Type": "application/x-lz4",
                    "Content-Encoding": "lz4",
                    "Content-Length": str(len(body)),
                    "X-Uncompressed-Content-Length": str(len(payload)),
                },
                body,
                path,
                entry,
            )
        elif path in self.lz4_naive:
            # Byte-space trap: slice the COMPRESSED representation and keep
            # the lz4 markers — a client that appends these to a decoded
            # .sgpart corrupts the file.
            wire = slz4_wire(payload, corrupt_header=corrupt)
            body = wire[start:]
            entry["status"] = 206
            self._respond(
                206,
                {
                    "Content-Type": "application/x-lz4",
                    "Content-Encoding": "lz4",
                    "Content-Length": str(len(body)),
                    "Content-Range": f"bytes {start}-{len(wire) - 1}/{len(wire)}",
                    "X-Uncompressed-Content-Length": str(len(payload)),
                },
                body,
                path,
                entry,
            )
        else:
            # Protocol-correct: Range is answered in identity (decoded)
            # byte-space; the SLZ4 header SHA-256 is relayed so a resumed
            # client can still verify end-to-end integrity.
            body = payload[start:]
            sha = hashlib.sha256(payload).hexdigest()
            if path in self.lz4_bad_resume_sha:
                sha = "00" * 32
            entry["slz4_sha"] = sha
            entry["status"] = 206
            self._respond(
                206,
                {
                    "Content-Type": "application/octet-stream",
                    "Content-Length": str(len(body)),
                    "Content-Range": f"bytes {start}-{len(payload) - 1}/{len(payload)}",
                    "X-SLZ4-SHA256": sha,
                },
                body,
                path,
                entry,
            )

    def _respond(
        self, status: int, headers: dict, body: bytes, path: str, entry: dict
    ) -> None:
        """Send status+headers+body, honoring ``truncate_once`` crash injection."""
        self.send_response(status)
        for k, v in headers.items():
            self.send_header(k, v)
        self.send_header("ETag", '"static"')
        self.end_headers()

        n = RangeHandler.truncate_once.pop(path, None)
        if n is not None:
            # Mid-stream crash: promise the full Content-Length, send a
            # prefix, then slam the connection shut.
            self.wfile.write(body[:n])
            self.wfile.flush()
            self.connection.close()
            return
        for i in range(0, len(body), 2048):
            self.wfile.write(body[i : i + 2048])

    def do_HEAD(self):
        head = self.head_responses.get(self._path())
        if head is not None:
            self.send_response(head.get("status", 200))
            for k, v in head.get("headers", {}).items():
                self.send_header(k, v)
            self.end_headers()
            return
        payload = self._payload_for()
        if payload is None:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        # Identity size always — matches edge UCL semantics.
        self.send_response(200)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Content-Type", self._content_type_for())
        if self._path() in self.encodings:
            self.send_header("X-Uncompressed-Content-Length", str(len(payload)))
        self.end_headers()

    def log_message(self, *args):
        pass


def _range_url(server, path: str) -> str:
    """Build a URL for a path on the running range server."""
    return f"http://127.0.0.1:{server.server_port}{path}"
