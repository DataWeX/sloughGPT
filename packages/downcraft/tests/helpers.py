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

from http.server import BaseHTTPRequestHandler


class RangeHandler(BaseHTTPRequestHandler):
    """Serves per-path payloads with HTTP Range + HEAD support.

    Class attributes (set per test):
        payloads: ``{path: bytes}`` — any other path returns 404.
        content_types: ``{path: str}`` — per-path Content-Type override.
        head_responses: ``{path: dict}`` — per-path HEAD response headers.
        encodings: ``{path: str}`` — ``gzip`` | ``zstd`` → edge-compress GET.
    """

    payloads: dict = {}
    content_types: dict = {}
    head_responses: dict = {}
    encodings: dict = {}

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
        payload = self._payload_for()
        if payload is None:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        # Edge-compressed full body: Range must NOT be used with encoding
        # (gateway returns identity for Range — mirror that here).
        if self._path() in self.encodings and not self.headers.get("Range"):
            wire, extra = self._encode(payload)
            self.send_response(200)
            self.send_header("Content-Length", str(len(wire)))
            self.send_header("Content-Type", self._content_type_for())
            for k, v in extra.items():
                self.send_header(k, v)
            self.send_header("ETag", '"static"')
            self.end_headers()
            self.wfile.write(wire)
            return

        start = 0
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            spec = rng[len("bytes=") :].split("-")[0]
            if spec.isdigit():
                start = int(spec)
        data = payload[start:]
        if start > 0:
            self.send_response(206)
            self.send_header(
                "Content-Range",
                f"bytes {start}-{len(payload) - 1}/{len(payload)}",
            )
        else:
            self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Content-Type", self._content_type_for())
        self.send_header("ETag", '"static"')
        self.end_headers()
        for i in range(0, len(data), 2048):
            self.wfile.write(data[i : i + 2048])

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
