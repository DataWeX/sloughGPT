#!/usr/bin/env python3
"""Minimal Range/HEAD file sidecar for the edge gateway (local host).

Serves files from --root at any GET/HEAD path under that root.
Gateway points MAN_CORE_URL here so downloads go sidecar → gateway → client.
Also answers GET /health as JSON so the gateway's sidecar probe succeeds.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Force octet-stream for model/checkpoint bytes — mimetypes maps .pt to
# junk (vnd.snesdev…) which confuses clients; gateway still compresses it.
OCTET_SUFFIXES = {
    ".pt",
    ".pth",
    ".bin",
    ".safetensors",
    ".gguf",
    ".npz",
    ".onnx",
    ".so",
    ".wasm",
    ".apk",
    ".zip",
}


class Handler(BaseHTTPRequestHandler):
    root: Path = Path(".")

    def _resolve(self) -> Path | None:
        raw = self.path.split("?", 1)[0]
        rel = raw.lstrip("/")
        if not rel or rel.endswith("/"):
            return None
        candidate = (self.root / rel).resolve()
        try:
            candidate.relative_to(self.root.resolve())
        except ValueError:
            return None
        if candidate.is_file():
            return candidate
        return None

    def _content_type(self, path: Path | None) -> str:
        if path and path.suffix.lower() in OCTET_SUFFIXES:
            return "application/octet-stream"
        if path:
            guess, _ = mimetypes.guess_type(str(path))
            if guess:
                return guess
        return "application/octet-stream"

    def _send_headers(
        self,
        status: int,
        path: Path | None,
        length: int,
        extra: dict | None = None,
    ):
        self.send_response(status)
        self.send_header("Content-Type", self._content_type(path))
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("ETag", '"sidecar-static"')
        if extra:
            for k, v in extra.items():
                self.send_header(k, v)
        self.end_headers()

    def do_HEAD(self):
        path = self._resolve()
        if path is None:
            self._send_headers(404, None, 0)
            return
        self._send_headers(200, path, path.stat().st_size)

    def do_GET(self):
        raw = self.path.split("?", 1)[0]
        if raw == "/health":
            body = json.dumps(
                {"status": "ok", "healthy": True, "model_loaded": False, "model": "file-sidecar"}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        path = self._resolve()
        if path is None:
            self._send_headers(404, None, 0)
            return
        size = path.stat().st_size
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        status = 200
        extra: dict = {}
        if rng and rng.startswith("bytes="):
            spec = rng[len("bytes=") :].split(",")[0].strip()
            if "-" in spec:
                a, b = spec.split("-", 1)
                if a:
                    start = int(a)
                if b:
                    end = min(int(b), size - 1)
                if start > end or start >= size:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                status = 206
                extra["Content-Range"] = f"bytes {start}-{end}/{size}"

        length = end - start + 1
        self._send_headers(status, path, length, extra)
        with open(path, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def log_message(self, fmt, *args):
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path.cwd())
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18000)
    args = ap.parse_args()
    Handler.root = args.root.resolve()
    if not Handler.root.is_dir():
        raise SystemExit(f"root not a directory: {Handler.root}")
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"sidecar root={Handler.root} on {args.host}:{args.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
