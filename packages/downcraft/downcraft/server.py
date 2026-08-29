"""
downcraft.server — Local server for browser extension integration.

Endpoints:
    - POST /capture — receive download URLs
    - GET /captures — list recent captures
    - GET /health — server status

Usage::

    downcraft capture --port 6400
"""

import json
import logging
import time
import threading
from dataclasses import dataclass, field, asdict
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

_entry_counter = 0
_entry_lock = threading.Lock()


@dataclass
class CaptureEntry:
    """A captured download link."""

    url: str
    title: str = ""
    referrer: str = ""
    timestamp: float = field(default_factory=time.time)
    id: str = ""

    def __post_init__(self):
        if not self.id:
            global _entry_counter
            with _entry_lock:
                _entry_counter += 1
                self.id = f"{int(self.timestamp * 1000)}-{_entry_counter}"

    def to_dict(self):
        return asdict(self)


class CaptureQueue:
    """Thread-safe queue for captured URLs."""

    def __init__(self, max_size: int = 1000):
        self._entries: List[CaptureEntry] = []
        self._max_size = max_size
        self._lock = threading.Lock()
        self._listeners: List[Callable[[CaptureEntry], None]] = []

    def add(self, entry: CaptureEntry) -> None:
        with self._lock:
            self._entries.append(entry)
            if len(self._entries) > self._max_size:
                self._entries = self._entries[-self._max_size:]

        for listener in self._listeners:
            try:
                listener(entry)
            except Exception as e:
                logger.warning("Listener error: %s", e)

    def list(self, limit: int = 50) -> List[dict]:
        with self._lock:
            entries = self._entries[-limit:]
        return [e.to_dict() for e in reversed(entries)]

    def count(self) -> int:
        with self._lock:
            return len(self._entries)

    def add_listener(self, fn: Callable[[CaptureEntry], None]) -> None:
        self._listeners.append(fn)

    def remove_listener(self, fn: Callable[[CaptureEntry], None]) -> None:
        self._listeners.remove(fn)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


_capture_queue = CaptureQueue()


def get_capture_queue() -> CaptureQueue:
    return _capture_queue


class CaptureHandler(BaseHTTPRequestHandler):

    server_ref: Optional[HTTPServer] = None

    def do_GET(self):
        path = self.path.split("?")[0]

        if path == "/health":
            self._respond(200, {
                "status": "ok",
                "captures": _capture_queue.count(),
                "uptime": time.time() - self.server_ref._start_time if self.server_ref else 0,
            })
        elif path == "/captures":
            self._respond(200, _capture_queue.list(limit=50))
        else:
            self._respond(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/capture":
            self._handle_capture()
        else:
            self._respond(404, {"error": "not found"})

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _handle_capture(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
        except (json.JSONDecodeError, ValueError):
            self._respond(400, {"error": "invalid json"})
            return

        url = body.get("url", "")
        if not url:
            self._respond(400, {"error": "missing url"})
            return

        entry = CaptureEntry(url=url, title=body.get("title", ""), referrer=body.get("referrer", ""))
        _capture_queue.add(entry)

        logger.info("Captured: %s", url)
        self._respond(200, {"status": "captured", "url": url, "id": entry.id})

    def _respond(self, status: int, data: dict):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self._cors()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")

    def log_message(self, format, *args):
        pass


def start_capture_server(
    port: int = 6400,
    on_capture: Optional[Callable[[CaptureEntry], None]] = None,
) -> HTTPServer:
    """Start the capture server."""

    if on_capture:
        _capture_queue.add_listener(on_capture)

    server = HTTPServer(("127.0.0.1", port), CaptureHandler)
    server._start_time = time.time()
    CaptureHandler.server_ref = server

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    logger.info("Capture server listening on http://127.0.0.1:%d", port)
    return server
