"""Headless-shell identity tests — the request-fingerprint gate opens.

The fixture is a stdlib ``ThreadingHTTPServer`` standing in for the
servers that fingerprint request identity (x.com does: it 403'd avion
while curl and a real Chrome got 200 from the same host). The gate
answers 403 unless the ``User-Agent`` carries no ``HeadlessChrome``
marker and the ``sec-ch-ua`` hints, when sent, name ``Google Chrome`` —
the UA header alone is already proven insufficient (an avion
``--user-agent`` launch still got 403, because Chromium keeps
advertising Chromium-only hints and the server cross-checks).

Two proofs, both against a real browser launched by avion:

1. the browser's own view of itself is headed — no ``HeadlessChrome``
   in ``navigator.userAgent`` and a ``Google Chrome`` brand in
   ``navigator.userAgentData.brands`` (what ``sec-ch-ua`` is built
   from);
2. the gate returns the open page, i.e. the server accepted the
   document request.

A real Chrome passes both untouched: the override only fires when the
live build reports the headless shell. Degrades explicitly (stated
skip) when no Chromium is available.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from avion.backends.cdp import _headed_metadata
from avion.core.element import ElementLocator
from avion.core.session import Arken
from avion.launch import Profile, find_chromium, launch_chromium


class _GateHandler(BaseHTTPRequestHandler):
    """403 unless the request looks like a headed Chrome."""

    def log_message(self, *args):  # silence the server thread
        pass

    def do_GET(self):
        ua = self.headers.get("User-Agent", "")
        ch = self.headers.get("Sec-CH-UA", "")
        ok = "HeadlessChrome" not in ua and (not ch or "Google Chrome" in ch)
        body = "<h1>gate-open</h1>" if ok else "<h1>gate-closed</h1>"
        payload = body.encode()
        self.send_response(200 if ok else 403)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture
def gate():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _GateHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def _chromium_or_skip() -> str:
    try:
        return find_chromium()
    except RuntimeError:
        pytest.skip("no Chromium available — set AVION_CHROMIUM or install chromium")


_LAUNCH_ARGS = ("--no-sandbox", "--disable-gpu")


class TestHeadedMetadata:
    """The override's payload — a protocol error here fails silently (fail-open)."""

    _UA = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "HeadlessChrome/151.0.7922.34 Safari/537.36"
    )

    def test_carries_every_field_the_cdp_binding_requires(self):
        """architecture/model/mobile are mandatory — omitting one is a protocol error."""
        meta = _headed_metadata(self._UA, "151.0.7922.34")
        for key in (
            "brands",
            "fullVersionList",
            "platform",
            "platformVersion",
            "architecture",
            "model",
            "mobile",
            "bitness",
        ):
            assert key in meta, f"mandatory field missing: {key}"
        assert meta["brands"][0] == {"brand": "Google Chrome", "version": "151"}
        assert meta["fullVersionList"][0]["version"] == "151.0.7922.34"
        assert meta["platform"] == "Linux"
        assert meta["architecture"] in ("x86", "arm")
        assert meta["mobile"] is False


class TestHeadlessIdentity:
    async def test_headed_identity_opens_the_fingerprint_gate(self, gate, tmp_path):
        """The shell presents a headed identity, so a fingerprinting server says yes."""
        binary = _chromium_or_skip()
        launched = launch_chromium(
            Profile("uagate", root=tmp_path / "profiles"),
            headless=True,
            binary=binary,
            extra_args=_LAUNCH_ARGS,
            timeout=30.0,
        )
        try:
            session = Arken(base_url=gate)
            await session.start(backend=launched.backend)
            try:
                await session.goto("/")

                # The cause — what the browser claims about itself.
                identity = await launched.backend.evaluate(
                    "({ua: navigator.userAgent,"
                    " brands: (navigator.userAgentData"
                    " && navigator.userAgentData.brands.map(b => b.brand)) || []})"
                )
                assert "HeadlessChrome" not in identity["ua"], (
                    f"still identifies as headless: {identity['ua']}"
                )
                assert "Google Chrome" in identity["brands"], (
                    f"client hints lack the Google Chrome brand: {identity['brands']}"
                )

                # The effect — the server accepted the document request.
                open_page = await session.find(ElementLocator.text("gate-open"))
                assert open_page is not None, (
                    "fingerprint gate closed — the document request was refused"
                )
            finally:
                await session.stop()
        finally:
            launched.stop()
