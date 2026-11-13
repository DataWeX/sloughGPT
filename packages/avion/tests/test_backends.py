"""Backends tests — api + cli live, driver backends degraded gracefully."""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from avion.backends.api import ApiBackend
from avion.backends.cdp import selector_to_js
from avion.backends.cli import CliBackend
from avion.core.element import ElementLocator


def run(coro):
    return asyncio.run(coro)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"ok": True})
        else:
            self._send(404, {"error": "nope"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length).decode() or "{}")
        self._send(200, {"echo": body})

    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def server():
    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    httpd.shutdown()


class TestApi:
    def test_get_ok(self, server):
        b = ApiBackend(base_url=server)
        run(b.start())
        r = run(b.get("/health"))
        run(b.stop())
        assert r.ok and r.body == {"ok": True} and r.status == 200

    def test_404_not_ok(self, server):
        b = ApiBackend(base_url=server)
        r = run(b.get("/missing"))
        assert not r.ok and r.status == 404

    def test_post_echo(self, server):
        b = ApiBackend(base_url=server)
        r = run(b.post("/echo", {"a": 1}))
        assert r.ok and r.body == {"echo": {"a": 1}}

    def test_connection_refused(self):
        b = ApiBackend(base_url="http://127.0.0.1:1")
        r = run(b.get("/x"))
        assert not r.ok and r.error != ""


class TestCli:
    def test_spawn_echo_expect(self):
        b = CliBackend()
        run(b.start())
        run(b.spawn("cat"))
        assert b.running
        run(b.send("hello arken"))
        result = run(b.expect("hello arken"))
        run(b.stop())
        assert result.matched
        assert not b.running

    def test_expect_timeout(self):
        b = CliBackend()
        run(b.spawn("cat"))
        result = run(b.expect("never-comes", timeout=0.3))
        run(b.stop())
        assert not result.matched and "timeout" in result.error

    def test_send_without_process_raises(self):
        b = CliBackend()
        with pytest.raises(RuntimeError):
            run(b.send("x"))


class TestDriverBackends:
    def test_selenium_missing_dep(self):
        try:
            __import__("selenium")
            pytest.skip("selenium installed")
        except ImportError:
            from avion.backends.selenium import SeleniumBackend

            with pytest.raises(ImportError, match="pip install selenium"):
                run(SeleniumBackend().start())

    def test_cdp_missing_dep(self):
        try:
            __import__("websockets")
            pytest.skip("websockets installed")
        except ImportError:
            from avion.backends.cdp import CdpBackend

            with pytest.raises(ImportError, match="pip install websockets"):
                run(CdpBackend().start())

    def test_appium_missing_dep(self):
        try:
            __import__("appium")
            pytest.skip("appium installed")
        except ImportError:
            from avion.backends.appium import AppiumBackend

            with pytest.raises(ImportError, match="Appium-Python-Client"):
                run(AppiumBackend().start())

    def test_desktop_missing_dep(self):
        try:
            __import__("pyautogui")
            pytest.skip("pyautogui installed")
        except ImportError:
            from avion.backends.desktop import DesktopBackend

            with pytest.raises(ImportError, match="pip install pyautogui"):
                run(DesktopBackend().start())

    def test_cdp_selector_js(self):
        js = selector_to_js(ElementLocator.css(".btn").selectors[0])
        assert js == "document.querySelector('.btn')"
        js = selector_to_js(ElementLocator.text("Go").selectors[0])
        assert "Go" in js
        assert selector_to_js(ElementLocator.role("x").selectors[0]) is None
