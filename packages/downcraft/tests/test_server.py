"""
Tests for downcraft.server — minimal capture server.
"""

import json
import urllib.request
import urllib.error
import pytest

from downcraft.server import CaptureEntry, CaptureQueue, start_capture_server


class TestCaptureEntry:
    def test_creates_with_defaults(self):
        entry = CaptureEntry(url="https://example.com/file.zip")
        assert entry.url == "https://example.com/file.zip"
        assert entry.id

    def test_to_dict(self):
        entry = CaptureEntry(url="https://example.com/file.zip", title="Test")
        d = entry.to_dict()
        assert d["url"] == "https://example.com/file.zip"
        assert d["title"] == "Test"

    def test_unique_ids(self):
        e1 = CaptureEntry(url="https://a.com")
        e2 = CaptureEntry(url="https://b.com")
        assert e1.id != e2.id


class TestCaptureQueue:
    def test_add_and_list(self):
        q = CaptureQueue()
        q.add(CaptureEntry(url="https://a.com/file.zip"))
        q.add(CaptureEntry(url="https://b.com/file.mp4"))
        items = q.list()
        assert len(items) == 2
        assert items[0]["url"] == "https://b.com/file.mp4"

    def test_count(self):
        q = CaptureQueue()
        assert q.count() == 0
        q.add(CaptureEntry(url="https://a.com"))
        assert q.count() == 1

    def test_max_size(self):
        q = CaptureQueue(max_size=3)
        for i in range(5):
            q.add(CaptureEntry(url=f"https://{i}.com"))
        assert q.count() == 3

    def test_clear(self):
        q = CaptureQueue()
        q.add(CaptureEntry(url="https://a.com"))
        q.clear()
        assert q.count() == 0

    def test_listener(self):
        q = CaptureQueue()
        received = []
        q.add_listener(lambda e: received.append(e))
        q.add(CaptureEntry(url="https://a.com"))
        assert len(received) == 1

    def test_remove_listener(self):
        q = CaptureQueue()
        received = []
        fn = lambda e: received.append(e)
        q.add_listener(fn)
        q.add(CaptureEntry(url="https://a.com"))
        assert len(received) == 1
        q.remove_listener(fn)
        q.add(CaptureEntry(url="https://b.com"))
        assert len(received) == 1


class TestCaptureServer:
    @pytest.fixture(autouse=True)
    def _clear(self):
        import downcraft.server as mod
        mod._capture_queue.clear()
        mod._entry_counter = 0
        yield

    @pytest.fixture
    def server(self):
        srv = start_capture_server(port=0)
        yield srv
        srv.shutdown()

    @pytest.fixture
    def port(self, server):
        return server.server_address[1]

    def _get(self, port, path):
        req = urllib.request.Request(f"http://127.0.0.1:{port}{path}")
        return json.loads(urllib.request.urlopen(req).read())

    def _post(self, port, path, data):
        body = json.dumps(data).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=body,
                                     headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req).read())

    def test_health(self, port):
        r = self._get(port, "/health")
        assert r["status"] == "ok"
        assert "captures" in r
        assert "uptime" in r

    def test_capture(self, port):
        r = self._post(port, "/capture", {"url": "https://example.com/file.zip", "title": "Test"})
        assert r["status"] == "captured"
        assert r["url"] == "https://example.com/file.zip"
        assert "id" in r

    def test_captures_list(self, port):
        self._post(port, "/capture", {"url": "https://a.com/file.zip"})
        self._post(port, "/capture", {"url": "https://b.com/file.mp4"})
        r = self._get(port, "/captures")
        assert len(r) == 2
        assert r[0]["url"] == "https://b.com/file.mp4"

    def test_capture_missing_url(self, port):
        with pytest.raises(urllib.error.HTTPError) as exc:
            self._post(port, "/capture", {"title": "no url"})
        assert exc.value.code == 400

    def test_cors(self, port):
        req = urllib.request.Request(f"http://127.0.0.1:{port}/health")
        resp = urllib.request.urlopen(req)
        assert resp.headers.get("Access-Control-Allow-Origin") == "*"

    def test_404(self, port):
        with pytest.raises(urllib.error.HTTPError) as exc:
            self._get(port, "/nonexistent")
        assert exc.value.code == 404
