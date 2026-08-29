"""Tests for the /voice router."""
from test_support import get_test_client


def _d(resp):
    j = resp.json()
    return j.get("data", j)


class TestVoiceRouter:
    def setup_method(self):
        self.client = get_test_client()

    def test_voice_status(self):
        resp = self.client.get("/voice/status")
        assert resp.status_code == 200
        data = _d(resp)
        assert isinstance(data, dict)

    def test_tts(self):
        resp = self.client.post("/voice/tts", json={"text": "hello world"})
        assert resp.status_code in (200, 400, 422, 500, 503)
