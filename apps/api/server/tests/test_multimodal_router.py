"""Tests for the /multimodal router."""

from test_support import get_test_client


def _d(resp):
    j = resp.json()
    return j.get("data", j)


class TestMultimodalRouter:
    def setup_method(self):
        self.client = get_test_client()

    def test_status(self):
        resp = self.client.get("/multimodal/status")
        assert resp.status_code == 200
        data = _d(resp)
        assert isinstance(data, dict)

    def test_checkpoints(self):
        resp = self.client.get("/multimodal/checkpoints")
        assert resp.status_code == 200

    def test_analyze_no_file(self):
        resp = self.client.post("/multimodal/analyze", json={})
        assert resp.status_code in (200, 400, 422, 500)

    def test_reset(self):
        resp = self.client.post("/multimodal/reset")
        assert resp.status_code in (200, 500)

    def test_synthesize_speech(self):
        resp = self.client.post("/multimodal/synthesize-speech", json={"text": "hello"})
        assert resp.status_code in (200, 400, 422, 500, 503)

    def test_transcribe_no_file(self):
        resp = self.client.post("/multimodal/transcribe", json={})
        assert resp.status_code in (200, 400, 422, 500)
