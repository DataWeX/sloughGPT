"""Tests for the /companion router."""
from test_support import get_test_client


def _d(resp):
    j = resp.json()
    return j.get("data", j)


class TestCompanionRouter:
    def setup_method(self):
        self.client = get_test_client()

    def test_get_companion(self):
        resp = self.client.get("/companion/")
        assert resp.status_code == 200

    def test_companion_presets(self):
        resp = self.client.get("/companion/presets")
        assert resp.status_code == 200
        data = _d(resp)
        assert isinstance(data, (list, dict))

    def test_companion_prompt(self):
        resp = self.client.get("/companion/prompt")
        assert resp.status_code == 200

    def test_set_personality(self):
        resp = self.client.post("/companion/personality", json={"personality": "friendly"})
        assert resp.status_code in (200, 400, 422)

    def test_companion_chat(self):
        resp = self.client.post("/companion/chat", json={"message": "hello"})
        assert resp.status_code in (200, 400, 422, 503)

    def test_apply_preset(self):
        resp = self.client.post("/companion/preset", json={"preset": "default"})
        assert resp.status_code in (200, 400, 404, 422)
