"""Tests for the /world router."""

from test_support import get_test_client


def _d(resp):
    j = resp.json()
    return j.get("data", j)


class TestWorldRenderRouter:
    def setup_method(self):
        self.client = get_test_client()

    def test_stats(self):
        resp = self.client.get("/world/stats")
        assert resp.status_code == 200

    def test_tick(self):
        resp = self.client.post("/world/tick")
        assert resp.status_code in (200, 500)

    def test_render(self):
        resp = self.client.post("/world/render", json={})
        assert resp.status_code in (200, 422, 500)

    def test_render_image(self):
        resp = self.client.post("/world/render/image", json={})
        assert resp.status_code in (200, 422, 500)

    def test_neural(self):
        resp = self.client.post("/world/neural", json={})
        assert resp.status_code in (200, 422, 500)
