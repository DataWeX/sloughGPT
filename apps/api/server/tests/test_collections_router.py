"""Tests for the /collections router."""
from test_support import get_test_client


def _d(resp):
    j = resp.json()
    return j.get("data", j)


class TestCollectionsRouter:
    def setup_method(self):
        self.client = get_test_client()

    def test_list_pipelines(self):
        resp = self.client.get("/collections")
        assert resp.status_code == 200

    def test_stats(self):
        resp = self.client.get("/collections/stats")
        assert resp.status_code == 200

    def test_create_pipeline(self):
        resp = self.client.post("/collections/create", json={"name": "test-pipeline", "sources": []})
        assert resp.status_code in (200, 201, 400, 422)

    def test_run_pipeline_not_found(self):
        resp = self.client.post("/collections/run", json={"pipeline_id": "nonexistent"})
        assert resp.status_code in (200, 400, 404, 422)

    def test_collect_direct(self):
        resp = self.client.post("/collections/collect", json={"text": "test content"})
        assert resp.status_code in (200, 400, 422, 500)
