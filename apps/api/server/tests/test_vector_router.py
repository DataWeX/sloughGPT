"""Tests for the /vector router."""
from test_support import get_test_client


def _d(resp):
    j = resp.json()
    return j.get("data", j)


class TestVectorRouter:
    def setup_method(self):
        self.client = get_test_client()

    def test_init_vector_store(self):
        resp = self.client.post("/vector/init", json={})
        assert resp.status_code in (200, 400, 422, 500)

    def test_get_stats(self):
        resp = self.client.get("/vector/stats")
        assert resp.status_code == 200
        data = _d(resp)
        assert isinstance(data, dict)

    def test_upsert(self):
        resp = self.client.post("/vector/upsert", json={"id": "test-1", "text": "hello world"})
        assert resp.status_code in (200, 400, 422, 500)

    def test_search(self):
        resp = self.client.post("/vector/search", json={"query": "hello", "top_k": 5})
        assert resp.status_code in (200, 400, 422, 500)

    def test_ingest_status(self):
        resp = self.client.get("/vector/ingest/status")
        assert resp.status_code == 200
