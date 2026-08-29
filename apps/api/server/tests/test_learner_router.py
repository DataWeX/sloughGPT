"""Tests for the /learn (learner) router."""
from test_support import get_test_client


def _d(resp):
    j = resp.json()
    return j.get("data", j)


class TestLearnerRouter:
    def setup_method(self):
        self.client = get_test_client()

    def test_status(self):
        resp = self.client.get("/learn/status")
        assert resp.status_code == 200

    def test_feed(self):
        resp = self.client.get("/learn/feed", params={"action": "list"})
        assert resp.status_code == 200

    def test_knowledge(self):
        resp = self.client.get("/learn/knowledge")
        assert resp.status_code == 200

    def test_search(self):
        resp = self.client.post("/learn/search", json={"query": "test"})
        assert resp.status_code in (200, 400, 422)

    def test_ingest(self):
        resp = self.client.post("/learn/ingest", json={"text": "test content"})
        assert resp.status_code in (200, 400, 422, 500)
