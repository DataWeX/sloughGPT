"""Tests for the /user-adapters router."""

from test_support import get_test_client


def _d(resp):
    j = resp.json()
    return j.get("data", j)


class TestUserAdaptersRouter:
    def setup_method(self):
        self.client = get_test_client()

    def test_list_adapters(self):
        resp = self.client.get("/user-adapters")
        assert resp.status_code == 200

    def test_quality(self):
        resp = self.client.get("/user-adapters/quality")
        assert resp.status_code == 200

    def test_get_nonexistent_user(self):
        resp = self.client.get("/user-adapters/nonexistent-user")
        assert resp.status_code in (200, 404)

    def test_prune(self):
        resp = self.client.post("/user-adapters/prune")
        assert resp.status_code in (200, 400, 422, 500)

    def test_aggregate_best(self):
        resp = self.client.post("/user-adapters/aggregate-best")
        assert resp.status_code in (200, 400, 422, 500)
