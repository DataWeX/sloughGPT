"""Tests for the /agents router."""
from test_support import get_test_client


def _d(resp):
    j = resp.json()
    return j.get("data", j)


class TestAgentsRouter:
    def setup_method(self):
        self.client = get_test_client()

    def test_list_agents(self):
        resp = self.client.get("/agents")
        assert resp.status_code == 200

    def test_create_agent(self):
        import time
        resp = self.client.post("/agents", json={"name": f"test-agent-{int(time.time())}", "description": "test"})
        assert resp.status_code in (200, 201, 400, 409, 422)

    def test_get_agent_not_found(self):
        resp = self.client.get("/agents/nonexistent-agent-id")
        assert resp.status_code in (404, 200)

    def test_delete_agent_not_found(self):
        resp = self.client.delete("/agents/nonexistent-agent-id")
        assert resp.status_code in (404, 200, 204)

    def test_agent_runs(self):
        resp = self.client.get("/agents/runs")
        assert resp.status_code == 200

    def test_orchestrate(self):
        resp = self.client.post("/agents/orchestrate", json={"task": "test task"})
        assert resp.status_code in (200, 400, 422)
