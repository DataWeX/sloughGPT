"""
Tests for the health router endpoints.

Unlike other router tests, this module registers only the health router
(not all routers) to avoid pulling in heavy dependencies (transformers,
torch, peft) that would slow test startup to >10s.
"""
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.main import app as _app
from routers.health import router as health_router

# Register only the health router so tests start fast.
_app.include_router(health_router)
client = TestClient(_app)


class TestHealthRouter:
    """Tests for /health/* endpoints."""

    def _data(self, resp):
        body = resp.json()
        return body.get("data", body)

    def test_basic_health(self):
        """GET /health returns status, timestamp, model info."""
        resp = client.get('/health')
        assert resp.status_code == 200
        data = self._data(resp)
        assert data['status'] in ('healthy', 'degraded', 'unhealthy')
        assert 'timestamp' in data
        assert 'model_loaded' in data
        assert 'inference_count' in data

    def test_liveness(self):
        """GET /health/live returns status: alive."""
        resp = client.get('/health/live')
        assert resp.status_code == 200
        assert self._data(resp)['status'] == 'alive'

    def test_readiness(self):
        """GET /health/ready returns status."""
        resp = client.get('/health/ready')
        assert resp.status_code == 200
        data = self._data(resp)
        assert 'status' in data

    def test_detailed_health_structure(self):
        """GET /health/detailed returns system metrics."""
        resp = client.get('/health/detailed')
        assert resp.status_code == 200
        data = self._data(resp)
        assert 'status' in data
        assert 'uptime_seconds' in data
        assert 'timestamp' in data
        assert 'system' in data
        assert isinstance(data['system'], dict)
        assert 'cpu_percent' in data['system']
        assert 'memory_percent' in data['system']
        assert 'model_loaded' in data
        assert 'inference' in data

    def test_startup_progress(self):
        """GET /health/startup-progress returns phase."""
        resp = client.get('/health/startup-progress')
        assert resp.status_code == 200
        data = self._data(resp)
        assert 'step' in data or 'phase' in data

    def test_debug_info(self):
        """GET /health/debug returns debug snapshot."""
        resp = client.get('/health/debug')
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)

    def test_all_health_endpoints_respond_without_crash(self):
        """All health endpoints return 200 without crashing."""
        for path in ('/health', '/health/live', '/health/ready',
                     '/health/detailed', '/health/startup-progress',
                     '/health/debug'):
            resp = client.get(path)
            assert resp.status_code == 200, f'{path} returned {resp.status_code}'
