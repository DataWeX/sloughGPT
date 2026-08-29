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

    def test_detailed_health_process_metrics(self):
        """GET /health/detailed includes real process metrics."""
        data = self._data(client.get('/health/detailed'))
        system = data['system']
        assert 'threads' in system
        assert system['threads'] >= 1
        assert 'gc_gen0' in system
        assert 'gc_gen1' in system
        assert 'gc_gen2' in system
        assert 'rss_mb' in system
        assert system['rss_mb'] > 0

    def test_detailed_health_records_trend_snapshots(self):
        """Each cache rebuild records a health + memory trend point."""
        from controllers.health import HealthController
        from domains.infrastructure.server_state import get_server_state

        hc = HealthController()
        # Reset the trend throttle so the first call records
        get_server_state()._last_trend_ts = 0.0
        hc._cache_time = 0.0
        d1 = hc.get_detailed_health()
        assert len(d1['health_history']) >= 1
        assert len(d1['memory_history']) >= 1

        # Force both the controller cache and the trend throttle open so
        # the next call rebuilds and records again.
        get_server_state()._last_trend_ts = 0.0
        hc._cache_time = 0.0
        d2 = hc.get_detailed_health()
        assert len(d2['health_history']) > len(d1['health_history'])
        assert len(d2['memory_history']) > len(d1['memory_history'])

    def test_health_stream_envelope_shape(self):
        """SSE stream emits a standard envelope with real fields, not a nested envelope."""
        import asyncio
        import json
        from routers.health import HealthRouter
        from controllers.health import get_health_controller

        class _Req:
            async def is_disconnected(self):
                return False

        async def _read_first():
            resp = await HealthRouter().health_stream(_Req())
            gen = resp.body_iterator
            try:
                async for chunk in gen:
                    return chunk
            finally:
                await gen.aclose()
            return None

        raw = asyncio.run(_read_first())
        assert raw is not None
        assert raw.startswith('data: ')
        envelope = json.loads(raw[6:].strip())
        assert envelope['stream'] == 'health'
        assert envelope['phase'] == 'HEALTH'
        data = envelope['data']
        # Real snapshot fields live directly under data — no nested envelope.
        assert 'model_loaded' in data
        assert isinstance(data['model_loaded'], bool)
        assert 'health_status' in data
        assert 'cpu_percent' in data
        assert 'requests_per_minute' in data
        assert 'total_tokens' in data
        assert 'avg_tokens_per_request' in data
        assert 'model_metrics' in data
        assert isinstance(data['model_metrics'], list)
        assert 'path_latencies' in data
        assert isinstance(data['path_latencies'], list)
        assert 'recent_errors' in data
        assert isinstance(data['recent_errors'], list)
        assert 'model_events' in data
        assert isinstance(data['model_events'], list)
        assert 'rate_violations' in data
        assert isinstance(data['rate_violations'], list)
        assert 'health_history' in data
        assert isinstance(data['health_history'], list)
        assert 'memory_history' in data
        assert isinstance(data['memory_history'], list)
        assert 'data' not in data

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

    def test_detailed_health_has_kv_sessions(self):
        """GET /health/detailed includes a kv_sessions block."""
        data = self._data(client.get('/health/detailed'))
        assert 'kv_sessions' in data
        assert isinstance(data['kv_sessions'], dict)
        assert 'enabled' in data['kv_sessions']

    def test_basic_health_kv_sessions_disabled_by_default(self):
        """Without a slonet provider, kv_sessions stays out of basic health."""
        data = self._data(client.get('/health'))
        assert data.get('kv_sessions') is None

    def test_basic_health_kv_sessions_reflects_provider(self):
        """A provider exposing session_stats surfaces stats in /health."""
        from domains.models.provider import register_provider

        class _FakeProvider:
            def session_stats(self):
                return {
                    "active_sessions": 2,
                    "cached_tokens": 128,
                    "ttl_seconds": 3600.0,
                }

        register_provider("slonet-native", _FakeProvider())
        try:
            data = self._data(client.get('/health'))
            kv = data.get('kv_sessions')
            assert kv is not None
            assert kv["enabled"] is True
            assert kv["active_sessions"] == 2
            assert kv["cached_tokens"] == 128
        finally:
            from domains.models.provider import _providers
            _providers.pop("slonet-native", None)
