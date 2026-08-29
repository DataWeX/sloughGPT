"""
Tests for the request middleware pipeline in infrastructure/middleware.py.

Covers: correlation-ID propagation, log level mapping, timeout behaviour,
metrics recording, and the client-extension DEBUG note.
"""
import logging

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from infrastructure.middleware import register_all_middleware
from infrastructure.exception_handlers import register_app_error_handler


def _make_app(timeout: float = 5.0) -> FastAPI:
    app = FastAPI()
    register_app_error_handler(app)
    register_all_middleware(app, request_timeout=timeout)

    @app.get("/ok")
    async def ok(request: Request):
        return {"corr": request.scope.get("correlation_id", "-")}

    @app.get("/fail")
    async def fail():
        from starlette.responses import JSONResponse
        return JSONResponse({"detail": "nope"}, status_code=404)

    @app.post("/boom")
    async def boom():
        from starlette.responses import JSONResponse
        return JSONResponse({"detail": "broken"}, status_code=500)

    @app.get("/slow")
    async def slow():
        import asyncio
        await asyncio.sleep(2.0)
        return {"ok": 1}

    return app


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)


def _capture_logger() -> _Capture:
    logger = logging.getLogger("slo.middleware")
    cap = _Capture()
    logger.addHandler(cap)
    logger.setLevel(logging.DEBUG)
    return cap


def _with_capture(app, fn):
    """Run fn with the slo.middleware logger captured, removing the handler after."""
    logger = logging.getLogger("slo.middleware")
    cap = _Capture()
    logger.addHandler(cap)
    logger.setLevel(logging.DEBUG)
    try:
        return fn(cap)
    finally:
        logger.removeHandler(cap)


class TestCorrelationId:

    def test_echoes_incoming_header(self):
        app = _make_app()
        client = TestClient(app)
        resp = client.get("/ok", headers={"X-Correlation-ID": "abc123"})
        assert resp.status_code == 200
        assert resp.json()["corr"] == "abc123"
        assert resp.headers["X-Correlation-ID"] == "abc123"

    def test_generates_id_when_absent(self):
        app = _make_app()
        client = TestClient(app)
        resp = client.get("/ok")
        corr = resp.json()["corr"]
        assert len(corr) == 8
        assert resp.headers["X-Correlation-ID"] == corr

    def test_id_propagates_to_log_context(self):
        app = _make_app()
        client = TestClient(app)

        def run(cap):
            resp = client.get("/ok", headers={"X-Correlation-ID": "zz99"})
            assert resp.status_code == 200
            req_logs = [r for r in cap.records if " corr=zz99" in r.getMessage()]
            assert len(req_logs) == 1
            ctx = req_logs[0].context
            assert ctx["corr"] == "zz99"

        _with_capture(app, run)


class TestLogLevels:

    def _collect_for(self, app, method, path, headers=None):
        def run(cap):
            resp = client.request(method, path, headers=headers or {})
            return resp.status_code, cap.records

        client = TestClient(app)
        return _with_capture(app, run)

    def test_ok_is_debug(self):
        app = _make_app()
        status, records = self._collect_for(app, "GET", "/ok")
        assert status == 200
        assert all(r.levelno == logging.DEBUG for r in records if "GET /ok 200" in r.getMessage())

    def test_4xx_is_warning(self):
        app = _make_app()
        status, records = self._collect_for(app, "GET", "/fail")
        assert status == 404
        assert any(r.levelno == logging.WARNING and "404 on GET /fail" in r.getMessage() for r in records)

    def test_5xx_is_error(self):
        app = _make_app()
        status, records = self._collect_for(app, "POST", "/boom")
        assert status == 500
        assert any(r.levelno == logging.ERROR and "500 on POST /boom" in r.getMessage() for r in records)


class TestTimeout:

    def test_slow_request_returns_504(self):
        app = _make_app(timeout=0.2)
        client = TestClient(app)
        resp = client.get("/slow")
        assert resp.status_code == 504
        assert resp.json()["code"] == "E_INFRA_TIMEOUT"


class TestMetrics:

    def test_request_recorded(self):
        from domains.infrastructure.metrics import get_metrics_collector
        collector = get_metrics_collector()
        before = collector._request_count.get("/ok", 0)
        app = _make_app()
        client = TestClient(app)
        client.get("/ok")
        assert collector._request_count.get("/ok", 0) == before + 1

    def test_error_recorded(self):
        from domains.infrastructure.metrics import get_metrics_collector
        collector = get_metrics_collector()
        before = collector._request_errors.get("/fail", 0)
        app = _make_app()
        client = TestClient(app)
        client.get("/fail")
        assert collector._request_errors.get("/fail", 0) == before + 1

    def test_active_requests_returns_to_baseline(self):
        from domains.infrastructure.metrics import get_metrics_collector
        collector = get_metrics_collector()
        baseline = collector.get_active_requests()
        app = _make_app()
        client = TestClient(app)
        client.get("/ok")
        assert collector.get_active_requests() == baseline


class TestClientExtensionFilter:

    def test_extension_origin_emits_debug_note(self):
        app = _make_app()
        client = TestClient(app)

        def run(cap):
            resp = client.get("/fail", headers={"Origin": "chrome-extension://abcdef"})
            assert resp.status_code == 404
            notes = [r for r in cap.records if "Extension error suppressed" in r.getMessage()]
            assert len(notes) == 1
            assert notes[0].levelno == logging.DEBUG

        _with_capture(app, run)

    def test_non_extension_origin_emits_no_note(self):
        app = _make_app()
        client = TestClient(app)

        def run(cap):
            resp = client.get("/fail", headers={"Origin": "http://localhost:3000"})
            assert resp.status_code == 404
            notes = [r for r in cap.records if "Extension error suppressed" in r.getMessage()]
            assert len(notes) == 0

        _with_capture(app, run)
