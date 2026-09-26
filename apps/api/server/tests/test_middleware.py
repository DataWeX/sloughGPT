"""
Tests for the request middleware pipeline in infrastructure/middleware.py.

Covers: correlation-ID propagation, log level mapping, timeout behaviour,
metrics recording, and the client-extension DEBUG note.
"""

import logging

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from infrastructure.exception_handlers import register_app_error_handler
from infrastructure.middleware import register_all_middleware


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


class _MultiCapture(logging.Handler):
    """Captures records from several loggers into one list."""

    def __init__(self, names):
        super().__init__(logging.DEBUG)
        self.records = []
        self._loggers = [logging.getLogger(n) for n in names]
        for lg in self._loggers:
            lg.addHandler(self)
            lg.setLevel(logging.DEBUG)

    def emit(self, record):
        self.records.append(record)

    def close(self):
        for lg in self._loggers:
            lg.removeHandler(self)
        super().close()


def _both_capture(fn):
    """Run fn with slo.middleware + slo.exception_handlers records captured."""
    cap = _MultiCapture(["slo.middleware", "slo.exception_handlers"])
    try:
        return fn(cap)
    finally:
        cap.close()


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
            # PayloadLoggingMiddleware + UnifiedRequestMiddleware both log the corr ID
            # via extra={"http": {"corr": corr_id, ...}}.  Filter by the structured
            # extra dict rather than message text, since multiple log lines now carry
            # the correlation ID.
            req_logs = [r for r in cap.records if getattr(r, "http", {}).get("corr") == "zz99"]
            assert len(req_logs) >= 1
            assert req_logs[0].http["corr"] == "zz99"

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
        assert any(
            r.levelno == logging.WARNING and "404 on GET /fail" in r.getMessage() for r in records
        )

    def test_5xx_is_error(self):
        app = _make_app()
        status, records = self._collect_for(app, "POST", "/boom")
        assert status == 500
        assert any(
            r.levelno == logging.ERROR and "500 on POST /boom" in r.getMessage() for r in records
        )


class TestTimeout:
    def test_slow_request_returns_504(self):
        app = _make_app(timeout=0.2)
        client = TestClient(app)
        resp = client.get("/slow")
        assert resp.status_code == 504
        assert resp.json()["code"] == "E_INFRA_TIMEOUT"


class TestColdStartPrefixSuppression:
    """SLOW warnings for prefix-based cold-start paths show as debug during warm-up."""

    def _slow_app(self):
        app = FastAPI()
        register_app_error_handler(app)
        register_all_middleware(app, request_timeout=20.0)

        @app.get("/token-tree/stats")
        async def stats():
            import asyncio

            await asyncio.sleep(1.2)
            return {"ok": 1}

        return app

    def test_cold_start_prefix_is_debug_in_warmup(self, monkeypatch):
        import time

        import infrastructure.middleware as mw

        monkeypatch.setattr(mw, "_SERVER_START", time.monotonic())
        app = self._slow_app()
        client = TestClient(app)

        def run(cap):
            resp = client.get("/token-tree/stats")
            assert resp.status_code == 200
            slow = [r for r in cap.records if "token-tree/stats" in r.getMessage()]
            assert len(slow) >= 1
            assert all(r.levelno == logging.DEBUG for r in slow)
            assert any("cold-start" in r.getMessage() for r in slow)

        _with_capture(app, run)

    def test_cold_start_prefix_warns_after_warmup(self, monkeypatch):
        import time

        import infrastructure.middleware as mw

        monkeypatch.setattr(mw, "_SERVER_START", time.monotonic() - 1000.0)
        app = self._slow_app()
        client = TestClient(app)

        def run(cap):
            resp = client.get("/token-tree/stats")
            assert resp.status_code == 200
            slow = [r for r in cap.records if "token-tree/stats" in r.getMessage()]
            assert len(slow) >= 1
            assert all(r.levelno == logging.WARNING for r in slow)
            assert not any("cold-start" in r.getMessage() for r in slow)

        _with_capture(app, run)


class TestMetrics:
    def test_request_recorded(self):
        from domain.infrastructure.metrics import get_metrics_collector

        collector = get_metrics_collector()
        before = collector._request_count.get("/ok", 0)
        app = _make_app()
        client = TestClient(app)
        client.get("/ok")
        assert collector._request_count.get("/ok", 0) == before + 1

    def test_error_recorded(self):
        from domain.infrastructure.metrics import get_metrics_collector

        collector = get_metrics_collector()
        before = collector._request_errors.get("/fail", 0)
        app = _make_app()
        client = TestClient(app)
        client.get("/fail")
        assert collector._request_errors.get("/fail", 0) == before + 1

    def test_active_requests_returns_to_baseline(self):
        from domain.infrastructure.metrics import get_metrics_collector

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


class TestSingleOwner4xxLogging:
    """Exception handlers own the 4xx WARN line; the middleware must not
    duplicate it (its timing line drops to DEBUG).  5xx keeps both lines —
    handler ERROR + middleware ERROR are different severities."""

    @staticmethod
    def _raising_app():
        app = FastAPI()
        from infrastructure.exception_handlers import register_all_handlers

        register_all_handlers(app)
        register_all_middleware(app, request_timeout=5.0)

        @app.get("/raise404")
        async def raise404():
            from fastapi import HTTPException

            raise HTTPException(status_code=404, detail="nope")

        @app.get("/raise500")
        async def raise500():
            from fastapi import HTTPException

            raise HTTPException(status_code=500, detail="kaput")

        return app

    def test_4xx_handler_warn_not_duplicated_by_middleware(self):
        app = self._raising_app()
        client = TestClient(app)

        def run(cap):
            resp = client.get("/raise404")
            assert resp.status_code == 404
            handler_warns = [
                r
                for r in cap.records
                if r.name == "slo.exception_handlers"
                and r.levelno == logging.WARNING
                and "HTTP 404" in r.getMessage()
            ]
            middleware_warns = [
                r
                for r in cap.records
                if r.name == "slo.middleware"
                and r.levelno == logging.WARNING
                and "404 on GET /raise404" in r.getMessage()
            ]
            middleware_debugs = [
                r
                for r in cap.records
                if r.name == "slo.middleware"
                and r.levelno == logging.DEBUG
                and "404 on GET /raise404" in r.getMessage()
            ]
            assert len(handler_warns) == 1, "handler must emit exactly one WARN"
            assert middleware_warns == [], "middleware must not repeat the WARN"
            assert len(middleware_debugs) == 1, "timing line survives at DEBUG"

        _both_capture(run)

    def test_4xx_without_handler_still_warns(self):
        """Routes that return 4xx without raising keep the middleware WARN."""
        app = _make_app()
        client = TestClient(app)

        def run(cap):
            resp = client.get("/fail")
            assert resp.status_code == 404
            warns = [
                r
                for r in cap.records
                if r.levelno == logging.WARNING and "404 on GET /fail" in r.getMessage()
            ]
            assert len(warns) == 1

        _with_capture(app, run)

    def test_5xx_keeps_both_error_lines(self):
        app = self._raising_app()
        client = TestClient(app)

        def run(cap):
            resp = client.get("/raise500")
            assert resp.status_code == 500
            handler_errors = [
                r
                for r in cap.records
                if r.name == "slo.exception_handlers" and r.levelno >= logging.ERROR
            ]
            middleware_errors = [
                r
                for r in cap.records
                if r.name == "slo.middleware"
                and r.levelno == logging.ERROR
                and "500 on GET /raise500" in r.getMessage()
            ]
            assert len(handler_errors) == 1
            assert len(middleware_errors) == 1

        _both_capture(run)

    def test_body_parse_400_logs_cause(self, caplog):
        """Starlette body-parse 400 carries __cause__ type as `cause` context."""
        import asyncio

        import tests.test_support  # noqa: F401  (registers feature routers on `main.app`)
        from main import app

        received = []
        calls = 0

        async def receive():
            nonlocal calls
            calls += 1
            if calls == 1:
                return {"type": "http.request", "body": b'{"partial"', "more_body": True}
            return {"type": "http.disconnect"}

        async def send(message):
            received.append(message)

        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "PUT",
            "scheme": "http",
            "path": "/docstore/kv/_cause_probe",
            "raw_path": b"/docstore/kv/_cause_probe",
            "query_string": b"",
            "root_path": "",
            "headers": [
                (b"host", b"testserver"),
                (b"content-type", b"application/json"),
                (b"content-length", b"999"),
            ],
            "client": ("127.0.0.1", 12345),
            "server": ("testserver", 80),
        }

        with caplog.at_level(logging.WARNING, logger="slo.exception_handlers"):
            asyncio.run(app(scope, receive, send))

        start = next(m for m in received if m["type"] == "http.response.start")
        assert start["status"] == 400
        ctxs = [
            getattr(r, "context", {})
            for r in caplog.records
            if r.name == "slo.exception_handlers" and "HTTP 400" in r.getMessage()
        ]
        assert any("cause" in c for c in ctxs), ctxs


class TestStormEndToEnd:
    """The original bug: 24 client-disconnect 400s produced 48 WARN lines
    (handler + middleware, one each per request).  With single-owner logging
    plus RepeatSuppressionFilter the whole storm collapses to 1 handler WARN
    + 1 middleware DEBUG line."""

    @staticmethod
    def _storm_app():
        app = FastAPI()
        from infrastructure.exception_handlers import register_all_handlers

        register_all_handlers(app)
        register_all_middleware(app, request_timeout=5.0)

        @app.put("/docstore/kv/{key}")
        async def put_kv(key: str, body: dict):
            return {"key": key, "ok": True}

        return app

    @staticmethod
    def _disconnect_once(app, key: str) -> int:
        """Drive the ASGI app with a mid-body client disconnect (the real
        ClientDisconnect -> HTTP 400 path) and return the status code."""
        import asyncio

        received: list[dict] = []
        calls = 0

        async def receive():
            nonlocal calls
            calls += 1
            if calls == 1:
                return {"type": "http.request", "body": b'{"partial"', "more_body": True}
            return {"type": "http.disconnect"}

        async def send(message):
            received.append(message)

        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "PUT",
            "scheme": "http",
            "path": f"/docstore/kv/{key}",
            "raw_path": f"/docstore/kv/{key}".encode(),
            "query_string": b"",
            "root_path": "",
            "headers": [
                (b"host", b"testserver"),
                (b"content-type", b"application/json"),
                (b"content-length", b"999"),
            ],
            "client": ("127.0.0.1", 12345),
            "server": ("testserver", 80),
        }
        asyncio.run(app(scope, receive, send))
        start = next(m for m in received if m["type"] == "http.response.start")
        return int(start["status"])

    def test_twenty_four_400s_emit_one_warn_line(self):
        from domain.logging._internal.config import RepeatSuppressionFilter

        app = self._storm_app()
        dedup = RepeatSuppressionFilter(window_s=60)

        class Capture(logging.Handler):
            def __init__(self):
                super().__init__(logging.DEBUG)
                self.records = []

            def emit(self, record):
                self.records.append(record)

        caps = []
        loggers = [logging.getLogger("slo.exception_handlers"), logging.getLogger("slo.middleware")]
        try:
            for lg in loggers:
                cap = Capture()
                cap.addFilter(dedup)  # shared instance, like setup_logging
                lg.addHandler(cap)
                lg.setLevel(logging.DEBUG)
                caps.append(cap)

            for i in range(24):
                assert self._disconnect_once(app, f"conv_{i}") == 400
        finally:
            for lg, cap in zip(loggers, caps):
                lg.removeHandler(cap)

        handler_warns = [
            r
            for r in caps[0].records
            if r.levelno == logging.WARNING and "HTTP 400" in r.getMessage()
        ]
        middleware_warns = [r for r in caps[1].records if r.levelno == logging.WARNING]
        middleware_debugs = [r for r in caps[1].records if r.levelno == logging.DEBUG]

        assert len(handler_warns) == 1, f"storm not collapsed: {len(handler_warns)}"
        assert middleware_warns == [], "middleware must not add WARN lines to the storm"
        # Timing lines are DEBUG (below the dedup threshold) — one per request
        # is fine, they never reach the WARN stream.
        assert len(middleware_debugs) == 24, f"timing line count: {len(middleware_debugs)}"
        # 48 WARN lines (old) -> 1 WARN with the shared dedup filter.
