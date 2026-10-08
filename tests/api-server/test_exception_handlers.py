"""Tests for exception handlers — serialization guard middleware, HTTP, and unhandled errors."""

import logging

import pytest
from pydantic import BaseModel
from test_support import get_test_client

client = get_test_client()


class DummyResponse(BaseModel):
    text: str
    count: int


def _register_dummy_route(app):
    """Add a route that deliberately returns a non-serializable value."""
    from fastapi import APIRouter

    router = APIRouter(prefix="/_test", tags=["_test"])

    class _Broken:
        def __call__(self):
            pass

    broken = _Broken()

    @router.get("/serialization-error", response_model=DummyResponse)
    async def _broken_endpoint():
        return {"text": broken.__call__, "count": 0}

    app.include_router(router)


# ── Serialization guard middleware ──


class TestSerializationGuardMiddleware:
    @pytest.fixture(autouse=True)
    def _register(self):
        from main import app

        _register_dummy_route(app)

    def test_returns_500(self):
        resp = client.get("/_test/serialization-error")
        assert resp.status_code == 500

    def test_returns_json_not_html(self):
        resp = client.get("/_test/serialization-error")
        content_type = resp.headers.get("content-type", "")
        assert "application/json" in content_type

    def test_returns_error_envelope(self):
        resp = client.get("/_test/serialization-error")
        body = resp.json()
        assert body["code"] == "E_SERIALIZATION"
        assert "error" in body

    def test_user_friendly_message(self):
        resp = client.get("/_test/serialization-error")
        body = resp.json()
        msg = body["error"].lower()
        # Should NOT expose raw exception details
        assert "lambda" not in msg
        assert "broken" not in msg
        assert "pydantic" not in msg
        # Should contain a helpful message
        assert "server" in msg or "error" in msg

    def test_no_raw_exception_leaked(self):
        resp = client.get("/_test/serialization-error")
        body = resp.json()
        raw = str(body)
        assert "boom" not in raw.lower()
        assert "__call__" not in raw


# ── Known endpoints respond correctly ──


class TestKnownEndpoints:
    def test_health_endpoint_returns_json(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert "status" in body

    def test_infer_health_returns_structured(self):
        resp = client.get("/infer/health")
        assert resp.status_code == 200
        body = resp.json()
        assert "status" in body
        assert "model_loaded" in body


# ── starlette-raised routing/body-parse exceptions (404/405/400) ──


def _handler_warnings(caplog) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == "slo.exception_handlers" and r.levelno >= logging.WARNING
    ]


class TestStarletteRoutingExceptions:
    """``starlette.exceptions.HTTPException`` is a DIFFERENT class from
    ``fastapi.exceptions.HTTPException``.

    404 (no matching route) and 405 (method not allowed) are raised by the
    router itself, and 400 "There was an error parsing the body" by FastAPI's
    body parser on a client disconnect. They used to be served silently by
    FastAPI's built-in handler — no log line, which made the docstore
    400/405 storms invisible. They must now be logged while keeping the
    ``{"detail": ...}`` body shape clients rely on.
    """

    def test_unknown_route_404_is_logged_keeps_detail_body(self, caplog):
        with caplog.at_level(logging.WARNING, logger="slo.exception_handlers"):
            resp = client.get("/__definitely_not_a_route__")
        assert resp.status_code == 404
        assert "detail" in resp.json()
        assert any("HTTP 404" in msg for msg in _handler_warnings(caplog))

    def test_wrong_method_405_is_logged_keeps_detail_body(self, caplog):
        with caplog.at_level(logging.WARNING, logger="slo.exception_handlers"):
            resp = client.post("/health", json={})
        assert resp.status_code == 405
        assert "detail" in resp.json()
        assert any("HTTP 405" in msg for msg in _handler_warnings(caplog))

    @pytest.mark.asyncio
    async def test_body_parse_400_is_logged(self, caplog):
        """Client disconnect mid-body → FastAPI raises starlette HTTPException(400).

        httpx's TestClient cannot drop the connection mid-request, so drive the
        ASGI app directly: announce a 999-byte JSON body, deliver a fragment,
        then send http.disconnect — Request.stream() raises ClientDisconnect and
        FastAPI converts it to the 400 "error parsing the body" storm we saw in
        the docstore logs.
        """
        from main import app

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
            # PUT /docstore/{collection}/{doc_id} exists (put_doc) and parses a
            # JSON body — the exact route from the 400 storms in the logs.
            "method": "PUT",
            "scheme": "http",
            "path": "/docstore/kv/_disconnect_probe",
            "raw_path": b"/docstore/kv/_disconnect_probe",
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
            await app(scope, receive, send)

        start = next(m for m in received if m["type"] == "http.response.start")
        assert start["status"] == 400
        assert any("HTTP 400" in msg for msg in _handler_warnings(caplog))
