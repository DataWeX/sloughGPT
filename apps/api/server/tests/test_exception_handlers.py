"""Tests for exception handlers — serialization guard middleware, HTTP, and unhandled errors."""

import pytest
from pydantic import BaseModel
from tests.test_support import get_test_client

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
