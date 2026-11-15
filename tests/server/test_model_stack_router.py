"""Router-level tests for the model-stack routes (parity gate: w/o tests: 2 → 0).

Five routes, previously untested — absent coverage is what let
POST /model-stack/push report a MISSING REQUEST FIELD as HTTP 404
E_NOT_FOUND (card 3bd79e78 BUG 2): a malformed body told the client the
URL did not exist. A missing required field is a 422 validation failure.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.routers import model_stack as model_stack_module
from apps.api.server.routers.model_stack import ModelStackRouter


@pytest.fixture(autouse=True)
def _reset_stack():
    """The router keeps a process-global singleton; isolate every test."""
    model_stack_module._stack = None
    yield
    model_stack_module._stack = None


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    from apps.api.server.infrastructure.exception_handlers import register_all_handlers

    register_all_handlers(app)
    app.include_router(ModelStackRouter().router)
    return TestClient(app)


class TestModelStackRoutes:
    def test_get_stack(self, client):
        r = client.get("/model-stack")
        assert r.status_code == 200
        assert "stack" in r.json()["data"] or "layers" in r.json()["data"]

    def test_set_base(self, client, tmp_path):
        r = client.post(
            "/model-stack/base", json={"model_id": "m1", "path": str(tmp_path / "m1.soul")}
        )
        assert r.status_code == 200

    def test_push_and_remove_layer(self, client, tmp_path):
        layer = tmp_path / "kb.txt"
        layer.write_text("knowledge")
        r = client.post(
            "/model-stack/push",
            json={"kind": "knowledge", "name": "kb1", "path": str(layer)},
        )
        assert r.status_code == 200
        r = client.delete("/model-stack/kb1")
        assert r.status_code == 200
        assert r.json()["data"]["removed"] is True

    def test_clear_layers(self, client):
        r = client.post("/model-stack/clear", json={})
        assert r.status_code == 200
        assert r.json()["data"]["cleared"] >= 0

    def test_route_count(self):
        routes = ModelStackRouter().router
        paths = {r.path for r in routes.routes}
        assert paths == {
            "/model-stack",
            "/model-stack/base",
            "/model-stack/push",
            "/model-stack/{name}",
            "/model-stack/clear",
        }


class TestPushValidation:
    """Card 3bd79e78 BUG 2 — missing required field must be 422, not 404.

    ``payload["name"]`` raised KeyError, which classify_and_raise reported
    as E_NOT_FOUND 404: the client was told the URL does not exist when the
    request BODY was malformed.
    """

    def test_push_missing_name_is_422(self, client, tmp_path):
        r = client.post("/model-stack/push", json={"path": str(tmp_path / "x")})
        assert r.status_code == 422, (
            f"a missing body field is a validation failure, got {r.status_code} {r.json()}"
        )
        assert r.json()["code"] != "E_NOT_FOUND"

    def test_push_missing_path_is_422(self, client):
        r = client.post("/model-stack/push", json={"name": "kb1"})
        assert r.status_code == 422
        assert r.json()["code"] != "E_NOT_FOUND"

    def test_push_missing_both_is_422(self, client):
        r = client.post("/model-stack/push", json={"kind": "knowledge"})
        assert r.status_code == 422

    def test_push_still_accepts_valid_body(self, client, tmp_path):
        layer = Path(tmp_path) / "kb.txt"
        layer.write_text("knowledge")
        r = client.post(
            "/model-stack/push",
            json={"kind": "knowledge", "name": "kb1", "path": str(layer)},
        )
        assert r.status_code == 200
