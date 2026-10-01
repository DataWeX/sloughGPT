"""
Tests for the ModelStack router endpoints (apps/api/server/routers/model_stack.py).

Covers GET /model-stack, POST /model-stack/base, POST /model-stack/push,
DELETE /model-stack/{name}, POST /model-stack/clear — happy path + 404/405/422
+ auth edge. The in-memory module singleton is reset per test.
"""

import pytest
from test_support import _data, get_test_client

client = get_test_client()


@pytest.fixture(autouse=True)
def _fresh_stack():
    """Reset the module-level singleton so tests never leak stack state."""
    import routers.model_stack as ms

    ms._stack = None
    yield
    ms._stack = None


class TestGetStack:
    """GET /model-stack — current stack view."""

    def test_get_stack_default(self):
        resp = client.get("/model-stack")
        assert resp.status_code == 200
        data = _data(resp)
        assert data["base"]["model_id"] == "base"
        assert data["layers"] == []

    def test_unknown_nested_path_is_404(self):
        resp = client.post("/model-stack/base/extra", json={})
        assert resp.status_code == 404


class TestSetBase:
    """POST /model-stack/base — swap the base model, keep layers."""

    def test_set_base_happy_path(self, tmp_path):
        resp = client.post(
            "/model-stack/base",
            json={"model_id": "m1", "path": str(tmp_path / "m1.soul")},
        )
        assert resp.status_code == 200
        assert _data(resp)["base"]["model_id"] == "m1"

    def test_set_base_preserves_layers(self, tmp_path):
        client.post(
            "/model-stack/push",
            json={"kind": "knowledge", "name": "facts", "path": str(tmp_path)},
        )
        resp = client.post(
            "/model-stack/base",
            json={"model_id": "m2", "path": str(tmp_path / "m2.soul")},
        )
        assert resp.status_code == 200
        data = _data(resp)
        assert data["base"]["model_id"] == "m2"
        assert [l["name"] for l in data["layers"]] == ["facts"]

    def test_set_base_missing_body_is_422(self):
        resp = client.post("/model-stack/base")
        assert resp.status_code == 422


class TestPushLayer:
    """POST /model-stack/push — overlay layer on top of the stack."""

    def test_push_happy_path(self, tmp_path):
        resp = client.post(
            "/model-stack/push",
            json={"kind": "knowledge", "name": "kb1", "path": str(tmp_path)},
        )
        assert resp.status_code == 200
        assert [l["name"] for l in _data(resp)["layers"]] == ["kb1"]

    def test_push_missing_body_is_422(self):
        resp = client.post("/model-stack/push")
        assert resp.status_code == 422


class TestRemoveAndClear:
    """DELETE /model-stack/{name} and POST /model-stack/clear."""

    def test_remove_existing_layer(self, tmp_path):
        client.post(
            "/model-stack/push",
            json={"kind": "cache", "name": "c1", "path": str(tmp_path)},
        )
        resp = client.delete("/model-stack/c1")
        assert resp.status_code == 200
        data = _data(resp)
        assert data["removed"] is True
        assert data["stack"]["layers"] == []

    def test_remove_missing_layer_is_not_an_error(self):
        resp = client.delete("/model-stack/ghost")
        assert resp.status_code == 200
        assert _data(resp)["removed"] is False

    def test_clear_reports_cleared_count(self, tmp_path):
        client.post(
            "/model-stack/push",
            json={"kind": "knowledge", "name": "a", "path": str(tmp_path)},
        )
        client.post(
            "/model-stack/push",
            json={"kind": "cache", "name": "b", "path": str(tmp_path)},
        )
        resp = client.post("/model-stack/clear", json={})
        assert resp.status_code == 200
        assert _data(resp)["cleared"] == 2

    def test_clear_filters_by_kind(self, tmp_path):
        client.post(
            "/model-stack/push",
            json={"kind": "knowledge", "name": "a", "path": str(tmp_path)},
        )
        client.post(
            "/model-stack/push",
            json={"kind": "cache", "name": "b", "path": str(tmp_path)},
        )
        resp = client.post("/model-stack/clear", json={"kind": "cache"})
        assert resp.status_code == 200
        data = _data(resp)
        assert data["cleared"] == 1
        assert [l["kind"] for l in data["stack"]["layers"]] == ["knowledge"]

    def test_get_with_unallowed_method_is_405(self):
        # /model-stack/clear exists as POST; /{name} as DELETE
        resp = client.get("/model-stack/clear")
        assert resp.status_code == 405


class TestModelStackAuth:
    """Auth edge: set_base carries the auth dependency."""

    def test_auth_disabled_allows_anonymous(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SLO_AUTH_REQUIRED", "false")
        resp = client.post(
            "/model-stack/base",
            json={"model_id": "m", "path": str(tmp_path / "m.soul")},
        )
        assert resp.status_code == 200

    def test_auth_enabled_rejects_missing_token(self, monkeypatch):
        monkeypatch.setenv("SLO_AUTH_REQUIRED", "true")
        resp = client.post("/model-stack/base", json={"model_id": "m"})
        assert resp.status_code == 401
