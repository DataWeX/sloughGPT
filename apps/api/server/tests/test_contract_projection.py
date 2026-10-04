"""Descriptor-projected transport — ``create_router`` (card 57b0ea27).

Covers the five things the card says the projection must derive from the one
descriptor: verb from idempotency, 422 contract violations, the shared
``success_response`` envelope, ``classify_and_raise`` errors, and the
auth scope -> ``require_auth_if_enabled`` wiring — plus the ``x-contract``
OpenAPI marker and the two boot-time guards.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from infrastructure.contract import (
    _dispatch,
    build_request_model,
    check_handler_matches,
    create_router,
    resolve_method,
)
from infrastructure.exception_handlers import register_all_handlers

from domain.agents._internal.tools import ToolParam, ToolSpec


async def _ok(**_kwargs):
    return {"hello": "world"}


async def _boom(**_kwargs):
    raise RuntimeError("kaboom")


def _spec(**overrides) -> ToolSpec:
    base: dict = {
        "name": "doctor/report",
        "description": "Read the doctor report",
        "parameters": [ToolParam("scope", "string", "Scope to read", required=True)],
        "execute": _ok,
        "idempotent": True,
        "auth_scope": "public",
        "version": "1",
    }
    base.update(overrides)
    return ToolSpec(**base)


def _client(router) -> TestClient:
    app = FastAPI()
    register_all_handlers(app)
    app.include_router(router)
    return TestClient(app, raise_server_exceptions=False)


# ── verb derivation ─────────────────────────────────────────────────────────


def test_idempotent_descriptor_projects_to_get():
    assert resolve_method(_spec(idempotent=True)) == "GET"
    assert resolve_method(_spec(idempotent=False)) == "POST"


def test_explicit_override_wins_and_is_logged(caplog):
    with caplog.at_level("INFO", logger="slo.contract"):
        verb = resolve_method(_spec(idempotent=True), override="post")
    assert verb == "POST"
    assert any("explicit method override" in r.message for r in caplog.records)


def test_projected_router_serves_the_derived_verb():
    router = create_router(_spec(idempotent=True), "doctor/report", _ok)
    client = _client(router)
    assert client.get("/doctor/report", params={"scope": "all"}).status_code == 200
    assert client.post("/doctor/report", json={"scope": "all"}).status_code == 405


# ── envelope + contract violations ──────────────────────────────────────────


def test_success_uses_the_shared_envelope():
    client = _client(create_router(_spec(), "doctor/report", _ok))
    resp = client.get("/doctor/report", params={"scope": "all"})
    assert resp.status_code == 200
    assert resp.json() == {"status": "success", "data": {"hello": "world"}}


def test_missing_required_param_is_422():
    client = _client(create_router(_spec(idempotent=False), "doctor/check", _ok))
    resp = client.post("/doctor/check", json={})
    assert resp.status_code == 422


def test_wrong_type_in_query_is_422():
    router = create_router(
        _spec(parameters=[ToolParam("depth", "integer", "How deep", required=True)]),
        "doctor/deep",
        _ok,
    )
    resp = _client(router).get("/doctor/deep", params={"depth": "not-a-number"})
    assert resp.status_code == 422


def test_optional_descriptor_param_defaults_in_query():
    router = create_router(
        _spec(
            idempotent=True,
            parameters=[ToolParam("scope", "string", "Scope", required=False)],
        ),
        "doctor/report",
        _ok,
    )
    resp = _client(router).get("/doctor/report")
    assert resp.status_code == 200


# ── error path ──────────────────────────────────────────────────────────────


def test_handler_error_is_classified_and_tagged_with_router_source():
    from domain.infrastructure._internal.errors import AppError

    spec = _spec()
    with pytest.raises(AppError) as excinfo:
        asyncio.run(_dispatch(spec, _boom, {}))
    assert excinfo.value.source == "router.doctor/report"


def test_handler_error_returns_error_status_over_http():
    client = _client(create_router(_spec(), "doctor/report", _boom))
    resp = client.get("/doctor/report", params={"scope": "all"})
    assert resp.status_code >= 400


# ── auth scope projection ───────────────────────────────────────────────────


def test_auth_scope_projects_to_require_auth_if_enabled(monkeypatch):
    client = _client(create_router(_spec(), "doctor/report", _ok))
    assert client.get("/doctor/report", params={"scope": "x"}).status_code == 200

    monkeypatch.setenv("SLO_AUTH_REQUIRED", "true")
    resp = client.get("/doctor/report", params={"scope": "x"})
    assert resp.status_code == 401


# ── OpenAPI: x-contract marker ──────────────────────────────────────────────


def test_openapi_carries_x_contract():
    client = _client(create_router(_spec(idempotent=False), "doctor/check", _ok))
    paths = client.get("/openapi.json").json()["paths"]
    op = paths["/doctor/check"]["post"]
    contract = op["x-contract"]
    assert contract["name"] == "doctor/report"
    assert contract["version"] == "1"
    assert contract["auth_scope"] == "public"
    assert contract["idempotent"] is False
    assert contract["emitted_by"] == "create_router"
    assert contract["params"]["type"] == "object"
    assert contract["params"]["required"] == ["scope"]


def test_openapi_emits_request_body_from_descriptor():
    client = _client(create_router(_spec(idempotent=False), "doctor/check", _ok))
    op = client.get("/openapi.json").json()["paths"]["/doctor/check"]["post"]
    assert "requestBody" in op


# ── boot-time guards ────────────────────────────────────────────────────────


def test_handler_requiring_undeclared_param_fails_at_build():
    def needs_undeclared(undeclared_arg):
        return undeclared_arg

    with pytest.raises(ValueError, match="does not declare"):
        check_handler_matches(_spec(), needs_undeclared)


def test_unsupported_param_type_fails_at_build():
    with pytest.raises(ValueError, match="unsupported type"):
        build_request_model(_spec(parameters=[ToolParam("weird", "decimal", "x")]))


def test_empty_route_is_rejected():
    with pytest.raises(ValueError, match="empty route"):
        create_router(_spec(), "   ", _ok)


def test_descriptor_without_parameters_needs_no_request_model():
    assert build_request_model(_spec(parameters=[])) is None
    router = create_router(_spec(parameters=[]), "doctor/ping", _ok)
    assert _client(router).get("/doctor/ping").status_code == 200
