"""Contract projection tests — what ``create_router`` derives so nothing else re-decides it.

Covers the invariants of AGENTS.md "Endpoint & Transport Rule": descriptor →
projection → registration, one auth story, envelope and error classification
shared by every projected route, and the boot-time registration (manifest)
that replaced the hand-edited router list.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from infrastructure.auth import require_auth_if_enabled
from infrastructure.contract import (
    REGISTRY,
    ContractRegistry,
    RouteSpec,
    create_router,
    derive_method,
)
from infrastructure.exception_handlers import register_app_error_handler
from routers.contracts import SPEC as CONTRACTS_SPEC

from domain.agents import ToolParam, ToolSpec

ROOT = Path(__file__).resolve().parents[4]
SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import check_contract  # noqa: E402
import gen_router_manifest  # noqa: E402

# ── helpers ────────────────────────────────────────────────────────────


async def _read_echo(value: str | None = None, count: int | None = None, flag: bool | None = None):
    return {"value": value, "count": count, "flag": flag}


async def _write_echo(text: str, count: int = 1):
    return {"text": text, "count": count}


async def _boom():
    raise ValueError("boom")


def _spec(
    name: str,
    *,
    idempotent: bool = True,
    auth_scope: str = "public",
    params: list[ToolParam] | None = None,
    execute=_read_echo,
) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=f"Test capability {name}",
        parameters=params or [],
        execute=execute,
        auth_scope=auth_scope,
        idempotent=idempotent,
        version="1",
    )


def _app_for(*routers, auth_user=None) -> FastAPI:
    app = FastAPI()
    register_app_error_handler(app)
    for router in routers:
        app.include_router(router)
    if auth_user is not None:
        app.dependency_overrides[require_auth_if_enabled] = lambda: auth_user
    return app


def _serve(spec: ToolSpec, route: RouteSpec, handler, *, prefix: str = "/t", auth_user=None):
    """Build a throwaway app backed by its own registry (never the global one)."""
    registry = ContractRegistry()
    router = create_router(spec, route, handler, prefix=prefix, registry=registry)
    return _app_for(router, auth_user=auth_user), registry


# ── verb derivation (the safety-relevant default) ──────────────────────


def test_idempotent_capability_projects_to_get():
    assert derive_method(_spec("t.read", idempotent=True)) == "GET"


def test_action_capability_projects_to_post():
    assert derive_method(_spec("t.write", idempotent=False)) == "POST"


def test_explicit_route_method_overrides_derivation():
    """An override is allowed (verb semantics are stated, not guessed) and wins."""
    registry = ContractRegistry()
    router = create_router(
        _spec("t.explicit", idempotent=True),
        RouteSpec(path="/", method="POST"),
        _write_echo,
        prefix="/t",
        registry=registry,
    )
    assert router.routes[0].methods == {"POST"}
    entry = registry.routes()[0]
    assert entry["method"] == "POST"
    # the registry key mirrors the route FastAPI actually serves
    assert entry["path"] == "/t/"
    assert ("POST", "/t/") in {(e["method"], e["path"]) for e in registry.routes()}


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="unknown HTTP method"):
        create_router(
            _spec("t.badverb"),
            RouteSpec(path="/", method="FETCH"),
            _read_echo,
            registry=ContractRegistry(),
        )


# ── registration invariants (fail at boot, not at 404 in production) ───


def test_duplicate_method_path_is_rejected():
    registry = ContractRegistry()
    spec = _spec("t.dup")
    create_router(spec, RouteSpec(path="/same"), _read_echo, prefix="/dup", registry=registry)
    with pytest.raises(ValueError, match="duplicate contract route GET /dup/same"):
        create_router(spec, RouteSpec(path="/same"), _read_echo, prefix="/dup", registry=registry)


def test_operation_ids_stay_unique():
    """Same capability, two crossings: the base id collides and must disambiguate."""
    registry = ContractRegistry()
    spec = _spec("t.ops")
    create_router(spec, RouteSpec(path="/a"), _read_echo, prefix="/ops", registry=registry)
    create_router(spec, RouteSpec(path="/b"), _read_echo, prefix="/ops", registry=registry)
    ids = [e["operation_id"] for e in registry.routes()]
    assert len(ids) == 2
    assert len(set(ids)) == 2


def test_handler_that_cannot_receive_the_contract_is_rejected():
    async def missing_value():
        return {}

    with pytest.raises(ValueError, match="does not accept declared parameter"):
        create_router(
            _spec("t.mismatch", params=[ToolParam("value", "string", "a value")]),
            RouteSpec(path=""),
            missing_value,
            registry=ContractRegistry(),
        )


def test_reserved_request_parameter_is_rejected():
    with pytest.raises(ValueError, match="reserved"):
        create_router(
            _spec("t.reserved", params=[ToolParam("request", "string", "oops")]),
            RouteSpec(path=""),
            _read_echo,
            registry=ContractRegistry(),
        )


def test_incomplete_descriptor_is_rejected():
    spec = _spec("t.noscope")
    spec.auth_scope = ""
    with pytest.raises(ValueError, match="incomplete descriptor"):
        create_router(spec, RouteSpec(path=""), _read_echo, registry=ContractRegistry())


def test_single_route_requires_a_handler():
    with pytest.raises(ValueError, match="handler is required"):
        create_router(_spec("t.nohandler"), RouteSpec(path=""), registry=ContractRegistry())


# ── read projection: envelope, coercion, contract validation ───────────


def _read_spec(name: str, auth_scope: str = "public") -> ToolSpec:
    return _spec(
        name,
        auth_scope=auth_scope,
        params=[
            ToolParam("count", "integer", "How many results", required=True),
            ToolParam("value", "string", "A free-text value"),
            ToolParam("flag", "boolean", "Whether to enable the flag"),
        ],
    )


def test_read_returns_the_shared_success_envelope():
    app, _ = _serve(_read_spec("t.read"), RouteSpec(path=""), _read_echo)
    resp = TestClient(app).get("/t", params={"count": "3"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["data"] == {"value": None, "count": 3, "flag": None}


def test_query_parameters_are_coerced_to_contract_types():
    app, _ = _serve(_read_spec("t.coerce"), RouteSpec(path=""), _read_echo)
    resp = TestClient(app).get("/t", params={"count": "7", "value": "hi", "flag": "true"})
    assert resp.status_code == 200
    assert resp.json()["data"] == {"value": "hi", "count": 7, "flag": True}


def test_missing_required_parameter_is_a_contract_violation():
    app, _ = _serve(_read_spec("t.required"), RouteSpec(path=""), _read_echo)
    resp = TestClient(app).get("/t")
    assert resp.status_code == 422
    assert resp.json()["detail"][0]["type"] == "missing"


def test_unparseable_parameter_is_a_contract_violation():
    app, _ = _serve(_read_spec("t.badtype"), RouteSpec(path=""), _read_echo)
    resp = TestClient(app).get("/t", params={"count": "many"})
    assert resp.status_code == 422


def test_parameters_outside_the_contract_are_ignored():
    """Cache-busters and such must not break a projected read."""
    app, _ = _serve(_read_spec("t.extra"), RouteSpec(path=""), _read_echo)
    resp = TestClient(app).get("/t", params={"count": "1", "noise": "x"})
    assert resp.status_code == 200


# ── write projection: generated body model from the same descriptor ────


def _write_spec(name: str, auth_scope: str = "public") -> ToolSpec:
    return _spec(
        name,
        idempotent=False,
        auth_scope=auth_scope,
        execute=_write_echo,
        params=[
            ToolParam("text", "string", "The text to write", required=True),
            ToolParam("count", "integer", "Repeat count"),
        ],
    )


def test_write_validates_body_against_the_descriptor():
    app, _ = _serve(_write_spec("t.write"), RouteSpec(path=""), _write_echo, auth_user={"sub": "u"})
    client = TestClient(app)
    ok = client.post("/t", json={"text": "hello", "count": 2})
    assert ok.status_code == 200
    assert ok.json() == {"status": "success", "data": {"text": "hello", "count": 2}}

    missing = client.post("/t", json={"count": 2})
    assert missing.status_code == 422

    bad_type = client.post("/t", json={"text": "x", "count": "twelve"})
    assert bad_type.status_code == 422


def test_action_projects_to_post_only():
    app, _ = _serve(
        _write_spec("t.action"), RouteSpec(path=""), _write_echo, auth_user={"sub": "u"}
    )
    client = TestClient(app)
    assert client.get("/t").status_code == 405
    assert client.post("/t", json={"text": "x"}).status_code == 200


# ── error classification (one story for every projected route) ─────────


def test_handler_failure_is_classified_into_the_error_envelope():
    app, _ = _serve(_spec("t.boom", execute=_boom), RouteSpec(path=""), _boom)
    resp = TestClient(app).get("/t")
    assert resp.status_code == 400
    body = resp.json()
    assert body["code"] == "E_BAD_REQUEST"
    assert body["error"]


# ── auth: one auth story, gated by the descriptor's scope ──────────────


def test_public_scope_serves_anonymous_traffic(monkeypatch):
    monkeypatch.setenv("SLO_AUTH_REQUIRED", "true")
    app, _ = _serve(_read_spec("t.pub", auth_scope="public"), RouteSpec(path="/pub"), _read_echo)
    assert TestClient(app).get("/t/pub", params={"count": "1"}).status_code == 200


def test_authenticated_scope_rejects_anonymous_traffic(monkeypatch):
    monkeypatch.setenv("SLO_AUTH_REQUIRED", "true")
    app, _ = _serve(
        _read_spec("t.priv", auth_scope="authenticated"), RouteSpec(path="/priv"), _read_echo
    )
    resp = TestClient(app).get("/t/priv", params={"count": "1"})
    assert resp.status_code == 401
    assert resp.json()["code"] == "E_AUTH_MISSING"


def test_projected_routes_still_honor_dependency_overrides(monkeypatch):
    """The house test pattern (override require_auth_if_enabled) keeps working."""
    monkeypatch.setenv("SLO_AUTH_REQUIRED", "true")
    app, _ = _serve(
        _read_spec("t.override", auth_scope="authenticated"),
        RouteSpec(path="/ovr"),
        _read_echo,
        auth_user={"sub": "user1"},
    )
    assert TestClient(app).get("/t/ovr", params={"count": "1"}).status_code == 200


# ── OpenAPI is a projection of the contract, not a second declaration ──


def test_openapi_publishes_the_contract():
    app, _ = _serve(_read_spec("t.openapi"), RouteSpec(path=""), _read_echo)
    op = app.openapi()["paths"]["/t"]["get"]
    assert op["x-contract"] == {
        "name": "t.openapi",
        "version": "1",
        "auth_scope": "public",
        "idempotent": True,
    }
    names = {p["name"]: p for p in op["parameters"]}
    assert set(names) == {"count", "value", "flag"}
    assert names["count"]["in"] == "query"
    assert names["count"]["required"] is True
    assert names["value"]["schema"]["type"] == "string"


def test_openapi_request_body_comes_from_the_generated_model():
    app, _ = _serve(_write_spec("t.body"), RouteSpec(path=""), _write_echo, auth_user={"sub": "u"})
    spec_doc = app.openapi()
    op = spec_doc["paths"]["/t"]["post"]
    schema = op["requestBody"]["content"]["application/json"]["schema"]
    if "$ref" in schema:  # generated models live in #/components/schemas
        schema = spec_doc["components"]["schemas"][schema["$ref"].rsplit("/", 1)[-1]]
    props = schema["properties"]
    assert set(props) == {"text", "count"}
    assert props["text"]["type"] == "string"


# ── boot self-registration (the manifest replaced the hand-edited list) ─


def test_router_manifest_is_fresh():
    assert gen_router_manifest.main(["--check"]) == 0


def test_contract_gate_static_checks_pass():
    assert check_contract.main(["--static"]) == 0


def test_importing_the_routers_package_stays_lazy():
    """Cold-start guard: no router module may load until get_all_routers()."""
    code = (
        "import sys, routers\n"
        "heavy = [m for m in sys.modules if m.startswith('routers.') and m != 'routers._manifest']\n"
        "print(heavy)\n"
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True
    )
    assert out.stdout.strip() == "[]", f"router modules imported eagerly: {out.stdout.strip()}"


# ── the pilot crossing: GET /contracts ─────────────────────────────────


def _pilot_app() -> FastAPI:
    from routers.contracts import router as contracts_router

    return _app_for(contracts_router, auth_user={"sub": "user1"})


def test_pilot_serves_the_contract_inventory():
    resp = TestClient(_pilot_app()).get("/contracts")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    data = body["data"]
    assert data["count"] == len(data["contracts"])
    entry = next(e for e in data["contracts"] if e["name"] == "contracts.list")
    assert entry["method"] == "GET"
    assert entry["path"] == "/contracts"
    assert entry["auth_scope"] == "authenticated"
    assert entry["idempotent"] is True
    assert entry["module"] == "routers.contracts"
    assert entry["version"] == CONTRACTS_SPEC.version
    assert "prefix" in entry["params"]["properties"]


def test_pilot_is_documented_in_openapi():
    op = _pilot_app().openapi()["paths"]["/contracts"]["get"]
    assert op["operationId"] == "contracts.list.get"
    assert op["x-contract"]["name"] == "contracts.list"
    assert [p["name"] for p in op["parameters"]] == ["prefix"]


def test_pilot_registry_entry_is_actually_served():
    """The registry's claim must match the router's route table (not a paper one)."""
    from routers.contracts import router as contracts_router

    served = {
        (route.path, method)
        for route in contracts_router.routes
        for method in route.methods
        if method != "HEAD"
    }
    assert ("/contracts", "GET") in served
    claimed = {(e["path"], e["method"]) for e in REGISTRY.routes()}
    assert claimed <= served, f"registry claims routes no router serves: {claimed - served}"


def test_pilot_prefix_filter():
    app = _pilot_app()
    resp = TestClient(app).get("/contracts", params={"prefix": "/nothing-matches"})
    assert resp.status_code == 200
    assert resp.json()["data"]["contracts"] == []
