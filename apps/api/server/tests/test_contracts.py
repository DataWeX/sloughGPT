"""Contract projection tests — what ``create_router`` derives so nothing else re-decides it.

Covers the invariants of AGENTS.md "Endpoint & Transport Rule": descriptor →
projection → registration, one auth story, envelope and error classification
shared by every projected route, and the boot-time registration (manifest)
that replaced the hand-edited router list.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from infrastructure.auth import require_auth_if_enabled
from infrastructure.contract import (
    REGISTRY,
    ContractRegistry,
    RouteEntry,
    RouteSpec,
    create_router,
    derive_method,
    params_from_model,
)
from infrastructure.exception_handlers import register_app_error_handler
from pydantic import BaseModel, Field
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


def test_registration_is_idempotent_across_import_identities():
    """Dual-identity imports re-execute the module; that is not a collision.

    ``training.control`` and ``apps.api.server.training.control`` are two
    names for one file (tests import and ``@patch`` via both) — the same
    capability re-claiming its own crossing from the second identity must be
    absorbed, while a different capability on that crossing, and a same-
    identity re-registration (a double projection), must still raise.
    """
    registry = ContractRegistry()
    entry = RouteEntry(
        name="t.ident",
        version="1",
        auth_scope="public",
        idempotent=True,
        description="",
        method="GET",
        path="/ident",
        operation_id="t.ident.get",
        module="training.control",
    )
    registry.register(entry)
    registry.register(replace(entry, module="apps.api.server.training.control"))
    assert [e["name"] for e in registry.routes()] == ["t.ident"]
    with pytest.raises(ValueError, match="duplicate contract route"):
        registry.register(replace(entry, name="t.other"))
    with pytest.raises(ValueError, match="duplicate contract route"):
        registry.register(entry)


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


def test_params_from_model_derives_the_model_fields():
    """Model-style routes document their model — one declaration, both views."""

    class _Req(BaseModel):
        name: str = Field(description="job name")
        epochs: int = 3
        device: str | None = None
        tags: list[str] = Field(default_factory=list)

    by_name = {p.name: p for p in params_from_model(_Req)}
    assert set(by_name) == {"name", "epochs", "device", "tags"}
    assert (by_name["name"].type, by_name["name"].required) == ("string", True)
    assert by_name["name"].description == "job name"
    assert (by_name["epochs"].type, by_name["epochs"].required) == ("integer", False)
    assert by_name["device"].type == "string"  # ``str | None`` unwraps
    assert by_name["tags"].type == "array"


async def _idle():
    return {"status": "idle"}


async def _ping():
    return {"ok": True}


def test_route_with_nothing_to_validate_publishes_no_422():
    """No params, no path placeholders, no body → 200 alone (control parity).

    FastAPI auto-422s on *any* body field, even an optional empty one, so a
    no-param write must not carry one — the hand-built twin published exactly
    ``{"200"}`` and so must the projection.
    """
    app, _ = _serve(
        _spec("t.idle", idempotent=False, execute=_idle),
        RouteSpec(path="/idle", method="POST"),
        _idle,
    )
    op = app.openapi()["paths"]["/t/idle"]["post"]
    assert sorted(op["responses"]) == ["200"]
    assert "requestBody" not in op
    assert TestClient(app).post("/t/idle").status_code == 200


def test_paramless_read_publishes_no_422():
    """A GET with nothing to validate serves 200 alone (builds parity)."""
    app, _ = _serve(_spec("t.ping", execute=_ping), RouteSpec(path="/ping"), _ping)
    op = app.openapi()["paths"]["/t/ping"]["get"]
    assert sorted(op["responses"]) == ["200"]
    assert not op.get("parameters")


# ── boot self-registration (the manifest replaced the hand-edited list) ─


def test_router_manifest_is_fresh():
    assert gen_router_manifest.main(["--check"]) == 0


def test_contract_gate_static_checks_pass():
    assert check_contract.main(["--static"]) == 0


# TS projection freshness lives in test_ts_contracts.py (subprocess): an
# in-process check here saw rows other tests had registered into the global
# REGISTRY and went red on suite order, duplicating that gate.


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


# ── OpenAPI drift report (path+method is not the whole contract) ───────


def _facts_for(app: FastAPI) -> dict:
    return check_contract._openapi_operations(app)


def test_openapi_facts_capture_params_body_and_responses():
    facts = _facts_for(_pilot_app())
    op = facts["/contracts"]["GET"]
    assert op["operationId"] == "contracts.list.get"
    assert "200" in op["responses"] and "422" in op["responses"]
    assert op["body"] is None  # GET has no request body
    by_name = {p["name"]: p for p in op["params"]}
    assert by_name["prefix"]["in"] == "query"
    assert {"name", "in", "required", "type"} == set(by_name["prefix"])


def test_openapi_diff_flags_shape_change_at_an_unchanged_path():
    """Same path, same verb, different contract — the signal a route diff misses."""
    before = {
        "/thing": {
            "GET": {
                "operationId": "thing_get",
                "params": [{"name": "q", "in": "query", "required": False, "type": "string"}],
                "body": None,
                "responses": ["200"],
            }
        }
    }
    after = {
        "/thing": {
            "GET": {
                "operationId": "thing_get",
                "params": [{"name": "q", "in": "query", "required": True, "type": "integer"}],
                "body": None,
                "responses": ["200", "401"],
            }
        }
    }
    changes = check_contract._diff_operations(after, before)
    assert changes == [
        "~ GET /thing — params: "
        "[{'name': 'q', 'in': 'query', 'required': False, 'type': 'string'}] -> "
        "[{'name': 'q', 'in': 'query', 'required': True, 'type': 'integer'}]; "
        "responses: ['200'] -> ['200', '401']"
    ]


def test_openapi_diff_reports_added_and_removed_operations():
    old = {"/gone": {"GET": {"operationId": "g", "params": [], "body": None, "responses": ["200"]}}}
    new = {
        "/fresh": {"POST": {"operationId": "f", "params": [], "body": None, "responses": ["200"]}}
    }
    assert check_contract._diff_operations(new, old) == [
        "- /gone removed",
        "+ /fresh added",
    ]
    # a verb swapped on an existing path reads as remove + add
    swapped = {
        "/gone": {"DELETE": {"operationId": "g", "params": [], "body": None, "responses": ["200"]}}
    }
    assert check_contract._diff_operations(swapped, old) == [
        "- GET /gone removed",
        "+ DELETE /gone added",
    ]


def test_openapi_diff_is_quiet_when_nothing_changed():
    facts = _facts_for(_pilot_app())
    assert check_contract._diff_operations(facts, facts) == []


def test_committed_baseline_carries_openapi_operations():
    """The baseline must ship contract facts, else the drift report is vacuous."""
    stored = json.loads((ROOT / "apps" / "api" / "server" / "contract-baseline.json").read_text())
    op = stored["operations"]["/contracts"]["GET"]
    assert op["operationId"] == "contracts.list.get"
    assert "200" in op["responses"]
    assert any(p["name"] == "prefix" for p in op["params"])


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
    """The registry's claim must match the router's route table (not a paper one).

    Scoped to the pilot module's own claims: the registry also carries rows
    from every other descriptor router in the boot set (e.g. llm_v1), whose
    served-or-not state is check_contract --runtime's invariant, not this
    router's.
    """
    from routers.contracts import router as contracts_router

    served = {
        (route.path, method)
        for route in contracts_router.routes
        for method in route.methods
        if method != "HEAD"
    }
    assert ("/contracts", "GET") in served
    claimed = {
        (e["path"], e["method"]) for e in REGISTRY.routes() if e["module"] == "routers.contracts"
    }
    assert claimed <= served, f"registry claims routes no router serves: {claimed - served}"


def test_pilot_prefix_filter():
    app = _pilot_app()
    resp = TestClient(app).get("/contracts", params={"prefix": "/nothing-matches"})
    assert resp.status_code == 200
    assert resp.json()["data"]["contracts"] == []


# ── declaration-style ratchet: hand-built routes are frozen, not free ──
#
# The endpoint rule (AGENTS.md) says routes are projections of a descriptor.
# The grandfathered backlog is declared once in handbuilt-routes-baseline.json;
# this gate keeps the count from ever growing again. See
# docs/TRANSPORT_PROJECTIONS.md.


def _write_py(root: Path, rel: str, src: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(src, encoding="utf-8")


def test_scanner_counts_handbuilt_but_not_projected_declarations():
    """Decorators and add_api_route are hand-built; create_router is the seam."""
    src = """\
from infrastructure.contract import create_router

router = APIRouter()

@router.get("/x")
async def x(): ...

def mount():
    r.add_api_route("/y", handler, methods=["POST"])
    create_router(spec, route, handler)
    app.include_router(other)
"""
    labels = sorted(label for _, label in check_contract.find_handbuilt_declarations(src))
    assert labels == sorted(["@router.get", "r.add_api_route"])


def test_scan_scope_excludes_tests_and_projection_internals(tmp_path: Path):
    """Test fixtures and create_router's own add_api_route are not the surface."""
    handbuilt = '@router.get("/x")\nasync def x(): ...\n'
    _write_py(tmp_path, "routers/a.py", handbuilt)
    _write_py(tmp_path, "tests/test_a.py", handbuilt)
    _write_py(tmp_path, "infrastructure/contract.py", 'r.add_api_route("/y", h)\n')
    _write_py(tmp_path, "infrastructure/other.py", handbuilt)
    _write_py(tmp_path, "routers/_generated.py", handbuilt)
    sites = check_contract.scan_handbuilt(tmp_path)
    assert set(sites) == {"routers/a.py", "infrastructure/other.py"}


def test_new_handbuilt_route_beyond_baseline_is_a_violation():
    sites = {"routers/new.py": [(3, "@router.get")]}
    violations = check_contract.check_declaration_style(sites, {})
    assert len(violations) == 1
    assert "routers/new.py" in violations[0]
    assert "create_router" in violations[0]  # tells the author the way out


def test_grandfathered_counts_pass_quietly():
    sites = {"routers/old.py": [(1, "@r.get"), (2, "@r.post")]}
    assert check_contract.check_declaration_style(sites, {"routers/old.py": 2}) == []


def test_one_extra_route_in_a_grandfathered_file_still_fails():
    """The ratchet bites where the backlog lives — counts may never grow."""
    sites = {"routers/old.py": [(1, "@r.get"), (2, "@r.post"), (9, "@r.delete")]}
    violations = check_contract.check_declaration_style(sites, {"routers/old.py": 2})
    assert len(violations) == 1
    assert "routers/old.py" in violations[0]


def test_shrunk_file_fails_until_the_baseline_entry_shrinks_with_it():
    """A migration must show up as a reviewed baseline edit, not silently."""
    sites = {"routers/old.py": [(1, "@r.get")]}
    violations = check_contract.check_declaration_style(sites, {"routers/old.py": 3})
    assert len(violations) == 1
    assert "shrink" in violations[0]


def test_leftover_baseline_entry_after_migration_fails():
    violations = check_contract.check_declaration_style({}, {"routers/gone.py": 5})
    assert len(violations) == 1
    assert "routers/gone.py" in violations[0]
    assert "remove" in violations[0]


def test_committed_declaration_baseline_matches_the_tree():
    """Integration: the real tree at rest must equal the committed baseline."""
    sites = check_contract.scan_handbuilt(check_contract.SERVER_DIR)
    stored = json.loads(
        (check_contract.SERVER_DIR / "handbuilt-routes-baseline.json").read_text(encoding="utf-8")
    )
    assert check_contract.check_declaration_style(sites, stored["files"]) == []


# ── projection facets (path params, raw wire, auth_user, body styles) ──
# What the training migration wave (card 20261104_151036) projects from:
# every capability the hand-built routes express, without re-deciding it.


async def _job_read(job_id: str, limit: int = 10):
    return {"id": job_id, "limit": limit}


def test_path_param_flows_to_the_handler():
    app, _ = _serve(
        _spec(
            "t.path.read",
            params=[ToolParam("job_id", "string", "id"), ToolParam("limit", "integer", "n")],
            execute=_job_read,
        ),
        RouteSpec(path="/jobs/{job_id}"),
        _job_read,
    )
    resp = TestClient(app).get("/t/jobs/abc?limit=5")
    assert resp.status_code == 200
    assert resp.json()["data"] == {"id": "abc", "limit": 5}


def test_path_param_is_not_required_from_the_query():
    """The URL template binds it — a missing query twin must not 422."""
    app, _ = _serve(
        _spec("t.path.only", params=[ToolParam("job_id", "string", "id")], execute=_job_read),
        RouteSpec(path="/jobs/{job_id}"),
        _job_read,
    )
    resp = TestClient(app).get("/t/jobs/xyz")
    assert resp.status_code == 200
    assert resp.json()["data"]["id"] == "xyz"


def test_path_params_are_published_as_path_parameters():
    app, _ = _serve(
        _spec("t.path.openapi", params=[ToolParam("job_id", "string", "id")], execute=_job_read),
        RouteSpec(path="/jobs/{job_id}"),
        _job_read,
    )
    op = app.openapi()["paths"]["/t/jobs/{job_id}"]["get"]
    assert any(p["in"] == "path" and p["name"] == "job_id" for p in op["parameters"])
    assert op["x-contract"]["name"] == "t.path.openapi"


def test_handler_must_accept_the_path_placeholder():
    with pytest.raises(ValueError, match="path parameter"):
        create_router(
            _spec("t.path.bad"),
            RouteSpec(path="/jobs/{job_id}"),
            _read_echo,  # takes value/count/flag — no job_id
            registry=ContractRegistry(),
        )


def test_envelope_facet_serves_the_raw_wire():
    """envelope=False keeps callers that parse a bare payload byte-identical."""

    async def _raw():
        return {"builds": [1, 2]}

    app, _ = _serve(
        _spec("t.raw", execute=_raw),
        RouteSpec(path="/raw", envelope=False),
        _raw,
    )
    resp = TestClient(app).get("/t/raw")
    assert resp.status_code == 200
    assert resp.json() == {"builds": [1, 2]}  # no {status, data} wrapper


def test_envelope_facet_passes_streaming_responses_through():
    """SSE survives the projection only on the raw wire — the stream is the body."""
    from starlette.responses import StreamingResponse

    async def _stream():
        async def _gen():
            yield b"data: hi\n\n"

        return StreamingResponse(_gen(), media_type="text/event-stream")

    app, _ = _serve(
        _spec("t.sse", execute=_stream),
        RouteSpec(path="/stream", envelope=False),
        _stream,
    )
    resp = TestClient(app).get("/t/stream")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert "data: hi" in resp.text


async def _who(job_id: str, auth_user: dict | None = None):
    return {"sub": (auth_user or {}).get("sub")}


def test_auth_user_is_forwarded_to_handlers_that_declare_it():
    app, _ = _serve(
        _spec(
            "t.who",
            params=[ToolParam("job_id", "string", "id")],
            auth_scope="authenticated",
            execute=_who,
        ),
        RouteSpec(path="/jobs/{job_id}"),
        _who,
        auth_user={"sub": "u1"},
    )
    resp = TestClient(app).get("/t/jobs/x")
    assert resp.status_code == 200
    assert resp.json()["data"] == {"sub": "u1"}


def test_auth_user_handler_on_a_public_scope_fails_at_boot():
    """No scope → no token enforcement → auth_user could never arrive."""
    with pytest.raises(ValueError, match="auth_user"):
        create_router(
            _spec("t.who.public", params=[ToolParam("job_id", "string", "id")], execute=_who),
            RouteSpec(path="/jobs/{job_id}"),
            _who,
            registry=ContractRegistry(),
        )


class _StartReq(BaseModel):
    epochs: int = Field(default=3, ge=1, le=1000)
    name: str = "model"


async def _start(req: _StartReq, auth_user: dict | None = None):
    return {"epochs": req.epochs, "is_model": isinstance(req, _StartReq)}


def test_model_style_body_passes_the_validated_instance():
    """The handler's pydantic annotation *is* the body validator (constraints on)."""
    app, _ = _serve(
        _spec(
            "t.start",
            params=[ToolParam("epochs", "integer", "e"), ToolParam("name", "string", "n")],
            idempotent=False,
            auth_scope="authenticated",
            execute=_start,
        ),
        RouteSpec(path="/start", method="POST"),
        _start,
        auth_user={"sub": "u"},
    )
    client = TestClient(app)
    ok = client.post("/t/start", json={"epochs": 5})
    assert ok.status_code == 200
    assert ok.json()["data"] == {"epochs": 5, "is_model": True}
    assert client.post("/t/start", json={"epochs": 0}).status_code == 422  # ge=1 holds
    assert client.post("/t/start", json={"epochs": 9999}).status_code == 422  # le=1000 holds
    assert client.post("/t/start", json={}).json()["data"]["epochs"] == 3  # default


def test_model_style_body_on_a_query_route_fails_at_boot():
    with pytest.raises(ValueError, match="body validator"):
        create_router(
            _spec("t.start.bad", idempotent=False, auth_scope="authenticated", execute=_start),
            RouteSpec(path="/start", method="POST", input="query"),
            _start,
            registry=ContractRegistry(),
        )


async def _pause():
    return {"status": "paused"}


def test_write_route_without_body_params_accepts_an_absent_body():
    """Hand-built no-arg POSTs never required a body; Body(...) would 422 them."""
    app, _ = _serve(
        _spec("t.pause", idempotent=False, execute=_pause),
        RouteSpec(path="/pause", method="POST"),
        _pause,
    )
    resp = TestClient(app).post("/t/pause")
    assert resp.status_code == 200
    assert resp.json()["data"] == {"status": "paused"}


async def _purge(status: str | None = None):
    return {"status": status or "all"}


def test_query_input_write_reads_the_query_and_ignores_the_body():
    """POST .../purge?status=... — today's working caller wire, preserved."""
    app, _ = _serve(
        _spec(
            "t.purge", params=[ToolParam("status", "string", "s")], idempotent=False, execute=_purge
        ),
        RouteSpec(path="/purge", method="POST", input="query"),
        _purge,
    )
    client = TestClient(app)
    assert client.post("/t/purge?status=failed").json()["data"] == {"status": "failed"}
    assert client.post("/t/purge", json={"status": "ignored"}).json()["data"] == {"status": "all"}


def test_mixed_body_styles_fail_at_boot():
    """One route, one body style — a model alongside flat params is refused."""

    async def _mixed(req: _StartReq, extra: str = ""):
        return {"extra": extra}

    with pytest.raises(ValueError, match="one body style"):
        create_router(
            _spec("t.mixed", idempotent=False, execute=_mixed),
            RouteSpec(path="/mixed", method="POST"),
            _mixed,
            registry=ContractRegistry(),
        )
