"""Contract Router — the descriptor-projected pilot (card 57b0ea27, deliverable 5).

The pilot exists to prove the acceptance criterion end to end: **one descriptor
entry yields a real route with the right verb, envelope and auth scope**, and
reading the contracts back works without anyone maintaining a list.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient
from infrastructure.contract import resolve_method
from infrastructure.exception_handlers import register_all_handlers
from routers.contracts import CONTRACTS_SPEC, router


def _client() -> TestClient:
    app = FastAPI()
    register_all_handlers(app)
    app.include_router(router)
    return TestClient(app, raise_server_exceptions=False)


REQUIRED_DESCRIPTOR_FIELDS = ("name", "version", "auth_scope", "idempotent", "params")


def test_descriptor_projects_to_get():
    """The descriptor says idempotent -> the projection must serve GET."""
    assert CONTRACTS_SPEC.idempotent is True
    assert resolve_method(CONTRACTS_SPEC) == "GET"

    client = _client()
    assert client.get("/contracts").status_code == 200
    assert client.post("/contracts", json={}).status_code == 405


def test_serves_its_own_descriptor():
    """The registry is populated by projection: the endpoint finds itself."""
    body = _client().get("/contracts?name=contracts/list").json()
    assert body["status"] == "success"
    entries = body["data"]["contracts"]
    assert entries, "create_router did not register its own contract"

    entry = entries[0]
    assert entry["method"] == "GET"
    assert entry["path"] == "/contracts"
    assert entry["name"] == "contracts/list"
    for field in REQUIRED_DESCRIPTOR_FIELDS:
        assert field in entry, f"contract entry missing descriptor field {field!r}"


def test_every_registered_contract_is_complete():
    """No projected contract may ship without auth_scope or a params schema.

    This is the blocking half of the CI contract gate (card deliverable 3)
    expressed as a test: a descriptor that cannot describe its own inputs or
    say who may call it is not a contract.
    """
    client = _client()
    entries = client.get("/contracts").json()["data"]["contracts"]
    assert entries, "expected at least the pilot's own contract"

    for entry in entries:
        assert entry["auth_scope"], f"{entry['name']} has no auth_scope"
        assert entry["params"].get("type") == "object", (
            f"{entry['name']} has no object params schema"
        )
        assert entry["version"], f"{entry['name']} has no version"


def test_openapi_carries_x_contract():
    """The contract is readable out of the OpenAPI document, not just JSON."""
    op = _client().get("/openapi.json").json()["paths"]["/contracts"]["get"]
    contract = op["x-contract"]
    assert contract["name"] == "contracts/list"
    assert contract["idempotent"] is True
    assert contract["auth_scope"] == "public"
    assert contract["emitted_by"] == "create_router"


def test_filter_matches_on_descriptor_name():
    """?name= filters case-insensitively; a miss returns an empty list, not 404."""
    client = _client()
    hit = client.get("/contracts?name=CONTRACTS").json()["data"]
    assert hit["count"] >= 1

    miss = client.get("/contracts?name=no-such-descriptor").json()["data"]
    assert miss["count"] == 0
    assert miss["contracts"] == []
