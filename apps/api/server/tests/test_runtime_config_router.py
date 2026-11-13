"""Tests for the descriptor-projected runtime-config crossing.

The route lives in ``routers/runtime_config.py`` (descriptor + projection, see
docs/TRANSPORT_PROJECTIONS.md) — its tests live here so the module name, the
test file name and the docs parity check agree. Card 20261108_134211: the
server's feature flags (versioned, env-overridable, ledgered) had zero exposure
to the browser; this crossing serves the resolved document with a sha256
``config_version`` as ETag so the client can cache it for 30s.
"""

from __future__ import annotations

import re

from fastapi import FastAPI
from fastapi.testclient import TestClient

HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _client() -> TestClient:
    """App serving only the runtime-config router with auth stubbed open."""
    from infrastructure.auth import require_auth_if_enabled
    from infrastructure.exception_handlers import register_app_error_handler
    from routers.runtime_config import router as runtime_config_router

    app = FastAPI()
    register_app_error_handler(app)
    app.include_router(runtime_config_router)
    app.dependency_overrides[require_auth_if_enabled] = lambda: None
    return TestClient(app)


def test_runtime_config_spec_is_descriptor_projected():
    """The capability is declared once as a ToolSpec and projects to GET."""
    from routers.runtime_config import SPEC

    assert SPEC.name == "runtime_config.get"
    assert SPEC.idempotent is True  # idempotent → GET
    assert SPEC.auth_scope == "authenticated"
    assert SPEC.version == "1"


def test_runtime_config_serves_sorted_resolved_flags():
    """The document is {config_version, flags} — sorted, no env internals."""
    res = _client().get("/runtime-config")
    assert res.status_code == 200

    body = res.json()
    # Bare payload (raw-wire facet): the caller parses the document itself.
    assert set(body) == {"config_version", "flags"}
    assert HEX64.match(body["config_version"])

    names = [f["name"] for f in body["flags"]]
    assert names == sorted(names), "flags must be sorted for a stable ETag"
    assert "native_c_inference" in names  # a known default-disabled flag

    for flag in body["flags"]:
        assert set(flag) == {"name", "status", "enabled", "version", "since", "description"}
        assert flag["status"] in {"enabled", "disabled", "experimental"}
        assert isinstance(flag["enabled"], bool)
        # env_var must NOT leak — the resolved value is what consumers act on.
        assert "env_var" not in flag


def test_runtime_config_env_override_wins_and_bumps_version(monkeypatch):
    """SLO_FF_* overrides the registered status; the version tracks it."""
    client = _client()

    monkeypatch.delenv("SLO_FF_NATIVE_C_INFERENCE", raising=False)
    baseline = client.get("/runtime-config").json()
    off = next(f for f in baseline["flags"] if f["name"] == "native_c_inference")
    assert off["status"] == "disabled"
    assert off["enabled"] is False

    monkeypatch.setenv("SLO_FF_NATIVE_C_INFERENCE", "1")
    overridden = client.get("/runtime-config").json()
    on = next(f for f in overridden["flags"] if f["name"] == "native_c_inference")
    assert on["status"] == "disabled"  # registered status untouched
    assert on["enabled"] is True  # ...but the env override wins
    # enabled feeds the hash — a consumer can key its cache off config_version.
    assert overridden["config_version"] != baseline["config_version"]


def test_runtime_config_serves_etag_and_cache_control():
    """ETag = quoted config_version; the card's cache ladder is private/30s."""
    res = _client().get("/runtime-config")
    assert res.status_code == 200

    version = res.json()["config_version"]
    assert res.headers["etag"] == f'"{version}"'
    assert res.headers["cache-control"] == "private, max-age=30"


def test_runtime_config_matching_if_none_match_is_304():
    """Revalidation with the current ETag costs no body."""
    client = _client()
    first = client.get("/runtime-config")
    etag = first.headers["etag"]

    revalidated = client.get("/runtime-config", headers={"If-None-Match": etag})
    assert revalidated.status_code == 304
    assert revalidated.headers["etag"] == etag
    assert revalidated.content == b""

    stale = client.get("/runtime-config", headers={"If-None-Match": '"not-current"'})
    assert stale.status_code == 200  # a stale tag gets a fresh document


def test_runtime_config_v_query_is_cache_buster_only():
    """?v=<hash> is accepted and ignored — undeclared query keys never 422."""
    res = _client().get("/runtime-config", params={"v": "deadbeef"})
    assert res.status_code == 200
    assert HEX64.match(res.json()["config_version"])
