"""Fail-closed on client/server config drift (card 20261108_222308).

VERIFY, straight from the card:
  * serving config version N while the client holds N-1 → the action is
    REFUSED with an explicit message;
  * a matching version → the action proceeds.

The version is the sibling contract's sha256 ``config_version``
(GET /runtime-config, card 20261108_134211) presented by the client as
``X-Slo-Config-Version``; the check lives in
``infrastructure/config_drift_middleware.py``, registered through the SAME
``register_all_middleware`` stack production uses — so these tests prove the
wiring, not just the class in isolation.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

HEADER = "X-Slo-Config-Version"


def _client() -> TestClient:
    """Full middleware stack + runtime-config route + one action probe."""
    from infrastructure.auth import require_auth_if_enabled
    from infrastructure.exception_handlers import register_app_error_handler
    from infrastructure.middleware import register_all_middleware
    from routers.runtime_config import router as runtime_config_router

    app = FastAPI()
    register_app_error_handler(app)
    register_all_middleware(app)
    app.include_router(runtime_config_router)

    @app.post("/__drift/probe")
    async def act():
        # Stands in for any irreversible action — must stay unreachable on a
        # refusal (412 comes from middleware, pre-handler).
        return {"acted": True}

    @app.get("/__drift/probe")
    async def read():
        return {"read": True}

    app.dependency_overrides[require_auth_if_enabled] = lambda: None
    return TestClient(app)


def _version(client: TestClient) -> str:
    return client.get("/runtime-config").json()["config_version"]


def test_matching_version_proceeds():
    """VERIFY half 2 — client holds N, server serves N → the action runs."""
    client = _client()
    version = _version(client)

    res = client.post("/__drift/probe", headers={HEADER: version})

    assert res.status_code == 200
    assert res.json() == {"acted": True}


def test_stale_version_refused_with_explicit_sentence(monkeypatch):
    """VERIFY half 1 — server serves N, client holds N-1 → refused 412.

    Real drift, not a fabricated string: the client reads the document, then
    the server's resolved flags change (SLO_FF_* env override bumps the sha256
    exactly as ``test_runtime_config_env_override_wins_and_bumps_version``
    proves), and the still-held N-1 claim hits the new server.
    """
    client = _client()
    monkeypatch.delenv("SLO_FF_NATIVE_C_INFERENCE", raising=False)
    held = _version(client)  # client loads N-1…

    monkeypatch.setenv("SLO_FF_NATIVE_C_INFERENCE", "1")  # …server moves to N
    current = _version(client)
    assert current != held  # the drift is real

    res = client.post("/__drift/probe", headers={HEADER: held})

    assert res.status_code == 412
    body = res.json()
    assert body["code"] == "E_CONFIG_DRIFT"
    # The card's required explicit sentence (registry user_message, verbatim).
    from domain.infrastructure._internal.errors import ERROR_REGISTRY, ErrorCode

    sentence = ERROR_REGISTRY[ErrorCode.E_CONFIG_DRIFT][3]
    assert body["error"] == sentence
    assert "disagree about the active configuration" in sentence
    assert "Reload the page" in sentence
    # Drift is diagnosable: both versions are in the details.
    assert body["details"] == {"client_version": held, "server_version": current}
    # Nothing acted — the probe's 200 body never appears.
    assert "acted" not in res.text


def test_unknown_version_refused():
    """A version the server never issued (garbage) fails closed the same way."""
    client = _client()

    res = client.post("/__drift/probe", headers={HEADER: "0" * 64})

    assert res.status_code == 412
    assert res.json()["code"] == "E_CONFIG_DRIFT"


def test_empty_claim_refused():
    """A PRESENTED claim is validated even when empty — malformed ≠ absent."""
    client = _client()

    res = client.post("/__drift/probe", headers={HEADER: ""})

    assert res.status_code == 412
    assert res.json()["code"] == "E_CONFIG_DRIFT"


def test_absent_header_passes():
    """No claim → no drift property: CLI/SDK/mobile keep working unchanged."""
    client = _client()

    res = client.post("/__drift/probe")

    assert res.status_code == 200
    assert res.json() == {"acted": True}


def test_reads_never_refused_even_with_stale_version():
    """GETs are the recovery path a stale client refreshes through."""
    client = _client()

    res = client.get("/__drift/probe", headers={HEADER: "f" * 64})

    assert res.status_code == 200
    assert res.json() == {"read": True}


def test_registry_entry_is_412_recoverable_with_reload_sentence():
    """The registry stays the single source for status + sentence."""
    from domain.infrastructure._internal.errors import ERROR_REGISTRY, ErrorCode

    entry = ERROR_REGISTRY[ErrorCode.E_CONFIG_DRIFT]
    class_name, http_status, recoverable, user_msg = entry
    assert class_name == "ConflictError"
    assert http_status == 412
    assert recoverable is True  # the remedy IS recoverable: reload and retry
    assert "Reload the page" in user_msg
