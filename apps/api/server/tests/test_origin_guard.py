"""First-party origin guard — behavior tests (card 20261108_142919).

Threat model (the card's corrected premise): with ``SLO_AUTH_REQUIRED``
defaulting to off, a cross-origin SIMPLE POST (form/text-plain — no
preflight) can EXECUTE state-changing actions on any endpoint. CORS only
blocks reading the response, not sending the request — so the send itself
is gated: unsafe methods carrying a foreign ``Origin`` get
``403 E_BAD_ORIGIN``.

Guaranteed passes (the guard's contract):
  * curl / non-browser clients (no Origin header) → allowed
  * loopback origins (localhost/127.0.0.1/[::1], any port) → allowed
  * same-origin (Origin netloc == request Host) → allowed
  * ``SLO_CORS_ORIGINS`` entries → allowed (mirror of the CORS allow list)
  * safe methods (GET/HEAD/OPTIONS) → allowed (state changes only)
  * ``/v1`` → exempt: OpenAI-compatible external consumers unaffected
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from infrastructure.exception_handlers import register_app_error_handler
from infrastructure.middleware import register_all_middleware
from test_support import get_test_client

from domain.infrastructure._internal.errors import AppError

_EVIL = "https://evil.example.com"


def _make_app() -> FastAPI:
    """A mini app wired through the REAL seam (``register_all_middleware``)."""
    app = FastAPI()
    register_app_error_handler(app)
    register_all_middleware(app)

    @app.post("/internal/do")
    async def do_post():
        return {"ok": True}

    @app.get("/internal/do")
    async def do_get():
        return {"ok": True}

    # /v1 mirror — the external surface must stay untouched by the guard.
    @app.post("/v1/chat/completions")
    async def v1_chat():
        return {"ok": True}

    @app.get("/v1/models")
    async def v1_models():
        return {"object": "list"}

    return app


client = TestClient(_make_app())


class TestUnsafeMethodOriginGuard:
    """Unsafe methods + foreign Origin → 403; everything else passes."""

    def test_foreign_origin_post_rejected(self):
        resp = client.post("/internal/do", headers={"Origin": _EVIL})
        assert resp.status_code == 403
        body = resp.json()
        assert body["code"] == "E_BAD_ORIGIN"
        assert _EVIL in body["error"]
        assert body["details"]["origin"] == _EVIL

    def test_foreign_origin_put_patch_delete_rejected(self):
        for method in (client.put, client.patch, client.delete):
            resp = method("/internal/do", headers={"Origin": _EVIL})
            assert resp.status_code == 403, f"{method} not guarded"

    def test_curl_without_origin_allowed(self):
        """No Origin header → non-browser client → allowed (the curl case)."""
        resp = client.post("/internal/do")
        assert resp.status_code == 200

    def test_loopback_origin_allowed(self):
        resp = client.post("/internal/do", headers={"Origin": "http://localhost:5173"})
        assert resp.status_code == 200

    def test_loopback_ipv6_any_port_allowed(self):
        resp = client.post("/internal/do", headers={"Origin": "http://[::1]:8000"})
        assert resp.status_code == 200

    def test_same_origin_allowed(self):
        # TestClient's default Host is "testserver" — a same-origin page posts
        # its own backend: Origin netloc == Host → allowed.
        resp = client.post("/internal/do", headers={"Origin": "http://testserver"})
        assert resp.status_code == 200

    def test_env_origin_list_allowed(self, monkeypatch):
        monkeypatch.setenv("SLO_CORS_ORIGINS", "http://lan-host.example:3000")
        resp = client.post("/internal/do", headers={"Origin": "http://lan-host.example:3000"})
        assert resp.status_code == 200

    def test_origin_outside_env_list_rejected(self, monkeypatch):
        monkeypatch.setenv("SLO_CORS_ORIGINS", "http://lan-host.example:3000")
        resp = client.post("/internal/do", headers={"Origin": _EVIL})
        assert resp.status_code == 403

    def test_safe_get_with_foreign_origin_allowed(self):
        """GET never changes state — the guard is unsafe-method-only."""
        resp = client.get("/internal/do", headers={"Origin": _EVIL})
        assert resp.status_code == 200

    def test_options_is_not_blocked_by_the_guard(self):
        """Preflight belongs to CORS (outermost in main); guard skips OPTIONS.
        The mini app has no CORS registered, so 405 from routing proves the
        guard passed it through instead of 403-ing it."""
        resp = client.options(
            "/internal/do",
            headers={"Origin": _EVIL, "Access-Control-Request-Method": "POST"},
        )
        assert resp.status_code != 403

    def test_v1_post_foreign_origin_exempt(self):
        """/v1 external consumers must be unaffected — path exemption."""
        resp = client.post("/v1/chat/completions", headers={"Origin": _EVIL})
        assert resp.status_code == 200

    def test_v1_get_foreign_origin_exempt(self):
        resp = client.get("/v1/models", headers={"Origin": _EVIL})
        assert resp.status_code == 200


class TestProductionStackWiring:
    """The guard runs in the REAL app via the register_all_middleware seam."""

    def test_real_app_foreign_origin_post_rejected(self):
        real = get_test_client()
        resp = real.post("/errors/logs/ingest", json={"logs": []}, headers={"Origin": _EVIL})
        assert resp.status_code == 403
        assert resp.json()["code"] == "E_BAD_ORIGIN"

    def test_real_app_curl_post_allowed(self):
        real = get_test_client()
        resp = real.post("/errors/logs/ingest", json={"logs": []})
        assert resp.status_code == 200

    def test_real_app_loopback_origin_allowed(self):
        real = get_test_client()
        resp = real.post(
            "/errors/logs/ingest",
            json={"logs": []},
            headers={"Origin": "http://localhost:3000"},
        )
        assert resp.status_code == 200


class TestRequireFirstPartyOriginDependency:
    """The exported dependency — the per-route/descriptor projection of the
    same rule (route-level use where middleware placement is wrong)."""

    def test_foreign_origin_raises_403(self):
        from infrastructure.origin_guard import require_first_party_origin

        with pytest.raises(AppError) as ei:
            require_first_party_origin(origin=_EVIL, host="app.example")
        assert ei.value.code == "E_BAD_ORIGIN"
        assert ei.value.http_status == 403

    def test_no_origin_passes(self):
        from infrastructure.origin_guard import require_first_party_origin

        assert require_first_party_origin(origin=None, host="app.example") is None

    def test_loopback_origin_passes(self):
        from infrastructure.origin_guard import require_first_party_origin

        assert (
            require_first_party_origin(origin="http://127.0.0.1:8000", host="app.example") is None
        )

    def test_same_origin_passes(self):
        from infrastructure.origin_guard import require_first_party_origin

        assert require_first_party_origin(origin="https://app.example", host="app.example") is None
