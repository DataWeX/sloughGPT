"""Rate-limit middleware behavior (card 20261119_091258).

Two defects pinned here:

1. The class docstring promises "Localhost requests get 10x the limit", but
   the boost was only ever applied to the *global* limiter — route buckets
   (``/models/`` at 30/60) ran at the raw limit for every client. All
   localhost co-tenants (user browser :3000 + parallel dev servers :5173/:5174
   + their headless e2e browsers) share 127.0.0.1, so they fought over one
   30 req/min budget and the /training journey logged dozens of 429s.
2. ``/models/hf`` (model listing, polled by the UI) shared the ``/models/``
   file-serving bucket — listing could starve file serving and vice versa.

``test_support`` no-ops ``RateLimitMiddleware.dispatch`` process-wide at
import (ordinary API tests must never see a 429); these tests exercise the
real dispatch and restore the suite's convention afterwards.
"""

from __future__ import annotations

import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from infrastructure.rate_limit_middleware import RateLimitMiddleware

REMOTE_IP = "10.9.8.7"
LOCAL_IP = "127.0.0.1"


def _build_app() -> FastAPI:
    """Minimal app exercising only the paths the middleware cares about."""
    app = FastAPI()

    @app.get("/models/hf")
    async def models_hf():
        return {"ok": "hf"}

    @app.get("/models/downloads")
    async def models_downloads():
        return {"ok": "downloads"}

    @app.get("/health")
    async def health():
        return {"ok": "health"}

    app.add_middleware(RateLimitMiddleware)
    return app


def _client(ip: str) -> TestClient:
    # TestClient stamps this tuple into scope["client"], which is exactly
    # what the middleware keys its buckets on.
    return TestClient(_build_app(), client=(ip, 40000))


@pytest.fixture(autouse=True)
def _real_dispatch():
    """Run each test against the REAL dispatch; restore the suite's after.

    ``tests/test_support.py`` replaces ``dispatch`` with a pass-through at
    import time so ordinary API tests never trip a 429. This module tests the
    middleware itself, so it swaps the pristine function in per test and puts
    whatever the suite had installed back afterwards. Save/restore is
    order-independent: pytest imports every collected module before running
    any test, so by fixture time ``test_support``'s patch (if it applies to
    this session at all) is already in place.
    """
    saved = RateLimitMiddleware.dispatch
    pristine = saved
    for name, mod in sys.modules.items():
        if name.rsplit(".", 1)[-1] == "test_support":
            stash = getattr(mod, "_orig_dispatch", None)
            if stash is not None:
                pristine = stash
                break
    RateLimitMiddleware.dispatch = pristine
    try:
        yield
    finally:
        RateLimitMiddleware.dispatch = saved


class TestRateLimitMiddleware:
    def test_local_route_limit_is_10x_the_raw_cap(self):
        """Docstring promise: localhost route buckets get 10x the raw limit.

        Raw ``/models/`` budget is 30/60 — a local client must sail past it.
        """
        client = _client(LOCAL_IP)
        statuses = [client.get("/models/downloads").status_code for _ in range(35)]
        assert statuses == [200] * 35

    def test_remote_route_limit_still_enforces_raw_cap(self):
        """The 10x boost must not leak to non-local clients."""
        client = _client(REMOTE_IP)
        for _ in range(30):
            assert client.get("/models/downloads").status_code == 200
        resp = client.get("/models/downloads")
        assert resp.status_code == 429
        assert "Rate limit exceeded for /models/" in resp.json()["detail"]

    def test_models_listing_bucket_is_independent_of_file_serving(self):
        """Exhausting ``/models/`` file serving must not starve ``/models/hf``."""
        client = _client(REMOTE_IP)
        for _ in range(30):
            assert client.get("/models/downloads").status_code == 200
        assert client.get("/models/downloads").status_code == 429
        assert client.get("/models/hf").status_code == 200

    def test_models_listing_cap_is_120_per_minute(self):
        """``/models/hf`` runs on its own (120, 60) budget, not the shared 30."""
        client = _client(REMOTE_IP)
        for _ in range(120):
            assert client.get("/models/hf").status_code == 200
        resp = client.get("/models/hf")
        assert resp.status_code == 429
        assert "Rate limit exceeded for /models/hf" in resp.json()["detail"]

    def test_health_bypasses_rate_limiting(self):
        """Health probes stay allowed even with every bucket saturated."""
        client = _client(REMOTE_IP)
        for _ in range(31):
            client.get("/models/downloads")
        assert client.get("/models/downloads").status_code == 429
        assert client.get("/health").status_code == 200

    def test_options_preflight_never_429s(self):
        """A 429 before CORSMiddleware fires TypeError: NetworkError."""
        client = _client(REMOTE_IP)
        for _ in range(31):
            client.get("/models/downloads")
        resp = client.request("OPTIONS", "/models/downloads")
        assert resp.status_code != 429
