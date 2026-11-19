"""Shared test configuration for API server tests.

Ensures all feature routers are registered on the FastAPI app before any
test runs, even if the lifespan context manager didn't fully complete
(e.g. during TestClient usage where model loading or W&B startup hangs).
"""

import os
import sys
from pathlib import Path

# Ensure the server package is on sys.path so `from main import app` works
# regardless of where pytest is invoked from.
_server_dir = str(Path(__file__).resolve().parents[1])
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)

os.environ.setdefault("SLO_AUTO_WORKFLOW", "false")
os.environ.setdefault("SLO_AUTOLOAD_MODEL", "")

# Disable rate limiting for tests to avoid 429s when running the full suite.
try:
    from infrastructure.rate_limit_middleware import RateLimitMiddleware as _RLM

    if _RLM is not None:
        _orig_dispatch = _RLM.dispatch

        async def _noop_dispatch(self, request, call_next):
            return await call_next(request)

        _RLM.dispatch = _noop_dispatch
except Exception:
    pass

# Disable the readiness gate for tests so schema-validation tests can
# reach the router and trigger 422s instead of being blocked with 503.
# The real dispatch is stashed on the class so suites that DO test the gate
# (tests/server/test_readiness_gate.py) can restore it — this import-time
# patch is process-wide and leaks into any test module that runs after it.
try:
    from infrastructure.middleware import ReadinessGateMiddleware as _RGM

    if not hasattr(_RGM, "_real_dispatch"):
        _RGM._real_dispatch = _RGM.dispatch

    async def _noop_readiness(self, request, call_next):
        return await call_next(request)

    _RGM.dispatch = _noop_readiness
except Exception:
    pass


def _ensure_routers_registered():
    """Register all feature routers on the app through the single assembler.

    ``register_routers`` (infrastructure/route_registration.py) is the same
    path production boot and scripts/check_contract.py use, and it is
    idempotent — a router whose declared paths are all already served is
    skipped — so the old ``/benchmark``-prefix guard is unnecessary. The
    three assemblers used to diverge: this one included unconditionally,
    production prefix-skipped (dropping the whole /agents router), and the
    gate included unconditionally — tests stayed green while live 404'd
    (card 20261119_105501).
    """
    from infrastructure.route_registration import register_routers
    from main import app
    from routers import get_all_routers

    register_routers(app, get_all_routers())
    try:
        from training.router import router as training_router

        register_routers(app, [training_router])
    except Exception:
        pass


_ensure_routers_registered()


def get_test_client():
    """Create a TestClient with all routes registered."""
    from fastapi.testclient import TestClient
    from main import app

    return TestClient(app)


def _data(resp):
    """Unwrap the success_response() envelope from a TestClient response.

    Handles three shapes:
    - {"status":"success","data":{...}}  → returns the inner data dict/list
    - [list] or {dict} returned directly → returns as-is
    """
    body = resp.json()
    if isinstance(body, dict):
        return body.get("data", body)
    return body
