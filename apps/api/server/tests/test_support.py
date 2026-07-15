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


def _ensure_routers_registered():
    """Register all feature routers on the app if they haven't been yet."""
    from main import app

    # Check if benchmark router is already registered (proxy for "all routers")
    has_routers = any(
        hasattr(r, "path") and r.path.startswith("/benchmark")
        for r in app.routes
    )
    if has_routers:
        return

    from routers import get_all_routers
    for r in get_all_routers():
        app.include_router(r)
    try:
        from training.router import router as training_router
        app.include_router(training_router)
    except Exception:
        pass


_ensure_routers_registered()


def get_test_client():
    """Create a TestClient with all routes registered."""
    from fastapi.testclient import TestClient
    from main import app
    return TestClient(app)
