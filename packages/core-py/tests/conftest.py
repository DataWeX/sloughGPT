"""Shared test fixtures and helpers for router tests."""

import asyncio
import sys
from pathlib import Path

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "live: marks tests that require running servers")


_server_dir = str(Path(__file__).resolve().parents[2] / "apps" / "api" / "server")
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)

# Journey tests drive the shared avion computer-use library. rootdir for this
# package is packages/core-py (it has its own pytest.ini), so the repo-root
# conftest never loads here — add avion from source instead of installing it
# into the shared .venv (shared-workspace rule: no new installs).
_avion_src = str(Path(__file__).resolve().parents[3] / "packages" / "avion" / "src")
if _avion_src not in sys.path:
    sys.path.insert(0, _avion_src)


@pytest.fixture(autouse=True)
def _ensure_event_loop():
    """Ensure a working event loop exists for each test."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("closed")
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
    yield


def build_test_app(*routers):
    """Build a FastAPI app with exception handlers registered.

    Usage::

        app = build_test_app(my_router)
        client = TestClient(app)

    This ensures raise_error() exceptions are properly caught and
    converted to JSON responses, matching production behavior.
    """
    from fastapi import FastAPI

    app = FastAPI()
    for r in routers:
        app.include_router(r)

    from infrastructure.exception_handlers import register_all_handlers

    register_all_handlers(app)

    return app
