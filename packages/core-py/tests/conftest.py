"""Shared test fixtures and helpers for router tests."""

import asyncio
import pytest
import sys
from pathlib import Path

_server_dir = str(Path(__file__).resolve().parents[2] / "apps" / "api" / "server")
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)


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
