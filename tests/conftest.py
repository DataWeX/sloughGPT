"""Pytest configuration: path bootstrap and shared fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
# Add project sub-packages at high priority (position 0).
# _REPO_ROOT itself goes at the end to avoid shadowing sub-package modules.
for _p in (
    _REPO_ROOT / "packages" / "core-py",
    _REPO_ROOT / "packages" / "sdk-py",
    _REPO_ROOT / "apps" / "api" / "server",
    _REPO_ROOT / "apps" / "cli" / "src",
):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)
# Add repo root last — don't insert at front to avoid shadowing cli module.
_s_root = str(_REPO_ROOT)
if _s_root not in sys.path:
    sys.path.append(_s_root)


@pytest.fixture(autouse=True)
def _isolated_pooled_stores(tmp_path, monkeypatch):
    """Keep pooled MogDB stores out of the live ``<repo>/data`` tree.

    ``db_pool.get_db()`` resolves the real repo data dir, so any suite that
    POSTs fixtures (``test-*``/``dup-*``/``del-*`` presets, uploaded files)
    persisted them into production stores and every run left junk behind — the
    companion store alone held 204 fixture records against 4 real presets.

    ``SLO_DATA_DIR`` (same shape as the existing ``SLO_CACHE_DIR``) points every
    pooled store at a per-test tmp dir instead.  Lived here rather than in a
    per-suite conftest because the pollution came from suites in *both*
    ``tests/api-server/`` and ``tests/server/``.

    The pool is a process-wide singleton keyed by ``(db_path, sync_path)``, so
    it is cleared before and after: entries created here must not leak into
    other tests, and pre-existing entries must not leak in here.  The import is
    guarded — non-server suites must not fail if the server package is absent.
    """
    try:
        import infrastructure.db_pool as db_pool
    except Exception:
        yield
        return

    monkeypatch.setenv("SLO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(db_pool, "_POOL", {})
    yield
    monkeypatch.setattr(db_pool, "_POOL", {})


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


@pytest.fixture
def api_base_url() -> str:
    """Base URL for API tests."""
    return "http://localhost:8000"


@pytest.fixture
def test_user() -> dict:
    return {
        "username": "testuser",
        "email": "test@example.com",
        "password": "testpass123",
    }


@pytest.fixture
def test_model() -> str:
    return "gpt2"


@pytest.fixture
def test_dataset() -> str:
    return "wikitext-103"


@pytest.fixture
def test_training_config() -> dict:
    return {
        "name": "Test Training",
        "model": "gpt2",
        "dataset": "wikitext-103",
        "epochs": 1,
        "batch_size": 2,
        "learning_rate": 1e-5,
    }


class TestHelpers:
    @staticmethod
    def assert_response_success(response, status_code=200):
        assert response.status_code == status_code, (
            f"Expected {status_code}, got {response.status_code}"
        )

    @staticmethod
    def assert_has_keys(data: dict, keys: list):
        for key in keys:
            assert key in data, f"Missing key: {key}"


@pytest.fixture
def helpers() -> TestHelpers:
    return TestHelpers()
