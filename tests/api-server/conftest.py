"""Shared pytest fixtures for API server tests."""

import pytest


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path, monkeypatch):
    """Keep pooled MogDB stores out of the live ``<repo>/data`` tree.

    ``db_pool.get_db()`` used to resolve the real repo data dir, so suites that
    POST fixtures (``test-*``/``dup-*``/``del-*`` presets, uploaded files)
    persisted them into production stores and every run left junk behind.
    ``SLO_DATA_DIR`` — same shape as the existing ``SLO_CACHE_DIR`` — points
    every pooled store at a per-test tmp dir instead.

    The pool is a process-wide singleton keyed by ``(db_path, sync_path)``, so
    clear it before and after: entries created here must not leak into other
    tests, and pre-existing entries must not leak in here.
    """
    import infrastructure.db_pool as db_pool

    monkeypatch.setenv("SLO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(db_pool, "_POOL", {})
    yield
    monkeypatch.setattr(db_pool, "_POOL", {})


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Reset the rate limiter before every test so the full suite doesn't 429."""
    try:
        from main import app

        # Walk the ASGI middleware stack to find RateLimitMiddleware and reset it.
        current = getattr(app, "middleware_stack", None) or getattr(app, "_middleware_stack", None)
        while current is not None:
            if hasattr(current, "app") and type(current).__name__ == "RateLimitMiddleware":
                current.limiter.reset("127.0.0.1")
                current._local_limiter.reset("127.0.0.1")
                break
            current = getattr(current, "app", None)
    except Exception:
        pass
    yield
