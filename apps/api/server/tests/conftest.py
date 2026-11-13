"""Shared pytest fixtures for API server tests."""

import pytest


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
