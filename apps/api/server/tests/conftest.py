"""Shared pytest fixtures for API server tests."""

import pytest


# Re-exported so `from conftest import build_test_app` in router tests keeps
# working no matter which tests/ directory wins the top-level `conftest`
# name under --import-mode=importlib (namespace shadowing).
try:
    import importlib.util as _ilu
    from pathlib import Path as _P

    _bt_spec = _ilu.spec_from_file_location(
        "_core_py_tests_conftest",
        _P(__file__).resolve().parents[4]
        / "packages"
        / "core-py"
        / "tests"
        / "conftest.py",
    )
    _bt_mod = _ilu.module_from_spec(_bt_spec)
    _bt_spec.loader.exec_module(_bt_mod)
    build_test_app = _bt_mod.build_test_app
    del _bt_spec, _bt_mod
except Exception:
    pass


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
