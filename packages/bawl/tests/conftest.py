"""Shared fixtures for bawl tests — resets shared state between tests."""

from __future__ import annotations

import sys
from pathlib import Path

# The fixtures below import from bawl at module scope, so the import path has
# to be settled before they are collected. This file previously imported bawl
# directly, so when the package was not installed editable the conftest itself
# raised ModuleNotFoundError and took the entire suite down with it — no test
# in this directory ever ran.
_SRC = Path(__file__).resolve().parents[1] / "src"

try:  # pragma: no cover - trivial import guard
    import bawl  # noqa: F401
except ImportError:  # pragma: no cover
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

import pytest  # noqa: E402
from bawl.crawl import _robots  # noqa: E402
from bawl.fetch import _hits  # noqa: E402


@pytest.fixture(autouse=True)
def reset_rate_limit():
    """Clear the per-domain rate-limit tracking between tests."""
    _hits.clear()
    _robots.clear()
    yield
