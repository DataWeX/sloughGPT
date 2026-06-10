"""Shared fixtures for bawl tests — resets shared state between tests."""

import pytest
from bawl.fetch import _hits
from bawl.crawl import _robots


@pytest.fixture(autouse=True)
def reset_rate_limit():
    """Clear the per-domain rate-limit tracking between tests."""
    _hits.clear()
    _robots.clear()
    yield
