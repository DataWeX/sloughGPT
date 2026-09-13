"""
Shell domain configuration.

Reads from environment variables with sensible defaults.
"""

from __future__ import annotations

import os

DEFAULT_API_BASE = os.environ.get("MAN_API_URL", "http://localhost:8000")


def get_api_base() -> str:
    """Return the API base URL, respecting MAN_API_URL env var."""
    return os.environ.get("MAN_API_URL", DEFAULT_API_BASE)
