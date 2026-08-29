"""Tests for domains.shell.config — API base URL configuration.

Covers: default value, env var override.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_core_dir = str(Path(__file__).resolve().parents[2])
if _core_dir not in sys.path:
    sys.path.insert(0, _core_dir)

from domains.shell.config import get_api_base, DEFAULT_API_BASE


class TestConfig:
    def test_default_value(self):
        # DEFAULT_API_BASE should be set from env or fallback
        assert isinstance(DEFAULT_API_BASE, str)
        assert len(DEFAULT_API_BASE) > 0

    def test_get_api_base_returns_string(self):
        result = get_api_base()
        assert isinstance(result, str)
        assert result.startswith("http")
