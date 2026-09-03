"""Tests for model_size formatting functions."""
from __future__ import annotations

from domains.infrastructure.model_size import format_size_gb, format_size_mb


class TestFormatSizeGb:
    def test_none_returns_dash(self):
        assert format_size_gb(None) == "—"

    def test_formats(self):
        assert format_size_gb(1.5) == "1.50 GB"

    def test_zero(self):
        assert format_size_gb(0) == "0.00 GB"

    def test_custom_decimals(self):
        assert format_size_gb(1.23456, decimals=3) == "1.235 GB"


class TestFormatSizeMb:
    def test_none_returns_none(self):
        assert format_size_mb(None) is None

    def test_conversion(self):
        assert format_size_mb(1.0) == 1024.0

    def test_fractional(self):
        assert format_size_mb(0.5) == 512.0
