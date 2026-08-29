"""Tests for domains.collections.config — pipeline configuration dataclasses.

Covers: SourceConfig, StoreConfig, FilterConfig, PipelineConfig defaults and custom values.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_core_dir = str(Path(__file__).resolve().parents[2])
if _core_dir not in sys.path:
    sys.path.insert(0, _core_dir)

from domains.collections.config import (
    SourceConfig,
    StoreConfig,
    FilterConfig,
    PipelineConfig,
)


class TestSourceConfig:
    def test_defaults(self):
        c = SourceConfig()
        assert c.type == "file"
        assert c.path == ""
        assert c.url == ""
        assert c.timeout == 30
        assert c.poll_interval == 60.0
        assert c.headers == {}

    def test_custom(self):
        c = SourceConfig(type="url", url="https://example.com", timeout=60)
        assert c.type == "url"
        assert c.url == "https://example.com"
        assert c.timeout == 60


class TestStoreConfig:
    def test_defaults(self):
        c = StoreConfig()
        assert c.type == "file"
        assert c.path == ""
        assert c.max_size == 10000

    def test_custom(self):
        c = StoreConfig(type="sqlite", path="/tmp/db.sqlite", max_size=50000)
        assert c.type == "sqlite"
        assert c.max_size == 50000


class TestFilterConfig:
    def test_defaults(self):
        c = FilterConfig()
        assert c.type == "length"
        assert c.min_length == 10
        assert c.max_length == 100000
        assert c.keywords == []
        assert c.mode == "include"

    def test_custom(self):
        c = FilterConfig(type="keyword", keywords=["python", "test"], mode="exclude")
        assert c.keywords == ["python", "test"]
        assert c.mode == "exclude"


class TestPipelineConfig:
    def test_defaults(self):
        c = PipelineConfig()
        assert c.name == ""
        assert c.collect_interval == 0.0
        assert c.max_rounds is None

    def test_nested(self):
        c = PipelineConfig(
            name="test_pipeline",
            source=SourceConfig(type="url"),
            store=StoreConfig(type="memory"),
            filters=[FilterConfig(type="length")],
        )
        assert c.source.type == "url"
        assert c.store.type == "memory"
        assert len(c.filters) == 1
