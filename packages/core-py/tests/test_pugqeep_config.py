"""Tests for domains.infrastructure.pugqeep.config — compression configuration dataclasses.

Covers: PointConfig, CompressorConfig, LibraryConfig, TreeConfig, QueueConfig defaults.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_core_dir = str(Path(__file__).resolve().parents[2])
if _core_dir not in sys.path:
    sys.path.insert(0, _core_dir)

from domains.infrastructure.pugqeep.config import (
    PointConfig,
    CompressorConfig,
    LibraryConfig,
    TreeConfig,
    QueueConfig,
)


class TestPointConfig:
    def test_defaults(self):
        c = PointConfig()
        assert c.function_type == "cluster"
        assert c.n_clusters == 16
        assert c.residual_threshold == 0.99

    def test_custom(self):
        c = PointConfig(function_type="periodic", n_clusters=32)
        assert c.function_type == "periodic"
        assert c.n_clusters == 32


class TestCompressorConfig:
    def test_defaults(self):
        c = CompressorConfig()
        assert c.n_clusters == 16
        assert c.lloyd_iterations == 5
        assert c.method == "cluster"

    def test_custom(self):
        c = CompressorConfig(n_clusters=64, method="function")
        assert c.n_clusters == 64
        assert c.method == "function"


class TestLibraryConfig:
    def test_defaults(self):
        c = LibraryConfig()
        assert c.name == "default"
        assert c.storage_dir is None
        assert c.auto_save is False

    def test_custom(self):
        c = LibraryConfig(name="mylib", storage_dir=Path("/tmp"), auto_save=True)
        assert c.name == "mylib"
        assert c.auto_save is True


class TestTreeConfig:
    def test_defaults(self):
        c = TreeConfig()
        assert c.name == "model"
        assert c.n_clusters == 16
        assert c.skip_embeddings is True
        assert c.skip_biases is True

    def test_custom(self):
        c = TreeConfig(name="tree1", skip_embeddings=False)
        assert c.name == "tree1"
        assert c.skip_embeddings is False


class TestQueueConfig:
    def test_defaults(self):
        c = QueueConfig()
        assert c.max_trees == 10
        assert c.default_n_clusters == 16
        assert c.dedup is True

    def test_custom(self):
        c = QueueConfig(max_trees=20, dedup=False)
        assert c.max_trees == 20
        assert c.dedup is False
