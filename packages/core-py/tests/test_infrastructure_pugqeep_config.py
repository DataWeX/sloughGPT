"""Tests for pugqeep config dataclasses."""
from __future__ import annotations

from pathlib import Path

from domains.infrastructure.pugqeep.config import (
    CompressorConfig,
    EngineConfig,
    LibraryConfig,
    MonitorConfig,
    PointConfig,
    QueueConfig,
    RestartPolicy,
    SubprocessConfig,
    TreeConfig,
)


class TestPointConfig:
    def test_defaults(self):
        c = PointConfig()
        assert c.function_type == "cluster"
        assert c.n_clusters == 16
        assert c.residual_threshold == 0.99


class TestCompressorConfig:
    def test_defaults(self):
        c = CompressorConfig()
        assert c.n_clusters == 16
        assert c.lloyd_iterations == 5
        assert c.method == "cluster"


class TestLibraryConfig:
    def test_defaults(self):
        c = LibraryConfig()
        assert c.name == "default"
        assert c.storage_dir is None
        assert c.auto_save is False


class TestTreeConfig:
    def test_defaults(self):
        c = TreeConfig()
        assert c.name == "model"
        assert c.skip_embeddings is True
        assert c.skip_biases is True


class TestQueueConfig:
    def test_defaults(self):
        c = QueueConfig()
        assert c.max_trees == 10
        assert c.dedup is True


class TestSubprocessConfig:
    def test_defaults(self):
        c = SubprocessConfig()
        assert c.enabled is True
        assert c.python_exe == "python3"
        assert c.max_workers == 4
        assert c.start_method == "fork"
        assert c.terminate_grace == 3.0


class TestRestartPolicy:
    def test_defaults(self):
        c = RestartPolicy()
        assert c.max_restarts == 0
        assert c.backoff == "exponential"
        assert c.max_backoff == 30.0


class TestMonitorConfig:
    def test_defaults(self):
        c = MonitorConfig()
        assert c.enabled is True
        assert c.poll_interval == 1.0
        assert c.on_stall == "restart"


class TestEngineConfig:
    def test_defaults(self):
        c = EngineConfig()
        assert c.name == "main"
        assert c.max_trees == 16
        assert c.tree_workers == 4
        assert isinstance(c.subprocess, SubprocessConfig)
        assert isinstance(c.restart, RestartPolicy)
        assert isinstance(c.monitor, MonitorConfig)

    def test_override_subprocess(self):
        sp = SubprocessConfig(max_workers=8)
        c = EngineConfig(subprocess=sp)
        assert c.subprocess.max_workers == 8
