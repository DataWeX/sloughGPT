"""Tests for domains.infrastructure.pugqeep.config — compression configuration dataclasses.

Covers: PointConfig, CompressorConfig, LibraryConfig, TreeConfig, QueueConfig,
SubprocessConfig, RestartPolicy, MonitorConfig, EngineConfig defaults and edges.
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
    SubprocessConfig,
    RestartPolicy,
    MonitorConfig,
    EngineConfig,
)


# ---------------------------------------------------------------------------
# PointConfig
# ---------------------------------------------------------------------------

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

    def test_custom_residual_threshold(self):
        c = PointConfig(residual_threshold=0.85)
        assert c.residual_threshold == 0.85

    def test_all_function_types(self):
        for ft in ["cluster", "periodic", "linear", "polynomial", "raw"]:
            c = PointConfig(function_type=ft)
            assert c.function_type == ft

    def test_n_clusters_zero(self):
        c = PointConfig(n_clusters=0)
        assert c.n_clusters == 0

    def test_n_clusters_large(self):
        c = PointConfig(n_clusters=10000)
        assert c.n_clusters == 10000

    def test_residual_threshold_zero(self):
        c = PointConfig(residual_threshold=0.0)
        assert c.residual_threshold == 0.0

    def test_residual_threshold_one(self):
        c = PointConfig(residual_threshold=1.0)
        assert c.residual_threshold == 1.0

    def test_residual_threshold_above_one(self):
        c = PointConfig(residual_threshold=1.5)
        assert c.residual_threshold == 1.5

    def test_independent_instances(self):
        c1 = PointConfig(n_clusters=8)
        c2 = PointConfig(n_clusters=32)
        assert c1.n_clusters == 8
        assert c2.n_clusters == 32

    def test_function_type_mutable(self):
        c = PointConfig()
        c.function_type = "raw"
        assert c.function_type == "raw"


# ---------------------------------------------------------------------------
# CompressorConfig
# ---------------------------------------------------------------------------

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

    def test_gap_fill_iterations_default(self):
        c = CompressorConfig()
        assert c.gap_fill_iterations == 4

    def test_gap_fill_max_elements_default(self):
        c = CompressorConfig()
        assert c.gap_fill_max_elements == 100_000

    def test_custom_gap_fill(self):
        c = CompressorConfig(gap_fill_iterations=10, gap_fill_max_elements=500_000)
        assert c.gap_fill_iterations == 10
        assert c.gap_fill_max_elements == 500_000

    def test_lloyd_iterations_custom(self):
        c = CompressorConfig(lloyd_iterations=20)
        assert c.lloyd_iterations == 20

    def test_method_values(self):
        for m in ["cluster", "function"]:
            c = CompressorConfig(method=m)
            assert c.method == m

    def test_n_clusters_zero(self):
        c = CompressorConfig(n_clusters=0)
        assert c.n_clusters == 0

    def test_lloyd_iterations_zero(self):
        c = CompressorConfig(lloyd_iterations=0)
        assert c.lloyd_iterations == 0

    def test_gap_fill_iterations_zero(self):
        c = CompressorConfig(gap_fill_iterations=0)
        assert c.gap_fill_iterations == 0

    def test_independent_instances(self):
        c1 = CompressorConfig(n_clusters=8)
        c2 = CompressorConfig(n_clusters=64)
        assert c1.n_clusters != c2.n_clusters


# ---------------------------------------------------------------------------
# LibraryConfig
# ---------------------------------------------------------------------------

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

    def test_storage_dir_type(self):
        c = LibraryConfig(storage_dir=Path("/data"))
        assert isinstance(c.storage_dir, Path)

    def test_name_empty(self):
        c = LibraryConfig(name="")
        assert c.name == ""

    def test_auto_save_toggle(self):
        c = LibraryConfig()
        c.auto_save = True
        assert c.auto_save is True

    def test_storage_dir_none_default(self):
        c = LibraryConfig()
        assert c.storage_dir is None

    def test_independent_instances(self):
        c1 = LibraryConfig(name="a")
        c2 = LibraryConfig(name="b")
        assert c1.name != c2.name

    def test_custom_storage_dir(self):
        p = Path("/var/lib/pugqeep")
        c = LibraryConfig(storage_dir=p)
        assert c.storage_dir == p


# ---------------------------------------------------------------------------
# TreeConfig
# ---------------------------------------------------------------------------

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

    def test_skip_biases_false(self):
        c = TreeConfig(skip_biases=False)
        assert c.skip_biases is False

    def test_method_default(self):
        c = TreeConfig()
        assert c.method == "cluster"

    def test_method_function(self):
        c = TreeConfig(method="function")
        assert c.method == "function"

    def test_n_clusters_custom(self):
        c = TreeConfig(n_clusters=64)
        assert c.n_clusters == 64

    def test_name_custom(self):
        c = TreeConfig(name="custom_tree")
        assert c.name == "custom_tree"

    def test_skip_embeddings_toggle(self):
        c = TreeConfig()
        c.skip_embeddings = False
        assert c.skip_embeddings is False

    def test_independent_instances(self):
        c1 = TreeConfig(name="t1")
        c2 = TreeConfig(name="t2")
        assert c1.name != c2.name

    def test_all_fields_settable(self):
        c = TreeConfig()
        c.name = "new"
        c.n_clusters = 32
        c.method = "function"
        c.skip_embeddings = False
        c.skip_biases = False
        assert c.name == "new"
        assert c.n_clusters == 32
        assert c.method == "function"
        assert c.skip_embeddings is False
        assert c.skip_biases is False


# ---------------------------------------------------------------------------
# QueueConfig
# ---------------------------------------------------------------------------

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

    def test_storage_dir_default(self):
        c = QueueConfig()
        assert c.storage_dir is None

    def test_storage_dir_custom(self):
        c = QueueConfig(storage_dir=Path("/data/queue"))
        assert c.storage_dir == Path("/data/queue")

    def test_default_n_clusters_custom(self):
        c = QueueConfig(default_n_clusters=64)
        assert c.default_n_clusters == 64

    def test_dedup_toggle(self):
        c = QueueConfig()
        c.dedup = False
        assert c.dedup is False

    def test_max_trees_zero(self):
        c = QueueConfig(max_trees=0)
        assert c.max_trees == 0

    def test_max_trees_large(self):
        c = QueueConfig(max_trees=1000)
        assert c.max_trees == 1000

    def test_independent_instances(self):
        c1 = QueueConfig(max_trees=5)
        c2 = QueueConfig(max_trees=20)
        assert c1.max_trees != c2.max_trees

    def test_all_fields_custom(self):
        c = QueueConfig(max_trees=15, default_n_clusters=32, dedup=False)
        assert c.max_trees == 15
        assert c.default_n_clusters == 32
        assert c.dedup is False


# ---------------------------------------------------------------------------
# SubprocessConfig
# ---------------------------------------------------------------------------

class TestSubprocessConfig:
    def test_defaults(self):
        c = SubprocessConfig()
        assert c.enabled is True
        assert c.python_exe == "python3"
        assert c.max_workers == 4
        assert c.memory_limit_mb is None
        assert c.cpu_affinity is None
        assert c.start_method == "fork"
        assert c.env is None
        assert c.cwd is None
        assert c.capture_output is False
        assert c.terminate_grace == 3.0

    def test_custom(self):
        c = SubprocessConfig(enabled=False, max_workers=8, memory_limit_mb=1024)
        assert c.enabled is False
        assert c.max_workers == 8
        assert c.memory_limit_mb == 1024

    def test_cpu_affinity(self):
        c = SubprocessConfig(cpu_affinity=[0, 1, 2, 3])
        assert c.cpu_affinity == [0, 1, 2, 3]

    def test_start_method_spawn(self):
        c = SubprocessConfig(start_method="spawn")
        assert c.start_method == "spawn"

    def test_env_dict(self):
        c = SubprocessConfig(env={"HOME": "/root"})
        assert c.env["HOME"] == "/root"

    def test_cwd(self):
        c = SubprocessConfig(cwd="/tmp")
        assert c.cwd == "/tmp"

    def test_capture_output_true(self):
        c = SubprocessConfig(capture_output=True)
        assert c.capture_output is True

    def test_terminate_grace_custom(self):
        c = SubprocessConfig(terminate_grace=10.0)
        assert c.terminate_grace == 10.0

    def test_preexec_fn_default(self):
        c = SubprocessConfig()
        assert c.preexec_fn is None

    def test_max_workers_zero(self):
        c = SubprocessConfig(max_workers=0)
        assert c.max_workers == 0

    def test_independent_instances(self):
        c1 = SubprocessConfig(max_workers=1)
        c2 = SubprocessConfig(max_workers=8)
        assert c1.max_workers != c2.max_workers


# ---------------------------------------------------------------------------
# RestartPolicy
# ---------------------------------------------------------------------------

class TestRestartPolicy:
    def test_defaults(self):
        c = RestartPolicy()
        assert c.max_restarts == 0
        assert c.restart_delay == 1.0
        assert c.backoff == "exponential"
        assert c.max_backoff == 30.0

    def test_custom(self):
        c = RestartPolicy(max_restarts=5, restart_delay=2.0, backoff="fixed")
        assert c.max_restarts == 5
        assert c.restart_delay == 2.0
        assert c.backoff == "fixed"

    def test_backoff_linear(self):
        c = RestartPolicy(backoff="linear")
        assert c.backoff == "linear"

    def test_max_backoff_custom(self):
        c = RestartPolicy(max_backoff=60.0)
        assert c.max_backoff == 60.0

    def test_max_restarts_zero(self):
        c = RestartPolicy(max_restarts=0)
        assert c.max_restarts == 0

    def test_restart_delay_zero(self):
        c = RestartPolicy(restart_delay=0.0)
        assert c.restart_delay == 0.0

    def test_backoff_values(self):
        for b in ["fixed", "linear", "exponential"]:
            c = RestartPolicy(backoff=b)
            assert c.backoff == b

    def test_independent_instances(self):
        c1 = RestartPolicy(max_restarts=1)
        c2 = RestartPolicy(max_restarts=10)
        assert c1.max_restarts != c2.max_restarts

    def test_all_fields_settable(self):
        c = RestartPolicy()
        c.max_restarts = 5
        c.restart_delay = 3.0
        c.backoff = "fixed"
        c.max_backoff = 10.0
        assert c.max_restarts == 5
        assert c.restart_delay == 3.0
        assert c.backoff == "fixed"
        assert c.max_backoff == 10.0


# ---------------------------------------------------------------------------
# MonitorConfig
# ---------------------------------------------------------------------------

class TestMonitorConfig:
    def test_defaults(self):
        c = MonitorConfig()
        assert c.enabled is True
        assert c.poll_interval == 1.0
        assert c.stall_timeout == 60.0
        assert c.on_stall == "restart"
        assert c.on_restart == "log"

    def test_custom(self):
        c = MonitorConfig(enabled=False, poll_interval=5.0, stall_timeout=120.0)
        assert c.enabled is False
        assert c.poll_interval == 5.0
        assert c.stall_timeout == 120.0

    def test_on_stall_kill(self):
        c = MonitorConfig(on_stall="kill")
        assert c.on_stall == "kill"

    def test_on_stall_alert(self):
        c = MonitorConfig(on_stall="alert")
        assert c.on_stall == "alert"

    def test_on_restart_alert(self):
        c = MonitorConfig(on_restart="alert")
        assert c.on_restart == "alert"

    def test_poll_interval_zero(self):
        c = MonitorConfig(poll_interval=0.0)
        assert c.poll_interval == 0.0

    def test_stall_timeout_zero(self):
        c = MonitorConfig(stall_timeout=0.0)
        assert c.stall_timeout == 0.0

    def test_on_stall_values(self):
        for v in ["restart", "kill", "alert"]:
            c = MonitorConfig(on_stall=v)
            assert c.on_stall == v

    def test_on_restart_values(self):
        for v in ["log", "alert"]:
            c = MonitorConfig(on_restart=v)
            assert c.on_restart == v

    def test_independent_instances(self):
        c1 = MonitorConfig(poll_interval=1.0)
        c2 = MonitorConfig(poll_interval=10.0)
        assert c1.poll_interval != c2.poll_interval

    def test_all_fields_settable(self):
        c = MonitorConfig()
        c.enabled = False
        c.poll_interval = 2.0
        c.stall_timeout = 30.0
        c.on_stall = "kill"
        c.on_restart = "alert"
        assert c.enabled is False
        assert c.poll_interval == 2.0
        assert c.stall_timeout == 30.0
        assert c.on_stall == "kill"
        assert c.on_restart == "alert"


# ---------------------------------------------------------------------------
# EngineConfig
# ---------------------------------------------------------------------------

class TestEngineConfig:
    def test_defaults(self):
        c = EngineConfig()
        assert c.name == "main"
        assert c.max_trees == 16
        assert c.tree_workers == 4
        assert c.max_stems == 8
        assert c.queue_size == 128
        assert c.poll_interval == 0.1

    def test_subconfig_defaults(self):
        c = EngineConfig()
        assert isinstance(c.subprocess, SubprocessConfig)
        assert isinstance(c.restart, RestartPolicy)
        assert isinstance(c.monitor, MonitorConfig)

    def test_custom(self):
        c = EngineConfig(name="worker", max_trees=32, tree_workers=8)
        assert c.name == "worker"
        assert c.max_trees == 32
        assert c.tree_workers == 8

    def test_custom_subprocess(self):
        sp = SubprocessConfig(max_workers=16)
        c = EngineConfig(subprocess=sp)
        assert c.subprocess.max_workers == 16

    def test_custom_restart(self):
        rp = RestartPolicy(max_restarts=5)
        c = EngineConfig(restart=rp)
        assert c.restart.max_restarts == 5

    def test_custom_monitor(self):
        mc = MonitorConfig(poll_interval=2.0)
        c = EngineConfig(monitor=mc)
        assert c.monitor.poll_interval == 2.0

    def test_queue_size_custom(self):
        c = EngineConfig(queue_size=256)
        assert c.queue_size == 256

    def test_poll_interval_custom(self):
        c = EngineConfig(poll_interval=0.5)
        assert c.poll_interval == 0.5

    def test_max_stems_custom(self):
        c = EngineConfig(max_stems=16)
        assert c.max_stems == 16

    def test_independent_instances(self):
        c1 = EngineConfig(name="a")
        c2 = EngineConfig(name="b")
        assert c1.name != c2.name
        assert c1.subprocess is not c2.subprocess

    def test_subconfig_independence(self):
        c1 = EngineConfig()
        c2 = EngineConfig()
        c1.subprocess.max_workers = 99
        assert c2.subprocess.max_workers == 4

    def test_all_fields_settable(self):
        c = EngineConfig()
        c.name = "new"
        c.max_trees = 64
        c.tree_workers = 16
        c.max_stems = 32
        c.queue_size = 512
        c.poll_interval = 1.0
        assert c.name == "new"
        assert c.max_trees == 64
        assert c.tree_workers == 16
        assert c.max_stems == 32
        assert c.queue_size == 512
        assert c.poll_interval == 1.0
