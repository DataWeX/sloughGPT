"""Tests for ResourceManager, ResourceAllocation, compute_allocation."""

import os
import sys
import threading
import types
from unittest.mock import patch

import pytest
from domains.infrastructure.resource_manager import (
    ResourceAllocation,
    ResourceManager,
    compute_allocation,
    get_resource_manager,
    reset_resource_manager,
)
from domains.infrastructure.cpu_topology import CpuTopology, detect_topology


# ── Fixtures ──


@pytest.fixture(autouse=True)
def _reset_singleton():
    reset_resource_manager()
    yield
    reset_resource_manager()


@pytest.fixture
def known_topology() -> CpuTopology:
    """A known topology so tests are deterministic."""
    return CpuTopology(
        physical_cores=4,
        logical_cores=8,
        has_hyperthreading=True,
        l3_cache_kb=8192,
        cpu_freq_mhz=2400.0,
        model_name="Intel Core i5",
        numa_nodes=1,
    )


# ── ResourceAllocation ──


class TestResourceAllocation:
    def test_default_values(self):
        alloc = ResourceAllocation()
        assert alloc.compute_threads == 0
        assert alloc.io_threads == 0
        assert alloc.inference_pool_size == 0
        assert alloc.train_pool_size == 0
        assert alloc.task_queue_workers == 0

    def test_summary_contains_mode(self):
        alloc = ResourceAllocation(workload_mode="training")
        assert "training" in alloc.summary()

    def test_apply_env_no_overrides(self, monkeypatch):
        alloc = ResourceAllocation(compute_threads=4)
        result = alloc.apply_env()
        assert result.compute_threads == 4

    def test_apply_env_with_overrides(self, monkeypatch):
        monkeypatch.setenv("SLO_COMPUTE_THREADS", "2")
        monkeypatch.setenv("SLO_IO_THREADS", "4")
        alloc = ResourceAllocation(compute_threads=4, io_threads=1)
        result = alloc.apply_env()
        assert result.compute_threads == 2
        assert result.io_threads == 4

    def test_apply_env_partial_overrides(self, monkeypatch):
        monkeypatch.setenv("SLO_COMPUTE_THREADS", "8")
        alloc = ResourceAllocation(compute_threads=4, io_threads=2)
        result = alloc.apply_env()
        assert result.compute_threads == 8
        assert result.io_threads == 2  # unchanged

    def test_apply_env_invalid_value_ignored(self, monkeypatch):
        monkeypatch.setenv("SLO_COMPUTE_THREADS", "invalid")
        alloc = ResourceAllocation(compute_threads=4)
        result = alloc.apply_env()
        assert result.compute_threads == 4


# ── compute_allocation ──


class TestComputeAllocation:
    def test_balanced_mode(self, known_topology):
        alloc = compute_allocation(topology=known_topology, mode="balanced")
        assert alloc.workload_mode == "balanced"
        assert alloc.compute_threads == 4  # min(phys, 4)
        assert alloc.io_threads == 2  # max(1, min(4//2, 2))
        assert alloc.inference_pool_size == 1  # max(1, 5//4)
        assert alloc.train_pool_size == 1  # max(1, 5//3)
        assert alloc.task_queue_workers == 2  # max(2, 5//4)

    def test_inference_mode(self, known_topology):
        alloc = compute_allocation(topology=known_topology, mode="inference")
        assert alloc.workload_mode == "inference"
        assert alloc.compute_threads == 2  # clamp(4//2, 1, 4)
        assert alloc.inference_pool_size == 5  # eff cores
        assert alloc.train_pool_size == 1  # min for inference

    def test_training_mode(self, known_topology):
        alloc = compute_allocation(topology=known_topology, mode="training")
        assert alloc.workload_mode == "training"
        assert alloc.compute_threads == 4  # clamp(4, 1, 8)
        assert alloc.inference_pool_size == 1  # min for training
        assert alloc.train_pool_size == 2  # max(2, 5//2)

    def test_defaults_to_balanced(self, known_topology):
        alloc = compute_allocation(topology=known_topology)
        assert alloc.workload_mode == "balanced"

    def test_env_override_in_formula(self, known_topology, monkeypatch):
        monkeypatch.setenv("SLO_COMPUTE_THREADS", "1")
        alloc = compute_allocation(topology=known_topology, mode="balanced")
        assert alloc.compute_threads == 1

    def test_single_core_topology(self):
        topo = CpuTopology(physical_cores=1, logical_cores=1)
        alloc = compute_allocation(topology=topo, mode="balanced")
        assert alloc.compute_threads == 1
        assert alloc.inference_pool_size == 1
        assert alloc.task_queue_workers == 2  # max(2, 1//4) = 2

    def test_many_core_topology(self):
        topo = CpuTopology(physical_cores=64, logical_cores=128)
        alloc = compute_allocation(topology=topo, mode="balanced")
        assert alloc.compute_threads == 4  # clamped to 4
        assert alloc.inference_pool_size == 16  # clamped to 16
        assert alloc.train_pool_size == 8  # clamped to 8
        assert alloc.task_queue_workers == 16  # clamped to 16


# ── ResourceManager singleton ──


class TestResourceManager:
    def test_singleton(self):
        rm1 = get_resource_manager()
        rm2 = get_resource_manager()
        assert rm1 is rm2

    def test_reset(self):
        rm1 = get_resource_manager()
        rm2 = reset_resource_manager()
        assert rm1 is not rm2

    def test_default_mode(self):
        rm = get_resource_manager()
        assert rm.mode == "balanced"

    def test_mode_property(self):
        rm = get_resource_manager(mode="balanced")
        assert rm.mode == "balanced"

    def test_properties_reflect_allocation(self):
        rm = get_resource_manager()
        alloc = rm.allocation
        assert rm.compute_threads == alloc.compute_threads
        assert rm.io_threads == alloc.io_threads
        assert rm.inference_pool_size == alloc.inference_pool_size
        assert rm.train_pool_size == alloc.train_pool_size
        assert rm.task_queue_workers == alloc.task_queue_workers
        assert rm.dataloader_workers == alloc.dataloader_workers
        assert rm.concurrent_writes == alloc.concurrent_writes
        assert rm.concurrent_reads == alloc.concurrent_reads
        assert rm.process_guard_concurrent == alloc.process_guard_concurrent

    def test_blas_properties(self):
        rm = get_resource_manager()
        assert rm.omp_num_threads == rm.compute_threads
        assert rm.mkl_num_threads == rm.compute_threads
        assert rm.numexpr_num_threads == rm.compute_threads
        assert rm.openblas_num_threads == 1

    def test_topology_property(self):
        rm = get_resource_manager()
        assert rm.topology is not None
        assert rm.topology.physical_cores > 0

    def test_recompute_changes_mode(self):
        rm = get_resource_manager()
        rm.recompute("inference")
        assert rm.mode == "inference"
        assert rm.compute_threads != 4 or rm.inference_pool_size > 1  # mode-specific

    def test_summary(self):
        rm = get_resource_manager()
        s = rm.summary()
        assert "balanced" in s
        assert "compute=" in s

    # ── mode_override ──

    def test_mode_override_temporary(self):
        rm = get_resource_manager()
        original = rm.mode
        with rm.mode_override("training"):
            assert rm.mode == "training"
        assert rm.mode == original

    def test_mode_override_applies_blas_env(self):
        rm = get_resource_manager()
        with rm.mode_override("inference"):
            omp = int(os.environ.get("OMP_NUM_THREADS", "0"))
            assert omp == rm.compute_threads

    def test_mode_override_restores_blas_env(self):
        rm = get_resource_manager()
        with rm.mode_override("training"):
            pass
        omp = int(os.environ.get("OMP_NUM_THREADS", "0"))
        assert omp == rm.compute_threads

    def test_mode_override_exception_safe(self):
        rm = get_resource_manager()
        original_mode = rm.mode
        try:
            with rm.mode_override("training"):
                raise ValueError("boom")
        except ValueError:
            pass
        assert rm.mode == original_mode
        # BLAS env should also be restored
        omp = int(os.environ.get("OMP_NUM_THREADS", "0"))
        assert omp == rm.compute_threads

    # ── apply_blas_env / apply_compute_limits ──

    def test_apply_blas_env_sets_omp(self):
        rm = get_resource_manager()
        rm.apply_blas_env()
        assert os.environ["OMP_NUM_THREADS"] == str(rm.omp_num_threads)

    def test_apply_blas_env_overrides_previous(self):
        rm = get_resource_manager()
        os.environ["OMP_NUM_THREADS"] = "99"
        rm.apply_blas_env()
        assert os.environ["OMP_NUM_THREADS"] == str(rm.omp_num_threads)

    def test_apply_compute_limits_calls_np(self):
        import numpy as np
        rm = get_resource_manager()
        # Should not raise
        rm.apply_compute_limits()

    def test_apply_compute_limits_calls_numexpr(self, monkeypatch):
        import numpy as np

        calls = []
        fake_numexpr = types.ModuleType("numexpr")
        fake_numexpr.set_num_threads = lambda n: calls.append(n)
        monkeypatch.setitem(sys.modules, "numexpr", fake_numexpr)
        rm = get_resource_manager()
        rm.apply_compute_limits()
        assert calls == [rm.numexpr_num_threads]

    def test_apply_compute_limits_numexpr_missing_is_silent(self, monkeypatch):
        monkeypatch.delitem(sys.modules, "numexpr", raising=False)
        rm = get_resource_manager()
        rm.apply_compute_limits()  # must not raise

    def test_lazy_init_when_singleton_none(self, monkeypatch):
        import domains.infrastructure.resource_manager as rm_mod

        monkeypatch.setattr(rm_mod, "_global_manager", None)
        rm = rm_mod.get_resource_manager()
        assert rm_mod._global_manager is rm

    def test_singleton_is_global_variable(self, monkeypatch):
        import domains.infrastructure.resource_manager as rm_mod

        rm = get_resource_manager()
        assert rm_mod._global_manager is rm

    def test_apply_environment(self, monkeypatch):
        monkeypatch.setenv("SLO_COMPUTE_THREADS", "1")
        rm = get_resource_manager()
        rm.apply_environment()
        assert rm.compute_threads == 1

    # ── apply_environment ──

    def test_apply_environment_updates_allocation(self, monkeypatch):
        monkeypatch.setenv("SLO_INFERENCE_POOL_SIZE", "8")
        rm = get_resource_manager()
        old = rm.inference_pool_size
        rm.apply_environment()
        # If env var value differs from default, allocation changes
        if old != 8:
            assert rm.inference_pool_size == 8
        else:
            # Env says 8 and default was 8 — still works
            assert rm.inference_pool_size == 8
