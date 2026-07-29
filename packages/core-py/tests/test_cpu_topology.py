"""Tests for CpuTopology, detect_topology, and topology utilities."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from domains.infrastructure.cpu_topology import CpuTopology, detect_topology


# ── CpuTopology dataclass ──


class TestCpuTopology:
    def test_minimal_construction(self):
        t = CpuTopology(physical_cores=2, logical_cores=4)
        assert t.physical_cores == 2
        assert t.logical_cores == 4
        assert t.effective_cores == 2  # base + 0.25*(4-2) = 2

    def test_default_values(self):
        t = CpuTopology(physical_cores=4, logical_cores=8)
        assert t.l1d_cache_kb == 0
        assert t.cpu_freq_mhz == 0.0
        assert t.model_name == ""
        assert t.numa_nodes == 1
        assert not t.has_hyperthreading

    def test_full_construction(self):
        t = CpuTopology(
            physical_cores=4,
            logical_cores=8,
            has_hyperthreading=True,
            l1d_cache_kb=32,
            l2_cache_kb=256,
            l3_cache_kb=8192,
            numa_nodes=2,
            cpu_freq_mhz=2400.0,
            model_name="Intel Core i5",
        )
        assert t.l1d_cache_kb == 32
        assert t.l2_cache_kb == 256
        assert t.l3_cache_kb == 8192
        assert t.cpu_freq_mhz == 2400.0
        assert t.model_name == "Intel Core i5"
        assert t.numa_nodes == 2
        assert t.has_hyperthreading

    def test_summary_includes_key_fields(self):
        t = CpuTopology(physical_cores=4, logical_cores=8, has_hyperthreading=True)
        s = t.summary()
        assert "8L" in s
        assert "4P" in s
        assert "HT" in s

    def test_effective_cores_no_ht(self):
        t = CpuTopology(physical_cores=4, logical_cores=4)
        assert t.effective_cores == 4  # no HT

    def test_effective_cores_with_ht(self):
        t = CpuTopology(physical_cores=4, logical_cores=8)
        assert t.effective_cores == 5  # 4 + 0.25*(8-4)

    def test_threads_per_core(self):
        t = CpuTopology(physical_cores=4, logical_cores=8)
        assert t.threads_per_core == 2

    def test_threads_per_core_no_ht(self):
        t = CpuTopology(physical_cores=4, logical_cores=4)
        assert t.threads_per_core == 1


# ── detect_topology ──


class TestDetectTopology:
    def test_detects_positive_values(self):
        t = detect_topology()
        assert t.physical_cores >= 1
        assert t.logical_cores >= t.physical_cores
        assert t.effective_cores >= 1

    def test_physical_cores_reasonable(self):
        t = detect_topology()
        assert t.physical_cores <= 256  # no known CPU has more

    def test_effective_cores_between_phys_and_logical(self):
        t = detect_topology()
        # effective is physical + (logical - physical) * 0.25
        assert t.physical_cores <= t.effective_cores <= t.logical_cores

    def test_different_topologies_independent(self):
        t1 = detect_topology()
        t2 = detect_topology()
        # Same host → same topology
        assert t1.physical_cores == t2.physical_cores
        assert t1.logical_cores == t2.logical_cores


# ── Edge cases ──


class TestTopologyEdgeCases:
    def test_single_threaded_topology(self):
        t = CpuTopology(physical_cores=1, logical_cores=1)
        assert t.effective_cores == 1

    def test_hyperthreaded_topology(self):
        t = CpuTopology(physical_cores=4, logical_cores=8)
        assert t.effective_cores == 5  # 4 + 0.25*(8-4)

    def test_large_topology(self):
        t = CpuTopology(physical_cores=64, logical_cores=128)
        assert t.effective_cores <= t.logical_cores

    def test_physical_exceeds_logical(self):
        t = CpuTopology(physical_cores=8, logical_cores=4)
        # No validation — dataclass is frozen, not validated


# ── detect_topology with mocked cpuinfo ──


# detect_topology uses real /proc/cpuinfo and os.cpu_count — tested
# on the actual host.  Mocking open() is fragile because _linux_cpuinfo
# opens multiple proc files.  Instead, we verify that the real detection
# is internally consistent.
