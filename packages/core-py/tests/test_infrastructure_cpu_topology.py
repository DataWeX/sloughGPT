"""Tests for CpuTopology — CPU detection dataclass and properties."""
from __future__ import annotations

from domains.infrastructure.cpu_topology import CpuTopology


class TestCpuTopologyDefaults:
    def test_default_values(self):
        topo = CpuTopology()
        assert topo.logical_cores == 1
        assert topo.physical_cores == 1
        assert topo.has_hyperthreading is False
        assert topo.l1d_cache_kb == 0
        assert topo.l2_cache_kb == 0
        assert topo.l3_cache_kb == 0
        assert topo.numa_nodes == 1
        assert topo.cpu_freq_mhz == 0.0
        assert topo.model_name == ""


class TestThreadsPerCore:
    def test_no_ht(self):
        topo = CpuTopology(logical_cores=8, physical_cores=8)
        assert topo.threads_per_core == 1

    def test_with_ht(self):
        topo = CpuTopology(logical_cores=16, physical_cores=8)
        assert topo.threads_per_core == 2

    def test_zero_physical(self):
        topo = CpuTopology(physical_cores=0)
        assert topo.threads_per_core == 1


class TestEffectiveCores:
    def test_no_ht(self):
        topo = CpuTopology(logical_cores=8, physical_cores=8)
        assert topo.effective_cores == 8

    def test_with_ht(self):
        topo = CpuTopology(logical_cores=16, physical_cores=8)
        # 8 + (8 * 0.25) = 10
        assert topo.effective_cores == 10

    def test_single_core(self):
        topo = CpuTopology(logical_cores=2, physical_cores=1)
        # 1 + (1 * 0.25) = 1 (int)
        assert topo.effective_cores == 1


class TestSummary:
    def test_summary_contains_key_info(self):
        topo = CpuTopology(logical_cores=8, physical_cores=4, model_name="Test CPU")
        s = topo.summary()
        assert "8" in s
        assert "4" in s
        assert "Test CPU" in s

    def test_summary_is_string(self):
        topo = CpuTopology()
        assert isinstance(topo.summary(), str)


class TestImmutability:
    def test_frozen_dataclass(self):
        topo = CpuTopology()
        try:
            topo.logical_cores = 99
            assert False, "Should be frozen"
        except AttributeError:
            pass
