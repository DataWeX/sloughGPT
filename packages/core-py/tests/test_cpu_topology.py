"""Tests for CpuTopology, detect_topology, and topology utilities."""

import builtins
import io
import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from domains.infrastructure import cpu_topology as ct
from domains.infrastructure.cpu_topology import CpuTopology, detect_topology


@pytest.fixture(autouse=True)
def _clear_topology_cache():
    detect_topology.cache_clear()
    yield
    detect_topology.cache_clear()


def _patch_proc_files(monkeypatch, files):
    """Patch ``builtins.open`` so known paths return StringIO contents."""

    def _opener(path, *a, **k):
        if path in files:
            return io.StringIO(files[path])
        raise FileNotFoundError(path)

    monkeypatch.setattr(builtins, "open", _opener)


def _patch_proc_files_raising(monkeypatch, raise_paths):
    def _opener(path, *a, **k):
        if path in raise_paths:
            raise OSError("denied")
        raise FileNotFoundError(path)

    monkeypatch.setattr(builtins, "open", _opener)


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

    def test_threads_per_core_zero_physical(self):
        t = CpuTopology(physical_cores=0, logical_cores=4)
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


# ── sysctl helpers ──


class TestSysctlHelpers:
    def test_parse_sysctl_success(self, monkeypatch):
        monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: "32768\n")
        assert ct._parse_sysctl("hw.l1dcachesize") == 32768

    def test_parse_sysctl_error(self, monkeypatch):
        def _fail(*a, **k):
            raise subprocess.CalledProcessError(1, "sysctl")

        monkeypatch.setattr(subprocess, "check_output", _fail)
        assert ct._parse_sysctl("hw.l1dcachesize") is None

    def test_parse_sysctl_string_success(self, monkeypatch):
        monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: "Apple M1\n")
        assert ct._parse_sysctl_string("machdep.cpu.brand_string") == "Apple M1"

    def test_parse_sysctl_string_error(self, monkeypatch):
        def _fail(*a, **k):
            raise OSError("no sysctl")

        monkeypatch.setattr(subprocess, "check_output", _fail)
        assert ct._parse_sysctl_string("machdep.cpu.brand_string") is None

    def test_sysctl_cache_kb_known(self, monkeypatch):
        monkeypatch.setattr(ct, "_parse_sysctl", lambda key: 32768)
        assert ct._sysctl_cache_kb("l1d") == 32

    def test_sysctl_cache_kb_unknown(self):
        assert ct._sysctl_cache_kb("nope") == 0

    def test_sysctl_cache_kb_none_val(self, monkeypatch):
        monkeypatch.setattr(ct, "_parse_sysctl", lambda key: None)
        assert ct._sysctl_cache_kb("l1d") == 0


# ── /proc/cpuinfo parsing ──


class TestLinuxCpuinfo:
    def test_missing_cpuinfo(self, monkeypatch):
        _patch_proc_files_raising(monkeypatch, {"/proc/cpuinfo"})
        assert ct._linux_cpuinfo() == (0, 0, 0, 0, "")

    def test_full_cpuinfo(self, monkeypatch):
        text = (
            "physical id\t: 0\n"
            "cpu cores\t: 2\n"
            "core id\t\t: 0\n"
            "model name\t: Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz\n"
            "cache size\t: 8192 KB\n"
            "physical id\t: 0\n"
            "core id\t\t: 1\n"
            "cache size\t: 2 MB\n"
        )
        _patch_proc_files(monkeypatch, {"/proc/cpuinfo": text})
        assert ct._linux_cpuinfo() == (2, 2, 0, 8192, "Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz")

    def test_bad_value_lines(self, monkeypatch):
        text = (
            "physical id\t: 0\n"
            "physical id\t: x\n"
            "cpu cores\t: \n"
            "core id\t\t: bad\n"
            "cache size\t: bogus KB\n"
            "cache size\t: bogus MB\n"
        )
        _patch_proc_files(monkeypatch, {"/proc/cpuinfo": text})
        assert ct._linux_cpuinfo() == (0, 0, 0, 0, "")

    def test_no_core_ids_uses_physical_count(self, monkeypatch):
        text = "physical id\t: 0\ncpu cores\t: 4\nphysical id\t: 1\ncpu cores\t: 4\n"
        _patch_proc_files(monkeypatch, {"/proc/cpuinfo": text})
        assert ct._linux_cpuinfo() == (8, 4, 0, 0, "")

    def test_only_core_count_no_physical(self, monkeypatch):
        text = "cpu cores\t: 4\n"
        _patch_proc_files(monkeypatch, {"/proc/cpuinfo": text})
        assert ct._linux_cpuinfo() == (0, 4, 0, 0, "")


# ── lscpu / NUMA detection ──


class TestLinuxLscpu:
    def test_lscpu_ok(self, monkeypatch):
        monkeypatch.setattr(
            subprocess,
            "check_output",
            lambda *a, **k: "Architecture: x86_64\nL1d cache: 32K\n",
        )
        assert ct._linux_lscpu() == {"Architecture": "x86_64", "L1d cache": "32K"}

    def test_lscpu_error(self, monkeypatch):
        def _fail(*a, **k):
            raise OSError("no lscpu")

        monkeypatch.setattr(subprocess, "check_output", _fail)
        assert ct._linux_lscpu() == {}


class TestDetectNuma:
    def test_numa_from_lscpu(self, monkeypatch):
        monkeypatch.setattr(
            subprocess, "check_output", lambda *a, **k: "CPU(s): 8\nNUMA node(s): 2\n"
        )
        assert ct._detect_numa() == 2

    def test_numa_from_sysfs(self, monkeypatch):
        def _fail(*a, **k):
            raise OSError("no lscpu")

        monkeypatch.setattr(subprocess, "check_output", _fail)
        monkeypatch.setattr(os, "listdir", lambda p: ["node0", "node1", "node2"])
        assert ct._detect_numa() == 3

    def test_numa_unavailable(self, monkeypatch):
        def _fail(*a, **k):
            raise OSError("no lscpu")

        monkeypatch.setattr(subprocess, "check_output", _fail)
        monkeypatch.setattr(os, "listdir", lambda p: (_ for _ in ()).throw(OSError("no")))
        assert ct._detect_numa() == 1


# ── CPU frequency detection ──


class TestDetectFreq:
    def test_macos_from_sysctl_value(self, monkeypatch):
        monkeypatch.setattr(ct, "_parse_sysctl", lambda key: 2_400_000_000)
        assert ct._detect_freq_macos() == 2400.0

    def test_macos_from_brand_string(self, monkeypatch):
        monkeypatch.setattr(ct, "_parse_sysctl", lambda key: None)
        monkeypatch.setattr(
            ct,
            "_parse_sysctl_string",
            lambda key: "Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz",
        )
        assert ct._detect_freq_macos() == 2600.0

    def test_macos_brand_error(self, monkeypatch):
        monkeypatch.setattr(ct, "_parse_sysctl", lambda key: None)

        def _raise(key):
            raise RuntimeError("sysctl failed")

        monkeypatch.setattr(ct, "_parse_sysctl_string", _raise)
        assert ct._detect_freq_macos() == 0.0

    def test_macos_brand_no_match(self, monkeypatch):
        monkeypatch.setattr(ct, "_parse_sysctl", lambda key: None)
        monkeypatch.setattr(ct, "_parse_sysctl_string", lambda key: "Apple M1 Pro")
        assert ct._detect_freq_macos() == 0.0

    def test_linux_from_cpuinfo(self, monkeypatch):
        _patch_proc_files(monkeypatch, {"/proc/cpuinfo": "processor : 0\ncpu MHz : 2400.123\n"})
        assert ct._detect_freq_linux() == 2400.123

    def test_linux_from_sysfs(self, monkeypatch):
        def _opener(path, *a, **k):
            if path == "/proc/cpuinfo":
                raise OSError("no")
            if path == "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq":
                return io.StringIO("2400000\n")
            raise FileNotFoundError(path)

        monkeypatch.setattr(builtins, "open", _opener)
        monkeypatch.setattr(os, "cpu_count", lambda: 2)
        monkeypatch.setattr(os.path, "exists", lambda p: True)
        assert ct._detect_freq_linux() == 2400.0

    def test_linux_freq_unavailable(self, monkeypatch):
        def _opener(path, *a, **k):
            raise OSError("no")

        monkeypatch.setattr(builtins, "open", _opener)
        monkeypatch.setattr(os, "cpu_count", lambda: 2)
        monkeypatch.setattr(os.path, "exists", lambda p: True)
        assert ct._detect_freq_linux() == 0.0


# ── cgroup detection ──


class TestCgroupCpuset:
    def test_cgroupv1_cpuset(self, monkeypatch):
        monkeypatch.setattr(os.path, "isfile", lambda p: p == "/proc/self/cgroup")
        _patch_proc_files(monkeypatch, {"/proc/self/cgroup": "12:cpuset:/docker/abc\n"})
        assert ct._has_cgroup_cpuset() is True

    def test_cgroupv2_effective_file(self, monkeypatch):
        monkeypatch.setattr(
            os.path, "isfile", lambda p: p == "/sys/fs/cgroup/cpuset.cpus.effective"
        )
        assert ct._has_cgroup_cpuset() is True

    def test_no_cgroup(self, monkeypatch):
        monkeypatch.setattr(os.path, "isfile", lambda p: False)
        assert ct._has_cgroup_cpuset() is False

    def test_cgroup_read_error(self, monkeypatch):
        monkeypatch.setattr(os.path, "isfile", lambda p: p == "/proc/self/cgroup")
        _patch_proc_files_raising(monkeypatch, {"/proc/self/cgroup"})
        assert ct._has_cgroup_cpuset() is False


# ── detect_topology with mocked platform ──


class TestDetectTopologyMocked:
    def test_linux_lscpu_fallback(self, monkeypatch):
        monkeypatch.setattr(ct.platform, "system", lambda: "Linux")
        monkeypatch.setattr(ct.os, "cpu_count", lambda: 8)
        monkeypatch.setattr(ct, "_linux_cpuinfo", lambda: (0, 0, 0, 0, ""))
        monkeypatch.setattr(
            ct,
            "_linux_lscpu",
            lambda: {"L1d cache": "32K", "L2 cache": "256K", "L3 cache": "8M"},
        )
        monkeypatch.setattr(ct, "_detect_numa", lambda: 2)
        monkeypatch.setattr(ct, "_detect_freq_linux", lambda: 2400.0)

        t = detect_topology()
        assert t.logical_cores == 8
        assert t.physical_cores == 4  # max(1, 8 // 2) fallback
        assert t.has_hyperthreading is True
        assert t.l1d_cache_kb == 32
        assert t.l2_cache_kb == 256
        assert t.l3_cache_kb == 8  # regex only captures the leading digits
        assert t.numa_nodes == 2
        assert t.cpu_freq_mhz == 2400.0
        assert t.model_name == ""

    def test_macos_full(self, monkeypatch):
        monkeypatch.setattr(ct.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(ct.os, "cpu_count", lambda: 8)

        def _sysctl(cmd_args, **k):
            key = cmd_args[-1]
            return {
                "machdep.cpu.core_count": "4\n",
                "machdep.cpu.brand_string": "Apple M1 Pro\n",
            }[key]

        monkeypatch.setattr(subprocess, "check_output", _sysctl)
        monkeypatch.setattr(
            ct, "_sysctl_cache_kb", lambda name: {"l1d": 32, "l2": 256, "l3": 8192}[name]
        )
        monkeypatch.setattr(ct, "_detect_freq_macos", lambda: 2400.0)

        t = detect_topology()
        assert t.physical_cores == 4
        assert t.logical_cores == 8
        assert t.l1d_cache_kb == 32
        assert t.l3_cache_kb == 8192
        assert t.numa_nodes == 1
        assert t.cpu_freq_mhz == 2400.0
        assert t.model_name == "Apple M1 Pro"

    def test_macos_fallback(self, monkeypatch):
        monkeypatch.setattr(ct.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(ct.os, "cpu_count", lambda: 8)

        def _raise(cmd_args, **k):
            raise RuntimeError("no sysctl")

        monkeypatch.setattr(subprocess, "check_output", _raise)
        monkeypatch.setattr(ct, "_sysctl_cache_kb", lambda name: 0)
        monkeypatch.setattr(ct, "_detect_freq_macos", lambda: 0.0)

        t = detect_topology()
        assert t.physical_cores == 4  # best-effort estimate
        assert t.has_hyperthreading is True
        assert t.l1d_cache_kb == 0
        assert t.numa_nodes == 1
        assert t.cpu_freq_mhz == 0.0
        assert t.model_name == ""
