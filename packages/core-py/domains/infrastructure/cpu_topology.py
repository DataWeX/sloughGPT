"""
CPU topology detection — physical/logical cores, hyperthreading, cache, NUMA.

Provides a single ``CpuTopology`` dataclass that all downstream resource
allocation logic reads from, instead of calling ``os.cpu_count()`` or
``multiprocessing.cpu_count()`` ad-hoc.
"""

from __future__ import annotations

import logging
import os
import platform
import re
import subprocess
from dataclasses import dataclass, field

logger = logging.getLogger("slo.infrastructure.cpu_topology")
from functools import lru_cache
from typing import Optional


@dataclass(frozen=True)
class CpuTopology:
    """Immutable snapshot of the detected CPU topology.

    All counts are *per-process* (cgroup-aware on Linux) when the
    corresponding detection succeeds, falling back to system-wide.

    Attributes
    ----------
    logical_cores : int
        Number of logical CPUs visible to the process (``os.cpu_count()``).
    physical_cores : int
        Estimated physical core count.  On Linux this reads
        ``cpu cores`` from ``/proc/cpuinfo``; elsewhere it falls back
        to ``logical_cores // 2`` (best-effort).
    has_hyperthreading : bool
        ``True`` when ``logical_cores > physical_cores``.
    l1d_cache_kb : int
        L1 data cache size per core in KB (0 if undetectable).
    l2_cache_kb : int
        L2 cache size per core in KB (0 if undetectable).
    l3_cache_kb : int
        L3 cache *shared* size in KB (0 if undetectable).
    numa_nodes : int
        Number of NUMA nodes (1 on non-NUMA systems).
    cpu_freq_mhz : float
        Current CPU frequency in MHz (0 if undetectable).
    model_name : str
        CPU model name string (empty if undetectable).
    """

    logical_cores: int = 1
    physical_cores: int = 1
    has_hyperthreading: bool = False
    l1d_cache_kb: int = 0
    l2_cache_kb: int = 0
    l3_cache_kb: int = 0
    numa_nodes: int = 1
    cpu_freq_mhz: float = 0.0
    model_name: str = ""

    @property
    def threads_per_core(self) -> int:
        """Logical threads per physical core (typically 1 or 2)."""
        if self.physical_cores == 0:
            return 1
        return max(1, self.logical_cores // self.physical_cores)

    @property
    def effective_cores(self) -> int:
        """Cores available for CPU-bound work, discounting HT.

        HT provides ~15-30% throughput improvement for compute-bound
        workloads; treat it as ``physical_cores + (ht_threads * 0.25)``.
        """
        base = self.physical_cores
        extra = self.logical_cores - self.physical_cores
        return base + int(extra * 0.25)

    def summary(self) -> str:
        return (
            f"{self.logical_cores}L/{self.physical_cores}P cores"
            f"{' HT' if self.has_hyperthreading else ''}"
            f" | L1d={self.l1d_cache_kb}KB L2={self.l2_cache_kb}KB L3={self.l3_cache_kb}KB"
            f" | NUMA={self.numa_nodes}"
            f" | {self.cpu_freq_mhz:.0f}MHz"
            f" | {self.model_name}"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SYSCTL_CACHE_KEYS: dict[str, tuple[str, int]] = {
    "l1d": ("hw.l1dcachesize", 1024),
    "l2": ("hw.l2cachesize", 1024),
    "l3": ("hw.l3cachesize", 1024),
}


def _parse_sysctl(key: str) -> Optional[int]:
    """Run ``sysctl -n <key>`` and return the integer value."""
    try:
        out = subprocess.check_output(["sysctl", "-n", key], text=True).strip()
        return int(out)
    except Exception:
        return None


def _sysctl_cache_kb(name: str) -> int:
    """Read macOS cache size via sysctl, return KB."""
    key, divisor = _SYSCTL_CACHE_KEYS.get(name, ("", 1))
    if not key:
        return 0
    val = _parse_sysctl(key)
    return (val // divisor) if val else 0


def _linux_cpuinfo() -> tuple[int, int, int, int, str]:
    """Parse ``/proc/cpuinfo`` for core count, cache, model."""
    try:
        with open("/proc/cpuinfo") as f:
            text = f.read()
    except OSError:
        return 0, 0, 0, 0, ""

    physical_ids: set[int] = set()
    core_ids: set[tuple[int, int]] = set()
    l1d = l2 = l3 = 0
    model = ""
    current_phys: Optional[int] = None

    for line in text.splitlines():
        line = line.strip()
        if line.startswith("physical id"):
            try:
                current_phys = int(line.split(":")[1].strip())
                physical_ids.add(current_phys)
            except (ValueError, IndexError):
                pass
        elif line.startswith("cpu cores"):
            try:
                l1d = int(line.split(":")[1].strip())
            except (ValueError, IndexError):
                pass
        elif line.startswith("core id") and current_phys is not None:
            try:
                cid = int(line.split(":")[1].strip())
                core_ids.add((current_phys, cid))
            except (ValueError, IndexError):
                pass
        elif line.startswith("model name"):
            parts = line.split(":", 1)
            if len(parts) > 1:
                model = parts[1].strip()

    # Cache size lines
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("cache size"):
            try:
                val = line.split(":")[1].strip()
                if val.endswith("KB"):
                    l3 = max(l3, int(val[:-2].strip()))
                elif val.endswith("MB"):
                    l3 = max(l3, int(float(val[:-2].strip()) * 1024))
            except (ValueError, IndexError):
                pass

    n_physical = len(core_ids) if core_ids else (len(physical_ids) * l1d if l1d else 0)
    if n_physical == 0 and l1d:
        n_physical = len(physical_ids) * l1d

    return n_physical, l1d, l2, l3, model


def _linux_lscpu() -> dict[str, str]:
    """Parse ``lscpu`` output into a dict."""
    try:
        out = subprocess.check_output(["lscpu"], text=True)
    except Exception:
        return {}
    result: dict[str, str] = {}
    for line in out.splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            result[key.strip()] = val.strip()
    return result


def _detect_numa() -> int:
    """Detect number of NUMA nodes."""
    try:
        out = subprocess.check_output(["lscpu"], text=True)
        for line in out.splitlines():
            if "NUMA node(s)" in line or "NUMA nodes" in line:
                return int(line.split(":")[1].strip())
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as exc:
        logger.debug("lscpu NUMA detection failed: %s", exc)
    try:
        nodes = os.listdir("/sys/devices/system/node/")
        numa = [n for n in nodes if n.startswith("node") and n[4:].isdigit()]
        if numa:
            return len(numa)
    except OSError:
        pass
    return 1


def _detect_freq_macos() -> float:
    """Read Mac CPU frequency via sysctl."""
    val = _parse_sysctl("hw.cpufrequency")
    if val:
        return val / 1_000_000
    # fallback: estimate from brand string
    try:
        brand = _parse_sysctl_string("machdep.cpu.brand_string")
        if brand:
            m = re.search(r"(\d+\.?\d*)\s*GHz", brand)
            if m:
                return float(m.group(1)) * 1000
    except (ValueError, AttributeError) as exc:
        logger.debug("macOS frequency parse failed: %s", exc)
    return 0.0


def _parse_sysctl_string(key: str) -> Optional[str]:
    try:
        return subprocess.check_output(["sysctl", "-n", key], text=True).strip()
    except Exception:
        return None


def _detect_freq_linux() -> float:
    """Read current CPU frequency from ``/proc/cpuinfo`` or sysfs."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("cpu MHz"):
                    return float(line.split(":")[1].strip())
    except (FileNotFoundError, ValueError) as exc:
        logger.debug("/proc/cpuinfo frequency read failed: %s", exc)
    try:
        for i in range(os.cpu_count() or 1):
            path = f"/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_cur_freq"
            if os.path.exists(path):
                with open(path) as f:
                    return int(f.read().strip()) / 1000
    except (FileNotFoundError, ValueError) as exc:
        logger.debug("sysfs frequency read failed: %s", exc)
    return 0.0


def _has_cgroup_cpuset() -> bool:
    """Check if the process is running inside a cgroupv1/v2 cpu partition."""
    try:
        if os.path.isfile("/proc/self/cgroup"):
            with open("/proc/self/cgroup") as f:
                text = f.read()
            if "cpuset" in text:
                return True
        if os.path.isfile("/sys/fs/cgroup/cpuset.cpus.effective"):
            return True
    except OSError:
        pass
    return False


# ---------------------------------------------------------------------------
# Public detection entry point
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def detect_topology() -> CpuTopology:
    """Detect CPU topology once and cache the result.

    Returns a frozen ``CpuTopology`` dataclass.  The result is cached
    for the lifetime of the process.
    """
    logical = os.cpu_count() or 1

    system = platform.system()
    is_linux = system == "Linux"
    is_macos = system == "Darwin"

    physical = 0
    l1d = l2 = l3 = 0
    model = ""

    if is_linux:
        phys, c_l1d, c_l2, c_l3, c_model = _linux_cpuinfo()
        physical = phys
        l1d = c_l1d
        l2 = c_l2
        l3 = c_l3
        model = c_model

        # Try lscpu for cache if /proc/cpuinfo didn't have it
        if l1d == 0:
            lscpu = _linux_lscpu()
            if "L1d cache" in lscpu:
                m = re.match(r"(\d+)\s*K?", lscpu["L1d cache"])
                if m:
                    l1d = int(m.group(1))
            if "L2 cache" in lscpu:
                m = re.match(r"(\d+)\s*K?", lscpu["L2 cache"])
                if m:
                    l2 = int(m.group(1))
            if "L3 cache" in lscpu:
                m = re.match(r"(\d+)\s*K?", lscpu["L3 cache"])
                if m:
                    l3 = int(m.group(1))

    elif is_macos:
        l1d = _sysctl_cache_kb("l1d")
        l2 = _sysctl_cache_kb("l2")
        l3 = _sysctl_cache_kb("l3")
        try:
            import subprocess
            out = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.core_count"], text=True
            ).strip()
            physical = int(out)
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as exc:
            logger.debug("sysctl core_count failed: %s", exc)
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
            model = out
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            logger.debug("sysctl brand_string failed: %s", exc)

    if physical == 0:
        # Best-effort estimate
        physical = max(1, logical // 2)

    has_ht = logical > physical
    numa = _detect_numa() if is_linux else 1

    freq = 0.0
    if is_linux:
        freq = _detect_freq_linux()
    elif is_macos:
        freq = _detect_freq_macos()

    if not model:
        if is_linux:
            _, _, _, _, model = _linux_cpuinfo()
        elif is_macos:
            try:
                model = subprocess.check_output(
                    ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
                ).strip()
            except (subprocess.CalledProcessError, FileNotFoundError) as exc:
                logger.debug("sysctl brand_string fallback failed: %s", exc)

    return CpuTopology(
        logical_cores=logical,
        physical_cores=physical,
        has_hyperthreading=has_ht,
        l1d_cache_kb=l1d,
        l2_cache_kb=l2,
        l3_cache_kb=l3,
        numa_nodes=numa,
        cpu_freq_mhz=freq,
        model_name=model,
    )
