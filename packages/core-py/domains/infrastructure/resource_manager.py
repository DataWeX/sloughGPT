"""
Centralized resource allocation — auto-tunes every pool and thread count
from the detected CPU topology for the SloNet / NumPy stack.

Usage::

    from domains.infrastructure.resource_manager import get_resource_manager

    rm = get_resource_manager()
    pool_size = rm.inference_pool_size

All values are computable from ``CpuTopology`` and overridable via env vars:

    ============================== ======================= =========================
    Env var                        Default formula         Controls
    ============================== ======================= =========================
    SLO_COMPUTE_THREADS            min(phys, 4)            Threads per compute op
    SLO_IO_THREADS                 2                       Background I/O threads
    SLO_INFERENCE_POOL_SIZE        eff_cores // compute    Concurrent inference slots
    SLO_TRAIN_POOL_SIZE            max(1, eff_cores // 2)  Training thread pool
    SLO_TASK_QUEUE_WORKERS         max(2, eff_cores // 4)  Async task queue workers
    SLO_DATALOADER_WORKERS         max(0, min(4, phys//2)) DataLoader prefetch workers
    SLO_OMP_NUM_THREADS            compute_threads         OpenMP threads (numpy)
    SLO_MKL_NUM_THREADS            compute_threads         Intel MKL threads (numpy)
    SLO_OPENBLAS_NUM_THREADS       1                       OpenBLAS threads (numpy)
    SLO_NUMEXPR_NUM_THREADS        compute_threads         NumExpr threads (numpy)
    SLO_CONCURRENT_READS           pool * 4                ModelServer read semaphore
    ============================== ======================= =========================
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

from domains.infrastructure.cpu_topology import CpuTopology, detect_topology

logger = logging.getLogger("slo.infrastructure.resource_manager")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Resource allocation profile
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResourceAllocation:
    """All computed resource numbers for the current host."""

    topology: CpuTopology = field(default_factory=detect_topology)

    # --- Compute / I/O threads (our own stack) ---
    compute_threads: int = 0   # threads per compute op (numpy/loop-level)
    io_threads: int = 0        # threads for background I/O

    # --- BLAS / numpy thread controls ---
    omp_num_threads: int = 0
    mkl_num_threads: int = 0
    openblas_num_threads: int = 0
    numexpr_num_threads: int = 0

    # --- Pool sizes ---
    inference_pool_size: int = 0
    train_pool_size: int = 0
    task_queue_workers: int = 0
    dataloader_workers: int = 0

    # --- Concurrency gates ---
    concurrent_writes: int = 0  # ModelServer generate semaphore
    concurrent_reads: int = 0   # ModelServer tokenize/health semaphore
    process_guard_concurrent: int = 0  # subprocess guard semaphore

    # --- Workload mode ---
    workload_mode: str = "balanced"  # "inference" | "training" | "balanced"

    def summary(self) -> str:
        return (
            f"ResourceAllocation[mode={self.workload_mode}] "
            f"compute={self.compute_threads} io={self.io_threads} "
            f"infer={self.inference_pool_size} train={self.train_pool_size} "
            f"queue={self.task_queue_workers} dl={self.dataloader_workers} "
            f"writes={self.concurrent_writes} reads={self.concurrent_reads}"
        )

    def apply_env(self) -> ResourceAllocation:
        """Override fields from environment variables."""
        kwargs: dict = {}
        for env_key, attr in [
            ("SLO_COMPUTE_THREADS", "compute_threads"),
            ("SLO_IO_THREADS", "io_threads"),
            ("SLO_INFERENCE_POOL_SIZE", "inference_pool_size"),
            ("SLO_TRAIN_POOL_SIZE", "train_pool_size"),
            ("SLO_TASK_QUEUE_WORKERS", "task_queue_workers"),
            ("SLO_DATALOADER_WORKERS", "dataloader_workers"),
            ("SLO_OMP_NUM_THREADS", "omp_num_threads"),
            ("SLO_MKL_NUM_THREADS", "mkl_num_threads"),
            ("SLO_OPENBLAS_NUM_THREADS", "openblas_num_threads"),
            ("SLO_NUMEXPR_NUM_THREADS", "numexpr_num_threads"),
            ("SLO_CONCURRENT_WRITES", "concurrent_writes"),
            ("SLO_CONCURRENT_READS", "concurrent_reads"),
            ("SLO_PROCESS_GUARD_CONCURRENT", "process_guard_concurrent"),
        ]:
            val = _env_int(env_key, -1)
            if val >= 0:
                kwargs[attr] = val
        if kwargs:
            return ResourceAllocation(**{**self.__dict__, **kwargs})
        return self


# ---------------------------------------------------------------------------
# Allocation formulas
# ---------------------------------------------------------------------------


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def compute_allocation(
    topology: Optional[CpuTopology] = None,
    mode: str = "balanced",
) -> ResourceAllocation:
    """Compute optimal resource allocation from CPU topology.

    Parameters
    ----------
    topology:
        Detected topology; auto-detected if ``None``.
    mode:
        ``"inference"`` — favour inference throughput (max concurrent slots,
        fewer threads per compute op).
        ``"training"`` — favour training throughput (large train pool,
        more threads per compute op, limited inference concurrency).
        ``"balanced"`` — reasonable compromise for mixed workloads.
    """
    topo = topology or detect_topology()
    phys = topo.physical_cores
    eff = topo.effective_cores

    # --- Compute threads (our own per-op parallelism) ---
    if mode == "inference":
        # Many concurrent requests; keep per-op threads low
        compute = _clamp(phys // 2, 1, 4)
    elif mode == "training":
        # Fewer requests; maximise single-op throughput
        compute = _clamp(phys, 1, 8)
    else:
        compute = _clamp(min(phys, 4), 1, 4)

    io = _clamp(min(phys // 2, 2), 1, 4)

    # --- Pool sizes ---
    if mode == "inference":
        infer_pool = _clamp(eff, 1, 16)
    elif mode == "training":
        infer_pool = _clamp(1, 1, 4)
    else:
        infer_pool = _clamp(max(1, eff // compute), 1, 16)

    if mode == "training":
        train_pool = _clamp(max(2, eff // 2), 1, 8)
    elif mode == "inference":
        train_pool = _clamp(1, 1, 2)
    else:
        train_pool = _clamp(max(1, eff // 3), 1, 8)

    task_q = _clamp(max(2, eff // 4), 1, 16)
    dl_workers = _clamp(min(4, phys // 2), 0, 16)

    # --- Concurrency gates ---
    concurrent_writes = _clamp(infer_pool, 1, 16)
    concurrent_reads = _clamp(infer_pool * 4, 1, 64)
    pg_concurrent = _clamp(1, 1, 4)

    return ResourceAllocation(
        topology=topo,
        compute_threads=compute,
        io_threads=io,
        omp_num_threads=compute,
        mkl_num_threads=compute,
        openblas_num_threads=1,
        numexpr_num_threads=compute,
        inference_pool_size=infer_pool,
        train_pool_size=train_pool,
        task_queue_workers=task_q,
        dataloader_workers=dl_workers,
        concurrent_writes=concurrent_writes,
        concurrent_reads=concurrent_reads,
        process_guard_concurrent=pg_concurrent,
        workload_mode=mode,
    ).apply_env()


# ---------------------------------------------------------------------------
# Singleton manager
# ---------------------------------------------------------------------------


class ResourceManager:
    """Central resource manager — auto-tunes everything from CPU topology.

    Thread-safe (read-only after init).  Call ``recompute()`` to
    switch workload mode at runtime.

    Usage::

        rm = ResourceManager()
        pool = rm.inference_pool_size
        rm.apply_environment()  # set env vars once at startup
    """

    def __init__(self, mode: str = "balanced"):
        self._alloc = compute_allocation(mode=mode)
        self._mode = mode

    # --- Read-only properties ---

    @property
    def topology(self) -> CpuTopology:
        return self._alloc.topology

    @property
    def compute_threads(self) -> int:
        """Threads per compute operation (numpy / loop-level parallelism)."""
        return self._alloc.compute_threads

    @property
    def io_threads(self) -> int:
        """Threads for background I/O operations."""
        return self._alloc.io_threads

    @property
    def inference_pool_size(self) -> int:
        return self._alloc.inference_pool_size

    @property
    def train_pool_size(self) -> int:
        return self._alloc.train_pool_size

    @property
    def task_queue_workers(self) -> int:
        return self._alloc.task_queue_workers

    @property
    def dataloader_workers(self) -> int:
        return self._alloc.dataloader_workers

    @property
    def omp_num_threads(self) -> int:
        return self._alloc.omp_num_threads

    @property
    def mkl_num_threads(self) -> int:
        return self._alloc.mkl_num_threads

    @property
    def openblas_num_threads(self) -> int:
        return self._alloc.openblas_num_threads

    @property
    def numexpr_num_threads(self) -> int:
        return self._alloc.numexpr_num_threads

    @property
    def concurrent_writes(self) -> int:
        return self._alloc.concurrent_writes

    @property
    def concurrent_reads(self) -> int:
        return self._alloc.concurrent_reads

    @property
    def process_guard_concurrent(self) -> int:
        return self._alloc.process_guard_concurrent

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def allocation(self) -> ResourceAllocation:
        return self._alloc

    # --- Runtime adjustment ---

    def recompute(self, mode: str = "balanced") -> ResourceAllocation:
        """Re-compute allocation for a different workload mode."""
        self._alloc = compute_allocation(mode=mode)
        self._mode = mode
        return self._alloc

    @contextmanager
    def mode_override(self, mode: str):
        """Context manager — temporarily switch mode and restore on exit.

        Usage::

            with rm.mode_override("training"):
                ...  # pool sizes are optimised for training
            # mode restores to previous value

        Applies BLAS env vars and numpy thread limits so the new thread
        counts take effect immediately.
        """
        previous = self._mode
        self.recompute(mode)
        self.apply_blas_env()
        self.apply_compute_limits()
        try:
            yield
        finally:
            self.recompute(previous)
            self.apply_blas_env()
            self.apply_compute_limits()

    def apply_environment(self) -> None:
        """Re-apply env var overrides (call once at startup)."""
        self._alloc = self._alloc.apply_env()

    # --- Apply to current process ---

    def apply_blas_env(self) -> None:
        """Set BLAS / numpy environment variables for the current process.

        Call once at startup before importing numpy to set defaults.
        Subsequent calls override the values (useful after ``mode_override``).
        """
        os.environ["OMP_NUM_THREADS"] = str(self.omp_num_threads)
        os.environ["MKL_NUM_THREADS"] = str(self.mkl_num_threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(self.openblas_num_threads)
        os.environ["NUMEXPR_NUM_THREADS"] = str(self.numexpr_num_threads)

    def apply_compute_limits(self) -> None:
        """Set numpy-level compute thread limits in the current process.

        This is the replacement for ``torch.set_num_threads()`` — we use
        the same mechanisms but for numpy / our own stack.
        """
        try:
            import numpy as np
            np.set_num_threads(self.compute_threads)
        except (ImportError, AttributeError) as exc:
            logger.debug("Failed to set numpy thread count: %s", exc)
        # Apply to numexpr directly if available
        try:
            import numexpr
            numexpr.set_num_threads(self.numexpr_num_threads)
        except (ImportError, AttributeError) as exc:
            logger.debug("Failed to set numexpr thread count: %s", exc)

    def summary(self) -> str:
        return self._alloc.summary()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_global_manager: Optional[ResourceManager] = None
_global_manager_lock = threading.Lock()


def get_resource_manager(mode: str = "balanced") -> ResourceManager:
    """Return the global ``ResourceManager`` singleton.

    Creates it on first call with the given ``mode``.  Subsequent
    calls return the same instance regardless of ``mode``.
    """
    global _global_manager
    if _global_manager is None:
        with _global_manager_lock:
            if _global_manager is None:
                _global_manager = ResourceManager(mode=mode)
    return _global_manager


def reset_resource_manager(mode: str = "balanced") -> ResourceManager:
    """Reset the global singleton (useful for tests)."""
    global _global_manager
    with _global_manager_lock:
        _global_manager = ResourceManager(mode=mode)
    return _global_manager
