"""
Process-level crash guard for model inference workers.

Provides ProcessGuard — wraps ModelWorkerProcess with automatic crash
detection, configurable restart policy, health monitoring, and lifecycle
callbacks.

Two construction modes:

  SloNet (preferred — pure NumPy)::

      guard = ProcessGuard(slnc_path="models/gpt2.slnc", model_id="gpt2")
      guard.start()
      result = guard.generate("Hello")

  HF/Legacy (requires PyTorch)::

      guard = ProcessGuard(
          model_cls_path="transformers.AutoModelForCausalLM",
          model_kwargs={"pretrained_model_name_or_path": "gpt2"},
      )
"""

import time
import logging
import threading
import os
from typing import Any, Optional, Callable, Generator

from domains.infrastructure.model_worker import WorkerStreamStalledError


def resolve_memory_limit_mb(
    slnc_path: Optional[str], configured: Optional[float] = None
) -> Optional[float]:
    """Resolve a guard worker memory limit from an explicit value or the model size.

    When ``configured`` is set and > 0 it wins (operator override). Otherwise the
    limit is derived from the .slnc file size: the worker holds the weights in
    memory at several times the on-disk quantized size (numpy float32 arrays plus
    tokenizer and activation workspace), so use ``max(8192, slnc_mb * 8)``. The
    floor avoids an alarmingly low limit for tiny test models.

    Args:
        slnc_path: Path to the .slnc weight file used to size the limit.
        configured: Explicit operator-provided limit (MB), 0/None = auto.

    Returns:
        Memory limit in MB, or None when no explicit limit and the file cannot
        be sized.
    """
    if configured and configured > 0:
        return float(configured)
    if not slnc_path:
        return None
    try:
        size_mb = os.path.getsize(slnc_path) / (1024 * 1024)
    except OSError:
        return None
    return max(8192.0, size_mb * 8.0)

logger = logging.getLogger("slo.infrastructure.process_guard")


class ProcessGuard:
    """Wraps a ModelWorkerProcess with auto-restart and health monitoring.

    Usage::

        guard = ProcessGuard(
            model_cls_path="fake_model.FakeTestModel",
            model_kwargs={"reply": "hello"},
            worker_id="my-worker",
            max_restarts=3,
            restart_delay=1.0,
        )
        guard.start()
        result = guard.generate("Hello")
        guard.stop()
    """

    def __init__(
        self,
        worker_id: str = "guard",
        generate_timeout: float = 120.0,
        stall_timeout: float = 120.0,
        max_restarts: int = 3,
        restart_delay: float = 1.0,
        health_check_interval: float = 1.0,
        extra_sys_paths: Optional[list] = None,
        max_concurrent: Optional[int] = None,
        memory_limit_mb: Optional[float] = 4096.0,
        # SloNet mode (preferred)
        slnc_path: Optional[str] = None,
        model_id: Optional[str] = None,
        quantize: bool = False,
        quant_bits: int = 8,
        quant_mode: str = "symmetric",
        quant_clip: float = 0.999,
        # HF/Legacy mode
        model_cls_path: Optional[str] = None,
        model_kwargs: Optional[dict] = None,
    ):
        self.worker_id = worker_id
        self.generate_timeout = generate_timeout
        self.stall_timeout = stall_timeout
        self.max_restarts = max_restarts
        self.restart_delay = restart_delay
        self.health_check_interval = health_check_interval
        self.extra_sys_paths = extra_sys_paths or []
        self.memory_limit_mb = memory_limit_mb

        # SloNet params
        self._slnc_path = slnc_path
        self._model_id = model_id or "default"
        self._quantize = quantize
        self._quant_bits = quant_bits
        self._quant_mode = quant_mode
        self._quant_clip = quant_clip

        # HF params
        self.model_cls_path = model_cls_path
        self.model_kwargs = model_kwargs or {}

        self._worker: Optional[Any] = None
        self._restart_count = 0
        self._requests_served = 0
        self._crash_callbacks: list[Callable[[str], None]] = []
        self._restart_callbacks: list[Callable[[str], None]] = []
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitor = threading.Event()
        self._restart_lock = threading.Lock()
        if max_concurrent is None:
            from domains.infrastructure.resource_manager import get_resource_manager
            max_concurrent = get_resource_manager().process_guard_concurrent
        self._semaphore = threading.Semaphore(max_concurrent)

    @property
    def alive(self) -> bool:
        """True when the underlying worker process is alive and healthy."""
        return self._worker is not None and self._worker.alive

    def start(self) -> None:
        """Start the worker and begin health monitoring."""
        logger.info("process_guard: starting", extra={
            "worker_id": self.worker_id, "model_id": self._model_id,
        })
        self._launch_worker()
        self._stop_monitor.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name=f"guard-mon-{self.worker_id}"
        )
        self._monitor_thread.start()

    def stop(self) -> None:
        """Stop the worker and health monitoring."""
        logger.info("process_guard: stopping", extra={
            "worker_id": self.worker_id, "model_id": self._model_id,
        })
        self._stop_monitor.set()
        if self._worker is not None:
            self._worker.stop()
            self._worker = None
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=3.0)
            self._monitor_thread = None

    def generate(self, prompt: str, **kwargs: Any) -> dict:
        """Generate text via the worker process (thread-safe with semaphore).

        If the worker times out or stalls (wedge), the worker is restarted
        before the error propagates, so subsequent requests succeed.

        Raises RuntimeError if the worker is not alive.
        """
        if not self.alive:
            raise RuntimeError(f"Guard worker [{self.worker_id}] is not alive")
        with self._semaphore:
            try:
                result = self._worker.generate(prompt, **kwargs)
            except (TimeoutError, WorkerStreamStalledError) as e:
                self._recover_from_stall()
                raise
        self._requests_served += 1
        return result

    def generate_stream(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> Generator[str, None, dict]:
        """Stream generate via the worker process (thread-safe with semaphore).

        Yields:
            str: Each decoded token.

        Returns:
            dict: Final result with ``tokens_generated`` and ``elapsed_ms``.
        """
        if not self.alive:
            raise RuntimeError(f"Guard worker [{self.worker_id}] is not alive")
        with self._semaphore:
            gen = self._worker.generate_stream(prompt, **kwargs)
            try:
                token = next(gen)
                while True:
                    yield token
                    token = next(gen)
            except StopIteration as e:
                return e.value if hasattr(e, "value") else {}
            except (TimeoutError, WorkerStreamStalledError) as e:
                self._recover_from_stall()
                raise

    def _recover_from_stall(self) -> None:
        """Restart a wedged worker (stalled queue writes / no messages).

        Uses the same restart budget as crash recovery. Callbacks are fired
        as for a crash. Raises RuntimeError when the budget is exhausted.
        """
        if self._restart_count >= self.max_restarts:
            logger.error(
                "ProcessGuard[%s]: worker stalled and restart budget exhausted",
                self.worker_id, extra={"tag": "INFRA"},
            )
            raise RuntimeError(
                f"ProcessGuard[{self.worker_id}]: worker restart budget exhausted "
                f"({self.max_restarts} restarts)"
            )
        with self._restart_lock:
            self._restart_worker_locked("stalled")

    def _memory_mb(self) -> Optional[float]:
        """Return RSS memory usage of the worker process in MB, if available."""
        if self._worker is None or self._worker._process is None:
            return None
        try:
            import psutil
        except ImportError:
            return None
        try:
            proc = psutil.Process(self._worker._process.pid)
            return proc.memory_info().rss / (1024 * 1024)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None

    def health(self) -> dict:
        """Return a health snapshot dict with memory usage."""
        worker_requests = 0
        if self._worker is not None and hasattr(self._worker, "_health"):
            worker_requests = getattr(self._worker._health, "requests_served", 0)
        mem = self._memory_mb()
        return {
            "alive": self.alive,
            "worker_id": self.worker_id,
            "requests_served": self._requests_served + worker_requests,
            "restart_count": self._restart_count,
            "max_restarts": self.max_restarts,
            "exhausted": self._restart_count >= self.max_restarts,
            "memory_mb": round(mem, 1) if mem is not None else None,
            "memory_limit_mb": self.memory_limit_mb,
            "over_limit": mem is not None and self.memory_limit_mb is not None
                          and mem > self.memory_limit_mb,
        }

    def on_crash(self, cb: Callable[[str], None]) -> None:
        """Register a callback invoked on worker crash (receives worker_id)."""
        self._crash_callbacks.append(cb)

    def on_restart(self, cb: Callable[[str], None]) -> None:
        """Register a callback invoked after worker restart (receives worker_id)."""
        self._restart_callbacks.append(cb)

    def load_adapter(self, adapter_path: str, merge: bool = False, timeout: float = 120.0) -> dict:
        """Load a LoRA adapter into the worker's model.

        Delegates to ModelWorkerProcess.load_adapter() which sends the command
        to the subprocess.
        """
        if self._worker is None or not self._worker.alive:
            raise RuntimeError("Worker is not alive — cannot load adapter.")
        return self._worker.load_adapter(adapter_path, merge=merge, timeout=timeout)

    def unload_adapter(self, timeout: float = 60.0) -> dict:
        """Unload the LoRA adapter from the worker's model.

        Delegates to ModelWorkerProcess.unload_adapter() which sends the command
        to the subprocess.
        """
        if self._worker is None or not self._worker.alive:
            raise RuntimeError("Worker is not alive — cannot unload adapter.")
        return self._worker.unload_adapter(timeout=timeout)

    # ── Private ──────────────────────────────────────────────────────

    def _launch_worker(self) -> None:
        from domains.infrastructure.model_worker import ModelWorkerProcess

        if self._worker is not None:
            self._worker.stop()

        if self._slnc_path is not None:
            self._worker = ModelWorkerProcess(
                slnc_path=self._slnc_path,
                model_id=self._model_id,
                worker_id=self.worker_id,
                generate_timeout=self.generate_timeout,
                stall_timeout=self.stall_timeout,
                extra_sys_paths=self.extra_sys_paths,
                quantize=self._quantize,
                quant_bits=self._quant_bits,
                quant_mode=self._quant_mode,
                quant_clip=self._quant_clip,
            )
        else:
            self._worker = ModelWorkerProcess(
                model_cls_path=self.model_cls_path,
                model_kwargs=self.model_kwargs,
                worker_id=self.worker_id,
                generate_timeout=self.generate_timeout,
                stall_timeout=self.stall_timeout,
                extra_sys_paths=self.extra_sys_paths,
            )
        self._worker.start()
        logger.info("process_guard: worker launched", extra={
            "worker_id": self.worker_id, "model_id": self._model_id,
            "slnc_path": str(self._slnc_path) if self._slnc_path else None,
        })

    def _restart_worker(self, reason: str, fire_callbacks: bool = False) -> None:
        """Stop, relaunch, and count a worker restart under a lock.

        Args:
            reason: Short description of why the worker is being replaced
                (logged; used in restart/crash callbacks).
            fire_callbacks: When True, invokes crash callbacks before the
                restart and restart callbacks after (crash-recovery path).
        """
        with self._restart_lock:
            self._restart_worker_locked(reason, fire_callbacks=fire_callbacks)

    def _restart_worker_locked(self, reason: str, fire_callbacks: bool = False) -> None:
        """Unlocked core of ``_restart_worker`` (caller holds ``_restart_lock``)."""
        if fire_callbacks:
            for cb in self._crash_callbacks:
                try:
                    cb(self.worker_id)
                except Exception:
                    logger.exception("ProcessGuard crash callback failed", extra={"tag": "INFRA"})
        logger.info(
            "ProcessGuard[%s]: restarting worker (%s) (%d/%d)",
            self.worker_id,
            reason,
            self._restart_count + 1,
            self.max_restarts,
            extra={"tag": "INFRA"},
        )
        time.sleep(self.restart_delay)
        self._launch_worker()
        self._restart_count += 1
        if fire_callbacks:
            for cb in self._restart_callbacks:
                try:
                    cb(self.worker_id)
                except Exception:
                    logger.exception("ProcessGuard restart callback failed", extra={"tag": "INFRA"})

    def _monitor_loop(self) -> None:
        while not self._stop_monitor.is_set():
            time.sleep(self.health_check_interval)
            if self._stop_monitor.is_set():
                break
            # Snapshot worker reference under the restart lock to avoid racing
            # with _restart_worker_locked() which swaps self._worker via
            # _launch_worker().
            with self._restart_lock:
                worker = self._worker
                restarts_left = self._restart_count < self.max_restarts
            if worker is None:
                continue
            if not worker.alive and restarts_left:
                self._restart_worker("died", fire_callbacks=True)
            elif not worker.alive and not restarts_left:
                logger.error("process_guard: worker dead, restart budget exhausted", extra={
                    "worker_id": self.worker_id, "model_id": self._model_id,
                    "restarts": self._restart_count, "max_restarts": self.max_restarts,
                })


def create_model_guard(
    model_id: str,
    device: str = "cpu",
    worker_id: Optional[str] = None,
    max_restarts: int = 3,
    restart_delay: float = 2.0,
    memory_limit_mb: Optional[float] = None,
    generate_timeout: float = 120.0,
    stall_timeout: float = 120.0,
    max_concurrent: Optional[int] = None,
) -> ProcessGuard:
    """Create a ProcessGuard for an HF model (legacy path).

    Convenience factory that configures the guard to load the model
    in a subprocess via HF ``model.generate()``.

    Args:
        model_id: HuggingFace model ID (e.g. "gpt2").
        device: Device string ("cpu", "mps", "cuda", "auto").
        worker_id: Optional worker name. Defaults to ``f"guard-{model_id}"``.
        max_restarts: Max worker restarts before giving up.
        restart_delay: Seconds to wait between restarts.
        memory_limit_mb: Optional RSS memory limit (MB).
        generate_timeout: Max seconds per generate() call.
        stall_timeout: Max seconds without a worker message before restart.
        max_concurrent: Max concurrent requests.

    Returns:
        Started ProcessGuard instance.
    """
    guard = ProcessGuard(
        model_cls_path="domains.infrastructure.hf_model_worker.hf_model_loader",
        model_kwargs={"model_id": model_id, "device": device},
        worker_id=worker_id or f"guard-{model_id.split('/')[-1]}",
        max_restarts=max_restarts,
        restart_delay=restart_delay,
        memory_limit_mb=memory_limit_mb,
        generate_timeout=generate_timeout,
        stall_timeout=stall_timeout,
        max_concurrent=max_concurrent,
    )
    guard.start()
    return guard


def create_slo_guard(
    slnc_path: str,
    model_id: str = "default",
    worker_id: Optional[str] = None,
    max_restarts: int = 3,
    restart_delay: float = 2.0,
    memory_limit_mb: Optional[float] = None,
    generate_timeout: float = 120.0,
    stall_timeout: float = 120.0,
    max_concurrent: Optional[int] = None,
    quantize: bool = False,
    quant_bits: int = 8,
    quant_mode: str = "symmetric",
    quant_clip: float = 0.999,
) -> ProcessGuard:
    """Create a ProcessGuard for a SloNet model (pure NumPy).

    Loads the model in a subprocess via ``SloNetChatProvider.from_slnc()``
    and streams via ``generate_numpy_stream()``.

    Args:
        slnc_path: Path to the .slnc weight file.
        model_id: Model identifier (e.g. "gpt2").
        worker_id: Optional worker name.
        max_restarts: Max worker restarts before giving up.
        restart_delay: Seconds to wait between restarts.
        memory_limit_mb: Optional RSS memory limit (MB).
        generate_timeout: Max seconds per generate() call.
        stall_timeout: Max seconds without a worker message before restart.
        max_concurrent: Max concurrent requests.
        quantize: Apply quantization after loading.
        quant_bits: Bits for quantization (8 or 4).
        quant_mode: "symmetric" or "asymmetric".
        quant_clip: Outlier clipping percentile.

    Returns:
        Started ProcessGuard instance.
    """
    guard = ProcessGuard(
        slnc_path=slnc_path,
        model_id=model_id,
        worker_id=worker_id or f"slo-guard-{model_id.split('/')[-1]}",
        max_restarts=max_restarts,
        restart_delay=restart_delay,
        memory_limit_mb=memory_limit_mb,
        generate_timeout=generate_timeout,
        stall_timeout=stall_timeout,
        max_concurrent=max_concurrent,
        quantize=quantize,
        quant_bits=quant_bits,
        quant_mode=quant_mode,
        quant_clip=quant_clip,
    )
    guard.start()
    return guard
