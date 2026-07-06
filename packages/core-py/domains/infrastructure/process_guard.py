"""
Process-level crash guard for model inference workers.

Provides ProcessGuard — wraps ModelWorkerProcess with automatic crash
detection, configurable restart policy, health monitoring, and lifecycle
callbacks.
"""

import time
import logging
import threading
import os
from typing import Any, Optional, Callable, Generator

logger = logging.getLogger("man.infrastructure.process_guard")


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
        model_cls_path: str,
        model_kwargs: dict,
        worker_id: str = "guard",
        generate_timeout: float = 120.0,
        max_restarts: int = 3,
        restart_delay: float = 1.0,
        health_check_interval: float = 1.0,
        extra_sys_paths: Optional[list] = None,
        max_concurrent: int = 1,
        memory_limit_mb: Optional[float] = None,
    ):
        self.model_cls_path = model_cls_path
        self.model_kwargs = model_kwargs
        self.worker_id = worker_id
        self.generate_timeout = generate_timeout
        self.max_restarts = max_restarts
        self.restart_delay = restart_delay
        self.health_check_interval = health_check_interval
        self.extra_sys_paths = extra_sys_paths or []
        self.memory_limit_mb = memory_limit_mb

        self._worker: Optional[Any] = None
        self._restart_count = 0
        self._requests_served = 0
        self._crash_callbacks: list[Callable[[str], None]] = []
        self._restart_callbacks: list[Callable[[str], None]] = []
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitor = threading.Event()
        self._semaphore = threading.Semaphore(max_concurrent)

    @property
    def alive(self) -> bool:
        """True when the underlying worker process is alive and healthy."""
        return self._worker is not None and self._worker.alive

    def start(self) -> None:
        """Start the worker and begin health monitoring."""
        self._launch_worker()
        self._stop_monitor.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name=f"guard-mon-{self.worker_id}"
        )
        self._monitor_thread.start()

    def stop(self) -> None:
        """Stop the worker and health monitoring."""
        self._stop_monitor.set()
        if self._worker is not None:
            self._worker.stop()
            self._worker = None
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=3.0)
            self._monitor_thread = None

    def generate(self, prompt: str, **kwargs: Any) -> dict:
        """Generate text via the worker process (thread-safe with semaphore).

        Raises RuntimeError if the worker is not alive.
        """
        if not self.alive:
            raise RuntimeError(f"Guard worker [{self.worker_id}] is not alive")
        with self._semaphore:
            result = self._worker.generate(prompt, **kwargs)
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

    def _memory_mb(self) -> Optional[float]:
        """Return RSS memory usage of the worker process in MB, if available."""
        if self._worker is None or self._worker._process is None:
            return None
        try:
            import psutil
            proc = psutil.Process(self._worker._process.pid)
            return proc.memory_info().rss / (1024 * 1024)
        except (ImportError, psutil.NoSuchProcess, psutil.AccessDenied):
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

    # ── Private ──────────────────────────────────────────────────────

    def _launch_worker(self) -> None:
        from domains.infrastructure.model_worker import ModelWorkerProcess

        if self._worker is not None:
            self._worker.stop()
        self._worker = ModelWorkerProcess(
            model_cls_path=self.model_cls_path,
            model_kwargs=self.model_kwargs,
            worker_id=self.worker_id,
            generate_timeout=self.generate_timeout,
            extra_sys_paths=self.extra_sys_paths,
        )
        self._worker.start()

    def _monitor_loop(self) -> None:
        while not self._stop_monitor.is_set():
            time.sleep(self.health_check_interval)
            if self._stop_monitor.is_set():
                break
            if self._worker is None:
                continue
            if not self._worker.alive and self._restart_count < self.max_restarts:
                logger.info(
                    "ProcessGuard[%s]: worker died — restarting (%d/%d)",
                    self.worker_id,
                    self._restart_count + 1,
                    self.max_restarts,
                )
                for cb in self._crash_callbacks:
                    try:
                        cb(self.worker_id)
                    except Exception:
                        logger.exception("ProcessGuard crash callback failed")
                time.sleep(self.restart_delay)
                self._launch_worker()
                self._restart_count += 1
                for cb in self._restart_callbacks:
                    try:
                        cb(self.worker_id)
                    except Exception:
                        logger.exception("ProcessGuard restart callback failed")
