"""
ProcessGuard — monitors worker subprocess health, auto-restarts on crash.

Provides a drop-in backend for ModelServer that runs model inference in a
separate OS process, with automatic crash recovery.
"""

import logging
import time
import threading
from typing import Any, Optional, Callable

from .model_worker import ModelWorkerProcess, WorkerHealth

logger = logging.getLogger(__name__)


class ProcessGuard:
    """Guards a ModelWorkerProcess with health monitoring and auto-restart.

    Usage::

        guard = ProcessGuard(
            model_cls_path="transformers.AutoModelForCausalLM",
            model_kwargs={"from_pretrained": "gpt2"},
            worker_id="gpt2",
            max_restarts=3,
        )
        guard.start()
        result = guard.generate("Hello")
    """

    def __init__(
        self,
        model_cls_path: str,
        model_kwargs: dict,
        worker_id: str = "worker",
        generate_timeout: float = 120.0,
        max_restarts: int = 3,
        restart_delay: float = 2.0,
        health_check_interval: float = 5.0,
        extra_sys_paths: Optional[list] = None,
    ):
        self.model_cls_path = model_cls_path
        self.model_kwargs = dict(model_kwargs)
        self.worker_id = worker_id
        self._generate_timeout = generate_timeout
        self._max_restarts = max_restarts
        self._restart_delay = restart_delay
        self._extra_sys_paths = extra_sys_paths or []
        self._health_check_interval = health_check_interval

        self._worker: Optional[ModelWorkerProcess] = None
        self._restart_count = 0
        self._lock = threading.Lock()
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None

        # Callbacks
        self._on_crash: list[Callable[[str], None]] = []
        self._on_restart: list[Callable[[str], None]] = []

    # ── Callbacks ──────────────────────────────────────────────────────

    def on_crash(self, callback: Callable[[str], None]) -> None:
        self._on_crash.append(callback)

    def on_restart(self, callback: Callable[[str], None]) -> None:
        self._on_restart.append(callback)

    # ── Lifecycle ──────────────────────────────────────────────────────

    def _make_worker(self) -> ModelWorkerProcess:
        return ModelWorkerProcess(
            model_cls_path=self.model_cls_path,
            model_kwargs=self.model_kwargs,
            worker_id=self.worker_id,
            generate_timeout=self._generate_timeout,
            extra_sys_paths=self._extra_sys_paths,
        )

    def start(self) -> None:
        """Start the worker and monitoring thread."""
        with self._lock:
            if self._running:
                return
            self._running = True
            self._restart_count = 0

        self._worker = self._make_worker()
        try:
            self._worker.start()
        except Exception:
            self._running = False
            raise

        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name=f"guard-{self.worker_id}",
        )
        self._monitor_thread.start()

        logger.info(
            "ProcessGuard[%s]: started (pid=%d, max_restarts=%d)",
            self.worker_id, self._worker._process.pid if self._worker._process else 0,
            self._max_restarts,
        )

    def stop(self) -> None:
        """Stop the worker and monitoring thread."""
        self._running = False
        with self._lock:
            if self._worker is not None:
                self._worker.stop()
                self._worker = None
        logger.info("ProcessGuard[%s]: stopped", self.worker_id)

    def _monitor_loop(self) -> None:
        """Background loop: poll worker health, restart on crash."""
        while self._running:
            time.sleep(self._health_check_interval)
            with self._lock:
                worker = self._worker

            if worker is None:
                continue

            health = worker.health_check()

            if not health.alive and self._running:
                logger.warning(
                    "ProcessGuard[%s]: worker died (crashed=%s, restarts=%d/%d)",
                    self.worker_id, health.crashed,
                    self._restart_count, self._max_restarts,
                )

                # Fire crash callbacks
                for cb in self._on_crash:
                    try:
                        cb(self.worker_id)
                    except Exception as e:
                        logger.warning("Crash callback error: %s", e)

                if self._restart_count < self._max_restarts:
                    self._restart_count += 1
                    logger.info(
                        "ProcessGuard[%s]: restarting in %.1fs (attempt %d/%d)",
                        self.worker_id, self._restart_delay,
                        self._restart_count, self._max_restarts,
                    )
                    time.sleep(self._restart_delay)

                    try:
                        new_worker = self._make_worker()
                        new_worker.start()
                        with self._lock:
                            self._worker = new_worker

                        for cb in self._on_restart:
                            try:
                                cb(self.worker_id)
                            except Exception as e:
                                logger.warning("Restart callback error: %s", e)

                        logger.info(
                            "ProcessGuard[%s]: restarted (pid=%d)",
                            self.worker_id,
                            new_worker._process.pid if new_worker._process else 0,
                        )
                    except Exception as e:
                        logger.error(
                            "ProcessGuard[%s]: restart failed: %s",
                            self.worker_id, e,
                        )
                else:
                    logger.error(
                        "ProcessGuard[%s]: max restarts (%d) reached, stopping",
                        self.worker_id, self._max_restarts,
                    )
                    self._running = False

    # ── Generate ───────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        **kwargs: Any,
    ) -> dict:
        """Generate text via guarded worker process.

        If the worker is momentarily dead (e.g. mid-restart), retries
        until the monitor restarts it or the budget is exhausted.

        Delegates to the underlying ModelWorkerProcess.generate().
        """
        deadline = time.time() + max(self._restart_delay * 4 + 4.0, 10.0)
        last_error: Optional[str] = None

        while time.time() < deadline:
            with self._lock:
                worker = self._worker

            if worker is None:
                raise RuntimeError(f"ProcessGuard[{self.worker_id}] not started")

            health = worker.health_check()
            if health.alive:
                try:
                    return worker.generate(
                        prompt=prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        repetition_penalty=repetition_penalty,
                        **kwargs,
                    )
                except RuntimeError as e:
                    # Worker died mid-generation; retry if the guard can restart
                    if "not alive" in str(e) or "crashed during" in str(e):
                        last_error = str(e)
                        logger.warning(
                            "ProcessGuard[%s]: worker died during generate, retrying: %s",
                            self.worker_id, e,
                        )
                        time.sleep(1.0)
                        continue
                    raise

            # Worker is not alive — wait for monitor to restart
            time.sleep(0.5)

        raise RuntimeError(
            f"ProcessGuard[{self.worker_id}]: worker unavailable after retries"
            + (f": {last_error}" if last_error else "")
        )

    # ── Health ─────────────────────────────────────────────────────────

    def health(self) -> dict:
        """Return health summary."""
        with self._lock:
            worker = self._worker
            restarts = self._restart_count

        if worker is None:
            return {
                "worker_id": self.worker_id,
                "alive": False,
                "restart_count": restarts,
                "max_restarts": self._max_restarts,
                "running": self._running,
                "exhausted": restarts >= self._max_restarts,
            }

        h = worker.health_check()
        return {
            "worker_id": self.worker_id,
            "alive": h.alive,
            "pid": h.pid,
            "started_at": h.started_at,
            "last_heartbeat": h.last_heartbeat,
            "requests_served": h.requests_served,
            "errors": h.errors,
            "crashed": h.crashed,
            "crash_count": h.crash_count,
            "restart_count": restarts,
            "max_restarts": self._max_restarts,
            "running": self._running,
            "exhausted": restarts >= self._max_restarts,
        }

    # ── Context manager ────────────────────────────────────────────────

    @property
    def alive(self) -> bool:
        with self._lock:
            if self._worker is None:
                return False
            return self._worker.health_check().alive

    def __enter__(self) -> "ProcessGuard":
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()
