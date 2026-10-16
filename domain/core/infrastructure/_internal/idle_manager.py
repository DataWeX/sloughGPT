"""Background monitor that unloads idle models to save memory."""

from __future__ import annotations

import time
from collections.abc import Callable
from threading import Lock, Thread
from typing import Any

from .structured_log import StructuredLogger

logger = StructuredLogger("slo.infrastructure.idle_manager")


def _get_bg_queue() -> Any:
    """Lazily create the global background work queue."""
    from .producer_consumer import ProducerConsumerQueue

    q = ProducerConsumerQueue[Any](
        maxsize=32,
        num_consumers=2,
        handler=lambda item: item() if callable(item) else None,
        name="idle-manager-bg",
    )
    q.start()
    return q


class IdleManager:
    """Background monitor that unloads idle models to save memory.

    Tracks last-request timestamps per model. When a model exceeds
    ``idle_timeout_s`` without requests, calls the unload callback.
    On next request, auto-reloads via the reload callback before processing.

    Usage::

        manager = IdleManager(idle_timeout_s=300)
        manager.register("gpt2", unload_fn, reload_fn)
        # ... on each request:
        manager.touch("gpt2")
        # ... background thread handles unloading
    """

    def __init__(self, idle_timeout_s: float = 300.0, check_interval_s: float = 30.0):
        self._idle_timeout_s = idle_timeout_s
        self._check_interval_s = check_interval_s
        self._models: dict[
            str, dict
        ] = {}  # model_id → {last_touch, unload_fn, reload_fn, unloaded_at}
        self._lock = Lock()
        self._thread: Thread | None = None
        self._running = False
        self._on_unload: Callable[[str], None] | None = None
        self._on_reload: Callable[[str], None] | None = None
        self._logger = logger

    def register(
        self,
        model_id: str,
        unload_fn: Callable[[], None] | None = None,
        reload_fn: Callable[[], None] | None = None,
    ) -> None:
        """Register a model for idle tracking."""
        with self._lock:
            self._models[model_id] = {
                "last_touch": time.time(),
                "unload_fn": unload_fn,
                "reload_fn": reload_fn,
                "unloaded_at": None,
            }
        self._ensure_running()

    def unregister(self, model_id: str) -> None:
        """Stop tracking a model."""
        with self._lock:
            self._models.pop(model_id, None)

    def touch(self, model_id: str) -> bool:
        """Update last request time. Returns True if model was idle and reloaded.

        Reload happens synchronously in the calling thread. For async reload,
        use ``touch_async()`` instead.
        """
        reloaded = False
        with self._lock:
            entry = self._models.get(model_id)
            if entry is None:
                return False
            if entry["unloaded_at"] is not None:
                # Model was idle-unloaded — trigger reload
                reload_fn = entry.get("reload_fn")
                if reload_fn:
                    self._logger.info(
                        "Auto-reloading idle model %s",
                        model_id,
                        extra={"tag": "IDLE"},
                    )
                    try:
                        reload_fn()
                        entry["unloaded_at"] = None
                        reloaded = True
                    except Exception as e:
                        self._logger.error(
                            "Auto-reload failed for %s: %s",
                            model_id,
                            e,
                            extra={"tag": "IDLE"},
                        )
            entry["last_touch"] = time.time()
        return reloaded

    def touch_async(self, model_id: str) -> str:
        """Update last request time. Returns status: 'ok', 'reloading', 'reload_failed'.

        If the model was idle-unloaded, triggers reload in a background thread
        and returns 'reloading' immediately. The caller should return 503.
        """
        with self._lock:
            entry = self._models.get(model_id)
            if entry is None:
                return "ok"
            entry["last_touch"] = time.time()
            if entry["unloaded_at"] is not None:
                reload_fn = entry.get("reload_fn")
                if not reload_fn:
                    return "reload_failed"
                # Check if already reloading
                if entry.get("_reloading"):
                    return "reloading"
                entry["_reloading"] = True

                def _do_reload():
                    try:
                        self._logger.info(
                            "Auto-reloading idle model %s (background)",
                            model_id,
                            extra={"tag": "IDLE"},
                        )
                        reload_fn()
                        with self._lock:
                            entry["unloaded_at"] = None
                            entry["_reloading"] = False
                        self._logger.info(
                            "Auto-reload complete for %s",
                            model_id,
                            extra={"tag": "IDLE"},
                        )
                    except Exception as e:
                        self._logger.error(
                            "Auto-reload failed for %s: %s",
                            model_id,
                            e,
                            extra={"tag": "IDLE"},
                        )
                        with self._lock:
                            entry["_reloading"] = False

                try:
                    _get_bg_queue().put_nowait(_do_reload)
                except Exception:
                    Thread(target=_do_reload, daemon=True, name=f"idle-reload-{model_id}").start()
                return "reloading"
        return "ok"

    def is_reloading(self, model_id: str) -> bool:
        """Check if a model is currently being reloaded after idle timeout."""
        with self._lock:
            entry = self._models.get(model_id)
            return entry is not None and entry.get("_reloading", False)

    def is_idle_unloaded(self, model_id: str) -> bool:
        """Check if a model was unloaded due to idle timeout."""
        with self._lock:
            entry = self._models.get(model_id)
            return entry is not None and entry["unloaded_at"] is not None

    def get_idle_info(self, model_id: str) -> dict | None:
        """Get idle status info for a model."""
        with self._lock:
            entry = self._models.get(model_id)
            if entry is None:
                return None
            last_touch = entry["last_touch"]
            age_s = time.time() - last_touch if last_touch > 0 else 0
            return {
                "last_request_age_s": round(age_s, 1),
                "idle_timeout_s": self._idle_timeout_s,
                "unloaded": entry["unloaded_at"] is not None,
                "remaining_s": round(max(0, self._idle_timeout_s - age_s), 1),
            }

    def _ensure_running(self) -> None:
        """Start the background check thread if not already running."""
        with self._lock:
            if self._running:
                return
            self._running = True
        self._thread = Thread(target=self._check_loop, daemon=True, name="idle-manager")
        self._thread.start()

    def _check_loop(self) -> None:
        """Background loop: check for idle models and unload them."""
        while self._running:
            time.sleep(self._check_interval_s)
            now = time.time()
            with self._lock:
                for model_id, entry in self._models.items():
                    if entry["unloaded_at"] is not None:
                        continue  # already unloaded
                    age = now - entry["last_touch"]
                    if age >= self._idle_timeout_s:
                        unload_fn = entry.get("unload_fn")
                        if unload_fn:
                            self._logger.info(
                                "Idle timeout %.0fs reached for %s — unloading",
                                age,
                                model_id,
                                extra={"tag": "IDLE"},
                            )
                            try:
                                unload_fn()
                                entry["unloaded_at"] = now
                            except Exception as e:
                                self._logger.error(
                                    "Idle unload failed for %s: %s",
                                    model_id,
                                    e,
                                    extra={"tag": "IDLE"},
                                )

    def shutdown(self) -> None:
        """Stop the background check thread."""
        with self._lock:
            self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)

    def reset(self) -> None:
        """Stop background thread and clear all tracked models (for testing)."""
        self.shutdown()
        with self._lock:
            self._models.clear()
            self._running = False


_idle_manager: IdleManager | None = None
_idle_manager_lock = Lock()


def get_idle_manager() -> IdleManager:
    """Get or create the global IdleManager singleton."""
    global _idle_manager
    if _idle_manager is None:
        with _idle_manager_lock:
            if _idle_manager is None:
                _idle_manager = IdleManager()
    return _idle_manager
