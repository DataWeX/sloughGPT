"""Embeddable core facade — a host-agnostic ``SloughInstance``.

Phase 1 of card 20261116_095706 (embeddable core): an instance any host —
Node subprocess, Blender add-on, MCP adapter, a test — can start, call
contracts on in-process, and stop, with no FastAPI, no uvicorn, and no repo
checkout assumptions.

Design (the three invariants):

1. **Same state machine, different hook set.** Lifecycle reuses the domain
   ``LifecycleManager`` (profiles, dependency-ordered hooks, per-hook
   timeouts, graceful drain, health gates) — the exact machine the FastAPI
   host drives. The facade registers a domain-pure hook set built from
   ``instance_wiring``; the host keeps its decorated one. One lifecycle
   state machine, not a second bespoke boot sequence.
2. **Sync public API over a private loop.** The core is sync; its two
   async surfaces (``TaskQueue.start/stop``, ``ToolRegistry.execute``)
   live behind a private daemon loop thread driven with
   ``run_coroutine_threadsafe``. Embedders get plain blocking calls —
   no event loop leaks into Blender's main thread or a Node bridge.
3. **Model loading is a declared seam.** ``model_loader`` is injected, never
   copied: the host's autoload machinery (engine subprocess, lazy guard,
   progress reporting) stays in ``startup.py`` untouched, and phase 2 will
   attach it. Without a loader the instance runs the ``quick``-equivalent
   wiring model-free; with one, the ``full`` profile adds a ``model_load``
   hook (sync or async callable, receives the resolved config, its return
   value is the instance's ``model`` handle).

Usage::

    from domain.infrastructure import SloughInstance

    inst = SloughInstance.create(profile="quick")
    inst.start()
    result = inst.invoke("calculator", {"expression": "2 + 2"})
    inst.stop()
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable
from typing import Any

from domain.infrastructure._internal.instance_wiring import (
    shutdown_executor,
    unwire_agent_routines,
    unwire_registry,
    unwire_task_queue,
    wire_agent_routines,
    wire_config,
    wire_model_registry,
    wire_task_queue,
)
from domain.infrastructure._internal.lifecycle import (
    LifecycleManager,
    ShutdownHook,
    StartupHook,
    StartupProfile,
)

logger = logging.getLogger("slo.instance")

# The loader seam: receives the resolved config, returns the model/provider
# handle (or an awaitable of one). Phase 2 attaches the host's full autoload
# machinery behind this signature.
InstanceModelLoader = Callable[[Any], Any]

_VALID_PROFILES = frozenset(p.value for p in StartupProfile)


class SloughInstance:
    """A startable, contract-invocable, stoppable core instance."""

    def __init__(
        self,
        *,
        profile: str = "quick",
        model_loader: InstanceModelLoader | None = None,
        config: Any | None = None,
    ) -> None:
        if profile not in _VALID_PROFILES:
            raise ValueError(f"unknown profile {profile!r} — one of {sorted(_VALID_PROFILES)}")
        self._profile = profile
        self._model_loader = model_loader
        self._explicit_config = config
        self._cfg: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._lifecycle: LifecycleManager | None = None
        self._task_queue: Any = None
        self._model: Any = None
        self._state = "created"

    @classmethod
    def create(
        cls,
        *,
        profile: str = "quick",
        model_loader: InstanceModelLoader | None = None,
        config: Any | None = None,
    ) -> SloughInstance:
        """Create an instance in the ``created`` state (nothing wired yet)."""
        return cls(profile=profile, model_loader=model_loader, config=config)

    # ── Public surface ────────────────────────────────────────────────

    @property
    def state(self) -> str:
        """One of ``created`` / ``running`` / ``failed`` / ``stopped``."""
        return self._state

    @property
    def model(self) -> Any:
        """The handle returned by ``model_loader`` (``None`` without one)."""
        return self._model

    @property
    def task_queue(self) -> Any:
        """The in-process task queue once wired (``None`` before start)."""
        return self._task_queue

    @property
    def config(self) -> Any:
        """The resolved config after start (``None`` before)."""
        return self._cfg

    def start(self, timeout: float = 60.0) -> bool:
        """Wire the core and run startup hooks in dependency order.

        Idempotent: a running instance returns ``True`` immediately.
        Returns ``False`` if any (non-critical) hook failed — the instance
        state becomes ``failed`` but stays stoppable for cleanup.
        """
        if self._state == "running":
            return True
        self._ensure_loop()
        self._lifecycle = self._build_lifecycle()
        try:
            ok = self._submit(
                self._lifecycle.start(timeout=timeout, profile=StartupProfile(self._profile)),
                timeout + 10.0,
            )
        except Exception:
            logger.exception("Instance startup raised")
            self._state = "failed"
            return False
        self._state = "running" if ok else "failed"
        return bool(ok)

    def stop(self, timeout: float = 60.0) -> bool:
        """Drain and run shutdown hooks in reverse dependency order.

        Idempotent: stopping a ``created``/``stopped`` instance is a no-op
        that returns ``True``. Always safe to call after a failed start.
        """
        if self._state in ("created", "stopped"):
            self._state = "stopped"
            return True
        ok = True
        if self._lifecycle is not None:
            try:
                ok = self._submit(self._lifecycle.shutdown(timeout=timeout), timeout + 10.0)
            except Exception:
                logger.exception("Instance shutdown raised")
                ok = False
        self._teardown_loop()
        self._state = "stopped"
        return bool(ok)

    def invoke(
        self,
        name: str,
        args: dict[str, Any],
        *,
        approve: Callable[..., bool] | None = None,
        timeout: float = 30.0,
    ) -> Any:
        """Execute a registered contract in-process (same-process → call).

        Returns the registry's ``ToolResult`` — unknown names and
        approval-gated specs resolve to ``success=False`` results rather
        than raising, matching the registry's own contract.
        """
        if self._state != "running":
            raise RuntimeError(f"instance not invocable in state {self._state!r}")
        from domain.agents import get_tool_registry

        registry = get_tool_registry()
        return self._submit(registry.execute(name, args, approve=approve), timeout)

    def list_contracts(self) -> list[dict[str, Any]]:
        """Descriptor projection of every registered contract (name, params…)."""
        from domain.agents import get_tool_registry

        return get_tool_registry().list_tools()

    # ── Internals ─────────────────────────────────────────────────────

    def _build_lifecycle(self) -> LifecycleManager:
        """Register the domain-pure hook set on a fresh state machine.

        Mirrors the host orchestrator's hook names, dependencies, profiles
        and timeouts — minus every host decoration (progress reporting,
        routers, staged loader, terminal, webhooks).
        """
        from domain.infrastructure._internal.event_bus import EventBus

        mgr = LifecycleManager(event_bus=EventBus(max_history=200))
        quick_plus = frozenset({StartupProfile.FULL, StartupProfile.QUICK})
        full_only = frozenset({StartupProfile.FULL})

        async def _config() -> None:
            wired = wire_config(log=logger)
            self._cfg = self._explicit_config if self._explicit_config is not None else wired

        async def _task_queue() -> None:
            self._task_queue = await wire_task_queue(log=logger)

        async def _agent_routines() -> None:
            wire_agent_routines(log=logger)

        async def _model_registry() -> None:
            wire_model_registry(log=logger)

        async def _model_load() -> None:
            if self._model_loader is None:
                logger.info("Model load skipped (no model_loader — model-free instance)")
                return
            result = self._model_loader(self._cfg)
            if asyncio.iscoroutine(result):
                result = await result
            self._model = result

        mgr.register_startup_hook(
            StartupHook(
                "config",
                _config,
                depends_on=[],
                timeout=5.0,
                critical=False,
                profiles=quick_plus,
            )
        )
        mgr.register_startup_hook(
            StartupHook(
                "task_queue",
                _task_queue,
                depends_on=[],
                timeout=10.0,
                critical=False,
                profiles=quick_plus,
            )
        )
        mgr.register_startup_hook(
            StartupHook(
                "model_registry",
                _model_registry,
                depends_on=["task_queue", "config"],
                timeout=10.0,
                critical=False,
                profiles=quick_plus,
            )
        )
        mgr.register_startup_hook(
            StartupHook(
                "agent_routines",
                _agent_routines,
                depends_on=["task_queue"],
                timeout=10.0,
                critical=False,
                profiles=quick_plus,
            )
        )
        if self._model_loader is not None:
            cfg_timeout = float(
                getattr(self._explicit_config, "startup_model_load_timeout", 60.0) or 60.0
            )
            mgr.register_startup_hook(
                StartupHook(
                    "model_load",
                    _model_load,
                    depends_on=["config"],
                    timeout=cfg_timeout,
                    critical=False,
                    profiles=full_only,
                )
            )
            mgr.register_gate("model_loaded", lambda: self._model is not None)

        # Shutdown: same reverse-dependency shape as the host — the ticker
        # stops before the queue drains (no new enqueues once draining).
        async def _stop_task_queue() -> None:
            await unwire_task_queue(self._task_queue, log=logger)

        async def _stop_agent_routines() -> None:
            unwire_agent_routines(log=logger)

        async def _stop_registry() -> None:
            unwire_registry(log=logger)

        async def _stop_executor() -> None:
            shutdown_executor(log=logger)

        mgr.register_shutdown_hook(
            ShutdownHook("task_queue_shutdown", _stop_task_queue, depends_on=[], timeout=10.0)
        )
        mgr.register_shutdown_hook(
            ShutdownHook(
                "agent_routines_shutdown",
                _stop_agent_routines,
                depends_on=["task_queue_shutdown"],
                timeout=10.0,
            )
        )
        mgr.register_shutdown_hook(
            ShutdownHook("registry_cleanup", _stop_registry, depends_on=[], timeout=5.0)
        )
        mgr.register_shutdown_hook(
            ShutdownHook("executor_shutdown", _stop_executor, depends_on=[], timeout=10.0)
        )
        return mgr

    def _ensure_loop(self) -> None:
        if self._thread is not None:
            return
        loop = asyncio.new_event_loop()

        def _run() -> None:
            asyncio.set_event_loop(loop)
            loop.run_forever()

        self._loop = loop
        self._thread = threading.Thread(target=_run, name="slough-instance-loop", daemon=True)
        self._thread.start()

    def _submit(self, coro: Any, timeout: float) -> Any:
        loop = self._loop
        if loop is None:
            coro.close()
            raise RuntimeError("instance loop is not running")
        return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout)

    def _teardown_loop(self) -> None:
        loop, thread = self._loop, self._thread
        self._loop = self._thread = None
        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join(timeout=5.0)
        if loop is not None and not loop.is_closed():
            loop.close()


def create_instance(
    *,
    profile: str = "quick",
    model_loader: InstanceModelLoader | None = None,
    config: Any | None = None,
) -> SloughInstance:
    """Module-level shorthand for :meth:`SloughInstance.create`."""
    return SloughInstance.create(profile=profile, model_loader=model_loader, config=config)
