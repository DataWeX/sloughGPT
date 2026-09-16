"""PGQ AsyncEngine — asyncio-native bridge for the PGQ Engine.

Wraps the synchronous PGQ Engine so processes can be spawned and
dispatched without blocking the uvicorn event loop.  Sync callables
run in a thread pool; async callables run directly on the event loop.

Usage::

    from pugqeep.async_engine import AsyncEngine

    engine = AsyncEngine("startup")

    # Sync callable — runs in thread pool (event loop stays free)
    engine.spawn_sync(db_pool_init, name="db_pool")

    # Async callable — runs directly on event loop
    engine.spawn_async(health_check, name="health")

    # Dispatch all pending processes without blocking
    await engine.dispatch_async()

    # Or run the full dispatch loop until everything is done
    await engine.run_until_complete()
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from .engine import Engine, Process, ProcessStatus, Tree

logger = logging.getLogger("slo.pugqeep.async_engine")


class AsyncProcess(Process):
    """Process that can hold an async callable."""

    _async_fn: Callable[..., Any] | None = None
    _is_coroutine: bool = False

    def set_async(self, fn: Callable[..., Any]) -> None:
        """Mark this process as async-capable."""
        self._async_fn = fn
        self._is_coroutine = True


class AsyncEngine:
    """asyncio-native wrapper around PGQ Engine.

    Runs all sync work in ``asyncio.to_thread()`` so the event loop
    stays responsive.  Async callables execute directly on the loop.

    This is the bridge between PGQ's thread-pool-based Engine and
    uvicorn's async world.
    """

    def __init__(self, name: str = "async-engine", max_trees: int = 4, thread_pool_size: int = 4):
        self._engine = Engine(name=name, max_trees=max_trees)
        self._name = name
        self._thread_pool_size = thread_pool_size
        self._processes: dict[str, AsyncProcess] = {}
        self._running = False

    # ── Spawning ──

    def spawn_sync(
        self,
        fn: Callable[..., Any],
        *args: Any,
        name: str = "",
        timeout: float | None = None,
        tree: str | None = None,
        depends_on: list[str] | None = None,
        critical: bool = False,
        **kwargs: Any,
    ) -> AsyncProcess:
        """Spawn a sync callable — runs via ``asyncio.to_thread()``.

        The event loop stays free while ``fn`` executes.
        """
        proc = AsyncProcess(fn=fn, args=args, kwargs=kwargs, name=name, timeout=timeout)
        if tree:
            proc._tree_name = tree
        if depends_on:
            proc.depends_on = list(depends_on)
        proc._critical = critical
        self._processes[proc.id] = proc
        self._engine._processes[proc.id] = proc
        self._engine._pending.append(proc)
        self._engine._all_done_event.clear()
        self._engine._metrics.record_spawn()
        return proc

    def spawn_async(
        self,
        fn: Callable[..., Any],
        *args: Any,
        name: str = "",
        timeout: float | None = None,
        tree: str | None = None,
        depends_on: list[str] | None = None,
        critical: bool = False,
        **kwargs: Any,
    ) -> AsyncProcess:
        """Spawn an async callable — runs directly on the event loop."""
        proc = AsyncProcess(fn=fn, args=args, kwargs=kwargs, name=name, timeout=timeout)
        proc.set_async(fn)
        if tree:
            proc._tree_name = tree
        if depends_on:
            proc.depends_on = list(depends_on)
        proc._critical = critical
        self._processes[proc.id] = proc
        self._engine._processes[proc.id] = proc
        self._engine._pending.append(proc)
        self._engine._all_done_event.clear()
        self._engine._metrics.record_spawn()
        return proc

    def spawn_threaded(
        self,
        fn: Callable[..., Any],
        *args: Any,
        name: str = "",
        timeout: float | None = None,
        tree: str | None = None,
        depends_on: list[str] | None = None,
        critical: bool = False,
        **kwargs: Any,
    ) -> AsyncProcess:
        """Spawn a sync callable on a dedicated daemon thread.

        Unlike ``spawn_sync`` which uses ``asyncio.to_thread()`` (bounded
        by the default executor pool), this creates a dedicated thread.
        Use for long-running or GIL-heavy work that must not compete
        with other ``to_thread`` callers.
        """
        proc = AsyncProcess(fn=fn, args=args, kwargs=kwargs, name=name, timeout=timeout)
        if tree:
            proc._tree_name = tree
        if depends_on:
            proc.depends_on = list(depends_on)
        proc._critical = critical
        self._processes[proc.id] = proc
        self._engine._processes[proc.id] = proc
        self._engine._pending.append(proc)
        self._engine._all_done_event.clear()
        self._engine._metrics.record_spawn()
        # Mark as threaded so dispatch knows to run on daemon thread
        proc._threaded = True
        return proc

    # ── Dispatch ──

    def _deps_met(self, proc: AsyncProcess) -> bool:
        if not proc.depends_on:
            return True
        for dep_id in proc.depends_on:
            dep = self._processes.get(dep_id) or self._engine._processes.get(dep_id)
            if dep is None or dep.status != ProcessStatus.COMPLETED:
                return False
        return True

    async def dispatch_async(self) -> int:
        """Dispatch all pending processes whose dependencies are met.

        Sync processes run via ``asyncio.to_thread()`` (event loop stays free).
        Async processes run directly on the event loop.
        Returns the number of processes dispatched.
        """
        dispatchable: list[AsyncProcess] = []
        held: list[AsyncProcess] = []

        for proc in list(self._engine._pending):
            if proc.depends_on and not self._deps_met(proc):
                held.append(proc)
            else:
                dispatchable.append(proc)

        if not dispatchable:
            return 0

        tasks: list[asyncio.Task] = []
        dispatched_ids: list[str] = []

        for proc in dispatchable:
            self._engine._pending.remove(proc)
            dispatched_ids.append(proc.id)

            if getattr(proc, "_is_coroutine", False) and proc._async_fn is not None:
                # Async callable — run directly on event loop
                task = asyncio.create_task(
                    self._run_async_process(proc),
                    name=f"async-{proc.name or proc.id}",
                )
            elif getattr(proc, "_threaded", False):
                # Dedicated daemon thread
                task = asyncio.create_task(
                    self._run_threaded_process(proc),
                    name=f"thread-{proc.name or proc.id}",
                )
            else:
                # Sync callable — offload to thread pool
                task = asyncio.create_task(
                    self._run_sync_process(proc),
                    name=f"sync-{proc.name or proc.id}",
                )
            tasks.append(task)
            self._engine._metrics.record_dispatch(1)

        logger.info(
            "AsyncEngine[%s]: dispatched %d processes (%d async, %d sync)",
            self._name,
            len(dispatchable),
            sum(1 for p in dispatchable if getattr(p, "_is_coroutine", False)),
            sum(1 for p in dispatchable if not getattr(p, "_is_coroutine", False)),
        )

        # Wait for all dispatched processes (non-blocking — other events can fire)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        return len(dispatchable)

    async def _run_sync_process(self, proc: AsyncProcess) -> None:
        """Run a sync process via asyncio.to_thread (event loop stays free)."""
        proc.running()
        try:
            timeout = proc.timeout

            if timeout and timeout > 0:
                result = await asyncio.wait_for(
                    asyncio.to_thread(proc.fn, *proc.args, **proc.kwargs),
                    timeout=timeout,
                )
            else:
                result = await asyncio.to_thread(proc.fn, *proc.args, **proc.kwargs)

            proc.complete(result)
            logger.debug("AsyncEngine[%s]: sync process '%s' completed", self._name, proc.name)
        except TimeoutError:
            proc.fail(f"timed out after {proc.timeout}s")
            logger.warning("AsyncEngine[%s]: sync process '%s' timed out", self._name, proc.name)
        except Exception as e:
            proc.fail(str(e))
            logger.warning(
                "AsyncEngine[%s]: sync process '%s' failed: %s", self._name, proc.name, e
            )

    async def _run_async_process(self, proc: AsyncProcess) -> None:
        """Run an async process directly on the event loop."""
        proc.running()
        try:
            timeout = proc.timeout

            if proc.args or proc.kwargs:
                coro = proc._async_fn(*proc.args, **proc.kwargs)
            else:
                coro = proc._async_fn()

            if timeout and timeout > 0:
                result = await asyncio.wait_for(coro, timeout=timeout)
            else:
                result = await coro

            proc.complete(result)
            logger.debug("AsyncEngine[%s]: async process '%s' completed", self._name, proc.name)
        except TimeoutError:
            proc.fail(f"timed out after {proc.timeout}s")
            logger.warning("AsyncEngine[%s]: async process '%s' timed out", self._name, proc.name)
        except Exception as e:
            proc.fail(str(e))
            logger.warning(
                "AsyncEngine[%s]: async process '%s' failed: %s", self._name, proc.name, e
            )

    async def _run_threaded_process(self, proc: AsyncProcess) -> None:
        """Run a process on a dedicated daemon thread (for GIL-heavy work)."""
        proc.running()
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(None, proc.fn, *proc.args, **proc.kwargs)
            proc.complete(result)
            logger.debug("AsyncEngine[%s]: threaded process '%s' completed", self._name, proc.name)
        except Exception as e:
            proc.fail(str(e))
            logger.warning(
                "AsyncEngine[%s]: threaded process '%s' failed: %s", self._name, proc.name, e
            )

    # ── High-level dispatch ──

    async def run_until_complete(self, poll_interval: float = 0.1) -> None:
        """Dispatch all processes and wait until everything completes.

        Non-blocking — the event loop stays free.  Other coroutines
        can execute between poll cycles.
        """
        self._running = True
        while self._running:
            # Dispatch any ready processes
            await self.dispatch_async()

            # Check if everything is done
            all_procs = list(self._engine._processes.values())
            if all(p.is_done for p in all_procs) and not self._engine._pending:
                break

            await asyncio.sleep(poll_interval)

        self._running = False
        logger.info("AsyncEngine[%s]: all processes complete", self._name)

    # ── Delegation to inner Engine ──

    def tree(self, name: str, max_stems: int = 8, pool_workers: int = 4) -> Tree:
        return self._engine.tree(name, max_stems=max_stems, pool_workers=pool_workers)

    def route(self, process_name: str, tree_name: str) -> None:
        self._engine.route(process_name, tree_name)

    def get_process(self, proc_id: str) -> AsyncProcess | None:
        return self._processes.get(proc_id) or self._engine.get_process(proc_id)

    def list_processes(self, status: ProcessStatus | None = None) -> list[AsyncProcess]:
        procs = list(self._processes.values())
        if status:
            procs = [p for p in procs if p.status == status]
        return procs

    def dependency_graph(self) -> dict:
        return self._engine.dependency_graph()

    def metrics_snapshot(self) -> dict:
        return self._engine._metrics.snapshot()

    def health(self) -> dict:
        return self._engine.health()

    def stop(self) -> None:
        self._running = False
        self._engine.stop()

    @property
    def engine(self) -> Engine:
        """Access the inner synchronous Engine."""
        return self._engine
