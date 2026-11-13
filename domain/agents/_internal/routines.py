"""Agent phase 2 — background routine runtime (the factory layer).

A ``Routine`` is a persisted schedule: *run this prompt every N seconds* —
either on one agent (the factory leaf) or fanned out over the
``MultiAgentOrchestrator`` hierarchy (``mode="orchestrate"``).
``AgentRoutineRunner`` is the factory that turns due routines into
``TaskQueue`` work items, so many stateful agents can be produced and
managed on the pool without a request holding a connection open.

The registered ``agent_run`` handler replays each task through
``AgentSystem.execute_stream`` (leaf) or
``MultiAgentOrchestrator.async_execute`` (hierarchy) — the **same** bounded
loops the sync and HTTP paths use. There is deliberately no second loop:
one loop per shape, two projections (request-scoped SSE and background run).

Status leaves the runner through ``record_event``/``events_since`` and is
projected to clients by the ``agents.routines_stream`` SSE endpoint.

Lifecycle: ``start()`` spawns only a ticker; ``stop()`` cancels only that
ticker. In-flight runs are owned by the task queue, which drains them on
its own shutdown — the runner never reaches into a running task.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from domain.infrastructure._internal.repository import FileRepository, JsonSerializer

logger = logging.getLogger("slo.agents.routines")

# Mirrors ``AGENTS_DIR`` (data/agents) exactly — same four ``..`` hops, so
# agents and their routines share one ``data/`` root outside the checkout.
# One JSON file per routine.
ROUTINES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "data", "routines")

DEFAULT_TICK_INTERVAL = 5.0
MAX_RECORDED_EVENTS = 500


@dataclass
class Routine:
    """One scheduled background run.

    ``mode`` picks the shape: ``"agent"`` replays one agent (the factory
    leaf); ``"orchestrate"`` fans ``prompt`` out over the
    ``MultiAgentOrchestrator`` hierarchy (members filters the agent set,
    empty = every registered agent).
    """

    id: str
    agent_id: str = ""  # leaf identity; "" for orchestrate (hierarchy) routines
    prompt: str = ""
    schedule: float = 300.0  # seconds between runs
    enabled: bool = True
    created_at: float = field(default_factory=time.time)
    last_run_at: float = 0.0  # 0 = never run → due on first tick
    metadata: dict[str, Any] = field(default_factory=dict)
    mode: str = "agent"
    members: list[str] = field(default_factory=list)
    context: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Routine:
        # Tolerate records written by an older/newer schema.
        known = {key: data[key] for key in cls.__dataclass_fields__ if key in data}
        return cls(**known)


class RoutineStore:
    """File-backed routine persistence (``data/routines``)."""

    def __init__(self, directory: str | os.PathLike[str] | None = None) -> None:
        self._repo = FileRepository[dict](
            directory=str(directory or ROUTINES_DIR),
            serializer=JsonSerializer(dict),
            key_suffix=".json",
        )

    def list(self) -> list[Routine]:
        routines: list[Routine] = []
        for data in self._repo.list():
            if not isinstance(data, dict):
                continue
            try:
                routines.append(Routine.from_dict(data))
            except Exception:
                logger.warning("Skipping malformed routine record", extra={"tag": "AGENT"})
        return routines

    def get(self, routine_id: str) -> Routine | None:
        data = self._repo.get(routine_id)
        if not isinstance(data, dict):
            return None
        return Routine.from_dict(data)

    def save(self, routine: Routine) -> bool:
        return bool(self._repo.save(routine.id, routine.to_dict()))

    def delete(self, routine_id: str) -> bool:
        return bool(self._repo.delete(routine_id))


class AgentRoutineRunner:
    """Schedules routines onto the task queue and replays them.

    Construct with an explicit ``system``/``store``/``queue`` for tests;
    production code goes through :func:`get_routine_runner`.
    """

    def __init__(
        self,
        system: Any = None,
        store: RoutineStore | None = None,
        queue: Any = None,
        tick_interval: float = DEFAULT_TICK_INTERVAL,
    ) -> None:
        self._system = system
        self._store = store if store is not None else RoutineStore()
        self._queue = queue
        self._tick_interval = tick_interval
        self._task: asyncio.Task | None = None
        self._running = False
        self._events: list[dict[str, Any]] = []
        self._seq = 0
        self._lock = threading.Lock()

    # ── Accessors ────────────────────────────────────────────────────────

    @property
    def running(self) -> bool:
        """True while the ticker task exists."""
        return self._running

    @property
    def store(self) -> RoutineStore:
        return self._store

    @property
    def tick_interval(self) -> float:
        return self._tick_interval

    def _resolve_system(self) -> Any:
        if self._system is None:
            from domain.agents._internal.system import get_agent_system

            self._system = get_agent_system()
        return self._system

    def _resolve_queue(self) -> Any:
        if self._queue is None:
            from domain.infrastructure._internal.task_queue import get_task_queue

            self._queue = get_task_queue()
        return self._queue

    def register_handlers(self) -> None:
        """Attach this runner's handler to the shared task queue."""
        self._resolve_queue().register_handler("agent_run", self._handle_run)

    # ── Scheduling ───────────────────────────────────────────────────────

    async def tick(self) -> list[str]:
        """Enqueue every enabled routine that is due. Returns task ids.

        ``last_run_at`` advances at *enqueue* time, not completion: a run
        slower than its own schedule must not pile up behind itself.
        """
        now = time.time()
        enqueued: list[str] = []
        for routine in self._store.list():
            if not routine.enabled:
                continue
            if routine.last_run_at and (now - routine.last_run_at) < routine.schedule:
                continue
            # One broken routine must not block the others: log and keep
            # going. ``last_run_at`` stays put on failure so the routine
            # retries next tick (queue down, transient store error, ...).
            try:
                task_id = await self.enqueue_routine(routine)
                routine.last_run_at = now
                self._store.save(routine)
            except Exception as e:
                logger.warning(
                    "Failed to enqueue routine %s: %s", routine.id, e, extra={"tag": "AGENT"}
                )
                continue
            enqueued.append(task_id)
        return enqueued

    async def enqueue_routine(self, routine: Routine) -> str:
        """Put one routine on the queue as an ``agent_run`` task."""
        from domain.infrastructure._internal.task_queue import Priority, Task

        task = Task(
            name=f"routine:{routine.id}",
            task_type="agent_run",
            payload={
                "routine_id": routine.id,
                "agent_id": routine.agent_id,
                "prompt": routine.prompt,
                "mode": routine.mode,
                "members": list(routine.members),
                "context": routine.context,
            },
            priority=Priority.NORMAL,
        )
        return await self._resolve_queue().enqueue(task)

    async def run_now(self, routine_id: str) -> str | None:
        """Enqueue a routine immediately, bypassing its schedule."""
        routine = self._store.get(routine_id)
        if routine is None:
            return None
        task_id = await self.enqueue_routine(routine)
        routine.last_run_at = time.time()
        self._store.save(routine)
        return task_id

    # ── Execution (the ONE loop) ─────────────────────────────────────────

    async def _handle_run(self, task: Any) -> dict[str, Any]:
        """Consume an ``agent_run`` task — one agent, or the hierarchy fan-out.

        Both shapes replay their *existing* loop: ``AgentSystem.execute_stream``
        for a leaf, ``MultiAgentOrchestrator.async_execute`` for orchestrate.
        There is deliberately no third loop.
        """
        payload = getattr(task, "payload", None) or {}
        routine_id = str(payload.get("routine_id", ""))
        if str(payload.get("mode", "agent")) == "orchestrate":
            return await self._handle_orchestrate(payload)
        agent_id = str(payload.get("agent_id", ""))
        prompt = str(payload.get("prompt", ""))

        system = self._resolve_system()
        count = 0
        try:
            async for event in system.execute_stream(
                agent_id,
                prompt,
                session_id=f"routine:{routine_id}",
                user_id="routine",
            ):
                self.record_event({**event, "routine_id": routine_id, "agent_id": agent_id})
                count += 1
        except Exception as e:
            # A raised failure (execute_stream yields "error" events itself,
            # but a crash mid-iteration raises) still reaches the status
            # stream — then re-raise so the queue keeps its failure record.
            self.record_event(
                {"event": "error", "error": str(e), "routine_id": routine_id, "agent_id": agent_id}
            )
            raise
        return {"routine_id": routine_id, "events": count}

    async def _handle_orchestrate(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Fan a goal out over ``MultiAgentOrchestrator`` (hierarchy node).

        ``async_execute`` is the same plan → levels → compose engine the
        request-scoped ``/agents/orchestrate`` surface runs — projected here
        as coarse events (start → final → end) instead of SSE frames, with
        the same ``AgentRunStore`` durability (start/set_tasks/complete/fail).
        """
        from domain.agents import MultiAgentOrchestrator, get_agent_run_store

        routine_id = str(payload.get("routine_id", ""))
        goal = str(payload.get("prompt", ""))
        context = str(payload.get("context", ""))
        members = [str(m) for m in payload.get("members") or []]

        orch = MultiAgentOrchestrator()
        if members:
            filtered = {k: v for k, v in orch.agents.items() if k in members or v.name in members}
            if filtered:
                orch = MultiAgentOrchestrator(agents=filtered)

        store = get_agent_run_store()
        run_id = store.start(goal, context)
        count = 0
        try:
            self.record_event(
                {
                    "event": "run_start",
                    "routine_id": routine_id,
                    "mode": "orchestrate",
                    "goal": goal,
                    "run_id": run_id,
                }
            )
            count += 1
            result = await orch.async_execute(goal, context)
            tasks = result.get("tasks") or []
            store.set_tasks(run_id, tasks)
            if not tasks:
                store.fail(run_id, "Could not plan this goal")
            else:
                store.complete(run_id, str(result.get("response", "")), tasks)
            self.record_event(
                {
                    "event": "final",
                    "routine_id": routine_id,
                    "mode": "orchestrate",
                    "text": str(result.get("response", "")),
                    "run_id": run_id,
                }
            )
            count += 1
            self.record_event(
                {
                    "event": "run_end",
                    "routine_id": routine_id,
                    "mode": "orchestrate",
                    "run_id": run_id,
                }
            )
            count += 1
        except Exception as e:
            store.fail(run_id, str(e))
            self.record_event(
                {
                    "event": "error",
                    "error": str(e),
                    "routine_id": routine_id,
                    "mode": "orchestrate",
                    "run_id": run_id,
                }
            )
            raise
        return {"routine_id": routine_id, "events": count, "run_id": run_id}

    # ── Status stream ────────────────────────────────────────────────────

    def record_event(self, data: dict[str, Any]) -> dict[str, Any]:
        """Append a run event with a monotonic sequence number."""
        with self._lock:
            self._seq += 1
            event = {**data, "seq": self._seq}
            self._events.append(event)
            if len(self._events) > MAX_RECORDED_EVENTS:
                del self._events[: len(self._events) - MAX_RECORDED_EVENTS]
        return event

    def events_since(self, seq: int) -> list[dict[str, Any]]:
        """Events with ``seq`` greater than the cursor (no duplicates)."""
        with self._lock:
            return [event for event in self._events if event.get("seq", 0) > seq]

    # ── Ticker lifecycle ─────────────────────────────────────────────────

    def start(self) -> None:
        """Start the ticker. Idempotent; needs a running event loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._ticker())
        logger.info(
            "Routine runner ticker started (interval=%.1fs)",
            self._tick_interval,
            extra={"tag": "AGENT"},
        )

    def stop(self) -> None:
        """Cancel the ticker only — in-flight runs stay with the queue."""
        self._running = False
        task = self._task
        self._task = None
        if task is not None and not task.done():
            task.cancel()
        logger.info("Routine runner ticker stopped", extra={"tag": "AGENT"})

    async def _ticker(self) -> None:
        try:
            while self._running:
                await asyncio.sleep(self._tick_interval)
                if not self._running:
                    break
                try:
                    await self.tick()
                except Exception:
                    # A failed pass must not kill the ticker (it would leave
                    # running=True with no live task until restart): log,
                    # sleep, retry on the next tick.
                    logger.exception("Routine tick failed", extra={"tag": "AGENT"})
        except asyncio.CancelledError:
            pass  # expected on stop()


# ── Module singleton ─────────────────────────────────────────────────────

_default_runner: AgentRoutineRunner | None = None
_runner_lock = threading.Lock()


def get_routine_runner() -> AgentRoutineRunner:
    """Process-wide routine runner (constructed lazily)."""
    global _default_runner
    if _default_runner is None:
        with _runner_lock:
            if _default_runner is None:
                _default_runner = AgentRoutineRunner()
    return _default_runner


def reset_routine_runner_for_tests() -> None:
    """Drop the singleton so the next getter builds a fresh one."""
    global _default_runner
    with _runner_lock:
        _default_runner = None


__all__ = [
    "AgentRoutineRunner",
    "Routine",
    "RoutineStore",
    "get_routine_runner",
    "reset_routine_runner_for_tests",
]
