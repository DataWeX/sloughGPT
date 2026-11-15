"""Tests for agent phase 2 — background routine runtime.

Covers the factory layer that spawns stateful agent runs in the background:
Routine persistence, tick → TaskQueue enqueue, run_now, the queue handler
replaying ``AgentSystem.execute_stream`` (no second loop), and the
cache-first "fast" inference tier so lightweight bots do not always hit the
slow model.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

import domain.agents._internal.system as _sys_mod
from domain.agents._internal.routines import (
    AgentRoutineRunner,
    Routine,
    RoutineStore,
    get_routine_runner,
    reset_routine_runner_for_tests,
)
from domain.agents._internal.system import AgentSystem
from domain.infrastructure._internal.repository import FileRepository, JsonSerializer
from domain.infrastructure._internal.task_queue import Task

# ── Fakes ────────────────────────────────────────────────────────────────


class FakeQueue:
    """Captures enqueued tasks and registered handlers."""

    def __init__(self) -> None:
        self.tasks: list[Task] = []
        self.handlers: dict[str, object] = {}

    async def enqueue(self, task: Task) -> str:
        self.tasks.append(task)
        return task.id

    def register_handler(self, task_type: str, handler) -> None:
        self.handlers[task_type] = handler


class StubSystem:
    """AgentSystem stand-in whose execute_stream yields fixed events."""

    def __init__(self, events: list[dict] | None = None) -> None:
        self.events = events or []
        self.calls: list[tuple[str, str, str, str]] = []

    async def execute_stream(
        self, agent_id: str, request: str, session_id: str = "", user_id: str = "default"
    ):
        self.calls.append((agent_id, request, session_id, user_id))
        for ev in self.events:
            yield dict(ev)


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def store(tmp_path: Path) -> RoutineStore:
    return RoutineStore(directory=tmp_path / "routines")


@pytest.fixture
def queue() -> FakeQueue:
    return FakeQueue()


@pytest.fixture(autouse=True)
def _reset_singletons():
    reset_routine_runner_for_tests()
    yield
    reset_routine_runner_for_tests()


@pytest.fixture
def agent_sys(tmp_path: Path, monkeypatch) -> AgentSystem:
    """Isolated AgentSystem backed by a temp agent repo."""
    repo = FileRepository[dict](
        directory=str(tmp_path / "agents"),
        serializer=JsonSerializer(dict),
        key_suffix=".json",
    )
    monkeypatch.setattr(_sys_mod, "_agent_repo", repo)
    _sys_mod._default_system = None
    system = AgentSystem()
    yield system
    _sys_mod._default_system = None


# ── Routine persistence ──────────────────────────────────────────────────


class TestRoutineStore:
    def test_roundtrip(self, store: RoutineStore):
        routine = Routine(id="r1", agent_id="helper", prompt="summarize inbox")
        assert store.save(routine) is True

        loaded = store.get("r1")
        assert loaded is not None
        assert loaded.id == "r1"
        assert loaded.agent_id == "helper"
        assert loaded.prompt == "summarize inbox"
        assert loaded.enabled is True

    def test_list_and_delete(self, store: RoutineStore):
        store.save(Routine(id="r1", agent_id="a"))
        store.save(Routine(id="r2", agent_id="b"))

        assert {r.id for r in store.list()} == {"r1", "r2"}

        assert store.delete("r1") is True
        assert store.get("r1") is None
        assert [r.id for r in store.list()] == ["r2"]

    def test_get_missing_returns_none(self, store: RoutineStore):
        assert store.get("nope") is None

    def test_persists_to_disk(self, store: RoutineStore, tmp_path: Path):
        store.save(Routine(id="r1", agent_id="a"))
        files = list((tmp_path / "routines").glob("*.json"))
        assert len(files) == 1

        # A fresh store over the same directory sees the record.
        other = RoutineStore(directory=tmp_path / "routines")
        assert other.get("r1") is not None


# ── Tick → enqueue ───────────────────────────────────────────────────────


class TestTick:
    async def test_due_routine_is_enqueued(self, store, queue):
        store.save(Routine(id="r1", agent_id="helper", prompt="go", schedule=60.0))
        runner = AgentRoutineRunner(system=StubSystem(), store=store, queue=queue)

        await runner.tick()

        assert len(queue.tasks) == 1
        task = queue.tasks[0]
        assert task.task_type == "agent_run"
        assert task.payload["routine_id"] == "r1"
        assert task.payload["agent_id"] == "helper"
        assert task.payload["prompt"] == "go"

    async def test_tick_updates_last_run_at(self, store, queue):
        store.save(Routine(id="r1", agent_id="a", schedule=60.0))
        runner = AgentRoutineRunner(system=StubSystem(), store=store, queue=queue)
        before = time.time()

        await runner.tick()

        loaded = store.get("r1")
        assert loaded is not None
        assert loaded.last_run_at >= before

    async def test_disabled_routine_is_skipped(self, store, queue):
        store.save(Routine(id="r1", agent_id="a", enabled=False, schedule=0.0))
        runner = AgentRoutineRunner(system=StubSystem(), store=store, queue=queue)

        await runner.tick()

        assert queue.tasks == []

    async def test_not_due_routine_is_skipped(self, store, queue):
        routine = Routine(id="r1", agent_id="a", schedule=3600.0)
        routine.last_run_at = time.time() - 10  # ran 10s ago, interval 1h
        store.save(routine)
        runner = AgentRoutineRunner(system=StubSystem(), store=store, queue=queue)

        await runner.tick()

        assert queue.tasks == []

    async def test_only_due_routines_are_enqueued(self, store, queue):
        store.save(Routine(id="due", agent_id="a", schedule=0.0))
        stale = Routine(id="stale", agent_id="b", schedule=3600.0)
        stale.last_run_at = time.time() - 5
        store.save(stale)
        runner = AgentRoutineRunner(system=StubSystem(), store=store, queue=queue)

        await runner.tick()

        assert [t.payload["routine_id"] for t in queue.tasks] == ["due"]


# ── run_now ──────────────────────────────────────────────────────────────


class TestRunNow:
    async def test_run_now_enqueues_even_when_not_due(self, store, queue):
        routine = Routine(id="r1", agent_id="a", schedule=3600.0)
        routine.last_run_at = time.time() - 1
        store.save(routine)
        runner = AgentRoutineRunner(system=StubSystem(), store=store, queue=queue)

        task_id = await runner.run_now("r1")

        assert task_id is not None
        assert len(queue.tasks) == 1
        assert queue.tasks[0].payload["routine_id"] == "r1"

    async def test_run_now_missing_routine_returns_none(self, store, queue):
        runner = AgentRoutineRunner(system=StubSystem(), store=store, queue=queue)
        assert await runner.run_now("missing") is None
        assert queue.tasks == []


# ── Handler → execute_stream (the ONE loop) ──────────────────────────────


class TestHandler:
    async def test_handler_replays_execute_stream(self, store, queue):
        system = StubSystem(
            events=[
                {"event": "run_start", "run_id": "r1"},
                {"event": "final", "text": "done", "run_id": "r1"},
                {"event": "run_end", "run_id": "r1"},
            ]
        )
        runner = AgentRoutineRunner(system=system, store=store, queue=queue)
        runner.register_handlers()

        assert "agent_run" in queue.handlers

        task = Task(
            task_type="agent_run",
            payload={"routine_id": "r1", "agent_id": "helper", "prompt": "go"},
        )
        result = await queue.handlers["agent_run"](task)

        # Same loop as the sync/HTTP path — no second implementation.
        assert system.calls == [("helper", "go", "routine:r1", "routine")]
        assert result["routine_id"] == "r1"
        assert result["events"] == 3

    async def test_handler_records_events_with_routine_id(self, store, queue):
        system = StubSystem(events=[{"event": "run_start", "run_id": "x"}])
        runner = AgentRoutineRunner(system=system, store=store, queue=queue)
        runner.register_handlers()

        task = Task(
            task_type="agent_run",
            payload={"routine_id": "r7", "agent_id": "helper", "prompt": "p"},
        )
        await queue.handlers["agent_run"](task)

        events = runner.events_since(0)
        assert len(events) == 1
        assert events[0]["routine_id"] == "r7"
        assert events[0]["agent_id"] == "helper"
        assert events[0]["event"] == "run_start"

    async def test_events_since_is_monotonic(self, store, queue):
        system = StubSystem(events=[{"event": "run_start"}, {"event": "run_end"}])
        runner = AgentRoutineRunner(system=system, store=store, queue=queue)
        runner.register_handlers()
        task = Task(
            task_type="agent_run",
            payload={"routine_id": "r1", "agent_id": "a", "prompt": "p"},
        )
        await queue.handlers["agent_run"](task)

        first = runner.events_since(0)
        assert len(first) == 2
        last_seq = first[-1]["seq"]

        # Re-querying from the same cursor yields nothing new (no dupes).
        assert runner.events_since(last_seq) == []
        assert len(runner.events_since(0)) == 2


# ── Fast tier: cache-first inference ─────────────────────────────────────


class TestFastTier:
    @pytest.fixture(autouse=True)
    def _reset_cache(self):
        from domain.inference._internal.response_cache import (
            reset_response_cache_for_tests,
        )

        reset_response_cache_for_tests()
        yield
        reset_response_cache_for_tests()

    def test_fast_tier_serves_repeat_prompt_from_cache(self, agent_sys):
        from domain.inference._internal.response_cache import get_response_cache

        calls: list[str] = []

        def spy(prompt: str, max_tokens: int = 200) -> dict:
            calls.append(prompt)
            return {"text": "model-says"}

        agent_sys.create("lite", "Lite", "desc", inference_tier="fast")
        agent_sys._agent.set_inference_fn(spy)

        agent = agent_sys._fresh_agent("lite")
        assert agent is not None
        assert agent.config.inference_tier == "fast"

        first = agent._inference_fn("hello there")
        second = agent._inference_fn("hello there")

        assert first["text"] == "model-says"
        assert second["text"] == "model-says"
        # Second call must not touch the slow model.
        assert calls == ["hello there"]
        assert get_response_cache().get_stats()["hits"] >= 1

    def test_main_tier_always_calls_model(self, agent_sys):
        calls: list[str] = []

        def spy(prompt: str, max_tokens: int = 200) -> dict:
            calls.append(prompt)
            return {"text": "model-says"}

        agent_sys.create("heavy", "Heavy", "desc", inference_tier="main")
        agent_sys._agent.set_inference_fn(spy)

        agent = agent_sys._fresh_agent("heavy")
        assert agent is not None
        assert agent.config.inference_tier == "main"

        agent._inference_fn("same prompt")
        agent._inference_fn("same prompt")

        assert len(calls) == 2

    def test_default_tier_is_main(self, agent_sys):
        agent_sys.create("plain", "Plain", "desc")
        agent = agent_sys._fresh_agent("plain")
        assert agent is not None
        assert agent.config.inference_tier == "main"


# ── Ticker lifecycle ─────────────────────────────────────────────────────


class TestTicker:
    async def test_start_stop_cancels_ticker(self, store, queue):
        runner = AgentRoutineRunner(
            system=StubSystem(), store=store, queue=queue, tick_interval=0.01
        )
        assert runner.running is False

        runner.start()
        assert runner.running is True
        assert runner._task is not None

        runner.stop()
        assert runner.running is False
        assert runner._task is None
        await asyncio.sleep(0)

    async def test_ticker_enqueues_due_routines(self, store, queue):
        store.save(Routine(id="r1", agent_id="a", schedule=0.0))
        runner = AgentRoutineRunner(
            system=StubSystem(), store=store, queue=queue, tick_interval=0.01
        )

        runner.start()
        await asyncio.sleep(0.05)
        runner.stop()
        await asyncio.sleep(0)

        assert len(queue.tasks) >= 1
        assert queue.tasks[0].payload["routine_id"] == "r1"

    async def test_stop_does_not_enqueue_after_stop(self, store, queue):
        store.save(Routine(id="r1", agent_id="a", schedule=0.0))
        runner = AgentRoutineRunner(
            system=StubSystem(), store=store, queue=queue, tick_interval=0.01
        )
        runner.start()
        await asyncio.sleep(0.03)
        runner.stop()
        await asyncio.sleep(0)
        count_after_stop = len(queue.tasks)

        await asyncio.sleep(0.05)
        assert len(queue.tasks) == count_after_stop


# ── Singleton ────────────────────────────────────────────────────────────


class TestSingleton:
    def test_get_routine_runner_is_singleton(self):
        a = get_routine_runner()
        b = get_routine_runner()
        assert a is b

    def test_reset_returns_fresh_instance(self):
        a = get_routine_runner()
        reset_routine_runner_for_tests()
        b = get_routine_runner()
        assert a is not b


class TestHierarchy:
    """Orchestrate routines — the hierarchy loop reused, no third loop."""

    @pytest.fixture
    def fake(self, monkeypatch) -> dict:
        """Patch the facade imports ``_handle_orchestrate`` resolves at call time."""
        from types import SimpleNamespace

        import domain.agents as agents_facade

        calls: dict = {"init": [], "execute": [], "boom": False, "no_plan": False}

        class FakeStore:
            def __init__(self) -> None:
                self.started: list[tuple[str, str]] = []
                self.tasks_set: list[tuple[str, list]] = []
                self.completed: list[tuple[str, str, list | None]] = []
                self.failed: list[tuple[str, str]] = []

            def start(self, goal: str, context: str = "") -> str:
                self.started.append((goal, context))
                return "run-42"

            def set_tasks(self, run_id: str, tasks: list) -> None:
                self.tasks_set.append((run_id, tasks))

            def complete(self, run_id: str, response: str, tasks: list | None = None) -> None:
                self.completed.append((run_id, response, tasks))

            def fail(self, run_id: str, error: str) -> None:
                self.failed.append((run_id, error))

        class FakeOrch:
            agents = {
                "planner": SimpleNamespace(name="planner"),
                "writer": SimpleNamespace(name="writer"),
            }

            def __init__(self, agents: dict | None = None) -> None:
                calls["init"].append(agents)

            async def async_execute(self, goal: str, context: str = "") -> dict:
                calls["execute"].append((goal, context))
                if calls["boom"]:
                    raise RuntimeError("fan-out exploded")
                if calls["no_plan"]:
                    return {"response": "Could not plan this goal.", "tasks": []}
                return {"response": "composed", "tasks": [{"id": "t1", "status": "completed"}]}

        store = FakeStore()
        calls["store"] = store
        monkeypatch.setattr(agents_facade, "MultiAgentOrchestrator", FakeOrch)
        monkeypatch.setattr(agents_facade, "get_agent_run_store", lambda: store)
        return calls

    @staticmethod
    def _task(routine_id: str = "h1", **kw) -> Task:
        payload = {
            "routine_id": routine_id,
            "agent_id": "",
            "prompt": kw.pop("prompt", "ship the report"),
            "mode": "orchestrate",
            "members": kw.pop("members", []),
            "context": kw.pop("context", ""),
        }
        payload.update(kw)
        return Task(name=f"routine:{routine_id}", task_type="agent_run", payload=payload)

    async def test_orchestrate_routine_replays_async_execute(self, store, queue, fake):
        runner = AgentRoutineRunner(system=StubSystem(), store=store, queue=queue)
        result = await runner._handle_run(self._task(context="ctx"))

        assert fake["execute"] == [("ship the report", "ctx")]
        kinds = [e["event"] for e in runner.events_since(0)]
        assert kinds == ["run_start", "final", "run_end"]
        # Durable run record mirrors the HTTP projection's bookkeeping.
        assert fake["store"].started == [("ship the report", "ctx")]
        assert fake["store"].completed[0][0:2] == ("run-42", "composed")
        assert result["run_id"] == "run-42"

    async def test_member_filter_narrows_orchestrator(self, store, queue, fake):
        runner = AgentRoutineRunner(system=StubSystem(), store=store, queue=queue)
        await runner._handle_run(self._task(members=["writer"]))

        # First construction is the full registry (to read .agents), second
        # is the filtered one actually executed.
        assert fake["init"][0] is None
        assert list(fake["init"][1]) == ["writer"]

    async def test_orchestrate_failure_records_error_and_reraises(self, store, queue, fake):
        fake["boom"] = True
        runner = AgentRoutineRunner(system=StubSystem(), store=store, queue=queue)

        with pytest.raises(RuntimeError, match="fan-out exploded"):
            await runner._handle_run(self._task())

        kinds = [e["event"] for e in runner.events_since(0)]
        assert kinds == ["run_start", "error"]
        assert fake["store"].failed == [("run-42", "fan-out exploded")]

    async def test_plan_empty_marks_run_failed_not_completed(self, store, queue, fake):
        fake["no_plan"] = True
        runner = AgentRoutineRunner(system=StubSystem(), store=store, queue=queue)
        await runner._handle_run(self._task())

        assert fake["store"].failed == [("run-42", "Could not plan this goal")]
        assert fake["store"].completed == []

    async def test_tick_carries_hierarchy_payload(self, store, queue):
        routine = Routine(id="h9", prompt="g", mode="orchestrate", members=["writer"], schedule=1.0)
        store.save(routine)
        runner = AgentRoutineRunner(system=StubSystem(), store=store, queue=queue)

        await runner.tick()

        payload = queue.tasks[0].payload
        assert payload["mode"] == "orchestrate"
        assert payload["members"] == ["writer"]
        assert payload["context"] == ""

    def test_orchestrate_routine_roundtrips(self, store):
        routine = Routine(id="h2", prompt="g", mode="orchestrate", members=["a", "b"], context="c")
        store.save(routine)
        back = store.get("h2")
        assert back is not None
        assert (back.mode, back.members, back.context) == ("orchestrate", ["a", "b"], "c")


class TestRunnerResilience:
    """One broken routine or tick must not take the whole runtime down."""

    async def test_failing_routine_does_not_block_others(self, store, queue):
        bad = Routine(id="bad", agent_id="a", prompt="p", schedule=1.0)
        good = Routine(id="good", agent_id="a", prompt="p", schedule=1.0)
        store.save(bad)
        store.save(good)
        runner = AgentRoutineRunner(system=StubSystem(), store=store, queue=queue)

        original_enqueue = queue.enqueue

        async def flaky(task: Task) -> str:
            if task.payload.get("routine_id") == "bad":
                raise RuntimeError("queue down")
            return await original_enqueue(task)

        queue.enqueue = flaky  # type: ignore[method-assign]

        enqueued = await runner.tick()

        # The healthy routine ran; the broken one is retried next tick
        # (its schedule did not advance on failure).
        assert [t.payload["routine_id"] for t in queue.tasks] == ["good"]
        assert enqueued == [queue.tasks[0].id]
        assert store.get("good").last_run_at > 0
        assert store.get("bad").last_run_at == 0

    async def test_handler_records_error_event_then_reraises(self, store, queue):
        class BoomSystem:
            async def execute_stream(self, agent_id, request, session_id="", user_id="default"):
                yield {"event": "run_start"}
                raise RuntimeError("model exploded")

        runner = AgentRoutineRunner(system=BoomSystem(), store=store, queue=queue)
        task = Task(
            name="routine:r1",
            task_type="agent_run",
            payload={"routine_id": "r1", "agent_id": "a", "prompt": "p"},
        )

        # The queue keeps its failure semantics (re-raised) ...
        with pytest.raises(RuntimeError, match="model exploded"):
            await runner._handle_run(task)

        # ... but the status stream saw the failure, after the partial run.
        kinds = [e["event"] for e in runner.events_since(0)]
        assert kinds == ["run_start", "error"]
        assert runner.events_since(0)[-1]["error"] == "model exploded"

    async def test_ticker_survives_a_failed_tick(self, store, queue):
        runner = AgentRoutineRunner(
            system=StubSystem(), store=store, queue=queue, tick_interval=0.01
        )
        calls: list[int] = []

        async def boom() -> list[str]:
            calls.append(1)
            raise RuntimeError("transient failure")

        runner.tick = boom  # type: ignore[method-assign]
        runner.start()
        await asyncio.sleep(0.1)
        runner.stop()

        # A dead ticker would have recorded exactly one call (and left
        # running=True with no live task until restart).
        assert len(calls) >= 2
        assert runner._task is None
