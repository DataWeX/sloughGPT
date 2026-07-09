"""
Tests for LifecycleManager — startup/shutdown ordering, health gates, drain.
"""

from __future__ import annotations

import asyncio
import time
import pytest

from domains.infrastructure.lifecycle import (
    ALL_PROFILES,
    LifecycleManager,
    LifecyclePhase,
    StartupHook,
    StartupProfile,
    ShutdownHook,
    _topological_sort,
    get_lifecycle_manager,
    reset_lifecycle_manager,
)
from domains.infrastructure.event_bus import EventBus, EventPriority


# ── Topological sort ──


def test_topological_sort_empty():
    assert _topological_sort([]) == []


def test_topological_sort_no_deps():
    a = StartupHook(name="a", handler=lambda: asyncio.sleep(0))
    b = StartupHook(name="b", handler=lambda: asyncio.sleep(0))
    ordered = _topological_sort([a, b])
    assert len(ordered) == 2


def test_topological_sort_simple_chain():
    a = StartupHook(name="a", handler=lambda: asyncio.sleep(0))
    b = StartupHook(name="b", handler=lambda: asyncio.sleep(0), depends_on=["a"])
    c = StartupHook(name="c", handler=lambda: asyncio.sleep(0), depends_on=["b"])
    ordered = _topological_sort([c, a, b])
    names = [h.name for h in ordered]
    assert names.index("a") < names.index("b")
    assert names.index("b") < names.index("c")


def test_topological_sort_cycle_does_not_crash():
    a = StartupHook(name="a", handler=lambda: asyncio.sleep(0), depends_on=["b"])
    b = StartupHook(name="b", handler=lambda: asyncio.sleep(0), depends_on=["a"])
    ordered = _topological_sort([a, b])
    assert len(ordered) == 2  # best-effort


def test_topological_sort_missing_dep():
    a = StartupHook(name="a", handler=lambda: asyncio.sleep(0), depends_on=["nonexistent"])
    b = StartupHook(name="b", handler=lambda: asyncio.sleep(0))
    ordered = _topological_sort([a, b])
    assert len(ordered) == 2


# ── LifecycleManager ──


@pytest.fixture
def mgr():
    return LifecycleManager()


@pytest.mark.asyncio
async def test_initial_phase(mgr):
    assert mgr.phase == LifecyclePhase.INIT
    assert mgr.is_running() is False
    assert mgr.is_draining() is False


@pytest.mark.asyncio
async def test_empty_startup_succeeds(mgr):
    success = await mgr.start()
    assert success is True
    assert mgr.phase == LifecyclePhase.RUNNING
    assert mgr.is_running() is True
    assert mgr.uptime_seconds > 0


@pytest.mark.asyncio
async def test_simple_startup_hook(mgr):
    called = []

    async def my_hook():
        called.append(True)

    mgr.register_startup_hook(StartupHook("test", my_hook))
    success = await mgr.start()
    assert success is True
    assert len(called) == 1


@pytest.mark.asyncio
async def test_startup_hook_order(mgr):
    order: list[str] = []

    async def make_hook(name: str, delay: float = 0):
        async def _run():
            if delay:
                await asyncio.sleep(delay)
            order.append(name)
        return _run

    mgr.register_startup_hook(StartupHook("a", await make_hook("a"), depends_on=[]))
    mgr.register_startup_hook(StartupHook("b", await make_hook("b"), depends_on=["a"]))
    mgr.register_startup_hook(StartupHook("c", await make_hook("c"), depends_on=["b"]))

    await mgr.start()
    assert order == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_critical_hook_failure_crashes(mgr):
    async def fail_hook():
        raise RuntimeError("boom")

    mgr.register_startup_hook(StartupHook("fail", fail_hook, critical=True))
    success = await mgr.start()
    assert success is False
    assert mgr.phase == LifecyclePhase.CRASHED


@pytest.mark.asyncio
async def test_non_critical_hook_failure_allows_running(mgr):
    async def ok_hook():
        pass

    async def fail_hook():
        raise RuntimeError("non-critical")

    mgr.register_startup_hook(StartupHook("ok", ok_hook))
    mgr.register_startup_hook(StartupHook("fail", fail_hook, critical=False))

    success = await mgr.start()
    assert success is True
    assert mgr.phase == LifecyclePhase.RUNNING


@pytest.mark.asyncio
async def test_hook_timeout(mgr):
    async def slow_hook():
        await asyncio.sleep(10)

    mgr.register_startup_hook(StartupHook("slow", slow_hook, timeout=0.05, critical=True))
    success = await mgr.start()
    assert success is False
    assert mgr.phase == LifecyclePhase.CRASHED


@pytest.mark.asyncio
async def test_duplicate_hook_skipped(mgr):
    async def hook_a():
        pass

    async def hook_b():
        pass

    mgr.register_startup_hook(StartupHook("same", hook_a))
    mgr.register_startup_hook(StartupHook("same", hook_b))  # same name

    await mgr.start()
    assert mgr.phase == LifecyclePhase.RUNNING


@pytest.mark.asyncio
async def test_start_while_not_init_ignored(mgr):
    await mgr.start()
    success = await mgr.start()  # second call
    assert success is True
    assert mgr.phase == LifecyclePhase.RUNNING


@pytest.mark.asyncio
async def test_shutdown_before_start_ignored(mgr):
    success = await mgr.shutdown()
    assert success is False
    assert mgr.phase == LifecyclePhase.INIT  # unchanged


@pytest.mark.asyncio
async def test_shutdown_success(mgr):
    await mgr.start()
    success = await mgr.shutdown()
    assert success is True
    assert mgr.phase == LifecyclePhase.STOPPED


@pytest.mark.asyncio
async def test_shutdown_hook_runs(mgr):
    shutdown_called = []

    async def my_shutdown():
        shutdown_called.append(True)

    mgr.register_shutdown_hook(ShutdownHook("cleanup", my_shutdown))
    await mgr.start()
    await mgr.shutdown()
    assert len(shutdown_called) == 1


@pytest.mark.asyncio
async def test_shutdown_after_crash_works(mgr):
    async def fail_hook():
        raise RuntimeError("boom")

    mgr.register_startup_hook(StartupHook("fail", fail_hook, critical=True))
    await mgr.start()
    assert mgr.phase == LifecyclePhase.CRASHED

    # shutdown from crashed state should still work
    success = await mgr.shutdown()
    assert success is True
    assert mgr.phase == LifecyclePhase.STOPPED


# ── In-flight tracking ──


@pytest.mark.asyncio
async def test_acquire_release_in_flight(mgr):
    ok = await mgr.acquire_in_flight()
    assert ok is True
    assert mgr.in_flight_count == 1
    await mgr.release_in_flight()
    assert mgr.in_flight_count == 0


@pytest.mark.asyncio
async def test_acquire_rejected_during_drain(mgr):
    await mgr.start()
    await mgr.acquire_in_flight()

    # Start shutdown in background (blocks on drain)
    shutdown_task = asyncio.create_task(mgr.shutdown(timeout=2.0))
    await asyncio.sleep(0.05)  # let phase reach DRAINING

    # New acquires should be rejected during drain
    ok = await mgr.acquire_in_flight()
    assert ok is False

    # Release the stuck in-flight task so drain completes
    await mgr.release_in_flight()
    await shutdown_task
    assert mgr.phase == LifecyclePhase.STOPPED


@pytest.mark.asyncio
async def test_drain_waits_for_in_flight(mgr):
    await mgr.start()

    async def slow_task():
        await mgr.acquire_in_flight()
        await asyncio.sleep(0.1)
        await mgr.release_in_flight()

    task = asyncio.create_task(slow_task())
    await asyncio.sleep(0.02)  # let task acquire

    start = time.time()
    await mgr.shutdown(timeout=5.0)
    elapsed = time.time() - start
    assert elapsed >= 0.07  # waited for slow task
    assert mgr.phase == LifecyclePhase.STOPPED
    await task


# ── Health gates ──


@pytest.mark.asyncio
async def test_gate_ready(mgr):
    mgr.register_gate("db", lambda: True)
    assert mgr.gate_ready("db") is True


@pytest.mark.asyncio
async def test_gate_not_ready(mgr):
    mgr.register_gate("db", lambda: False)
    assert mgr.gate_ready("db") is False


@pytest.mark.asyncio
async def test_unknown_gate_is_ready(mgr):
    assert mgr.gate_ready("nonexistent") is True


@pytest.mark.asyncio
async def test_gates_ready_all_pass(mgr):
    mgr.register_gate("a", lambda: True)
    mgr.register_gate("b", lambda: True)
    assert mgr.gates_ready() is True


@pytest.mark.asyncio
async def test_gates_ready_one_fails(mgr):
    mgr.register_gate("a", lambda: True)
    mgr.register_gate("b", lambda: False)
    assert mgr.gates_ready() is False


@pytest.mark.asyncio
async def test_unregister_gate(mgr):
    mgr.register_gate("temp", lambda: False)
    mgr.unregister_gate("temp")
    assert mgr.gate_ready("temp") is True


@pytest.mark.asyncio
async def test_wait_for_gates_timeout(mgr):
    mgr.register_gate("never", lambda: False)
    ready = await mgr.wait_for_gates(timeout=0.1, poll_interval=0.05)
    assert ready is False


@pytest.mark.asyncio
async def test_wait_for_gates_success(mgr):
    gate_state = [False]

    def check():
        return gate_state[0]

    mgr.register_gate("eventually", check)

    async def enable_gate():
        await asyncio.sleep(0.05)
        gate_state[0] = True

    task = asyncio.create_task(enable_gate())
    ready = await mgr.wait_for_gates(timeout=1.0, poll_interval=0.02)
    assert ready is True
    await task


# ── Mark crashed ──


@pytest.mark.asyncio
async def test_mark_crashed(mgr):
    await mgr.start()
    await mgr.mark_crashed(reason="kernel panic")
    assert mgr.phase == LifecyclePhase.CRASHED


@pytest.mark.asyncio
async def test_mark_crashed_from_init(mgr):
    await mgr.mark_crashed("early failure")
    assert mgr.phase == LifecyclePhase.CRASHED


@pytest.mark.asyncio
async def test_mark_crashed_idempotent(mgr):
    await mgr.mark_crashed("first")
    await mgr.mark_crashed("second")
    assert mgr.phase == LifecyclePhase.CRASHED


# ── EventBus integration ──


@pytest.mark.asyncio
async def test_event_bus_emits_phase_changes():
    bus = EventBus()
    mgr = LifecycleManager(event_bus=bus)
    events: list[str] = []

    def collect(name, data):
        if name == "lifecycle.phase_changed":
            events.append(f"{data['from']}->{data['to']}")

    bus.on("lifecycle.phase_changed", collect, priority=EventPriority.CRITICAL)

    await mgr.start()
    assert "init->starting" in events
    assert "starting->running" in events

    await mgr.shutdown()
    assert "running->draining" in events
    assert "draining->stopping" in events
    assert "stopping->stopped" in events


@pytest.mark.asyncio
async def test_event_bus_hook_events():
    bus = EventBus()
    mgr = LifecycleManager(event_bus=bus)
    hook_events: list[str] = []

    def collect(name, data):
        hook_events.append(name)

    bus.on("lifecycle.hook_started", collect)
    bus.on("lifecycle.hook_completed", collect)

    async def my_hook():
        pass

    mgr.register_startup_hook(StartupHook("my", my_hook))
    await mgr.start()

    assert "lifecycle.hook_started" in hook_events
    assert "lifecycle.hook_completed" in hook_events


# ── Singleton ──


def test_get_lifecycle_manager_singleton():
    reset_lifecycle_manager()
    a = get_lifecycle_manager()
    b = get_lifecycle_manager()
    assert a is b


def test_reset_lifecycle_manager():
    reset_lifecycle_manager()
    a = get_lifecycle_manager()
    reset_lifecycle_manager()
    b = get_lifecycle_manager()
    assert a is not b


# ── Get results ──


@pytest.mark.asyncio
async def test_get_results(mgr):
    await mgr.start()
    results = mgr.get_results()
    assert results["phase"] == "running"
    assert results["uptime"] >= 0
    assert results["in_flight"] == 0
    assert "hooks" in results
    assert "gates" in results


@pytest.mark.asyncio
async def test_event_bus_on_startup_hook_failure():
    bus = EventBus()
    mgr = LifecycleManager(event_bus=bus)
    failed_events: list[str] = []

    def collect(name, data):
        if name == "lifecycle.hook_failed":
            failed_events.append(data["hook"])

    bus.on("lifecycle.hook_failed", collect)

    async def fail_hook():
        raise ValueError("broken")

    mgr.register_startup_hook(StartupHook("fail", fail_hook, critical=True))
    await mgr.start()

    assert "fail" in failed_events


@pytest.mark.asyncio
async def test_crashed_shutdown_with_in_flight():
    mgr = LifecycleManager()

    async def fail_hook():
        raise RuntimeError("critical failure")

    mgr.register_startup_hook(StartupHook("fail", fail_hook, critical=True))
    await mgr.start()
    assert mgr.phase == LifecyclePhase.CRASHED

    success = await mgr.shutdown()
    assert success is True
    assert mgr.phase == LifecyclePhase.STOPPED


@pytest.mark.asyncio
async def test_concurrent_acquires():
    mgr = LifecycleManager()
    count = 5
    tasks = [mgr.acquire_in_flight() for _ in range(count)]
    results = await asyncio.gather(*tasks)
    assert all(results)
    assert mgr.in_flight_count == count

    releases = [mgr.release_in_flight() for _ in range(count)]
    await asyncio.gather(*releases)
    assert mgr.in_flight_count == 0


@pytest.mark.asyncio
async def test_mark_crashed_emits_event():
    bus = EventBus()
    mgr = LifecycleManager(event_bus=bus)
    events: list[dict] = []

    def collect(name, data):
        if name == "lifecycle.crashed":
            events.append(data)

    bus.on("lifecycle.crashed", collect)
    await mgr.mark_crashed("test crash")
    assert len(events) == 1
    assert events[0]["reason"] == "test crash"


@pytest.mark.asyncio
async def test_duplicate_startup_hook_logged(mgr, caplog):
    async def hook():
        pass

    mgr.register_startup_hook(StartupHook("same", hook))
    mgr.register_startup_hook(StartupHook("same", hook))
    assert "Duplicate startup hook" in caplog.text


@pytest.mark.asyncio
async def test_duplicate_shutdown_hook_logged(mgr, caplog):
    async def hook():
        pass

    mgr.register_shutdown_hook(ShutdownHook("same", hook))
    mgr.register_shutdown_hook(ShutdownHook("same", hook))
    assert "Duplicate shutdown hook" in caplog.text


# ── StartupProfile ──


class TestStartupProfile:
    def test_enum_values(self):
        assert StartupProfile.FULL.value == "full"
        assert StartupProfile.QUICK.value == "quick"
        assert StartupProfile.MINIMAL.value == "minimal"

    def test_from_env_default(self, monkeypatch):
        monkeypatch.delenv("MAN_STARTUP_PROFILE", raising=False)
        assert StartupProfile.from_env() == StartupProfile.FULL

    def test_from_env_full(self, monkeypatch):
        monkeypatch.setenv("MAN_STARTUP_PROFILE", "full")
        assert StartupProfile.from_env() == StartupProfile.FULL

    def test_from_env_quick(self, monkeypatch):
        monkeypatch.setenv("MAN_STARTUP_PROFILE", "quick")
        assert StartupProfile.from_env() == StartupProfile.QUICK

    def test_from_env_minimal(self, monkeypatch):
        monkeypatch.setenv("MAN_STARTUP_PROFILE", "minimal")
        assert StartupProfile.from_env() == StartupProfile.MINIMAL

    def test_from_env_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("MAN_STARTUP_PROFILE", "QUICK")
        assert StartupProfile.from_env() == StartupProfile.QUICK

    def test_from_env_unknown_falls_back(self, monkeypatch):
        monkeypatch.setenv("MAN_STARTUP_PROFILE", "turbo")
        assert StartupProfile.from_env() == StartupProfile.FULL

    def test_all_profiles_includes_all(self):
        assert ALL_PROFILES == frozenset(StartupProfile)
        assert len(ALL_PROFILES) == 3


# ── Profile filtering ──


@pytest.mark.asyncio
async def test_profile_filter_hooks():
    """Only hooks matching the active profile run."""
    mgr = LifecycleManager()
    ran: list[str] = []

    async def make(name: str):
        async def _run():
            ran.append(name)
        return _run

    core = frozenset({StartupProfile.FULL, StartupProfile.QUICK, StartupProfile.MINIMAL})
    ai = frozenset({StartupProfile.FULL})

    mgr.register_startup_hook(StartupHook("logging", await make("logging"), profiles=core))
    mgr.register_startup_hook(StartupHook("model", await make("model"), profiles=ai, depends_on=["logging"]))

    await mgr.start(profile=StartupProfile.QUICK)
    assert ran == ["logging"]  # model skipped for quick


@pytest.mark.asyncio
async def test_profile_full_runs_all():
    mgr = LifecycleManager()
    ran: list[str] = []

    async def make(name: str):
        async def _run():
            ran.append(name)
        return _run

    core = frozenset({StartupProfile.FULL, StartupProfile.QUICK, StartupProfile.MINIMAL})
    ai = frozenset({StartupProfile.FULL})

    mgr.register_startup_hook(StartupHook("logging", await make("logging"), profiles=core))
    mgr.register_startup_hook(StartupHook("model", await make("model"), profiles=ai, depends_on=["logging"]))

    await mgr.start(profile=StartupProfile.FULL)
    assert ran == ["logging", "model"]


@pytest.mark.asyncio
async def test_profile_minimal_skips_ai():
    mgr = LifecycleManager()
    ran: list[str] = []

    async def make(name: str):
        async def _run():
            ran.append(name)
        return _run

    core = frozenset({StartupProfile.FULL, StartupProfile.QUICK, StartupProfile.MINIMAL})
    ai = frozenset({StartupProfile.FULL})

    mgr.register_startup_hook(StartupHook("logging", await make("logging"), profiles=core))
    mgr.register_startup_hook(StartupHook("model", await make("model"), profiles=ai, depends_on=["logging"]))

    await mgr.start(profile=StartupProfile.MINIMAL)
    assert ran == ["logging"]


@pytest.mark.asyncio
async def test_profile_defaults_to_env(mgr, monkeypatch):
    monkeypatch.setenv("MAN_STARTUP_PROFILE", "quick")
    ran: list[str] = []

    async def make(name: str):
        async def _run():
            ran.append(name)
        return _run

    core = frozenset({StartupProfile.FULL, StartupProfile.QUICK, StartupProfile.MINIMAL})
    ai = frozenset({StartupProfile.FULL})

    mgr.register_startup_hook(StartupHook("logging", await make("logging"), profiles=core))
    mgr.register_startup_hook(StartupHook("model", await make("model"), profiles=ai, depends_on=["logging"]))

    await mgr.start()  # no profile arg — reads from env
    assert ran == ["logging"]


# ── Preview ──


@pytest.mark.asyncio
async def test_preview_full():
    mgr = LifecycleManager()
    core = ALL_PROFILES
    ai = frozenset({StartupProfile.FULL})

    async def noop():
        pass

    mgr.register_startup_hook(StartupHook("logging", noop, profiles=core))
    mgr.register_startup_hook(StartupHook("model", noop, profiles=ai, depends_on=["logging"]))

    prev = mgr.preview(StartupProfile.FULL)
    names = [h["name"] for h in prev]
    assert names == ["logging", "model"]


@pytest.mark.asyncio
async def test_preview_quick():
    mgr = LifecycleManager()
    core = ALL_PROFILES
    ai = frozenset({StartupProfile.FULL})

    async def noop():
        pass

    mgr.register_startup_hook(StartupHook("logging", noop, profiles=core))
    mgr.register_startup_hook(StartupHook("model", noop, profiles=ai, depends_on=["logging"]))

    prev = mgr.preview(StartupProfile.QUICK)
    names = [h["name"] for h in prev]
    assert names == ["logging"]


@pytest.mark.asyncio
async def test_preview_uses_active_profile():
    mgr = LifecycleManager()
    core = ALL_PROFILES
    ai = frozenset({StartupProfile.FULL})

    async def noop():
        pass

    mgr.register_startup_hook(StartupHook("logging", noop, profiles=core))
    mgr.register_startup_hook(StartupHook("model", noop, profiles=ai, depends_on=["logging"]))

    await mgr.start(profile=StartupProfile.QUICK)
    prev = mgr.preview()  # no arg — uses active profile
    names = [h["name"] for h in prev]
    assert names == ["logging"]


@pytest.mark.asyncio
async def test_preview_returns_hook_details():
    mgr = LifecycleManager()

    async def noop():
        pass

    mgr.register_startup_hook(StartupHook("my_hook", noop, profiles=ALL_PROFILES, depends_on=[], timeout=15.0, critical=False))

    prev = mgr.preview(StartupProfile.FULL)
    assert len(prev) == 1
    hook = prev[0]
    assert hook["name"] == "my_hook"
    assert hook["critical"] is False
    assert hook["timeout"] == 15.0
    assert hook["depends_on"] == []


# ── get_results with profile info ──


@pytest.mark.asyncio
async def test_get_results_includes_profile():
    mgr = LifecycleManager()
    await mgr.start(profile=StartupProfile.QUICK)
    results = mgr.get_results()
    assert results["profile"] == "quick"
    assert "preview" in results["hooks"]


@pytest.mark.asyncio
async def test_get_results_preview_list():
    mgr = LifecycleManager()

    async def noop():
        pass

    mgr.register_startup_hook(StartupHook("a", noop, profiles=ALL_PROFILES))
    await mgr.start(profile=StartupProfile.MINIMAL)
    results = mgr.get_results()
    preview = results["hooks"]["preview"]
    assert isinstance(preview, list)
    assert preview[0]["name"] == "a"


# ── Server-level endpoint test ──


@pytest.mark.asyncio
async def test_lifecycle_endpoint():
    """Verify /system/lifecycle returns expected fields via TestClient."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()

    @app.get("/system/lifecycle")
    async def _lifecycle():
        from domains.infrastructure.lifecycle import get_lifecycle_manager
        mgr = get_lifecycle_manager()
        return mgr.get_results()

    client = TestClient(app)

    # No lifecycle manager running — returns default state
    resp = client.get("/system/lifecycle")
    assert resp.status_code == 200
    data = resp.json()
    # INIT phase by default
    assert "phase" in data
    assert "profile" in data
    assert "uptime" in data
    assert "in_flight" in data
    assert "hooks" in data
    assert "gates" in data


# ── Empty profiles field = all profiles ──


@pytest.mark.asyncio
async def test_empty_profiles_field_defaults_to_all():
    """A StartupHook with no profiles set should run under all profiles."""
    mgr = LifecycleManager()
    ran: list[str] = []

    async def my_hook():
        ran.append("ran")

    # No profiles field — defaults to ALL_PROFILES
    mgr.register_startup_hook(StartupHook("always", my_hook))

    await mgr.start(profile=StartupProfile.MINIMAL)
    assert ran == ["ran"]


# ── Hook result tracking ──


@pytest.mark.asyncio
async def test_startup_results_after_success():
    mgr = LifecycleManager()
    ran: list[str] = []

    async def hook_a():
        ran.append("a")

    async def hook_b():
        ran.append("b")

    mgr.register_startup_hook(StartupHook("a", hook_a))
    mgr.register_startup_hook(StartupHook("b", hook_b, depends_on=["a"]))

    await mgr.start()
    assert mgr.phase == LifecyclePhase.RUNNING

    results = mgr.startup_results
    assert len(results) == 2
    assert results[0]["name"] == "a"
    assert results[0]["success"] is True
    assert results[0]["error"] == ""
    assert results[1]["name"] == "b"
    assert results[1]["success"] is True


@pytest.mark.asyncio
async def test_startup_results_shows_failure():
    mgr = LifecycleManager()

    async def ok_hook():
        pass

    async def fail_hook():
        raise RuntimeError("broken")

    mgr.register_startup_hook(StartupHook("ok", ok_hook, critical=False))
    mgr.register_startup_hook(StartupHook("fail", fail_hook, critical=True, depends_on=["ok"]))

    await mgr.start()
    assert mgr.phase == LifecyclePhase.CRASHED

    results = mgr.startup_results
    assert len(results) == 2
    assert results[0]["name"] == "ok"
    assert results[0]["success"] is True
    assert results[1]["name"] == "fail"
    assert results[1]["success"] is False
    assert "broken" in results[1]["error"]


@pytest.mark.asyncio
async def test_startup_results_timeout_shown():
    mgr = LifecycleManager()

    async def slow_hook():
        await asyncio.sleep(10)

    mgr.register_startup_hook(StartupHook("slow", slow_hook, timeout=0.05, critical=True))

    await mgr.start()

    results = mgr.startup_results
    assert len(results) == 1
    assert results[0]["name"] == "slow"
    assert results[0]["success"] is False
    assert "timed out" in results[0]["error"]


@pytest.mark.asyncio
async def test_shutdown_results_after_success():
    mgr = LifecycleManager()

    async def noop():
        pass

    mgr.register_startup_hook(StartupHook("a", noop))
    mgr.register_shutdown_hook(ShutdownHook("clean", noop))

    await mgr.start()
    await mgr.shutdown()

    results = mgr.shutdown_results
    assert len(results) == 1
    assert results[0]["name"] == "clean"
    assert results[0]["success"] is True


@pytest.mark.asyncio
async def test_shutdown_results_with_failure():
    mgr = LifecycleManager()

    async def noop():
        pass

    async def fail_shutdown():
        raise ValueError("shutdown fail")

    mgr.register_startup_hook(StartupHook("a", noop))
    mgr.register_shutdown_hook(ShutdownHook("good", noop))
    mgr.register_shutdown_hook(ShutdownHook("bad", fail_shutdown))

    await mgr.start()
    await mgr.shutdown()

    results = mgr.shutdown_results
    assert len(results) == 2
    # reverse topological order: "bad" inserted after "good" but runs first
    bad_result = results[0] if results[0]["name"] == "bad" else results[1]
    assert bad_result["name"] == "bad"
    assert bad_result["success"] is False
    assert "shutdown fail" in bad_result["error"]


@pytest.mark.asyncio
async def test_get_results_contains_hook_results():
    mgr = LifecycleManager()

    async def noop():
        pass

    mgr.register_startup_hook(StartupHook("a", noop))
    mgr.register_shutdown_hook(ShutdownHook("z", noop))

    await mgr.start()
    info = mgr.get_results()
    assert "startup_results" in info["hooks"]
    assert "shutdown_results" in info["hooks"]
    assert len(info["hooks"]["startup_results"]) == 1

    await mgr.shutdown()
    info = mgr.get_results()
    assert len(info["hooks"]["shutdown_results"]) == 1


@pytest.mark.asyncio
async def test_startup_results_empty_before_start():
    mgr = LifecycleManager()
    assert mgr.startup_results == []


@pytest.mark.asyncio
async def test_shutdown_results_empty_before_shutdown():
    mgr = LifecycleManager()
    assert mgr.shutdown_results == []
