"""Agent loop tests — bounded awaits, retries, transcript safety, containment.

These lock in the tightened-loop contract:

* every await is bounded (``step_timeout`` capped by the remaining
  deadline) — a hung model/backend can wedge nothing;
* predictions retry, side-effecting actions never do;
* transcripts are claimed per run (never clobbered) and write failures
  are counted instead of killing the run;
* ``on_step`` callbacks are contained (counted, degraded, revived).
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import tempfile
import time

import pytest
from avion.ai import Action, ActionType, Agent, AgentConfig, EchoModel
from test_autoclicker import FakeBackend

HANG = 30.0  # seconds a wedged implementation would sleep for
BOUND = 5.0  # wall-clock ceiling the tightened loop must stay under


def run(coro):
    return asyncio.run(coro)


class AgentBackend(FakeBackend):
    """FakeBackend plus screenshot bytes the agent loop needs."""

    async def screenshot(self, path=None):
        return base64.b64decode("iVBORw0KGgo=")


class HangingBackend(AgentBackend):
    """A backend whose screenshot never returns in time."""

    async def screenshot(self, path=None):
        await asyncio.sleep(HANG)
        return b"png"


class HangingModel:
    """A model that never answers in time."""

    model_name = "hanging"
    supports_vision = False

    async def predict_action(self, **kwargs):
        await asyncio.sleep(HANG)
        return Action(ActionType.DONE)


class BrokenModel:
    """A model that always raises."""

    model_name = "broken"
    supports_vision = False

    def __init__(self):
        self.calls = 0

    async def predict_action(self, **kwargs):
        self.calls += 1
        raise RuntimeError("down")


class FlakyModel:
    """Fails the first `failures` calls, then answers DONE."""

    model_name = "flaky"
    supports_vision = False

    def __init__(self, failures: int):
        self.failures = failures
        self.calls = 0

    async def predict_action(self, **kwargs):
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("blip")
        return Action(ActionType.DONE)


class EndlessClickModel:
    """Always answers with a fresh, distinct click (EchoModel is
    single-use: run2 on the same agent would get its exhaustion answer)."""

    model_name = "endless"
    supports_vision = False

    def __init__(self):
        self.i = 0

    async def predict_action(self, **kwargs):
        self.i += 1
        return Action(ActionType.MOUSE_CLICK, {"x": self.i, "y": self.i})


def _config(**kwargs) -> AgentConfig:
    defaults = dict(
        max_steps=10,
        action_delay_ms=0,
        screenshot_on_each_step=False,
        output_dir=tempfile.mkdtemp(prefix="avion-loop-"),
    )
    defaults.update(kwargs)
    return AgentConfig(**defaults)


def _agent(actions=(), model=None, backend=None, **cfg_kwargs) -> Agent:
    agent = Agent(
        model=model if model is not None else EchoModel(list(actions)),
        config=_config(**cfg_kwargs),
    )
    run(agent.start(backend=backend or AgentBackend()))
    return agent


class TestTranscriptSafety:
    def test_rerun_never_clobbers_previous_transcript(self):
        agent = _agent([Action(ActionType.DONE)])
        first = run(agent.run("same task"))
        second = run(agent.run("same task"))
        run(agent.stop())

        assert first.transcript_path != second.transcript_path
        assert os.path.basename(first.transcript_path) == "same_task.jsonl"
        assert os.path.basename(second.transcript_path) == "same_task-1.jsonl"
        with open(first.transcript_path) as fh:
            lines_first = fh.read().splitlines()
        with open(second.transcript_path) as fh:
            lines_second = fh.read().splitlines()
        # Both runs survived intact: one DONE step each, nothing truncated.
        assert len(lines_first) == 1
        assert len(lines_second) == 1
        assert json.loads(lines_first[0])["step_number"] == 0
        assert json.loads(lines_second[0])["step_number"] == 0

    def test_transcript_path_is_returned_and_parseable(self):
        agent = _agent(
            [
                Action(ActionType.KEYBOARD_TYPE, {"text": "hi"}),
                Action(ActionType.DONE),
            ]
        )
        result = run(agent.run("write it"))
        run(agent.stop())

        assert result.transcript_path.endswith("write_it.jsonl")
        assert os.path.exists(result.transcript_path)
        assert result.to_dict()["transcript_path"] == result.transcript_path
        with open(result.transcript_path) as fh:
            lines = [json.loads(line) for line in fh]
        assert len(lines) == 2
        assert lines[0]["action"]["action_type"] == "keyboard_type"
        assert lines[0]["observation"] == "Typed 'hi'"
        assert result.transcript_failures == 0

    def test_save_disabled_writes_nothing(self):
        agent = _agent([Action(ActionType.DONE)], save_trajectories=False)
        result = run(agent.run("t"))
        run(agent.stop())
        assert result.transcript_path == ""
        assert os.listdir(agent._config.output_dir) == []  # dir exists, nothing in it

    def test_transcript_write_failure_is_counted_not_fatal(self, monkeypatch):
        class BrokenWriter:
            def write(self, *_args):
                raise OSError("disk full")

            def flush(self):
                raise OSError("disk full")

            def close(self):
                pass

        monkeypatch.setattr(os, "fdopen", lambda *_a, **_k: BrokenWriter())

        agent = _agent([Action(ActionType.DONE)])
        result = run(agent.run("t"))
        run(agent.stop())

        assert result.success is True  # the run survived its own log path
        assert result.transcript_failures == 1
        assert result.to_dict()["transcript_failures"] == 1


class TestModelRetries:
    def test_failure_retries_then_fails(self):
        model = BrokenModel()
        agent = _agent(model=model, max_steps=3, retry_on_failure=True, max_retries=2)
        start = time.monotonic()
        result = run(agent.run("t"))
        run(agent.stop())

        assert model.calls == 3  # 1 attempt + max_retries
        assert result.stop_reason == "fail"
        assert result.success is False
        assert result.error.startswith("Model error")
        assert time.monotonic() - start < BOUND

    def test_no_retry_when_disabled(self):
        model = BrokenModel()
        agent = _agent(model=model, retry_on_failure=False, max_retries=5)
        run(agent.run("t"))
        run(agent.stop())
        assert model.calls == 1

    def test_transient_failure_recovers_on_retry(self):
        model = FlakyModel(failures=2)
        agent = _agent(model=model, retry_on_failure=True, max_retries=2)
        result = run(agent.run("t"))
        run(agent.stop())

        assert model.calls == 3
        assert result.success is True
        assert result.stop_reason == "done"


class TestBoundedAwaits:
    def test_hanging_model_reports_step_timeout(self):
        # step_timeout (0.15s) is what fires; the run still has budget
        # left, so the failure is recorded and reported as a model error.
        agent = _agent(
            model=HangingModel(),
            deadline_ms=3000,
            step_timeout=0.15,
            retry_on_failure=False,
        )
        start = time.monotonic()
        result = run(agent.run("t"))
        wall = time.monotonic() - start
        run(agent.stop())

        assert result.stop_reason == "fail"  # timeout reported as model error
        assert result.error.startswith("Model error")
        assert "no result within" in result.error
        assert wall < BOUND  # before: slept the full 30s, unbounded

    def test_deadline_cuts_through_a_hung_model(self):
        # The step timeout is far beyond the run deadline: the deadline
        # itself must end the run (retries and awaits can't outlive it).
        agent = _agent(
            model=HangingModel(),
            deadline_ms=300,
            step_timeout=HANG,
            retry_on_failure=False,
        )
        start = time.monotonic()
        result = run(agent.run("t"))
        wall = time.monotonic() - start
        run(agent.stop())

        assert result.stop_reason == "deadline"
        assert result.error == "deadline exceeded"
        assert wall < BOUND

    def test_deadline_wins_over_retries(self):
        agent = _agent(
            model=HangingModel(),
            deadline_ms=200,
            step_timeout=HANG,
            retry_on_failure=True,
            max_retries=50,
        )
        start = time.monotonic()
        result = run(agent.run("t"))
        wall = time.monotonic() - start
        run(agent.stop())

        assert result.stop_reason == "deadline"
        assert result.error == "deadline exceeded"
        assert wall < BOUND  # retries must never outlive the budget

    def test_step_timeout_disabled_still_bounded_by_deadline(self):
        agent = _agent(
            model=HangingModel(),
            step_timeout=0,  # disabled: bound by deadline only
            deadline_ms=150,
            retry_on_failure=False,
        )
        start = time.monotonic()
        result = run(agent.run("t"))
        wall = time.monotonic() - start
        run(agent.stop())
        assert wall < BOUND
        assert result.stop_reason in ("fail", "deadline")

    def test_hanging_action_is_recorded_not_reissued(self):
        agent = _agent(
            [
                Action(ActionType.WAIT, {"ms": int(HANG * 1000)}),
                Action(ActionType.DONE),
            ],
            step_timeout=0.1,
            max_steps=3,
        )
        start = time.monotonic()
        result = run(agent.run("t"))
        run(agent.stop())

        first_step = result.trajectory.steps[0]
        assert first_step.observation.startswith("Timeout:")
        assert first_step.reward == -0.5
        assert result.success is True  # loop continued to the DONE step
        assert time.monotonic() - start < BOUND

    def test_hanging_capture_degrades_instead_of_wedging(self):
        agent = Agent(
            model=EchoModel([Action(ActionType.DONE)]),
            config=_config(step_timeout=0.1, screenshot_on_each_step=True),
        )
        run(agent.start(backend=HangingBackend()))
        start = time.monotonic()
        result = run(agent.run("t"))
        run(agent.stop())

        assert result.success is True
        assert result.trajectory.steps[0].screenshot_b64 == ""  # degraded frame
        assert time.monotonic() - start < BOUND


class TestCallbackContainment:
    def _clicks(self, n: int):
        return [Action(ActionType.MOUSE_CLICK, {"x": i, "y": i}) for i in range(n)]

    def test_failures_counted_then_degraded(self):
        agent = _agent(
            self._clicks(5),
            max_steps=5,
            no_progress_limit=0,
            callback_degrade_after=2,
        )
        healthy: list = []
        agent.on_step(healthy.append)
        agent.on_step(lambda _s: 1 / 0)

        result = run(agent.run("t"))
        run(agent.stop())

        assert result.steps_taken == 5
        assert len(healthy) == 5  # one bad observer never starves the good one
        status = agent.callback_status()
        assert status[0] == {"index": 0, "failures": 0, "disabled": False}
        assert status[1] == {"index": 1, "failures": 2, "disabled": True}

    def test_degraded_callback_revives_on_the_next_run(self):
        agent = _agent(
            model=EndlessClickModel(),  # distinct actions every run
            max_steps=4,
            no_progress_limit=0,
            callback_degrade_after=1,
        )
        healthy: list = []
        agent.on_step(healthy.append)
        agent.on_step(lambda _s: 1 / 0)

        run(agent.run("t"))
        assert agent.callback_status()[1]["disabled"] is True
        run(agent.run("t"))  # fresh run: degraded callback gets a new chance
        run(agent.stop())

        assert len(healthy) == 8  # both runs reached both callbacks
        status = agent.callback_status()[1]
        assert status["disabled"] is True  # failed again, degraded again
        assert status["failures"] == 2  # once per run (degrade_after=1 skips the rest)

    def test_callbacks_reset_state_per_run(self):
        agent = _agent([Action(ActionType.DONE)], callback_degrade_after=1)
        agent.on_step(lambda _s: 1 / 0)
        run(agent.run("t"))
        # Consecutive/disabled counters are per-run, totals are not:
        run(agent.run("t"))
        run(agent.stop())
        status = agent.callback_status()[0]
        assert status["failures"] == 2
        assert status["disabled"] is True


class TestReentrancy:
    def test_concurrent_run_raises(self):
        class SlowModel:
            model_name = "slow"
            supports_vision = False

            async def predict_action(self, **kwargs):
                await asyncio.sleep(0.05)
                return Action(ActionType.DONE)

        async def scenario():
            agent = Agent(model=SlowModel(), config=_config(max_steps=3))
            await agent.start(backend=AgentBackend())
            first = asyncio.create_task(agent.run("first"))
            await asyncio.sleep(0.01)  # first run is now mid-flight
            with pytest.raises(RuntimeError, match="already in progress"):
                await agent.run("second")
            result = await first
            await agent.stop()
            return result

        result = run(scenario())
        assert result.success is True
        assert result.transcript_path.endswith("first.jsonl")
