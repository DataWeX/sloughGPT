"""Agent loop tests — bounded awaits, retries, transcript safety, containment.

These lock in the tightened-loop contract:

* every await is bounded (``step_timeout`` capped by the remaining
  deadline) — a hung model/backend can wedge nothing;
* predictions retry, side-effecting actions never do;
* transcripts are claimed per run (never clobbered) and write failures
  are counted instead of killing the run;
* ``on_step`` callbacks are contained (counted, degraded, revived);
* DONE/FAIL are dispatched as ``on_event`` events, and a handler's
  ``continue_run`` veto is the only thing that keeps a run alive past a
  terminal action.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import tempfile
import time

import pytest
from avion.ai import Action, ActionType, Agent, AgentConfig, AgentEvent, EchoModel
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


class RecordingModel:
    """Answers DONE whenever asked, remembering the task it was given."""

    model_name = "recording"
    supports_vision = False

    def __init__(self):
        self.tasks: list[str] = []

    async def predict_action(self, *, task, **kwargs):
        self.tasks.append(task)
        return Action(ActionType.DONE)


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

    def test_default_flush_is_per_step_crash_durable(self):
        # Default policy keeps the old guarantee: a written step is on disk
        # before the step's callbacks run — a crash loses nothing.
        cfg = _config()
        agent = Agent(model=EchoModel([Action(ActionType.DONE)]), config=cfg)
        run(agent.start(backend=AgentBackend()))
        path = os.path.join(cfg.output_dir, "durable_task.jsonl")
        seen: list[int] = []

        def probe(_step):
            with open(path) as fh:  # separate handle: the OS view, unbuffered
                seen.append(len(fh.read()))

        agent.on_step(probe)
        result = run(agent.run("durable task"))
        run(agent.stop())
        assert result.transcript_failures == 0
        assert seen and seen[0] > 0

    def test_group_commit_holds_writes_until_the_flush_boundary(self):
        # flush_steps=1000 vs 3 written steps: nothing crosses the kernel
        # boundary during the run (one write(2) per N steps is the point),
        # and close() in finally flushes the remainder — a completed run's
        # transcript is always complete; a crash loses at most the batch.
        cfg = _config(transcript_flush_steps=1000)
        agent = Agent(
            model=EchoModel(
                [
                    Action(ActionType.KEYBOARD_TYPE, {"text": "a"}),
                    Action(ActionType.KEYBOARD_TYPE, {"text": "b"}),
                    Action(ActionType.DONE),
                ]
            ),
            config=cfg,
        )
        run(agent.start(backend=AgentBackend()))
        path = os.path.join(cfg.output_dir, "probe_task.jsonl")
        seen: list[int] = []

        def probe(_step):
            with open(path) as fh:
                seen.append(len(fh.read()))

        agent.on_step(probe)
        result = run(agent.run("probe task"))
        run(agent.stop())
        assert result.transcript_failures == 0
        # Probes run after each step's transcript write — all saw 0 bytes
        # on disk: the lines were still sitting in the writer's buffer.
        assert seen == [0, 0, 0]
        with open(result.transcript_path) as fh:
            lines = [json.loads(line) for line in fh]
        assert len(lines) == 3

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


class TestTerminalEvents:
    """DONE/FAIL as events: observed by default, extended on veto."""

    def test_done_event_is_dispatched_and_the_run_still_ends(self):
        # Registering a handler changes nothing about termination: without
        # a veto the loop breaks exactly as it always did.
        events: list[AgentEvent] = []
        agent = _agent([Action(ActionType.DONE)], save_trajectories=False)
        agent.on_event(events.append)

        result = run(agent.run("t"))
        run(agent.stop())

        assert [e.kind for e in events] == ["done"]
        assert events[0].continue_run is False
        assert result.stop_reason == "done"
        assert result.success is True

    def test_continue_run_veto_loops_the_agent_through_another_exercise(self):
        agent = _agent(
            [
                Action(ActionType.MOUSE_CLICK, {"x": 1, "y": 1}),
                Action(ActionType.DONE),
                Action(ActionType.MOUSE_CLICK, {"x": 2, "y": 2}),
                Action(ActionType.DONE),
            ],
            max_steps=10,
            save_trajectories=False,
        )
        seen: list[str] = []

        def next_exercise(event: AgentEvent) -> None:
            seen.append(event.kind)
            event.continue_run = len(seen) == 1  # claim the first, end on the second

        agent.on_event(next_exercise)
        result = run(agent.run("t"))
        run(agent.stop())

        assert seen == ["done", "done"]
        assert result.steps_taken == 4  # the run did not stop at step 2
        assert result.stop_reason == "done"

    def test_fail_event_carries_the_reason_and_the_last_terminal_wins(self):
        agent = _agent(
            [
                Action(ActionType.DONE),
                Action(ActionType.FAIL, reasoning="second exercise impossible"),
            ],
            max_steps=10,
            save_trajectories=False,
        )
        events: list[AgentEvent] = []

        def handler(event: AgentEvent) -> None:
            events.append(event)
            event.continue_run = event.kind == "done"

        agent.on_event(handler)

        result = run(agent.run("t"))
        run(agent.stop())

        assert [e.kind for e in events] == ["done", "fail"]
        assert events[1].reason == "second exercise impossible"
        assert result.stop_reason == "fail"
        assert result.success is False  # continuing past DONE did not bank the win

    def test_goal_check_reason_reaches_the_done_event(self):
        agent = _agent([Action(ActionType.DONE)], save_trajectories=False)
        events: list[AgentEvent] = []
        agent.on_event(events.append)

        result = run(agent.run("t", context={"goal": [{"kind": "text_present", "value": "nope"}]}))
        run(agent.stop())

        assert events[0].kind == "done"
        assert events[0].reason  # unverified goal reports why
        assert result.stop_reason == "unverified"

    def test_event_handler_is_contained_and_degrades(self):
        agent = _agent(
            [Action(ActionType.DONE)],
            save_trajectories=False,
            callback_degrade_after=1,
        )
        agent.on_event(lambda _e: 1 / 0)

        result = run(agent.run("t"))
        run(agent.stop())

        assert result.success is True  # a broken handler never breaks the run
        status = agent.callback_status("event")[0]
        assert status["failures"] == 1
        assert status["disabled"] is True

    def test_step_and_event_callback_status_are_separate_lists(self):
        agent = _agent([Action(ActionType.DONE)], save_trajectories=False)
        agent.on_step(lambda _s: None)
        agent.on_event(lambda _e: None)

        run(agent.run("t"))
        run(agent.stop())

        assert agent.callback_status() == [{"index": 0, "failures": 0, "disabled": False}]
        assert agent.callback_status("event") == [{"index": 0, "failures": 0, "disabled": False}]

    def test_handover_reprompts_the_agent_with_the_next_exercise(self):
        model = RecordingModel()
        agent = _agent(model=model, max_steps=5, save_trajectories=False)
        claims: list[str] = []

        def next_exercise(event: AgentEvent) -> None:
            claims.append(event.kind)
            if len(claims) == 1:
                event.continue_run = True
                event.task = "two"  # the agent must work on this one now

        agent.on_event(next_exercise)
        result = run(agent.run("one"))
        run(agent.stop())

        assert model.tasks == ["one", "two"]
        assert result.task == "two"
        assert result.steps_taken == 2
        assert result.exercises_claimed == {"done": 1, "fail": 0}
        assert result.to_dict()["exercises_claimed"] == {"done": 1, "fail": 0}

    def test_goal_handover_verifies_the_new_exercise_and_spares_the_caller(self):
        agent = _agent(
            [Action(ActionType.DONE), Action(ActionType.DONE)],
            max_steps=5,
            save_trajectories=False,
        )
        seen: list[int] = []

        def next_exercise(event: AgentEvent) -> None:
            seen.append(len(seen))
            if len(seen) == 1:
                event.continue_run = True
                event.goal = [{"kind": "text_present", "value": "never on this page"}]

        agent.on_event(next_exercise)
        caller_context = {"goal": []}
        result = run(agent.run("one", context=caller_context))
        run(agent.stop())

        # First DONE had no goal to check; the handed-over goal dooms the second.
        assert result.stop_reason == "unverified"
        assert result.success is False
        assert result.exercises_claimed["done"] == 1
        assert caller_context == {"goal": []}  # the caller's dict was copied, not used


# Lives here rather than in test_curriculum.py because the runner needs a
# started Agent: AgentBackend/_agent/RecordingModel are already defined above.
class TestExerciseRunner:
    def test_one_agent_is_driven_through_every_proposal(self):
        from avion.ai.curriculum import ExerciseRunner, TaskProposal

        model = RecordingModel()
        agent = _agent(model=model, max_steps=10, save_trajectories=False)
        runner = ExerciseRunner(
            [
                TaskProposal(task="alpha", source="untried"),
                TaskProposal(task="beta", source="retry-failure"),
                TaskProposal(task="gamma", source="extend"),
            ]
        )
        agent.on_event(runner.handle)
        first = runner.start()

        result = run(agent.run(first.task))
        run(agent.stop())

        assert first.task == "alpha"
        assert model.tasks == ["alpha", "beta", "gamma"]  # re-prompted each time
        assert [r.task for r in runner.records] == ["alpha", "beta", "gamma"]
        assert [r.outcome for r in runner.records] == ["done", "done", "done"]
        assert [r.source for r in runner.records] == ["untried", "retry-failure", "extend"]
        assert [r.steps_taken for r in runner.records] == [1, 1, 1]
        assert runner.summary() == {"done": 3}
        assert result.steps_taken == 3
        assert result.task == "gamma"  # only the last exercise is on the result

    def test_a_handed_over_goal_records_unverified(self):
        from avion.ai.curriculum import ExerciseRunner, TaskProposal

        agent = _agent(
            [Action(ActionType.DONE), Action(ActionType.DONE)],
            max_steps=10,
            save_trajectories=False,
        )
        runner = ExerciseRunner(
            [
                TaskProposal(task="alpha"),
                TaskProposal(task="beta", goal=[{"kind": "text_present", "value": "never here"}]),
            ]
        )
        agent.on_event(runner.handle)
        result = run(agent.run(runner.start().task))
        run(agent.stop())

        assert [r.outcome for r in runner.records] == ["done", "unverified"]
        assert runner.records[1].reason  # says why the page disagreed
        assert runner.summary() == {"done": 1, "unverified": 1}
        assert result.stop_reason == "unverified"
        assert result.success is False

    def test_an_exercise_that_never_ended_is_still_recorded(self):
        from avion.ai.curriculum import ExerciseRunner, TaskProposal

        agent = _agent(model=EndlessClickModel(), max_steps=3, save_trajectories=False)
        runner = ExerciseRunner(
            [TaskProposal(task="alpha"), TaskProposal(task="beta", source="extend")]
        )
        agent.on_event(runner.handle)
        result = run(agent.run(runner.start().task))
        run(agent.stop())

        runner.finish(result.steps_taken, result.stop_reason)
        assert [r.outcome for r in runner.records] == ["unfinished"]
        assert runner.records[0].reason == "max_steps"
        assert runner.records[0].steps_taken == 3
        assert runner.summary() == {"unfinished": 1}

        # finish() is idempotent: the in-flight exercise is no longer open.
        runner.finish(result.steps_taken, result.stop_reason)
        assert len(runner.records) == 1

    def test_an_empty_queue_never_vetoes(self):
        from avion.ai.curriculum import ExerciseRunner

        agent = _agent([Action(ActionType.DONE)], save_trajectories=False)
        runner = ExerciseRunner([])
        agent.on_event(runner.handle)

        assert runner.start() is None
        result = run(agent.run("alpha"))
        run(agent.stop())

        assert runner.records == []  # nothing was in flight to record
        assert result.stop_reason == "done"
