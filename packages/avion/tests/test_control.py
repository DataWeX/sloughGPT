"""Control-plane tests — a running agent is directed from outside its loop.

The contract locked in here (the inbox is the only way outside state
reaches the loop):

* commands are typed, validated on entry (``Command.from_dict``), and
  applied **at step boundaries only** — the agent loop stays the single
  writer of its own state (``ControlState`` / ``AgentConfig``);
* ``pause`` holds the loop at a boundary and **stops the deadline
  clock** — being paused must never age a run into ``stop_reason
  "deadline"``;
* ``stop`` ends the run as ``stop_reason "stopped"`` without executing
  another action;
* ``set_instructions`` / ``inject_message`` reach the model inside the
  task text it sees on the *next* prediction;
* ``set_params`` merges into ``AgentConfig`` with type checking —
  unknown or mistyped params are rejected and reported, never applied.
"""

from __future__ import annotations

import asyncio
import base64
import tempfile
import time

import pytest
from avion.ai import Action, ActionType, Agent, AgentConfig, EchoModel
from avion.ai.control import Command, CommandInbox, CommandKind, ControlState
from test_autoclicker import FakeBackend


def run(coro):
    return asyncio.run(coro)


class AgentBackend(FakeBackend):
    """FakeBackend plus screenshot bytes the agent loop needs."""

    async def screenshot(self, path=None):
        return base64.b64decode("iVBORw0KGgo=")


class RecordingModel:
    """Records the task text each prediction sees; answers from a queue."""

    model_name = "recording"
    supports_vision = False

    def __init__(self, actions=None):
        self.tasks: list[str] = []
        self._actions = list(actions or [Action(ActionType.DONE)])

    async def predict_action(self, **kwargs):
        self.tasks.append(kwargs.get("task", ""))
        if len(self.tasks) <= len(self._actions):
            return self._actions[len(self.tasks) - 1]
        return Action(ActionType.DONE, reasoning="queue exhausted")


def _config(**kwargs) -> AgentConfig:
    defaults = dict(
        max_steps=10,
        action_delay_ms=0,
        screenshot_on_each_step=False,
        output_dir=tempfile.mkdtemp(prefix="avion-control-"),
    )
    defaults.update(kwargs)
    return AgentConfig(**defaults)


def _agent(inbox, model=None, **cfg_kwargs) -> Agent:
    agent = Agent(
        model=model if model is not None else EchoModel([Action(ActionType.DONE)]),
        config=_config(**cfg_kwargs),
        inbox=inbox,
    )
    run(agent.start(backend=AgentBackend()))
    return agent


async def _wait_until(predicate, timeout=2.0):
    """Poll until predicate() is truthy (bounded — never hangs a test)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return predicate()


class TestCommandValidation:
    def test_roundtrip_and_defaults(self):
        cmd = Command.from_dict({"kind": "pause"})
        assert cmd.kind is CommandKind.PAUSE
        assert cmd.payload == {}
        assert cmd.to_dict() == {"kind": "pause", "payload": {}}

    @pytest.mark.parametrize(
        "bad",
        [
            "not an object",
            {},
            {"kind": "explode"},
            {"kind": "set_instructions", "payload": {}},
            {"kind": "set_instructions", "payload": {"text": ""}},
            {"kind": "inject_message", "payload": {"text": 42}},
            {"kind": "set_params", "payload": "max_steps=3"},
        ],
    )
    def test_rejects_malformed(self, bad):
        with pytest.raises(ValueError):
            Command.from_dict(bad)

    def test_drain_order_and_clearing(self):
        inbox = CommandInbox()
        inbox.submit(Command(CommandKind.PAUSE))
        inbox.submit(Command(CommandKind.RESUME))
        assert inbox.pending == 2
        records = inbox.drain(ControlState(), _config())
        assert [r["kind"] for r in records] == ["pause", "resume"]
        assert inbox.pending == 0
        assert inbox.drain(ControlState(), _config()) == []


class TestAgentIntegration:
    def test_instructions_reach_the_model_at_next_step(self):
        model = RecordingModel()
        inbox = CommandInbox()
        inbox.submit_dict(
            {"kind": "set_instructions", "payload": {"text": "only inspect the header"}}
        )
        agent = _agent(inbox, model=model)
        result = run(agent.run("browse the page"))
        assert result.stop_reason == "done"
        assert model.tasks, "the model was never consulted"
        assert "only inspect the header" in model.tasks[0]
        assert "browse the page" in model.tasks[0], "base task must stay in the prompt"

    def test_injected_messages_join_the_task_text(self):
        model = RecordingModel()
        inbox = CommandInbox()
        inbox.submit_dict({"kind": "inject_message", "payload": {"text": "skip the login"}})
        agent = _agent(inbox, model=model)
        run(agent.run("do things"))
        assert "skip the login" in model.tasks[0]

    def test_set_params_merges_at_boundary_and_rejects_bad_keys(self):
        inbox = CommandInbox()
        inbox.submit_dict({"kind": "set_params", "payload": {"max_steps": 2, "bogus": 1}})
        agent = _agent(inbox)
        run(agent.run("t"))
        assert agent.config.max_steps == 2, "set_params did not reach AgentConfig"
        records = agent.control_records
        param_records = [r for r in records if r["kind"] == "set_params"]
        assert param_records, "no set_params record was kept for the log"
        rejected = param_records[-1]["payload"]["rejected"]
        assert any("bogus" in entry for entry in rejected), rejected

    def test_set_params_type_check(self):
        inbox = CommandInbox()
        inbox.submit_dict({"kind": "set_params", "payload": {"max_steps": "ten"}})
        agent = _agent(inbox)
        run(agent.run("t"))
        assert agent.config.max_steps == 10, "a mistyped param must not be applied"
        rejected = agent.control_records[-1]["payload"]["rejected"]
        assert any("max_steps" in entry for entry in rejected), rejected

    def test_stop_ends_run_before_any_action(self):
        inbox = CommandInbox()
        inbox.submit(Command(CommandKind.STOP))
        model = RecordingModel([Action(ActionType.MOUSE_CLICK, {"x": 1, "y": 1})])
        agent = _agent(inbox, model=model)
        result = run(agent.run("t"))
        assert result.stop_reason == "stopped"
        assert result.steps_taken == 0, "a stopped run must not act"
        assert model.tasks == [], "the model must not be consulted after a stop"

    def test_pause_holds_the_loop_until_resume(self):
        inbox = CommandInbox()
        inbox.submit(Command(CommandKind.PAUSE))
        agent = _agent(inbox)

        async def scenario():
            task = asyncio.ensure_future(agent.run("t"))
            assert await _wait_until(lambda: agent.control.paused), "pause never took effect"
            assert not task.done(), "the loop kept running while paused"
            inbox.submit(Command(CommandKind.RESUME))
            return await task

        result = run(scenario())
        assert result.stop_reason == "done", result.stop_reason
        assert result.steps_taken == 1, "the run must complete after resume"
        assert not agent.control.paused

    def test_pause_stops_the_deadline_clock(self):
        inbox = CommandInbox()
        inbox.submit(Command(CommandKind.PAUSE))
        # deadline is 300ms; we stay paused far longer than that. The run
        # must resume with its budget intact instead of reporting "deadline".
        agent = _agent(inbox, deadline_ms=300.0)

        async def scenario():
            task = asyncio.ensure_future(agent.run("t"))
            assert await _wait_until(lambda: agent.control.paused), "never paused"
            await asyncio.sleep(0.6)  # > 2x the deadline, spent paused
            inbox.submit(Command(CommandKind.RESUME))
            return await task

        result = run(scenario())
        assert result.stop_reason == "done", (
            f"pause aged the run into {result.stop_reason!r} — the clock did not stop"
        )

    def test_stop_during_pause_releases_the_hold(self):
        inbox = CommandInbox()
        inbox.submit(Command(CommandKind.PAUSE))
        agent = _agent(inbox)

        async def scenario():
            task = asyncio.ensure_future(agent.run("t"))
            assert await _wait_until(lambda: agent.control.paused)
            inbox.submit(Command(CommandKind.STOP))
            return await task

        result = run(scenario())
        assert result.stop_reason == "stopped"
        assert result.steps_taken == 0

    def test_agent_without_inbox_is_unaffected(self):
        agent = Agent(model=EchoModel([Action(ActionType.DONE)]), config=_config())
        run(agent.start(backend=AgentBackend()))
        result = run(agent.run("t"))
        assert result.stop_reason == "done"
        assert agent.control_records == []
