"""AI tests — models, learning buffer, agent loop with a fake backend."""

import asyncio
import base64
import os
import tempfile

from avion.ai import (
    Action,
    ActionType,
    Agent,
    AgentConfig,
    CompositeModel,
    EchoModel,
    Experience,
    ExperienceBuffer,
    FeedbackLoop,
    RuleBasedModel,
    Step,
    Trajectory,
)
from test_autoclicker import FakeBackend


def run(coro):
    return asyncio.run(coro)


class AgentBackend(FakeBackend):
    """FakeBackend plus screenshot bytes the agent loop needs."""

    async def screenshot(self, path=None):
        return base64.b64decode("iVBORw0KGgo=")


class TestModels:
    def test_action_roundtrip(self):
        a = Action(ActionType.MOUSE_CLICK, {"x": 1}, reasoning="r", confidence=0.5)
        assert Action.from_dict(a.to_dict()) == a

    def test_echo_then_done(self):
        m = EchoModel([Action(ActionType.WAIT, {"ms": 1})])
        assert run(m.predict_action("t", "")).action_type == ActionType.WAIT
        assert run(m.predict_action("t", "")).action_type == ActionType.DONE

    def test_rule_based_clicks_button(self):
        m = RuleBasedModel()
        tree = {
            "role": "root",
            "children": [
                {"role": "button", "bbox": {"x": 3, "y": 4}},
            ],
        }
        a = run(m.predict_action("t", "", accessibility_tree=tree))
        assert a.action_type == ActionType.MOUSE_CLICK
        assert (a.params["x"], a.params["y"]) == (3, 4)

    def test_rule_based_no_tree_screenshots(self):
        a = run(RuleBasedModel().predict_action("t", ""))
        assert a.action_type == ActionType.SCREENSHOT

    def test_composite_first_match_wins(self):
        m = CompositeModel()
        m.add_model("echo", EchoModel([Action(ActionType.WAIT)]), condition=lambda ctx: False)
        m.add_model("rules", RuleBasedModel())
        a = run(m.predict_action("t", ""))
        assert a.action_type == ActionType.SCREENSHOT

    def test_trajectory_success_from_reward(self):
        t = Trajectory(task="t")
        t.add_step(Step(step_number=0, reward=0.1))
        t.add_step(Step(step_number=1, reward=1.0, done=True))
        assert t.success is True
        assert t.total_reward == 1.1


class TestLearning:
    def test_buffer_filters(self):
        b = ExperienceBuffer()
        b.add(Experience(task="a", reward=1.0, done=True))
        b.add(Experience(task="a", reward=-1.0, done=True))
        b.add(Experience(task="b", reward=0.2))
        assert len(b.get_by_task("a")) == 2
        assert len(b.get_high_reward()) == 1
        assert len(b.get_recent(2)) == 2

    def test_trajectory_reconstruction(self):
        b = ExperienceBuffer()
        b.add(Experience(task="a", reward=0.1))
        b.add(Experience(task="a", reward=1.0, done=True))
        trajs = b.get_trajectories("a")
        assert len(trajs) == 1 and trajs[0].success is True
        assert b.successful_trajectories("b") == []

    def test_stats(self):
        b = ExperienceBuffer()
        assert b.stats() == {"total": 0}
        b.add(Experience(task="a", reward=1.0))
        b.add(Experience(task="a", reward=-1.0))
        assert b.stats()["success_rate"] == 0.5

    def test_save_load_roundtrip(self):
        b = ExperienceBuffer()
        b.add(Experience(task="a", action=Action(ActionType.DONE), reward=1.0, done=True))
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "exp.json")
            b.save(p)
            b2 = ExperienceBuffer()
            b2.load(p)
            assert b2.size == 1
            assert b2.get_recent()[0].action.action_type == ActionType.DONE

    def test_feedback_patterns(self):
        f = FeedbackLoop()
        step = Step(
            step_number=0, action=Action(ActionType.MOUSE_CLICK), observation="missed popup"
        )
        f.record_feedback(step, -1.0, "wrong target")
        f.record_feedback(step, -1.0)
        f.record_correction(step, Action(ActionType.DONE))
        assert len(f.failure_patterns()) == 1
        assert len(f.improvement_suggestions()) == 1
        assert f.stats()["corrections"] == 1
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "fb.json")
            f.save(p)
            f2 = FeedbackLoop()
            f2.load(p)
            assert f2.stats()["total"] == 3


class TestAgent:
    def _agent(self, actions, **kw):
        import tempfile

        cfg = AgentConfig(
            max_steps=10,
            action_delay_ms=0,
            screenshot_on_each_step=False,
            output_dir=tempfile.mkdtemp(prefix="arken-traj-"),
        )
        agent = Agent(model=EchoModel(actions), config=cfg, **kw)
        run(agent.start(backend=AgentBackend()))
        return agent

    def test_run_click_type_done(self):
        agent = self._agent(
            [
                Action(ActionType.MOUSE_CLICK, {"x": 5, "y": 6}, reasoning="click it"),
                Action(ActionType.KEYBOARD_TYPE, {"text": "hi"}, reasoning="type it"),
                Action(ActionType.DONE, reasoning="all good"),
            ]
        )
        result = run(agent.run("do things"))
        run(agent.stop())
        assert result.success is True
        assert result.steps_taken == 3
        assert agent.experience_buffer.size == 3
        assert "SUCCESS" in result.summary()

    def test_fail_action_marks_failed(self):
        agent = self._agent([Action(ActionType.FAIL, reasoning="stuck")])
        result = run(agent.run("t"))
        run(agent.stop())
        assert result.success is False
        assert result.error == "stuck"

    def test_model_error_becomes_fail(self):
        class Broken:
            model_name = "broken"
            supports_vision = False

            async def predict_action(self, **kw):
                raise RuntimeError("down")

        agent = Agent(
            model=Broken(),
            config=AgentConfig(max_steps=2, action_delay_ms=0, screenshot_on_each_step=False),
        )
        run(agent.start(backend=AgentBackend()))
        result = run(agent.run("t"))
        run(agent.stop())
        assert result.success is False

    def test_callbacks_and_feedback(self):
        agent = self._agent([Action(ActionType.DONE)])
        seen = []
        agent.on_step(seen.append)
        agent.on_step(lambda s: 1 / 0)
        run(agent.run("t"))
        run(agent.stop())
        assert len(seen) == 1
        agent.feedback(seen[0], 1.0, "nice")
        agent.correct(seen[0], Action(ActionType.DONE))

    def test_vision_helpers(self):
        agent = self._agent([])
        raw = base64.b64encode(b"fakepng").decode()
        info = agent.analyze_screenshot(raw)
        assert info["size_bytes"] == 7 and len(info["hash"]) == 16
        assert agent.pixel_diff(raw, raw) == 0.0

    def test_save_load_learning(self):
        agent = self._agent([Action(ActionType.DONE)])
        run(agent.run("t"))
        run(agent.stop())
        with tempfile.TemporaryDirectory() as d:
            agent.save_learning(d)
            agent.load_learning(d)
            assert agent.experience_buffer.size == 1


class TestToolCards:
    def test_every_action_has_card(self):
        from avion.ai.models import ACTION_CARDS, ActionCard

        assert set(ACTION_CARDS) == set(ActionType)
        for card in ACTION_CARDS.values():
            assert isinstance(card, ActionCard) and card.description

    def test_validate_ok(self):
        from avion.ai.models import validate_action

        assert validate_action(Action(ActionType.MOUSE_CLICK, {"x": 1, "y": 2})) is None

    def test_validate_missing_params(self):
        from avion.ai.models import validate_action

        err = validate_action(Action(ActionType.MOUSE_CLICK, {}))
        assert err is not None and "x" in err and "y" in err

    def test_validate_bad_types(self):
        from avion.ai.models import validate_action

        assert "number" in validate_action(Action(ActionType.MOUSE_CLICK, {"x": "a", "y": 1}))
        assert "non-empty list" in validate_action(Action(ActionType.KEYBOARD_HOTKEY, {"keys": []}))
        assert "empty" in validate_action(Action(ActionType.NAVIGATE, {"url": "  "}))

    def test_cards_text(self):
        from avion.ai.models import tool_cards_text

        assert "mouse_click" in tool_cards_text()


class TestStopRules:
    def _agent(self, actions, **cfg_kw):
        import tempfile

        cfg = AgentConfig(
            action_delay_ms=0,
            screenshot_on_each_step=False,
            output_dir=tempfile.mkdtemp(prefix="arken-traj-"),
            **cfg_kw,
        )
        agent = Agent(model=EchoModel(actions), config=cfg)
        run(agent.start(backend=AgentBackend()))
        return agent

    def test_no_progress_stops(self):
        click = Action(ActionType.MOUSE_CLICK, {"x": 1, "y": 1})
        agent = self._agent([click] * 5, max_steps=10, no_progress_limit=3)
        result = run(agent.run("looping"))
        run(agent.stop())
        assert result.stop_reason == "no_progress"
        assert result.steps_taken == 2
        assert result.success is False

    def test_deadline_stops(self):
        agent = self._agent([Action(ActionType.WAIT, {"ms": 1})], max_steps=10, deadline_ms=0)
        result = run(agent.run("slow"))
        run(agent.stop())
        assert result.stop_reason == "deadline"
        assert result.steps_taken == 0

    def test_max_steps_reason(self):
        agent = self._agent(
            [Action(ActionType.WAIT, {"ms": 1})] * 5, max_steps=2, no_progress_limit=0
        )
        result = run(agent.run("long"))
        run(agent.stop())
        assert result.stop_reason == "max_steps"
        assert result.steps_taken == 2

    def test_done_and_fail_reasons(self):
        agent = self._agent([Action(ActionType.DONE)])
        assert run(agent.run("t")).stop_reason == "done"
        run(agent.stop())
        agent = self._agent([Action(ActionType.FAIL, reasoning="x")])
        assert run(agent.run("t")).stop_reason == "fail"
        run(agent.stop())

    def test_invalid_action_rejected_not_executed(self):
        agent = self._agent(
            [
                Action(ActionType.MOUSE_CLICK, {}, reasoning="sloppy"),
                Action(ActionType.DONE),
            ]
        )
        result = run(agent.run("t"))
        run(agent.stop())
        first = result.trajectory.steps[0]
        assert first.observation.startswith("Invalid action:")
        assert first.reward == -0.5
        assert result.success is True

    def test_jsonl_transcript(self):
        import json

        agent = self._agent(
            [
                Action(ActionType.KEYBOARD_TYPE, {"text": "hi"}),
                Action(ActionType.DONE),
            ]
        )
        run(agent.run("transcript me!"))
        run(agent.stop())
        path = agent._config.output_dir + "/transcript_me.jsonl"
        lines = open(path).read().splitlines()
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first["action"]["action_type"] == "keyboard_type"
        assert first["observation"] == "Typed 'hi'"


class TestVerifier:
    def test_url_and_text_checks(self):
        from avion.ai.verifier import GoalCheck, Verifier

        v = Verifier()
        r = run(
            v.verify(
                [
                    GoalCheck("url_contains", "/chat"),
                    GoalCheck("text_present", "hi"),
                    {"kind": "text_absent", "value": "bye"},
                ],
                url="http://x/chat",
                text="hi there",
            )
        )
        assert r.passed is True and r.checked == 3

    def test_failures_listed(self):
        from avion.ai.verifier import GoalCheck, Verifier

        r = run(
            Verifier().verify(
                [GoalCheck("url_contains", "/missing"), GoalCheck("unknown_kind", "x")],
                url="http://x/chat",
                text="hi",
            )
        )
        assert r.passed is False and len(r.failed) == 2

    def test_element_present(self):
        from avion.ai.verifier import GoalCheck, Verifier
        from avion.core.element import ElementLocator

        backend = AgentBackend()

        async def find(locator):
            return await backend.find_element(locator)

        ok = run(
            Verifier().verify(
                [GoalCheck("element_present", ElementLocator.text("Start").describe())],
                find=find,
            )
        )
        assert ok.passed is True
        missing = run(Verifier().verify([GoalCheck("element_present", "css=.nope")], find=find))
        assert missing.passed is False

    def test_goal_pass_marks_done(self):
        agent = TestAgent()._agent([Action(ActionType.DONE)])
        agent._backend.url = "http://x/chat"
        result = run(agent.run("t", context={"goal": [{"kind": "url_contains", "value": "/chat"}]}))
        run(agent.stop())
        assert result.success is True
        assert result.stop_reason == "done"

    def test_goal_fail_marks_unverified(self):
        agent = TestAgent()._agent([Action(ActionType.DONE)])
        agent._backend.url = "http://x/other"
        result = run(agent.run("t", context={"goal": [{"kind": "url_contains", "value": "/chat"}]}))
        run(agent.stop())
        assert result.success is False
        assert result.stop_reason == "unverified"
        assert "Unverified" in result.trajectory.steps[0].observation
