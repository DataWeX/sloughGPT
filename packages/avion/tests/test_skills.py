"""Skills tests — distill, persist, find, replay through Agent."""

import asyncio
import os
import tempfile

from avion.ai import Agent, AgentConfig
from avion.ai.models import Action, ActionType, Step, Trajectory
from avion.ai.skills import Skill, SkillLibrary
from test_autoclicker import FakeBackend


def run(coro):
    return asyncio.run(coro)


def trajectory():
    t = Trajectory(task="open chat")
    t.add_step(
        Step(
            step_number=0,
            action=Action(ActionType.NAVIGATE, {"url": "http://x/chat"}),
            observation="nav",
            reward=0.0,
        )
    )
    t.add_step(
        Step(
            step_number=1, action=Action(ActionType.WAIT, {"ms": 5}), observation="wait", reward=0.0
        )
    )
    t.add_step(
        Step(
            step_number=2,
            action=Action(ActionType.KEYBOARD_TYPE, {"text": "hi"}),
            observation="typed",
            reward=0.1,
        )
    )
    t.add_step(
        Step(step_number=3, action=Action(ActionType.DONE), observation="ok", reward=1.0, done=True)
    )
    t.success = True
    return t


class TestDistill:
    def test_skips_waits_by_default(self):
        lib = SkillLibrary()
        skill = lib.add_from_trajectory(trajectory(), "open_chat", "go to chat")
        assert [a.action_type for a in skill.actions] == [
            ActionType.NAVIGATE,
            ActionType.KEYBOARD_TYPE,
            ActionType.DONE,
        ]
        assert skill.source_task == "open chat"

    def test_keep_waits_opt_in(self):
        lib = SkillLibrary()
        skill = lib.add_from_trajectory(trajectory(), "s", skip_waits=False)
        assert len(skill.actions) == 4

    def test_roundtrip(self):
        lib = SkillLibrary()
        skill = lib.add_from_trajectory(trajectory(), "s", "d")
        assert Skill.from_dict(skill.to_dict()).name == "s"


class TestLibrary:
    def test_find_ranked(self):
        lib = SkillLibrary()
        lib.add_from_trajectory(trajectory(), "open_chat", "go to chat page")
        lib.add_from_trajectory(trajectory(), "other", "unrelated thing")
        found = lib.find("chat page")
        assert found[0].name == "open_chat"  # best match first
        assert lib.find("zzz") == []

    def test_save_load(self):
        with tempfile.TemporaryDirectory() as d:
            lib = SkillLibrary(d)
            lib.add_from_trajectory(trajectory(), "open_chat", "go to chat")
            assert lib.save() == 1
            assert os.path.exists(os.path.join(d, "open_chat.json"))
            lib2 = SkillLibrary(d)
            assert lib2.load() == 1
            assert lib2.get("open_chat").description == "go to chat"

    def test_load_missing_dir(self):
        assert SkillLibrary("/nope/missing").load() == 0

    def test_delete(self):
        with tempfile.TemporaryDirectory() as d:
            lib = SkillLibrary(d)
            lib.add_from_trajectory(trajectory(), "s")
            lib.save()
            assert lib.delete("s") is True
            assert lib.delete("s") is False
            assert not os.path.exists(os.path.join(d, "s.json"))


class TestReplay:
    def test_skill_replays_through_agent(self):
        lib = SkillLibrary()
        skill = lib.add_from_trajectory(trajectory(), "open_chat")
        assert skill.uses == 0

        import tempfile

        cfg = AgentConfig(
            action_delay_ms=0,
            screenshot_on_each_step=False,
            output_dir=tempfile.mkdtemp(prefix="avion-skill-"),
        )

        class Backend(FakeBackend):
            async def screenshot(self, path=None):
                return b"png"

            async def navigate(self, url):
                self.url = url

        agent = Agent(model=skill.to_echo_model(), config=cfg)
        run(agent.start(backend=Backend()))
        try:
            result = run(agent.run("replay open chat"))
            assert result.success is True
            assert result.stop_reason == "done"
            assert agent._backend.url == "http://x/chat"
            assert agent._backend.typed == ["hi"]
            assert skill.uses == 1
        finally:
            run(agent.stop())
