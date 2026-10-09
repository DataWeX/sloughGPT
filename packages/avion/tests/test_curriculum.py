"""Curriculum tests — retry failures, untried tasks, extend, explore."""

from avion.ai import Experience, ExperienceBuffer, FeedbackLoop
from avion.ai.curriculum import Curriculum, TaskProposal
from avion.ai.models import Action, ActionType, Step, Trajectory
from avion.ai.skills import SkillLibrary


def trajectory(task="open chat"):
    t = Trajectory(task=task)
    t.add_step(
        Step(
            step_number=0,
            action=Action(ActionType.DONE),
            observation="ok",
            reward=1.0,
            done=True,
        )
    )
    t.success = True
    return t


class TestCurriculum:
    def test_retry_failure_first(self):
        from avion.ai.models import ActionType as AT

        fb = FeedbackLoop()
        step = Step(step_number=0, action=Action(AT.MOUSE_CLICK), observation="miss")
        fb.record_feedback(step, -1.0)
        fb.record_feedback(step, -1.0)
        proposals = Curriculum().suggest(feedback=fb)
        assert proposals[0].source == "retry-failure"
        assert "mouse_click" in proposals[0].task

    def test_untried_task(self):
        lib = SkillLibrary()
        lib.add_from_trajectory(trajectory("open chat"), "open_chat")
        proposals = Curriculum().suggest(
            known_tasks=["open chat", "search datasets"], skills=lib
        )
        assert proposals[0].source == "untried"
        assert proposals[0].task == "search datasets"

    def test_extend_mastered(self):
        lib = SkillLibrary()
        skill = lib.add_from_trajectory(trajectory("open chat"), "open_chat")
        skill.uses = 5
        proposals = Curriculum().suggest(skills=lib)
        assert proposals[0].source == "extend"
        assert "open chat" in proposals[0].task

    def test_explore_fallback(self):
        proposals = Curriculum().suggest()
        assert proposals[0].source == "explore"

    def test_max_suggestions(self):
        fb = FeedbackLoop()
        step = Step(step_number=0, action=Action(ActionType.MOUSE_CLICK), observation="x")
        for _ in range(3):
            fb.record_feedback(step, -1.0)
        step2 = Step(step_number=0, action=Action(ActionType.WAIT), observation="y")
        for _ in range(2):
            fb.record_feedback(step2, -1.0)
        proposals = Curriculum().suggest(feedback=fb, max_suggestions=2)
        assert len(proposals) == 2

    def test_proposal_roundtrip(self):
        p = TaskProposal(task="t", reason="r", goal=[{"kind": "text_present", "value": "x"}],
                         source="untried")
        assert p.to_dict()["goal"][0]["kind"] == "text_present"

    def test_uses_experience_param(self):
        buf = ExperienceBuffer()
        buf.add(Experience(task="a", reward=1.0, done=True))
        proposals = Curriculum().suggest(experience=buf)
        assert proposals[0].source == "explore"


class TestClosedLoop:
    def test_curriculum_skill_agent(self):
        """Full loop: fail -> curriculum proposes retry -> skill replays."""
        import asyncio
        import tempfile

        from avion.ai import Agent, AgentConfig

        def run(coro):
            return asyncio.run(coro)

        from test_autoclicker import FakeBackend

        class Backend(FakeBackend):
            async def screenshot(self, path=None):
                return b"png"

        # 1. Agent fails at clicking; feedback recorded.
        fb = FeedbackLoop()
        bad = Step(step_number=0, action=Action(ActionType.MOUSE_CLICK),
                   observation="missed")
        fb.record_feedback(bad, -1.0)
        fb.record_feedback(bad, -1.0)

        # 2. Curriculum proposes practice.
        proposal = Curriculum().suggest(feedback=fb)[0]
        assert proposal.source == "retry-failure"

        # 3. A skill exists; agent replays it successfully.
        lib = SkillLibrary()
        lib.add_from_trajectory(trajectory(), "open_chat")
        skill = lib.get("open_chat")
        cfg = AgentConfig(
            action_delay_ms=0,
            screenshot_on_each_step=False,
            output_dir=tempfile.mkdtemp(prefix="avion-loop-"),
        )
        agent = Agent(model=skill.to_echo_model(), config=cfg)
        run(agent.start(backend=Backend()))
        try:
            result = run(agent.run(proposal.task))
            assert result.success is True
        finally:
            run(agent.stop())
