"""Skills tests — distill, persist, find, replay through Agent."""

import asyncio
import json
import os
import tempfile
from pathlib import Path

import pytest
from avion.ai import Agent, AgentConfig
from avion.ai.models import Action, ActionType, Step, Trajectory
from avion.ai.skills import SCHEMA_VERSION, Skill, SkillLibrary, skill_filename
from test_autoclicker import FakeBackend


def run(coro):
    return asyncio.run(coro)


def trajectory():
    t = Trajectory(task="open chat")
    t.add_step(Step(step_number=0,
                    action=Action(ActionType.NAVIGATE, {"url": "http://x/chat"}),
                    observation="nav", reward=0.0))
    t.add_step(Step(step_number=1, action=Action(ActionType.WAIT, {"ms": 5}),
                    observation="wait", reward=0.0))
    t.add_step(Step(step_number=2,
                    action=Action(ActionType.KEYBOARD_TYPE, {"text": "hi"}),
                    observation="typed", reward=0.1))
    t.add_step(Step(step_number=3, action=Action(ActionType.DONE),
                    observation="ok", reward=1.0, done=True))
    t.success = True
    return t


class TestDistill:
    def test_skips_waits_by_default(self):
        lib = SkillLibrary()
        skill = lib.add_from_trajectory(trajectory(), "open_chat", "go to chat")
        assert [a.action_type for a in skill.actions] == [
            ActionType.NAVIGATE, ActionType.KEYBOARD_TYPE, ActionType.DONE]
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


# ── invariants ──────────────────────────────────────────────────────────


class TestSuccessRate:
    def test_unproven_is_none(self):
        assert Skill(name="s").success_rate is None

    def test_record_replay(self):
        skill = Skill(name="s")
        skill.record_replay(True)
        skill.record_replay(True)
        skill.record_replay(False)
        assert skill.success_rate == pytest.approx(2 / 3)

    def test_all_failures_rate_zero(self):
        skill = Skill(name="s")
        skill.record_replay(False)
        assert skill.success_rate == 0.0

    def test_replay_start_marks_use_not_outcome(self):
        skill = Skill(name="s", actions=[Action(ActionType.DONE)])
        skill.to_echo_model()
        assert skill.uses == 1
        assert skill.success_rate is None  # no outcome recorded yet


class TestFindRanksBySuccess:
    def test_success_rate_breaks_keyword_ties(self):
        lib = SkillLibrary()
        lib.add(Skill(name="chat_proven", description="chat page"))
        lib.add(Skill(name="chat_failing", description="chat page"))
        lib.get("chat_proven").record_replay(True)
        lib.get("chat_failing").record_replay(False)
        assert [s.name for s in lib.find("chat page")] == [
            "chat_proven",
            "chat_failing",
        ]

    def test_unproven_sits_between_proven_and_failing(self):
        lib = SkillLibrary()
        for name in ("proven", "fresh", "failing"):
            lib.add(Skill(name=name, description="chat page"))
        lib.get("proven").record_replay(True)
        lib.get("failing").record_replay(False)
        assert [s.name for s in lib.find("chat")] == [
            "proven",
            "fresh",
            "failing",
        ]

    def test_relevance_still_beats_success_rate(self):
        lib = SkillLibrary()
        lib.add(Skill(name="chat", description="chat page"))  # matches both words
        lib.add(Skill(name="other", description="chat"))  # matches one, proven
        lib.get("other").record_replay(True)
        assert lib.find("chat page")[0].name == "chat"


class TestTrajectoryDedup:
    def test_same_trajectory_twice_returns_existing(self):
        lib = SkillLibrary()
        first = lib.add_from_trajectory(trajectory(), "first_name")
        second = lib.add_from_trajectory(trajectory(), "second_name")
        assert second is first
        assert len(lib) == 1

    def test_different_task_is_a_different_skill(self):
        lib = SkillLibrary()
        lib.add_from_trajectory(trajectory(), "a")
        other = trajectory()
        other.task = "a different task"
        second = lib.add_from_trajectory(other, "b")
        assert second is not lib.get("a")
        assert len(lib) == 2

    def test_skip_waits_changes_the_hash(self):
        lib = SkillLibrary()
        kept = lib.add_from_trajectory(trajectory(), "a")
        lib.clear()
        skipped = lib.add_from_trajectory(trajectory(), "a", skip_waits=False)
        assert kept.source_hash != skipped.source_hash
        assert len(lib) == 1  # each variant dedups against itself only

    def test_hash_is_set_and_stable(self):
        lib = SkillLibrary()
        first = lib.add_from_trajectory(trajectory(), "a")
        assert first.source_hash and len(first.source_hash) == 16
        lib.clear()
        again = lib.add_from_trajectory(trajectory(), "b")
        assert again.source_hash == first.source_hash


class TestSchemaVersion:
    def test_save_writes_current_version(self):
        with tempfile.TemporaryDirectory() as d:
            lib = SkillLibrary(d)
            lib.add_from_trajectory(trajectory(), "s")
            lib.save()
            data = json.loads(Path(d, "s.json").read_text())
        assert data["schema_version"] == SCHEMA_VERSION

    def test_versionless_v0_file_migrates(self):
        skill = Skill.from_dict({"name": "legacy", "actions": []})
        assert skill.successes == 0
        assert skill.failures == 0
        assert skill.source_hash == ""

    def test_newer_version_is_rejected(self):
        with pytest.raises(ValueError, match="newer than this build"):
            Skill.from_dict({"schema_version": SCHEMA_VERSION + 1, "name": "x"})

    def test_bad_version_type_is_rejected(self):
        with pytest.raises(ValueError, match="must be an int"):
            Skill.from_dict({"schema_version": "two", "name": "x"})

    def test_corrupt_file_is_skipped_loudly(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "broken.json").write_text("{not json")
            lib = SkillLibrary(d)
            assert lib.load() == 0
            assert [e["file"] for e in lib.load_errors] == ["broken.json"]
            assert lib.load_errors[0]["reason"]

    def test_too_new_file_is_skipped_loudly(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "future.json").write_text(
                json.dumps({"schema_version": SCHEMA_VERSION + 1, "name": "x"})
            )
            lib = SkillLibrary(d)
            assert lib.load() == 0
            assert "newer" in lib.load_errors[0]["reason"]
            assert lib.get("x") is None

    def test_load_errors_reset_between_loads(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "broken.json").write_text("{not json")
            lib = SkillLibrary(d)
            lib.load()
            assert lib.load_errors
            empty = tempfile.mkdtemp()
            lib.load(empty)
            assert lib.load_errors == []


class TestAtomicSave:
    def test_no_temp_files_left_behind(self):
        with tempfile.TemporaryDirectory() as d:
            lib = SkillLibrary(d)
            lib.add_from_trajectory(trajectory(), "s")
            lib.save()
            assert not list(Path(d).glob("*.tmp"))
            assert Path(d, "s.json").exists()

    def test_resave_replaces_in_place(self):
        with tempfile.TemporaryDirectory() as d:
            lib = SkillLibrary(d)
            lib.add_from_trajectory(trajectory(), "s")
            lib.save()
            lib.get("s").description = "updated"
            lib.save()
            lib2 = SkillLibrary(d)
            assert lib2.load() == 1
            assert lib2.get("s").description == "updated"
            assert not list(Path(d).glob("*.tmp"))

    def test_hostile_name_cannot_escape_the_directory(self):
        with tempfile.TemporaryDirectory() as d:
            lib = SkillLibrary(d)
            lib.add(Skill(name="../escape"))
            lib.save()
            files = [p.name for p in Path(d).iterdir()]
            assert files == ["escape.json"]
            assert skill_filename("../escape") == "escape.json"
