"""Choice agent tests — grounding, parsing, full loop on fakes."""

import asyncio

from avion.ai.choice import (
    ChoiceAgent,
    build_choice_prompt,
    parse_pick,
    propose_candidates,
)
from test_autoclicker import FakeBackend, TestSession


def run(coro):
    return asyncio.run(coro)


class TreeBackend(FakeBackend):
    async def get_accessibility_tree(self):
        return {
            "role": "root",
            "children": [
                {"role": "button", "name": "Start"},
                {"role": "link", "name": "More"},
                {"role": "textbox", "name": "Search"},
            ],
        }


def session_with_tree():
    s = TestSession()._session()
    s._backend = TreeBackend()
    from avion.core.element import ElementFinder
    from avion.core.navigator import Navigator

    s._navigator = Navigator(s._backend, "http://x")
    s._finder = ElementFinder(s._backend)
    return s


class TestParse:
    def test_first_in_range_wins(self):
        assert parse_pick("I pick 2!", 4) == 2
        assert parse_pick("9 then 1", 4) == 1

    def test_none_when_no_digits(self):
        assert parse_pick("hello there", 4) is None
        assert parse_pick("", 4) is None
        assert parse_pick("99", 4) is None


class TestCandidates:
    def test_grounding(self):
        choices = propose_candidates({
            "role": "root",
            "children": [{"role": "button", "name": "Start"}],
        })
        labels = [c.label for c in choices]
        assert labels[0] == "click 'Start'"
        assert labels[-3:] == [
            "wait and look again",
            "task is done",
            "task cannot be done",
        ]

    def test_limit(self):
        tree = {
            "role": "root",
            "children": [{"role": "button", "name": f"b{i}"} for i in range(20)],
        }
        choices = propose_candidates(tree, limit=4)
        assert len(choices) == 4 + 3

    def test_empty_tree(self):
        choices = propose_candidates(None)
        assert [c.kind for c in choices] == ["wait", "done", "fail"]

    def test_prompt_compact(self):
        choices = propose_candidates(None, task="open chat")
        prompt = build_choice_prompt("open chat", choices)
        assert len(prompt) < 300
        assert prompt.startswith("Task: open chat")


class TestLoop:
    def test_click_then_done(self):
        s = session_with_tree()
        # choices: 1 click Start, 2 click More, 3 fill Search, 4 wait, 5 done, 6 fail
        script = ["1", "5"]

        async def fake(prompt):
            return script.pop(0)

        agent = ChoiceAgent(s, generate=fake)
        result = run(agent.run("press start"))
        assert result.success is True
        assert result.stop_reason == "done"
        assert s._backend.clicked == ["text(contains)='Start'"]

    def test_garbage_reobserves(self):
        s = session_with_tree()

        async def fake(prompt):
            return "no digits here"

        agent = ChoiceAgent(s, generate=fake, max_steps=3)
        result = run(agent.run("t"))
        assert result.success is False
        assert result.stop_reason == "max_steps"
        assert result.steps_taken == 3  # re-observed, never acted
        assert s._backend.clicked == []

    def test_give_up(self):
        s = session_with_tree()

        async def fake(prompt):
            return "6"  # fail is 6th of 6 choices

        agent = ChoiceAgent(s, generate=fake)
        result = run(agent.run("t"))
        assert result.stop_reason == "give_up"

    def test_inference_error(self):
        s = session_with_tree()

        async def broken(prompt):
            raise RuntimeError("down")

        agent = ChoiceAgent(s, generate=broken)
        result = run(agent.run("t"))
        assert result.stop_reason == "fail"
        assert "down" in result.error
