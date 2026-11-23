"""Choice agent tests — grounding, parsing, full loop on fakes."""

import asyncio
import re

from avion.ai.choice import (
    ALREADY_SATISFIED,
    ChoiceAgent,
    build_choice_prompt,
    build_next_action_prompt,
    parse_action,
    parse_pick,
    propose_options,
)
from avion.core.element import Element, ElementLocator
from test_autoclicker import FakeBackend, TestSession


def run(coro):
    return asyncio.run(coro)


class TreeBackend(FakeBackend):
    """FakeBackend whose page is the tree below.

    The elements dict comes from FakeBackend and only has Start and an input;
    a choice grounded on this tree offers click 'More' too, and an unbacked
    card can only ever answer "error: Element not found", which tests the
    error path rather than the pick.
    """

    def __init__(self):
        super().__init__()
        for name, raw in {"More": 3, "Search": 4}.items():
            self._elements[ElementLocator.text(name).describe()] = Element(
                raw={"id": raw}, locator=ElementLocator.text(name)
            )

    async def get_accessibility_tree(self):
        return {
            "role": "root",
            "children": [
                {"role": "button", "name": "Start"},
                {"role": "link", "name": "More"},
                {"role": "textbox", "name": "Search"},
            ],
        }


def session_with_tree(backend=None):
    s = TestSession()._session()
    s._backend = backend or TreeBackend()
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

    def test_copied_list_is_not_an_answer(self):
        """The served model's actual replies, recorded from /inference/generate.

        Every one of these used to parse as pick 1: the leading "1." is the
        prompt's own numbering being echoed back, not a choice.
        """
        echo = "1. type 'hello' into 'new note'\n2. click 'Add note'"
        assert parse_pick(echo, 6) is None
        # Truncated at a small budget, so only the first marker survives.
        assert parse_pick("1. type 'hello' into '", 6) is None
        assert parse_pick("1. Type 'hello' into the", 6) is None

    def test_bare_number_still_parsed_after_prose(self):
        assert parse_pick("Number:\n4", 6) == 4
        assert parse_pick("3.", 6) == 3

    def test_lettered_pick_is_read_as_an_answer(self):
        """The recorded reply to a lettered list, and the forms around it.

        Where digits are copied as a list to be restated, a letter is chosen:
        "B. click 'Add note'" names a card that is not the first one. So a
        punctuated letter is the answer here, where the same shape for a digit is
        the echo being skipped above.
        """
        assert parse_pick("B. click 'Add note'", 6) == 2
        assert parse_pick("B. task is done", 6) == 5
        assert parse_pick("A. type 'hello' into '", 6) == 1
        assert parse_pick("F", 6) == 6
        assert parse_pick("Letter:\nC", 6) == 3
        assert parse_pick("d)", 6) == 4

    def test_a_copied_lettered_list_lands_on_wait_not_done(self):
        """Matching stop cards by text is safe only because the list ends in order.

        A whole copied list names all three stop cards, so the match has to fall
        on the first one — "wait and look again", which re-observes. If it could
        reach "task is done", a page the model merely described would be scored
        as a task the model completed.
        """
        copy = "\n".join(
            [
                "A. type 'hello' into 'new note'",
                "B. click 'Add note'",
                "C. click 'Clear list'",
                "D. wait and look again",
                "E. task is done",
                "F. task cannot be done",
            ]
        )
        assert parse_pick(copy, 6) == 4

    def test_letter_out_of_range_falls_through(self):
        assert parse_pick("H. nothing", 6) is None
        assert parse_pick("Z", 6) is None

    def test_a_word_is_not_a_letter_pick(self):
        # Only a line-opening letter counts, so the articles and initials inside
        # prose are not read as cards A and H.
        assert parse_pick("have a look at the add button", 6) is None
        assert parse_pick("I would choose the add note action", 6) is None


class TestOptions:
    def test_grounding(self):
        choices = propose_options(
            {
                "role": "root",
                "children": [{"role": "button", "name": "Start"}],
            }
        )
        labels = [c.label for c in choices]
        assert labels[0] == "click 'Start'"
        assert labels[-3:] == [
            "wait and look again",
            "task is done",
            "task cannot be done",
        ]

    def test_fill_card_needs_a_task_that_enters_text(self):
        tree = {"role": "root", "children": [{"role": "textbox", "name": "Search"}]}
        assert [c.kind for c in propose_options(tree, "click the clear button")] == [
            "wait",
            "done",
            "fail",
        ]
        assert propose_options(tree, "type hi into Search")[0].label == ("type 'hi' into 'Search'")

    def test_limit_is_the_whole_list(self):
        """limit counts the stop cards too, so the last letter is always H at worst.

        It used to cap the page's cards and append three more, which on a busy
        page put done and fail at I, J, K. build_choice_prompt letters by
        position and parse_pick reads A-H, so those cards could never be picked:
        a task whose only remaining move is to declare it finished had no letter
        left to answer with.
        """
        tree = {
            "role": "root",
            "children": [{"role": "button", "name": f"b{i}"} for i in range(20)],
        }
        choices = propose_options(tree, limit=4)
        assert len(choices) == 4
        assert [c.kind for c in choices] == ["click", "wait", "done", "fail"]
        assert parse_pick("D. task is done", len(choices)) == 3

    def test_empty_tree(self):
        choices = propose_options(None)
        assert [c.kind for c in choices] == ["wait", "done", "fail"]

    def test_prompt_compact(self):
        choices = propose_options(None, task="open chat")
        prompt = build_choice_prompt("open chat", choices)
        assert len(prompt) < 300
        assert prompt.startswith("Task: open chat")

    def test_prompt_letters_the_cards(self):
        choices = propose_options(None, task="open chat")
        prompt = build_choice_prompt("open chat", choices)
        assert "A. wait and look again" in prompt
        assert "C. task cannot be done" in prompt
        assert "Letter:" in prompt
        # No digits at all: a numbered list is the one the sampler restarts and
        # copies, which is what pinned every step to card 1.
        assert not re.search(r"^\s*\d+[.)]", prompt, re.MULTILINE)

    def test_unavailable_keeps_its_letter(self):
        choices = propose_options(None, task="open chat")
        prompt = build_choice_prompt("open chat", choices, unavailable={"wait and look again"})
        assert "A. wait and look again (tried, it did not work)" in prompt
        assert "B. task is done" in prompt


class TestLoop:
    def test_click_then_done(self):
        s = session_with_tree()
        # "press start" asks for no text, so the Search field offers no fill
        # card: 1 click Start, 2 click More, 3 wait, 4 done, 5 fail
        script = ["1", "4"]

        async def fake(prompt):
            return script.pop(0)

        agent = ChoiceAgent(s, generate=fake)
        result = run(agent.run("press start"))
        assert result.success is True
        assert result.stop_reason == "done"
        assert s._backend.clicked == ["text(contains)='Start'"]

    def test_lettered_echoes_drive_the_loop(self):
        """The shape the served model actually answers in.

        "B. click 'Add note'" is what /inference/generate returns to a lettered
        list, rest-of-label included. The loop has to act on that, not on a bare
        letter only.
        """
        s = session_with_tree()
        script = ["B. click 'More'", "E. task is done"]

        async def fake(prompt):
            return script.pop(0)

        agent = ChoiceAgent(s, generate=fake)
        result = run(agent.run("press start"))
        assert result.success is True
        assert result.stop_reason == "done"
        assert s._backend.clicked == ["text(contains)='More'"]

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
            return "5"  # fail is 5th of 5 choices; no fill card for a textless task

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


class MutatingBackend(TreeBackend):
    """Fake page whose status text changes only once a click has landed."""

    async def click(self, element, *, force: bool = False) -> None:
        await super().click(element, force=force)
        self.page_text = "notes cleared"


def _goal(value):
    return [{"kind": "text_present", "value": value}]


class TestGoalVerification:
    """A goal is a transition, not a snapshot.

    The post-step check reads the page as it is now, so a page that already said
    the right words handed `done` to the first action that left them alone —
    including a no-op click. Without the pre-run comparison the malformed
    exercise and the solved exercise report the same thing.
    """

    def test_goal_already_true_earns_nothing(self):
        backend = MutatingBackend()
        backend.page_text = "notes cleared"
        s = session_with_tree(backend)
        script = ["1"]  # click Start; the done card is never reached

        async def fake(prompt):
            return script.pop(0)

        agent = ChoiceAgent(s, generate=fake)
        result = run(agent.run("press start", goal=_goal("cleared")))
        assert result.success is False
        assert result.stop_reason == "unverified"
        assert result.error == ALREADY_SATISFIED
        assert backend.clicked == ["text(contains)='Start'"]

    def test_goal_reached_by_a_step_is_a_pass(self):
        backend = MutatingBackend()
        backend.page_text = "nothing yet"
        s = session_with_tree(backend)
        script = ["1"]

        async def fake(prompt):
            return script.pop(0)

        agent = ChoiceAgent(s, generate=fake)
        result = run(agent.run("press start", goal=_goal("cleared")))
        assert result.success is True
        assert result.stop_reason == "done"

    def test_volunteered_done_on_an_unmet_goal_keeps_the_checkers_reason(self):
        backend = MutatingBackend()
        backend.page_text = "nothing yet"
        s = session_with_tree(backend)
        script = ["4"]  # done, straight away

        async def fake(prompt):
            return script.pop(0)

        agent = ChoiceAgent(s, generate=fake)
        result = run(agent.run("press start", goal=_goal("cleared")))
        assert result.success is False
        assert result.stop_reason == "unverified"
        assert result.error != ALREADY_SATISFIED
        assert "cleared" in result.error


NOTE_PAGE = {
    "role": "root",
    "children": [
        {"role": "textbox", "name": "new note"},
        {"role": "button", "name": "Add note"},
        {"role": "button", "name": "Clear list"},
    ],
}
NOTE_TASK = "Type hello into the note field, then click the Add note button"


class TestNextActionFraming:
    """The other way to ask: say what happens next, and match the words.

    Recorded against the live 0.5B: the lettered menu answers click 'Add note'
    even when the fill is card A, so the preference is for the button, not for a
    position, and no amount of reordering fixes it. Asked what comes next with no
    list in front of it, the same sampler named the fill.
    """

    def _choices(self):
        return propose_options(NOTE_PAGE, task=NOTE_TASK)

    def test_prompt_offers_no_list_to_continue(self):
        prompt = build_next_action_prompt(NOTE_TASK, self._choices())
        assert "What does it do next?" in prompt
        assert "type 'hello' into 'new note'" in prompt
        # Nothing line-structured: the shape the sampler restarts and copies.
        assert not re.search(r"^[ \t]*(?:\d+|[A-Ha-h])[.)][ \t]", prompt, re.MULTILINE)

    def test_a_restated_plan_resolves_to_its_first_step(self):
        reply = "Type 'hello' into the note field, then click the Add note button"
        assert parse_action(reply, self._choices()) == 1

    def test_the_second_step_is_not_the_first_answer(self):
        """The test that tells planning from reciting the task.

        A reply that names only the button scores as the button, and the same
        words naming both cards score as whichever comes first.
        """
        choices = self._choices()
        assert parse_action("Click the Add note button", choices) == 2

    def test_a_name_split_by_other_words_still_matches(self):
        choices = self._choices()
        assert parse_action("Clear the list", choices) == 3
        assert parse_action("clear list", choices) == 3

    def test_a_bare_verb_is_not_an_action(self):
        # 'enter' is a fill verb and "press Enter" is not a fill: nothing in this
        # reply names a card, so the loop re-observes instead of guessing.
        assert parse_action("press Enter", self._choices()) is None

    def test_stop_words_are_picks(self):
        choices = self._choices()
        assert parse_action("it is done", choices) == 5
        assert parse_action("I cannot do it", choices) == 6
        assert parse_action("wait and look at it again", choices) == 4

    def test_a_spent_card_lets_the_rest_of_the_sentence_answer(self):
        """The live model recites the whole plan every step, so the plan is read
        from left to right, skipping what the run already did. Measured: without
        this, framing=next named the fill correctly and then named it again until
        the step budget ran out (8 steps, 7 refusals, stop=max_steps).
        """
        choices = self._choices()
        reply = "Type 'hello' into the note field, then click the Add note button"
        assert parse_action(reply, choices) == 1
        assert parse_action(reply, choices, skip={"type 'hello' into 'new note'"}) == 2

    def test_the_loop_acts_on_described_actions(self):
        s = session_with_tree()
        script = ["click More", "task is done"]

        async def fake(prompt):
            return script.pop(0)

        agent = ChoiceAgent(s, generate=fake, framing="next")
        result = run(agent.run("press start"))
        assert result.success is True
        assert result.stop_reason == "done"
        assert s._backend.clicked == ["text(contains)='More'"]
