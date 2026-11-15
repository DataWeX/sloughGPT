"""Desktop observation tests — the AT-SPI tree algorithm, no display needed.

Everything here runs against _Node graphs and role strings, not against a
live accessibility bus: the walk is thin glue around these functions, and
keeping them pure is what makes the pruning decisions reviewable at all.
"""

import asyncio
import json

import pytest
from avion.ai.choice import propose_candidates
from avion.ai.slo import build_prompt, flatten_tree_text
from avion.backends.axtree import VOCABULARY
from avion.backends.desktop import (
    DesktopBackend,
    _collapse,
    _fit,
    _Node,
    _norm_role,
    _prune,
    _rank,
    _to_dict,
)


def run(coro):
    return asyncio.run(coro)


def node(role, name="", text="", extents=None, children=(), states=()):
    return _Node(
        role=role,
        name=name,
        text=text,
        extents=extents,
        states=frozenset(states),
        children=list(children),
    )


class TestRoleClassifier:
    """AT-SPI has ~95 role names; the model speaks eleven."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("push button", "button"),
            ("toggle button", "button"),
            ("check box", "button"),
            ("radio button", "button"),
            ("combo box", "combobox"),
            ("link", "link"),
            ("page tab", "tab"),
            ("menu item", "menuitem"),
            ("check menu item", "menuitem"),
            ("list item", "option"),
            ("tree item", "option"),
            ("entry", "textbox"),
            ("password text", "textbox"),
            ("spin button", "button"),
            ("frame", "window"),
            ("dialog", "dialog"),
            ("alert", "dialog"),
            ("document web", "text"),
            ("label", "text"),
            ("terminal", "text"),
            ("panel", "generic"),
            ("filler", "generic"),
            ("menu bar", "generic"),
            ("scroll bar", "generic"),
            ("status bar", "generic"),
            ("table cell", "generic"),
        ],
    )
    def test_maps_into_the_model_vocabulary(self, raw, expected):
        assert _norm_role(raw) == expected

    def test_everything_it_emits_is_in_the_vocabulary(self):
        for raw in ("push button", "frame", "panel", "nonsense role", ""):
            assert _norm_role(raw) in VOCABULARY

    def test_spin_button_is_clickable_not_a_field(self):
        # "spin button" contains both a control word and button; a click on it
        # is the useful interpretation, so it must not land in the textbox set.
        assert _norm_role("spin button") == "button"


class TestPruneAndCollapse:
    def test_empty_scaffolding_goes(self):
        tree = node("window", name="Note", children=[node("generic"), node("generic")])
        assert _prune(tree).children == []

    def test_shell_around_a_button_survives_then_collapses(self):
        btn = node("button", name="Add note", extents=(600, 500, 80, 28))
        shell = node("generic", children=[node("generic", children=[btn])])
        kept = _collapse(_prune(node("window", children=[shell])))
        # The window stops the collapse: it is the root the model reads from.
        assert kept.role == "window"
        assert kept.children[0].role == "button"

    def test_a_nameless_chain_of_one_collapses_away_entirely(self):
        btn = node("button", name="Add note")
        wrapper = node("generic", children=[node("generic", children=[btn])])
        assert _collapse(_prune(wrapper)).role == "button"

    def test_unnamed_window_with_nothing_under_it_is_dropped(self):
        assert _prune(node("window")) is None

    def test_named_window_is_kept_alone(self):
        kept = _prune(node("window", name="Untitled — Editor"))
        assert kept is not None and kept.name == "Untitled — Editor"

    def test_text_without_a_name_still_counts_as_signal(self):
        kept = _prune(node("text", text="3 notes saved"))
        assert kept is not None


class TestPackAndBudget:
    def test_only_actionable_nodes_get_coordinates(self):
        tree = node(
            "window",
            name="Notes",
            extents=(0, 0, 800, 600),
            children=[node("button", name="Add", extents=(10, 10, 50, 20))],
        )
        out = _to_dict(_collapse(_prune(tree)))
        assert "pos" not in out  # window extents are not a click target here
        assert out["children"][0]["pos"] == "10,10,50x20"

    def test_a_node_without_extents_never_claims_a_position(self):
        out = _to_dict(node("button", name="Add", extents=None))
        assert "pos" not in out

    def test_a_nameless_group_keeps_its_role_like_the_browser_tree(self):
        """A text node with something under it is a group, not prose.

        The key policy lives in axtree.pack and both backends call it, so this
        pins the desktop side against drifting from the browser side.
        """
        out = _to_dict(node("text", children=[node("button", name="Go", extents=(1, 2, 3, 4))]))
        assert out == {
            "role": "text",
            "children": [{"role": "button", "name": "Go", "pos": "1,2,3x4"}],
        }
        assert _to_dict(node("text", text="no notes yet")) == {"text": "no notes yet"}

    def test_json_stays_valid_and_inside_budget_when_trimmed(self):
        kids = [node("button", name=f"b{i}", extents=(i, i, 10, 10)) for i in range(500)]
        tree = _fit(_to_dict(node("window", name="W", children=kids)), 1200)
        assert len(json.dumps(tree)) <= 1200
        assert json.dumps(tree)  # parseable: whole nodes dropped, never text cut
        assert tree["_trimmed"] > 0

    def test_trim_keeps_the_earliest_highest_ranked_nodes(self):
        first = node("button", name="Keep me", extents=(1, 1, 9, 9))
        rest = [node("text", text=f"filler {i} " + "x" * 40) for i in range(60)]
        tree = _fit(_to_dict(node("window", name="W", children=[first, *rest])), 900)
        assert any(c.get("name") == "Keep me" for c in tree["children"])
        assert len(json.dumps(tree)) <= 900

    def test_no_budget_means_no_trim(self):
        kids = [node("button", name=f"b{i}") for i in range(50)]
        tree = _to_dict(node("window", name="W", children=kids))
        assert _fit(dict(tree), 0) == tree

    def test_ranking_puts_clickables_first(self):
        tree = node(
            "window",
            name="W",
            children=[
                node("text", text="a label"),
                node("button", name="Go", extents=(0, 0, 5, 5)),
            ],
        )
        _rank(tree)
        assert tree.children[0].role == "button"


class TestDesktopTreeIsTheRealContract:
    """The tree shape is a cross-backend promise, so test it as one."""

    def desktop_tree(self):
        return {
            "role": "window",
            "name": "Notes",
            "children": [
                {"role": "textbox", "name": "Note"},
                {"role": "button", "name": "Add note", "pos": "612,540,88x28"},
                {"role": "text", "text": "1 saved"},
            ],
        }

    def test_choices_are_proposed_from_a_desktop_tree(self):
        picks = propose_candidates(self.desktop_tree(), task="write hello")
        labels = [c.label for c in picks]
        assert "click 'Add note'" in labels
        assert any(label.startswith("type ") for label in labels)

    def test_the_goal_text_survives_the_prompt_budget(self):
        tree = self.desktop_tree()
        assert "1 saved" in flatten_tree_text(tree)
        prompt = build_prompt("add a note", tree, [], max_chars=2400)
        assert "Add note" in prompt


class TestNoBusDegradation:
    def test_observation_starts_without_pyautogui(self, monkeypatch):
        monkeypatch.setitem(__import__("sys").modules, "pyautogui", None)
        backend = DesktopBackend(require_actions=False)
        run(backend.start())
        assert backend._started is True

    def test_actions_still_demand_pyautogui_at_start(self):
        # Loud failure at start beats a blind run that discovers it at step 12.
        if __import__("importlib.util", fromlist=["util"]).find_spec("pyautogui"):
            pytest.skip("pyautogui installed")
        with pytest.raises(ImportError, match="pip install pyautogui"):
            run(DesktopBackend().start())

    def test_url_is_empty_not_a_window_title(self):
        # A title here would let url_contains "chat" pass on a window named
        # "Chat": an unverified goal must never read as a pass.
        assert run(DesktopBackend(require_actions=False).get_url()) == ""

    def test_no_bindings_means_no_tree_and_no_raise(self, monkeypatch):
        import avion.backends.desktop as desktop

        monkeypatch.setattr(desktop, "_atspi_or_none", lambda: None)
        backend = DesktopBackend(require_actions=False)
        assert run(backend.get_accessibility_tree()) is None
        assert run(backend.get_title()) == ""
