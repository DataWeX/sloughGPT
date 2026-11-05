"""The shared observation normalizer, tested against captured CDP shapes.

Nothing here launches a browser: Chrome's Accessibility.getFullAXTree output is
hand-written from a live capture (RootWebArea / StaticText / InlineTextBox are
the real role tokens), and the tree that comes out is the contract every
backend has to satisfy for ai/choice.py and ai/slo.py.
"""

import asyncio
import json

import pytest
from avion.ai.choice import propose_candidates
from avion.backends.axtree import (
    VOCABULARY,
    cap_nodes,
    count_nodes,
    fetch_ax_tree,
    fit,
    norm_role,
    normalize_ax_nodes,
    pack,
    parse_pos,
    pos_of,
    role_words,
    signal_count,
)


def run(coro):
    return asyncio.run(coro)


def ax(
    node_id,
    role,
    name="",
    value="",
    parent=None,
    ignored=False,
    dom=None,
    children=(),
):
    node = {
        "nodeId": str(node_id),
        "ignored": ignored,
        "role": {"type": "role", "value": role},
        "name": {"type": "computedString", "value": name},
        "childIds": [str(c) for c in children],
    }
    if value:
        node["value"] = {"type": "value", "value": value}
    if parent is not None:
        node["parentId"] = str(parent)
    if dom is not None:
        node["backendDOMNodeId"] = dom
    return node


class TestNodeShape:
    """The one place a node's keys are decided: both backends build with pack."""

    def test_a_text_leaf_drops_a_role_its_position_already_said(self):
        assert pack("text", text="no notes yet") == {"text": "no notes yet"}

    def test_a_group_keeps_its_role_even_when_it_is_only_text(self):
        tree = pack("text", children=[pack("button", name="Go", pos="1,2,3x4")])
        assert tree["role"] == "text"

    def test_a_label_is_not_repeated_as_content(self):
        assert pack("textbox", name="Note", text="Note") == {"role": "textbox", "name": "Note"}
        assert pack("textbox", name="Note", text="hello") == {
            "role": "textbox",
            "name": "Note",
            "text": "hello",
        }

    def test_no_empty_children_and_no_box_that_was_not_measured(self):
        assert "children" not in pack("window", name="W", children=[])
        assert "pos" not in pack("button", name="Go")

    def test_a_cut_subtree_is_counted_not_shown(self):
        assert pack("window", name="W", children=[], omitted=3) == {
            "role": "window",
            "name": "W",
            "_omitted": 3,
        }


class TestRoleVocabulary:
    """CDP spells roles in CamelCase internal names; AT-SPI in lowercase phrases."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("RootWebArea", "generic"),
            ("generic", "generic"),
            ("button", "button"),
            ("pushButton", "button"),
            ("CheckBox", "button"),
            ("RadioButton", "button"),
            ("ToggleButton", "button"),
            ("link", "link"),
            ("textbox", "textbox"),
            ("searchbox", "textbox"),
            ("TextInput", "textbox"),
            ("ComboBox", "combobox"),
            ("MenuItem", "menuitem"),
            ("MenuItemCheckBox", "menuitem"),
            ("Tab", "tab"),
            ("tab", "tab"),
            ("TabList", "generic"),
            ("tabpanel", "generic"),
            ("option", "option"),
            ("treeItem", "option"),
            ("heading", "text"),
            ("paragraph", "text"),
            ("StaticText", "text"),
            ("InlineTextBox", "text"),
            ("list", "generic"),
            ("none", "generic"),
            ("presentation", "generic"),
            ("", "generic"),
        ],
    )
    def test_cdp_role_tokens_land_in_the_vocabulary(self, raw, expected):
        assert norm_role(raw) == expected

    def test_it_only_ever_emits_the_shared_vocabulary(self):
        for raw in ("RootWebArea", "InlineTextBox", "nonsense role", "", "Dialog"):
            assert norm_role(raw) in VOCABULARY

    def test_camel_and_hyphen_spellings_agree(self):
        assert role_words("push-button") == role_words("pushButton") == {"push", "button"}


class TestNormalize:
    def test_ignored_parents_do_not_eat_their_children(self):
        # A real capture: the button hides under two ignored layout nodes whose
        # own childIds never mention it once they are dropped.
        nodes = [
            ax(1, "RootWebArea", name="Notes", children=[2]),
            ax(2, "none", parent=1, ignored=True, children=[3]),
            ax(3, "generic", parent=2, ignored=True, children=[4]),
            ax(4, "button", name="Add note", parent=3, dom=40),
        ]
        tree, targets = normalize_ax_nodes(nodes)
        assert tree["role"] == "generic"
        assert [c["name"] for c in tree["children"]] == ["Add note"]
        assert targets[0][1] == 40

    def test_text_that_repeats_its_parent_is_dropped(self):
        nodes = [
            ax(1, "RootWebArea", name="Release notes", children=[2, 3]),
            ax(2, "heading", name="Release notes", parent=1, children=[4]),
            ax(3, "button", name="Save", parent=1, dom=30),
            ax(4, "StaticText", name="Release notes", parent=2, children=[5]),
            ax(5, "InlineTextBox", name="Release notes", parent=4),
        ]
        tree, _ = normalize_ax_nodes(nodes)
        text = json.dumps(tree)
        assert text.count("Release notes") == 1  # the root says it once
        assert "Save" in text  # a control is never dropped for repeating

    def test_a_field_keeps_its_label_and_its_contents_apart(self):
        nodes = [
            ax(1, "RootWebArea", name="Notes", children=[2]),
            ax(2, "textbox", name="new note", value="hello", parent=1, dom=20),
        ]
        tree, _ = normalize_ax_nodes(nodes)
        field = tree["children"][0]
        assert field["name"] == "new note" and field["text"] == "hello"

    def test_a_node_reference_reads_as_empty_not_as_a_dict(self):
        node = ax(1, "textbox", name="x", dom=10)
        node["value"] = {"type": "nodeId", "value": {"nodeId": "7"}}
        tree, _ = normalize_ax_nodes([node])
        assert tree == {"role": "textbox", "name": "x"}

    def test_nothing_says_anything(self):
        assert normalize_ax_nodes([]) == (None, [])
        assert normalize_ax_nodes([ax(1, "generic", ignored=True)]) == (None, [])

    def test_a_page_and_a_frame_are_two_roots_under_one_reported_root(self):
        nodes = [
            ax(1, "RootWebArea", name="Notes", children=[]),
            ax(2, "RootWebArea", name="Widget", children=[]),
        ]
        tree, _ = normalize_ax_nodes(nodes)
        assert sorted(c["name"] for c in tree["children"]) == ["Notes", "Widget"]

    def test_a_parent_cycle_terminates(self):
        nodes = [
            ax(1, "RootWebArea", name="Notes", children=[4]),
            ax(2, "generic", parent=3, ignored=True),
            ax(3, "generic", parent=2, ignored=True),
            ax(4, "button", name="Add note", parent=3, dom=40),
        ]
        tree, _ = normalize_ax_nodes(nodes)
        assert "Add note" in json.dumps(tree)

    def test_internal_keys_never_reach_the_tree(self):
        nodes = [
            ax(1, "RootWebArea", name="Notes", children=[2]),
            ax(2, "button", name="Save", parent=1, dom=20),
        ]
        tree, targets = normalize_ax_nodes(nodes)
        assert "_dom" not in json.dumps(tree)
        assert len(targets) == 1


class TestBudget:
    def big_page(self, count=500):
        nodes = [ax(0, "RootWebArea", name="Notes", children=list(range(1, count + 1)))]
        nodes += [
            ax(i, "StaticText", name=f"line {i}", parent=0, dom=i if i % 40 == 0 else None)
            for i in range(1, count + 1)
        ]
        return nodes

    def test_controls_survive_a_cut_that_text_does_not(self):
        nodes = [ax(0, "RootWebArea", name="Notes", children=list(range(1, 60)))]
        nodes += [ax(i, "StaticText", name=f"line {i}", parent=0) for i in range(1, 59)]
        nodes.append(ax(59, "button", name="Add note", parent=0, dom=900))
        tree, _ = normalize_ax_nodes(nodes, max_nodes=4)
        assert "Add note" in json.dumps(tree)
        assert count_nodes(tree) <= 4
        assert signal_count(tree) == count_nodes(tree)  # nothing kept says nothing

    def test_node_cap_and_json_budget_hold(self):
        tree, _ = normalize_ax_nodes(self.big_page(), max_nodes=40)
        assert count_nodes(tree) <= 40
        assert json.loads(json.dumps(tree)) == tree
        fitted = fit(tree, 1200)
        assert len(json.dumps(fitted, default=str)) <= 1200
        assert fitted.get("_trimmed", 0) > 0

    def test_ranking_is_the_reason_a_cut_loses_the_weak_tail(self):
        tree, _ = normalize_ax_nodes(self.big_page(80), max_nodes=80)
        assert count_nodes(tree) == 80  # 81 nodes existed; one whole subtree went
        assert count_nodes(cap_nodes(tree, 20)) == 20


class FakeCdp:
    """A CDP sender that answers the three methods observation uses."""

    def __init__(self, nodes, *, border=(320, 114, 411, 114, 411, 153, 320, 153)):
        self.nodes = nodes
        self.border = border
        self.calls = []

    async def send(self, method, params=None):
        self.calls.append((method, params))
        if method == "Accessibility.getFullAXTree":
            return {"nodes": self.nodes}
        if method == "DOM.getBoxModel":
            return {"model": {"border": list(self.border)}}
        return {}


class BrokenCdp(FakeCdp):
    async def send(self, method, params=None):
        self.calls.append((method, params))
        if method in ("DOM.enable", "Accessibility.enable"):
            raise RuntimeError("no such domain")
        if method == "DOM.getBoxModel":
            raise RuntimeError("no node with given id")
        return await super().send(method, params)


class TestFetch:
    def test_positions_come_from_the_box_model(self):
        nodes = [
            ax(1, "RootWebArea", name="Notes", children=[2, 3]),
            ax(2, "textbox", name="new note", parent=1, dom=12),
            ax(3, "button", name="Add note", parent=1, dom=13),
        ]
        fake = FakeCdp(nodes)
        tree = run(fetch_ax_tree(fake.send))
        assert [c["pos"] for c in tree["children"]] == ["320,114,91x39", "320,114,91x39"]
        assert [call[0] for call in fake.calls[:2]] == ["DOM.enable", "Accessibility.enable"]
        assert ("DOM.getBoxModel", {"backendNodeId": 12}) in fake.calls

    def test_domains_that_will_not_enable_still_observe(self):
        nodes = [
            ax(1, "RootWebArea", name="Notes", children=[2]),
            ax(2, "button", name="Add note", parent=1, dom=13),
        ]
        fake = BrokenCdp(nodes)
        tree = run(fetch_ax_tree(fake.send))
        assert tree["children"][0]["name"] == "Add note"
        assert "pos" not in tree["children"][0]  # no box, no coordinate, no lie

    def test_a_page_of_noise_is_reported_as_no_tree(self):
        nodes = [ax(1, "generic", name="", ignored=False), ax(2, "none", ignored=True)]
        assert run(fetch_ax_tree(FakeCdp(nodes).send)) is None

    def test_an_empty_ax_response_is_no_tree(self):
        assert run(fetch_ax_tree(FakeCdp([]).send)) is None


class TestContractWithTheRestOfAvion:
    """choice.py and slo.py read this shape; they must not have to guess."""

    def test_choices_are_proposed_for_a_browser_tree(self):
        nodes = [
            ax(1, "RootWebArea", name="Notes", children=[2, 3, 4]),
            ax(2, "textbox", name="new note", parent=1, dom=2),
            ax(3, "button", name="Add note", parent=1, dom=3),
            ax(4, "link", name="Archive", parent=1, dom=4),
        ]
        tree, _ = normalize_ax_nodes(nodes)
        labels = [c.label for c in propose_candidates(tree, "write a note")]
        assert labels[0] == "type 'write a note' into 'new note'"
        assert "click 'Add note'" in labels and "click 'Archive'" in labels

    def test_the_cli_snapshot_reads_this_shape_not_raw_nodes(self):
        from avion.cli import _snapshot_lines

        nodes = [
            ax(1, "RootWebArea", name="Notes", children=[2, 3]),
            ax(2, "button", name="Add note", parent=1, dom=12),
            ax(3, "StaticText", name="no notes yet", parent=1),
        ]
        tree, _ = normalize_ax_nodes(nodes)
        lines = _snapshot_lines(tree)
        assert any("button" in line and "Add note" in line for line in lines)
        assert any("no notes yet" in line for line in lines)

    def test_positions_are_the_string_a_coordinate_agent_can_parse(self):
        nodes = [
            ax(1, "RootWebArea", name="Notes", children=[2]),
            ax(2, "button", name="Go", parent=1, dom=2),
        ]
        tree = run(fetch_ax_tree(FakeCdp(nodes).send))
        assert parse_pos(tree["children"][0]["pos"]) == (320, 114, 91, 39)

    def test_the_writer_and_the_reader_agree_and_nothing_else_is_a_box(self):
        assert parse_pos(pos_of(3, 4, 91, 39)) == (3, 4, 91, 39)
        assert parse_pos(None) is None
        assert parse_pos("320,114") is None
        assert parse_pos("320,114,wide x tall") is None


class FakePage:
    """A page that speaks CDP and nothing else — no Playwright, no browser."""

    def __init__(self, session):
        self._session = session
        self.sessions_made = 0
        self.closed = False

    async def close(self):
        self.closed = True

    @property
    def context(self):
        page = self

        class _Context:
            async def new_cdp_session(self, target):
                page.sessions_made += 1
                return page._session

        return _Context()


class TestBackendsShareIt:
    """Both browser backends, one shape: the shape that used to be wrong."""

    NODES = [
        ax(1, "RootWebArea", name="Notes", children=[2, 3]),
        ax(2, "button", name="Add note", parent=1, dom=12),
        ax(3, "StaticText", name="no notes yet", parent=1),
    ]

    def test_playwright_observes_over_cdp(self):
        from avion.backends.playwright import PlaywrightBackend

        backend = PlaywrightBackend()
        page = FakePage(FakeCdp(self.NODES))
        backend._page = page
        tree = run(backend.get_accessibility_tree())
        assert [c.get("role") for c in tree["children"]] == ["button", "text"]
        assert tree["children"][0]["pos"] == "320,114,91x39"
        # One session for the page, reused by the next step's observation.
        run(backend.get_accessibility_tree())
        assert page.sessions_made == 1

    def test_playwright_without_cdp_reports_no_tree(self):
        from avion.backends.playwright import PlaywrightBackend

        class NoCdp(FakePage):
            @property
            def context(self):
                raise RuntimeError("CDP is only available in Chromium")

        backend = PlaywrightBackend()
        backend._page = NoCdp(FakeCdp(self.NODES))
        assert run(backend.get_accessibility_tree()) is None

    def test_cdp_backend_stops_returning_raw_nodes(self):
        from avion.backends.cdp import CdpBackend

        backend = CdpBackend()
        fake = FakeCdp(self.NODES)
        backend._send = fake.send
        tree = run(backend.get_accessibility_tree())
        assert "nodes" not in tree
        assert [c.get("role") for c in tree["children"]] == ["button", "text"]

    def test_both_backends_emit_the_same_tree(self):
        from avion.backends.cdp import CdpBackend
        from avion.backends.playwright import PlaywrightBackend

        playwright = PlaywrightBackend()
        playwright._page = FakePage(FakeCdp(self.NODES))
        cdp = CdpBackend()
        cdp._send = FakeCdp(self.NODES).send
        assert run(playwright.get_accessibility_tree()) == run(cdp.get_accessibility_tree())
