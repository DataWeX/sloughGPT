"""Action provenance — every mutating action logs the element it resolved.

Unit half: descriptor formatting, the shared TEXT predicate (the log must
describe exactly what the action hits) and the CLI miss path with and
without a near-miss-capable backend. The real-Chromium half lives in
``test_cdp.py``.
"""

from __future__ import annotations

import asyncio

from avion.backends.cdp import (
    selector_probe_all_js,
    selector_probe_js,
    selector_to_all_js,
    selector_to_js,
)
from avion.core.element import Element, ElementLocator

# ── the log must describe what the action hits ───────────────────────────


def test_selector_to_js_text_is_byte_identical():
    """TEXT branch refactor must not shift a byte of the action expression."""
    contains = ElementLocator.text("Go").selectors[0]
    assert selector_to_js(contains) == (
        "[...document.querySelectorAll('body *')]"
        ".find(e => e.textContent?.includes('Go') && e.children.length === 0)"
    )
    exact = ElementLocator.text("Go", exact=True).selectors[0]
    assert selector_to_js(exact) == (
        "[...document.querySelectorAll('body *')].find(e => e.textContent?.trim() === 'Go')"
    )


def test_probe_embeds_the_action_expression():
    """The probe resolves with the SAME expression the click will fire —
    otherwise the log could describe a different element than the hit."""
    for locator in (
        ElementLocator.css("#go"),
        ElementLocator.text("Go"),
        ElementLocator.text("Go", exact=True),
        ElementLocator.test_id("send"),
    ):
        sel = locator.selectors[0]
        action = selector_to_js(sel)
        probe = selector_probe_js(sel)
        assert action is not None and probe is not None
        assert f"const el = ({action});" in probe


def test_probe_and_all_mirror_unsupported_strategies():
    role = ElementLocator.role("button").selectors[0]
    assert selector_to_js(role) is None
    assert selector_probe_js(role) is None
    assert selector_probe_all_js(role) is None
    assert selector_to_all_js(role) is None


def test_probe_all_shares_the_field_declaration_with_single_probe():
    """Both probes describe a match from ONE field declaration — the
    identify block and the click line can never drift apart about an
    element's identity."""
    sel = ElementLocator.text("Go").selectors[0]
    single = selector_probe_js(sel)
    all_probe = selector_probe_all_js(sel)
    assert single is not None and all_probe is not None
    assert f"const all = ({selector_to_all_js(sel)});" in all_probe
    shared = "text: ((el.innerText || el.textContent || '')).trim()"
    assert shared in single and shared in all_probe
    assert "idx: all.indexOf(el)" in single  # single probe: position of THE match
    assert "idx: i, total: all.length" in all_probe  # all probe: position in the set


def test_all_js_uses_the_same_predicate():
    """The match ordinal counts with the action's predicate, not its own rule."""
    sel = ElementLocator.text("Go").selectors[0]
    all_js = selector_to_all_js(sel)
    assert all_js is not None
    assert ".filter(e => e.textContent?.includes('Go') && e.children.length === 0)" in all_js


# ── provenance formatting ────────────────────────────────────────────────


def _element(descriptor: dict | None = None, **cache) -> Element:
    return Element(
        raw={},
        locator=ElementLocator.text("Dismiss"),
        backend="cdp",
        descriptor=descriptor or {},
        _cache=cache,
    )


def test_provenance_full_descriptor():
    el = _element(
        {
            "tag": "button",
            "id": "d1",
            "cls": "toast-btn danger",
            "text": "Dismiss",
            "rect": {"x": 378.0, "y": 240.0, "width": 62.0, "height": 18.0},
            "visible": True,
            "idx": 0,
            "total": 3,
        }
    )
    assert el.provenance() == "button#d1.toast-btn 'Dismiss' @378,240 62x18 (match 1/3)"


def test_provenance_unique_match_has_no_ordinal():
    el = _element(
        {
            "tag": "a",
            "text": "Home",
            "rect": {"x": 1.0, "y": 2.0, "width": 30.0, "height": 12.0},
            "visible": True,
            "idx": 0,
            "total": 1,
        }
    )
    assert el.provenance() == "a 'Home' @1,2 30x12"


def test_provenance_no_ordinal_when_index_unknown():
    """idx -1 (node missing from the match array) must not print match 0/n."""
    el = _element({"tag": "b", "text": "x", "visible": True, "idx": -1, "total": 3})
    assert "(match" not in el.provenance()


def test_provenance_hidden_flag_and_truncation():
    el = _element({"tag": "div", "text": "x" * 100, "visible": False, "idx": 0, "total": 1})
    out = el.provenance()
    assert out.endswith(" (hidden)")
    assert "…" in out
    assert out.count("x") <= 40


def test_provenance_uses_placeholder_for_inputs():
    el = _element(
        {
            "tag": "input",
            "placeholder": "Filter datasets...",
            "rect": {"x": 4.0, "y": 8.0, "width": 200.0, "height": 20.0},
            "visible": True,
            "idx": 0,
            "total": 1,
        }
    )
    assert "'Filter datasets...'" in el.provenance()


def test_provenance_falls_back_to_cache():
    """Playwright-style element: no descriptor, cache carries the identity."""
    el = _element(tag="a", text="Home", bounding_box={"x": 1, "y": 2, "width": 3, "height": 4})
    assert el.provenance() == "a 'Home' @1,2 3x4"


def test_provenance_bare_element_degrades():
    assert _element().provenance() == "?"


# ── CLI miss path ────────────────────────────────────────────────────────


class _MissBackend:
    """find_element misses; no near_miss (pre-provenance backend shape)."""

    name = "fake"

    async def find_element(self, locator):
        return None


class _NearMissBackend(_MissBackend):
    async def near_miss(self, locator):
        return ['<div#toast> "Dismiss OK" — has child elements (not a leaf)']


def test_miss_without_near_miss_degrades_to_plain_line(capsys):
    from avion.cli import _cmd_click

    code, _ = asyncio.run(_cmd_click(_MissBackend(), "#x", "css"))
    assert code == 1
    err = capsys.readouterr().err
    assert "element not found (css): #x" in err
    assert "near-miss" not in err


def test_miss_prints_near_miss_report(capsys):
    from avion.cli import _cmd_click

    code, _ = asyncio.run(_cmd_click(_NearMissBackend(), "Dismiss", "text"))
    assert code == 1
    err = capsys.readouterr().err
    assert "element not found (text): Dismiss" in err
    assert "near-miss:" in err
    assert "has child elements (not a leaf)" in err


def test_click_logs_provenance_from_resolved_element(capsys):
    """The success line reports the RESOLVED element, not the input echo."""
    from avion.cli import _cmd_click

    class _Hit(_MissBackend):
        async def find_element(self, locator):
            return Element(
                raw={"js": "document.querySelector('#go')"},
                locator=locator,
                backend="cdp",
                descriptor={
                    "tag": "button",
                    "id": "go",
                    "cls": "",
                    "text": "Go",
                    "rect": {"x": 8.0, "y": 8.0, "width": 62.0, "height": 18.0},
                    "visible": True,
                    "idx": 0,
                    "total": 1,
                },
            )

        async def click(self, element, *, force=False):
            pass

    code, hits = asyncio.run(_cmd_click(_Hit(), "#go", "css"))
    assert code == 0
    assert len(hits) == 1  # the hit list feeds per-hit recording
    out = capsys.readouterr().out
    assert out.strip() == "clicked css: #go → button#go 'Go' @8,8 62x18"


def test_fill_logs_provenance_from_resolved_element(capsys):
    from avion.cli import _cmd_fill

    class _Hit(_MissBackend):
        async def find_element(self, locator):
            return Element(
                raw={"js": "document.querySelector('input[name=q]')"},
                locator=locator,
                backend="cdp",
                descriptor={
                    "tag": "input",
                    "id": "",
                    "cls": "",
                    "text": "",
                    "rect": {"x": 12.0, "y": 34.0, "width": 100.0, "height": 20.0},
                    "visible": True,
                    "idx": 0,
                    "total": 1,
                },
            )

        async def fill(self, element, value, *, force=False):
            pass

    code = asyncio.run(_cmd_fill(_Hit(), "input[name=q]", "hello", "css"))
    assert code == 0
    out = capsys.readouterr().out
    assert out.strip() == "filled css: input[name=q] → input @12,34 100x20"


# ── multi-match click: identify first, then act ───────────────────────────


def _desc(idx: int, total: int, *, el_id: str, visible: bool = True) -> dict:
    return {
        "tag": "button",
        "id": el_id,
        "cls": "",
        "text": "Go",
        "rect": {"x": 8.0, "y": 8.0 + 32 * idx, "width": 62.0, "height": 18.0},
        "visible": visible,
        "idx": idx,
        "total": total,
    }


class _MultiBackend:
    """find_elements-capable fake: N descriptor-stamped matches + click log.

    Deliberately has NO ``find_element``: ``find_elements`` is the
    authoritative resolution when a backend offers it.
    """

    name = "fake"

    def __init__(self, descs: list[dict]) -> None:
        self._descs = descs
        self.clicked: list[int] = []

    async def find_elements(self, locator):
        return [
            Element(
                raw={"js": f"all[{i}]"},
                locator=locator,
                backend="cdp",
                descriptor=d,
                _cache={
                    "tag": d.get("tag", ""),
                    "text": d.get("text", ""),
                    "bounding_box": d.get("rect"),
                    "is_visible": bool(d.get("visible", True)),
                },
            )
            for i, d in enumerate(self._descs)
        ]

    async def click(self, element, *, force: bool = False) -> None:
        self.clicked.append(int(element.descriptor["idx"]))


def test_multi_click_identifies_then_clicks_all_rendered(capsys):
    from avion.cli import _cmd_click

    b = _MultiBackend(
        [
            _desc(0, 3, el_id="a"),
            _desc(1, 3, el_id="b"),
            _desc(2, 3, el_id="c", visible=False),
        ]
    )
    code, hits = asyncio.run(_cmd_click(b, "Go", "text"))
    assert code == 0
    # backward over rendered indices [0, 1] → 1 first; hidden 2 never clicked
    assert b.clicked == [1, 0]
    assert len(hits) == 2
    out = capsys.readouterr()
    # identify block: the WHOLE set with ordinals before any click
    assert "3 matches for text: Go:" in out.out
    assert "button#a 'Go'" in out.out and "(match 1/3)" in out.out
    assert "button#c 'Go'" in out.out and "(hidden)" in out.out
    # the hidden skip is reported on stderr
    assert "skipping hidden match 3/3" in out.err
    # click lines carry the ORIGINAL set ordinals, in execution order
    lines = [ln for ln in out.out.splitlines() if ln.startswith("clicked text:")]
    assert len(lines) == 2
    assert "button#b" in lines[0] and "(match 2/3)" in lines[0]
    assert "button#a" in lines[1] and "(match 1/3)" in lines[1]


def test_click_nth_picks_one_match(capsys):
    from avion.cli import _cmd_click

    b = _MultiBackend([_desc(0, 2, el_id="a"), _desc(1, 2, el_id="b")])
    code, hits = asyncio.run(_cmd_click(b, "Go", "text", nth=2))
    assert code == 0
    assert b.clicked == [1] and len(hits) == 1
    out = capsys.readouterr()
    # the set is still identified — nth picks from it, it does not hide it
    assert "2 matches for text: Go:" in out.out
    assert "clicked text: Go → button#b 'Go'" in out.out and "(match 2/2)" in out.out
    assert out.err == ""


def test_click_nth_out_of_range_fails_without_clicking(capsys):
    from avion.cli import _cmd_click

    b = _MultiBackend([_desc(0, 2, el_id="a"), _desc(1, 2, el_id="b")])
    code, hits = asyncio.run(_cmd_click(b, "Go", "text", nth=3))
    assert code == 1 and hits == [] and b.clicked == []
    assert "--nth 3 out of range" in capsys.readouterr().err


def test_multi_click_all_hidden_falls_back_to_first(capsys):
    from avion.cli import _cmd_click

    b = _MultiBackend(
        [_desc(0, 2, el_id="a", visible=False), _desc(1, 2, el_id="b", visible=False)]
    )
    code, hits = asyncio.run(_cmd_click(b, "Go", "text"))
    assert code == 0
    assert b.clicked == [0] and len(hits) == 1  # first match, like pre-multi
    err = capsys.readouterr().err
    assert "all 2 matches are hidden" in err
    assert "clicking the first match anyway" in err
