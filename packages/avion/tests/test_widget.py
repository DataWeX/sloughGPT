"""Live-state widget — unit half + real-Chromium invisibility.

The unit half proves three invariants:

1. **Projection, not replication** — the pushed frame mirrors the identify
   block exactly (same ``provenance()`` strings, same ordinals), stdout is
   byte-identical with the overlay on and off, and the frame is only built
   when the overlay is on.
2. **The widget can never fail a verb** — an evaluate failure disables the
   bridge and the click proceeds with untouched output.
3. **Capture hygiene** — ``shot`` hides then restores, and hide/restore
   never inject (a page that never had the panel stays panel-free).

The real-Chromium half (fixture imported from ``test_cdp``) proves the
panel is observationally invisible: present in the page, absent from the
light DOM and from ``find_elements``.
"""

from __future__ import annotations

import argparse
import asyncio
import json

from avion.cli import _cmd_click, _cmd_shot, _dispatch, _widget_bridge, _widget_enabled
from avion.interact.widget import (
    _ENSURE_JS,
    _HIDE_EXPR,
    _RESTORE_EXPR,
    WidgetBridge,
    _render_expr,
)
from test_cdp import chromium_ws  # noqa: F401 — pytest fixture for the real half

# ── fakes ────────────────────────────────────────────────────────────────


class _El:
    def __init__(self, desc: str, visible: bool = True) -> None:
        self._desc = desc
        self.is_visible = visible
        self.bounding_box = {"x": 0, "y": 0, "width": 10, "height": 10}

    def provenance(self) -> str:
        return self._desc


class _Backend:
    """Enough surface for click/fill/wait/shot + a recording evaluator."""

    def __init__(self, els: list[_El], evaluate_fails: bool = False) -> None:
        self.els = els
        self.clicked: list[_El] = []
        self.exprs: list[str] = []
        self.evaluate_attempts = 0
        self._evaluate_fails = evaluate_fails

    async def find_elements(self, loc):
        return list(self.els)

    async def find_element(self, loc):
        return self.els[0] if self.els else None

    async def click(self, el):
        self.clicked.append(el)

    async def screenshot(self) -> bytes:
        return b"PNG"

    async def evaluate(self, expr: str):
        self.evaluate_attempts += 1
        if self._evaluate_fails:
            raise RuntimeError("cdp gone")
        self.exprs.append(expr)
        return "widget_rendered"


class _Recorder:
    def __init__(self) -> None:
        self.events: list = []

    def record(self, *a, **k) -> None:
        self.events.append((a, k))


def _frame(expr: str) -> dict:
    """Extract the embedded frame JSON from a render expression."""
    assert expr.startswith("/*avion-widget*/"), "missing widget sentinel"
    # `.render(` matches the CALL site only — the ensure body defines the
    # method as `render(frame) {` with no preceding dot.
    start = expr.rindex(".render(") + len(".render(")
    end = expr.rindex("); return 'widget_rendered'")
    return json.loads(expr[start:end])


def _three_matches() -> tuple[_Backend, list[str]]:
    els = [
        _El("button#a 'Go' @0,0 10x10 (match 1/3)"),
        _El("button#b 'Go' @0,0 10x10 (match 2/3)"),
        _El("button#c 'Go' @0,0 0x0 (match 3/3)", visible=False),
    ]
    return _Backend(els), [e.provenance() for e in els]


# ── projection: frame == identify block, stdout untouched ────────────────


def test_frame_mirrors_identify_block(capsys):
    b, descs = _three_matches()
    w = WidgetBridge(b)
    code, hits = asyncio.run(_cmd_click(b, "Go", "text", widget=w))
    out = capsys.readouterr()

    assert code == 0 and len(hits) == 2
    assert "3 matches for text: Go:" in out.out
    identify = [ln.strip() for ln in out.out.splitlines() if ln.startswith("  ")]
    assert identify == descs  # the identify block IS the frame's desc list

    assert len(b.exprs) == 1
    frame = _frame(b.exprs[0])
    assert frame["action"] == "click" and frame["by"] == "text"
    assert frame["value"] == "Go" and frame["nth"] is None
    assert frame["recording"] is False and frame["code"] == 0
    assert [m["desc"] for m in frame["matches"]] == descs
    assert [m["n"] for m in frame["matches"]] == [1, 2, 3]
    assert [m["hidden"] for m in frame["matches"]] == [False, False, True]
    # clicked: both rendered matches, hidden one untouched
    assert [m["clicked"] for m in frame["matches"]] == [True, True, False]
    assert w.last == frame  # last frame retained (equal to what was pushed)


def test_stdout_byte_identical_with_widget_off_and_on(capsys):
    b_off, _ = _three_matches()
    asyncio.run(_cmd_click(b_off, "Go", "text"))
    out_off = capsys.readouterr()

    b_on, _ = _three_matches()
    w = WidgetBridge(b_on)
    asyncio.run(_cmd_click(b_on, "Go", "text", widget=w))
    out_on = capsys.readouterr()

    assert out_off.out == out_on.out
    assert out_off.err == out_on.err
    assert b_off.evaluate_attempts == 0  # off means ZERO overlay traffic
    assert len(b_on.exprs) == 1


def test_disabled_widget_builds_no_frame():
    b, _ = _three_matches()
    code, hits = asyncio.run(_cmd_click(b, "Go", "text"))
    assert code == 0 and len(hits) == 2
    assert b.exprs == []


# ── the widget can never fail a verb ─────────────────────────────────────


def test_evaluate_failure_never_fails_the_verb(capsys):
    b_off, _ = _three_matches()
    asyncio.run(_cmd_click(b_off, "Go", "text"))
    out_off = capsys.readouterr()

    b, _ = _three_matches()
    b._evaluate_fails = True
    w = WidgetBridge(b)
    code, hits = asyncio.run(_cmd_click(b, "Go", "text", widget=w))
    out_on = capsys.readouterr()

    assert code == 0 and len(hits) == 2
    assert out_on.out == out_off.out and out_on.err == out_off.err
    assert w.enabled is False  # one strike: bridge self-disables
    # a later verb's push is a cheap no-op, not a retry storm
    asyncio.run(_cmd_click(b, "Go", "text", widget=w))
    assert b.evaluate_attempts == 1


def test_bridge_without_evaluate_capability_is_skipped():
    class _NoEval:
        pass

    assert _widget_bridge(_NoEval(), True) is None
    assert _widget_bridge(_Backend([]), False) is None
    assert isinstance(_widget_bridge(_Backend([]), True), WidgetBridge)


# ── capture hygiene ──────────────────────────────────────────────────────


def test_shot_hides_then_restores_without_injecting(tmp_path, capsys):
    b = _Backend([_El("button#a 'Go'")])
    w = WidgetBridge(b)
    path = tmp_path / "s.png"
    code = asyncio.run(_cmd_shot(b, str(path), widget=w))
    out = capsys.readouterr()

    assert code == 0 and path.exists()
    assert f"wrote {path}" in out.out
    assert b.exprs == [_HIDE_EXPR, _RESTORE_EXPR]
    # hide/restore must NEVER inject: a capture of a panel-free page stays
    # panel-free (no createElement anywhere in either expression)
    assert "createElement" not in _HIDE_EXPR
    assert "createElement" not in _RESTORE_EXPR


def test_shot_without_widget_still_hides_for_capture(tmp_path, capsys):
    """Capture hygiene is unconditional: a plain shot after a --widget
    command must not capture the panel either — hide/restore run even
    without the flag (no-ops on a page that never had the panel)."""
    b = _Backend([_El("button#a 'Go'")])
    path = tmp_path / "s.png"
    assert asyncio.run(_cmd_shot(b, str(path))) == 0
    assert b.exprs == [_HIDE_EXPR, _RESTORE_EXPR]
    assert f"wrote {path}" in capsys.readouterr().out


def test_shot_on_backend_without_evaluate_is_untouched(tmp_path, capsys):
    """A backend with no evaluate() (pre-overlay fakes) gets no overlay
    traffic at all — the shot path is byte-for-byte the old one."""

    class _NoEval:
        async def screenshot(self) -> bytes:
            return b"PNG"

    b = _NoEval()
    path = tmp_path / "s.png"
    assert asyncio.run(_cmd_shot(b, str(path))) == 0
    assert f"wrote {path}" in capsys.readouterr().out


# ── recording state reaches the frame ────────────────────────────────────


def test_dispatch_frame_carries_recording_state(capsys):
    b, _ = _three_matches()
    w = WidgetBridge(b)
    code = asyncio.run(_dispatch(b, ["click", "--by", "text", "Go"], widget=w))
    assert code == 0
    assert _frame(b.exprs[0])["recording"] is False
    capsys.readouterr()

    b2, _ = _three_matches()
    w2 = WidgetBridge(b2)
    rec = _Recorder()
    code = asyncio.run(_dispatch(b2, ["click", "--by", "text", "Go"], recorder=rec, widget=w2))
    assert code == 0
    frame = _frame(b2.exprs[0])
    assert frame["recording"] is True
    # recording still works with the overlay on: one ELEMENT_FOUND barrier
    # for the set + one CLICK per hit (2 rendered matches) = 3 events
    assert len(rec.events) == 3


# ── the injected JS: shadow-only, no markup parsing of frame data ────────


def test_injection_is_shadow_dom_and_never_parses_html():
    assert "attachShadow" in _ENSURE_JS
    assert "innerHTML" not in _ENSURE_JS  # frame data is app text — textContent only
    assert "textContent" in _ENSURE_JS
    assert "aria-hidden" in _ENSURE_JS
    assert "pointer-events:none" in _ENSURE_JS  # panel can never eat a click

    expr = _render_expr({"action": "click", "value": "Go"})
    assert expr.startswith("/*avion-widget*/")
    # JSON is embedded as a JS literal with non-ASCII escaped (U+2028/9
    # would otherwise terminate the statement)
    assert "widget_rendered" in expr
    weird = _render_expr({"value": "\u2028\u2029"})
    assert "\u2028" not in weird and "\u2029" not in weird


def test_widget_enabled_flag_and_env(monkeypatch):
    monkeypatch.delenv("AVION_WIDGET", raising=False)
    assert _widget_enabled(argparse.Namespace(widget=True))
    assert not _widget_enabled(argparse.Namespace(widget=False))
    assert not _widget_enabled(argparse.Namespace())  # getattr default
    monkeypatch.setenv("AVION_WIDGET", "1")
    assert _widget_enabled(argparse.Namespace(widget=False))
    monkeypatch.delenv("AVION_WIDGET")
    assert not _widget_enabled(argparse.Namespace(widget=False))


# ── real Chromium: present in the page, invisible to avion's own probes ──


def test_widget_panel_is_invisible_to_avion_probes(chromium_ws, tmp_path, capsys):
    from avion.backends.cdp import CdpBackend
    from avion.cli import main
    from avion.core.element import ElementLocator

    page = tmp_path / "widget.html"
    page.write_text(
        "<!doctype html><title>W</title><body>"
        "<button id='a' onclick='window.__a=(window.__a||0)+1'>Go</button>"
        "<button id='b' onclick='window.__b=(window.__b||0)+1'>Go</button>"
        "</body>"
    )
    from test_cdp import _eval_js

    assert main(["--widget", "--cdp", chromium_ws, "open", page.as_uri()]) == 0
    capsys.readouterr()
    assert main(["--widget", "--cdp", chromium_ws, "click", "--by", "text", "Go"]) == 0
    capsys.readouterr()

    # the panel IS on the page...
    assert _eval_js(chromium_ws, "!!document.querySelector('avion-widget')") is True
    # ...its text never reaches the light DOM...
    assert _eval_js(chromium_ws, "document.body.textContent.includes('AVION')") is False
    assert _eval_js(chromium_ws, "document.body.innerText.includes('AVION')") is False

    # ...and avion's own probes see exactly the 2 buttons, never the host
    async def matches():
        b = CdpBackend(cdp_url=chromium_ws)
        await b.start()
        try:
            found = await b.find_elements(ElementLocator.text("Go"))
            return [e.provenance() for e in found]
        finally:
            await b.stop()

    found = asyncio.run(matches())
    assert len(found) == 2
    assert all("button#" in m for m in found)

    # the panel's state IS the identify frame — clicks landed on both
    last = json.loads(_eval_js(chromium_ws, "JSON.stringify(window.__avionWidget._last)"))
    assert last["action"] == "click"
    assert [m["n"] for m in last["matches"]] == [1, 2]
    assert [m["clicked"] for m in last["matches"]] == [True, True]


def test_shot_roundtrip_leaves_panel_visible(chromium_ws, tmp_path, capsys):
    from avion.cli import main
    from test_cdp import _eval_js

    page = tmp_path / "shot.html"
    page.write_text(
        "<!doctype html><title>S</title><body><button id='a' onclick='void 0'>Go</button></body>"
    )
    assert main(["--widget", "--cdp", chromium_ws, "open", page.as_uri()]) == 0
    capsys.readouterr()
    assert main(["--widget", "--cdp", chromium_ws, "click", "--by", "text", "Go"]) == 0
    capsys.readouterr()

    visible = "document.querySelector('avion-widget').style.display"
    assert _eval_js(chromium_ws, visible) == ""

    out = tmp_path / "cap.png"
    assert main(["--widget", "--cdp", chromium_ws, "shot", str(out)]) == 0
    assert out.exists() and out.stat().st_size > 0
    assert f"wrote {out}" in capsys.readouterr().out
    # restored after the capture — the panel keeps tracking live state
    assert _eval_js(chromium_ws, visible) == ""
    assert _eval_js(chromium_ws, "!!window.__avionWidget._last") is True
