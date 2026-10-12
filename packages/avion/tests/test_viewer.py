"""Flow-viewer tests — the render half of "map out flows".

Asserts the trace-viewer page: three panes (timeline / SVG map / detail),
step instructions in the side panel, picks showing their resolved selector,
failures marked, and every user string escaped (the page embeds recorded
names and data verbatim, so escaping is a security invariant, not a nit).
"""

from __future__ import annotations

from avion.events.models import Event, EventType
from avion.flow import build_flow, render_html, save_html


def _ev(seq, etype, name="", success=True, data=None, error="", duration_ms=0.0):
    return Event(
        seq=seq,
        type=etype,
        name=name,
        success=success,
        data=data or {},
        error=error,
        duration_ms=duration_ms,
    )


def _sample():
    return build_flow(
        [
            _ev(0, EventType.SESSION_STARTED, "session_start"),
            _ev(1, EventType.NAVIGATE, "go /home"),
            _ev(
                2,
                EventType.SELECT,
                "pick <button>",
                data={
                    "tag": "button",
                    "text": "Save document",
                    "best": "testId='save'",
                    "candidates": [
                        {"strategy": "test_id", "value": "save"},
                        {"strategy": "css", "value": "#save"},
                    ],
                },
            ),
            _ev(3, EventType.CLICK, "click save", duration_ms=12.5),
            _ev(4, EventType.NAVIGATE, "go /settings"),
            _ev(5, EventType.FILL, "fill email", success=False, error="timeout 15s"),
        ],
        session_name="arken_run",
    )


# ── structure: the three panes ───────────────────────────────────────────────


def test_renders_all_three_panes():
    html = render_html(_sample())
    assert 'class="timeline"' in html  # left: step list
    assert 'class="map"' in html and "<svg" in html  # centre: flow map
    assert 'class="details"' in html  # right: detail side panel


def test_renders_every_step_label_and_skips_lifecycle():
    html = render_html(_sample())
    for label in ("go /home", "click save", "fill email"):
        assert label in html
    # labels are escaped — the raw angle-bracket form must NOT appear
    assert "pick &lt;button&gt;" in html
    assert "pick <button>" not in html
    # session lifecycle events are excluded from the map (by build_flow)
    assert "session_start" not in html


def test_header_carries_session_and_counts():
    html = render_html(_sample())
    assert "arken_run" in html
    assert "5 steps" in html and "1 pick" in html and "1 failed" in html
    assert "1 picks" not in html  # pluralised correctly


def test_click_targets_line_up_across_panes():
    """List item, map node and detail panel must share the same index."""
    html = render_html(_sample())
    for i in range(5):
        assert f'id="li{i}"' in html
        assert f'id="n{i}"' in html
        assert f'id="d{i}"' in html
        assert f"selectStep({i})" in html


# ── side panel: step instructions ───────────────────────────────────────────


def test_pick_detail_shows_resolved_selector_and_candidates():
    html = render_html(_sample())
    assert "testId=&#x27;save&#x27;" in html  # best selector, escaped
    assert "picked element" in html
    assert "<code>test_id</code> save" in html


def test_failed_step_shows_error_and_is_marked():
    html = render_html(_sample())
    assert "timeout 15s" in html
    assert 'class="fail"' in html  # timeline row + map node styling
    assert "#dc2626" in html  # failure stroke/fill on the map


def test_duration_shown_when_present():
    html = render_html(_sample())
    assert "12 ms" in html


# ── escaping: recorded strings are attacker-controlled ──────────────────────


def test_script_in_step_name_is_escaped():
    evil = build_flow([_ev(1, EventType.CLICK, "<script>alert(1)</script>")])
    html = render_html(evil)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


def test_script_in_error_is_escaped():
    evil = build_flow([_ev(1, EventType.CLICK, "x", success=False, error="</section>")])
    html = render_html(evil)
    # the error string must appear only in escaped form (the page itself has
    # legitimate </section> closers, so assert the escaped variant instead)
    assert "&lt;/section&gt;" in html


def test_data_json_cannot_break_out_of_markup():
    evil = build_flow([_ev(1, EventType.CLICK, "x", data={"k": "</html><script>"})])
    html = render_html(evil)
    # the data row is inside HTML markup: raw breakout must be impossible
    assert "</html><script>" not in html
    assert "&lt;" in html  # escaped form is what actually renders
    # slash-shield applied before escaping (defence if ever embedded in JS)
    assert "<\\/html>" in html or "&lt;\\/html&gt;" in html


def test_label_with_braces_survives_intact():
    """Regression: the SVG used to patch a '{}' height placeholder AFTER
    assembly, corrupting any step label that contained braces."""
    g = build_flow([_ev(1, EventType.CLICK, "click {} now")])
    html = render_html(g)
    assert "click {} now" in html


# ── segments + empty state ──────────────────────────────────────────────────


def test_page_boundary_names_the_navigation_target():
    html = render_html(_sample())  # second segment opens with "go /settings"
    assert "page: go /settings" in html
    assert "page boundary" not in html  # generic label retired


def test_empty_graph_renders_empty_state():
    html = render_html(build_flow([]))
    assert "No steps recorded yet" in html
    assert "<svg" not in html  # no map for an empty flow


def test_svg_has_real_height_no_placeholder():
    html = render_html(_sample())
    assert 'viewBox="0 0 640 ' in html
    # extract height value: must be an integer > 0
    start = html.index('viewBox="0 0 640 ') + len('viewBox="0 0 640 ')
    height = ""
    for ch in html[start:]:
        if ch.isdigit():
            height += ch
        else:
            break
    assert int(height) > 0


# ── save_html ───────────────────────────────────────────────────────────────


def test_save_html_writes_openable_file(tmp_path):
    out = save_html(_sample(), str(tmp_path / "nested" / "trace.html"))
    text = out.read_text(encoding="utf-8")
    assert text.startswith("<!doctype html>")
    assert "selectStep" in text
    assert out.stat().st_size > 1000


# ── quirk regressions (viewer polish round) ─────────────────────────────────


def test_truncation_keeps_map_list_and_details_aligned():
    """A >500-step flow must cap the MAP at the same 500 the list renders —
    otherwise map nodes past the cap call selectStep() with no detail."""
    big = build_flow([_ev(i, EventType.CLICK, f"step {i}") for i in range(1, 503)])
    html = render_html(big)
    assert html.count('id="li') == 500  # list capped
    assert html.count('id="n') == 500  # map capped to the SAME count
    assert html.count('id="d') == 500  # details capped to the same count
    assert "selectStep(500)" not in html  # indices are 0..499 only
    assert "showing first 500 of 502 steps" in html  # truncation is visible
    assert "502 steps" in html  # header counts the FULL flow


def test_keyboard_navigation_attrs():
    html = render_html(_sample())
    # both panes are operable without a mouse
    assert html.count("onkeydown") == 10  # 5 list items + 5 map nodes
    assert 'tabindex="0"' in html
    assert 'role="button"' in html  # map nodes
    assert 'aria-label="Step ' in html  # map nodes carry accessible names
    assert "aria-current" in html  # selection announced, not just coloured


def test_focus_visible_and_narrow_window_styles():
    html = render_html(_sample())
    assert ":focus-visible" in html  # keyboard focus is visible, not silent
    assert "@media (max-width: 1100px)" in html  # side panels narrow on small screens
    # the media block must be CLOSED — an unclosed one swallows later rules
    assert "290px; }\n  }\n  .seq" in html


def test_long_label_truncated_in_map_full_in_detail():
    long_label = "L" * 60
    g = build_flow([_ev(1, EventType.CLICK, long_label)])
    html = render_html(g)
    # map text node: "[seq] " + 27 chars + ellipsis (fits _NODE_W)
    assert f"[1] {'L' * 27}\u2026" in html
    # the detail panel still shows the full instruction
    assert long_label in html
