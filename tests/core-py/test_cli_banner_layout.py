"""Banner/box layout regressions — boxes must stay square and the wordmark right.

Two defects fixed 2026-11-27:

* ``--help`` header (``apps/cli/src/cli.py::_show_welcome_banner``) rendered
  "TLO" — the T glyph (``████████╗ / ╚══██╔══╝``) where S belongs — and its
  art rows were 39–40 chars inside a 38-char box interior, so the right
  border was pushed out of alignment (the "looks weird / whack" report).
* The interactive welcome banner (``core/helpers.py::show_welcome_banner``)
  spelled "BLOCH", ran art rows 2 chars short of the box, and its version
  line overflowed to 94 chars against a 64-char box because the padding was
  hardcoded for a shorter string.

Both banners now project ``core.tui`` (one renderer for every CLI box), so
the invariants below are properties of the renderer, not of two string
literals: uniform width, visible-width padding (ANSI-safe), truncation
instead of overflow, and a narrow-terminal fallback.
"""

import re

import pytest

from apps.cli.src.core import tui
from apps.cli.src.core.helpers import show_welcome_banner

ANSI = re.compile(r"\x1b\[[0-9;]*m")
BOX_CHARS = "╔╗╚╝║═┌┐└┘│─"


def _box_rows(text: str) -> list[str]:
    return [ANSI.sub("", ln) for ln in text.splitlines() if any(c in ln for c in BOX_CHARS)]


def _assert_square(rows: list[str]) -> None:
    assert rows, "no box rows rendered"
    widths = {len(r) for r in rows}
    assert len(widths) == 1, f"box rows differ in width: {sorted(widths)}\n" + "\n".join(rows)


def _assert_wordmark_is_slo(rows: list[str]) -> None:
    art = "\n".join(rows)
    # S glyph (ANSI Regular). T and B share the "████████╗"-style top bar and
    # are the two wrong first glyphs this test exists to prevent.
    assert "███████╗" in art, "S glyph missing from wordmark"
    assert "████████╗" not in art, "T glyph present — wordmark reads TLO"
    assert "██████╗ ██╗" not in art, "B glyph present — wordmark reads BLOCH"


class TestRenderer:
    def test_pads_on_visible_width_not_bytes(self):
        # A coloured string is padded by its visible width, so geometry holds.
        coloured = "\033[36mabc\033[0m"
        assert tui.fit_visible(coloured, 6) == coloured + "   "

    def test_truncates_instead_of_overflowing(self):
        out = tui.render_box(["x" * 100], 10, color=False)
        _assert_square(_box_rows(out))

    def test_no_color_emits_no_escapes(self):
        out = tui.render_box(["hello", ("world", tui.FG.PRIMARY)], 20, color=False)
        assert "\033[" not in out

    def test_narrow_terminal_falls_back_to_one_line_wordmark(self):
        rows = tui.art_rows(10)
        assert rows == [" " * 3 + "SLO"]


class TestHelpHeaderBanner:
    def test_rows_are_uniform_and_wordmark_is_slo(self, capsys):
        from apps.cli.src.cli import _show_welcome_banner

        _show_welcome_banner()
        rows = _box_rows(capsys.readouterr().out)
        _assert_square(rows)
        _assert_wordmark_is_slo(rows)


class TestWelcomeBanner:
    def test_rows_are_uniform_and_wordmark_is_slo(self, capsys):
        show_welcome_banner("sloughgpt v0.1.0 (Linux x86_64, Python 3.12.14)")
        rows = _box_rows(capsys.readouterr().out)
        _assert_square(rows)
        _assert_wordmark_is_slo(rows)

    @pytest.mark.parametrize(
        "version",
        [
            "v0.1.0",
            "sloughgpt v0.1.0 (Linux x86_64, Python 3.12.14)",
            "x" * 200,
        ],
    )
    def test_version_never_breaks_the_box(self, capsys, version):
        show_welcome_banner(version)
        rows = _box_rows(capsys.readouterr().out)
        _assert_square(rows)

    @pytest.mark.parametrize("cols", [40, 60, 80, 120])
    def test_box_fits_the_terminal(self, capsys, monkeypatch, cols):
        monkeypatch.setattr(tui, "_scrn", lambda: (cols, 24))
        show_welcome_banner("sloughgpt v0.1.0")
        rows = _box_rows(capsys.readouterr().out)
        _assert_square(rows)
        assert max(len(r) for r in rows) <= cols, f"box wider than {cols} cols"
