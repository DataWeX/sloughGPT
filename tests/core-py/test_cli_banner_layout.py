"""Banner/box layout regressions — boxes must stay square and the wordmark right.

Three defects, all found the same way: look at what the terminal actually draws.

* 2026-11-27 — ``--help`` header (``apps/cli/src/cli.py::_show_welcome_banner``)
  rendered "TLO" (the T glyph where S belongs) and its art rows were 39–40
  chars inside a 38-char box interior, pushing the right border out of
  alignment (the "looks weird / whack" report).
* 2026-11-27 — the interactive welcome banner (``core/helpers.py``)
  spelled "BLOCH", ran art rows 2 chars short of the box, and its version line
  overflowed to 94 chars against a 64-char box.
* 2026-11-27 — **naming**: the banner is the CLI of sloughGPT, but the wordmark
  spelled "SLO", which is a *brand we build*, not the product. Showing the
  brand as the product's own logo is the conflation this test now prevents: the
  art must spell the product name exactly.

Both banners project ``core.tui`` (one renderer for every CLI box), so the
invariants below are properties of the renderer, not of two string literals:
uniform width, visible-width padding (ANSI-safe), truncation instead of
overflow, and a narrow-terminal fallback.
"""

import re

import pytest

from apps.cli.src.core import tui
from apps.cli.src.core.helpers import show_welcome_banner

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def set_terminal(monkeypatch, cols, rows=40):
    """Pin the terminal size for *both* tui module objects.

    ``apps.cli.src.core.tui`` (imported here) and ``core.tui`` (imported by
    ``apps/cli/src/cli.py`` off ``sys.path``) are distinct module objects, so
    patching ``_scrn`` on one leaves the banner reading the real terminal — the
    width assertions then pass on the runner's own width, not the width they
    asked for. ``shutil.get_terminal_size()`` honours COLUMNS/LINES in both.
    """
    monkeypatch.setenv("COLUMNS", str(cols))
    monkeypatch.setenv("LINES", str(rows))
    monkeypatch.setattr(tui, "_scrn", lambda: (cols, rows))


BOX_CHARS = "╔╗╚╝║═┌┐└┘│─"

#: The rendered product wordmark, pinned literally.
#:
#: A glyph table alone cannot catch a wrong *word* — asserting the composed
#: output is what makes "SLO" (or "TLO"/"BLOCH") a test failure rather than a
#: matter of taste. ``#`` is a filled block; rows are trailing-stripped.
EXPECTED_ART = (
    "      █                       █     ▄▀▀▀▄ █▀▀▀▄ ▀▀█▀▀",
    "▄▀▀▀  █     ▄▀▀▀▄ █   █ ▄▀▀▀▄ █▀▀▀▄ █ ▄▄▄ █▄▄▄▀   █",
    " ▀▀▀▄ █     █   █ █   █ ▀▄▄▄▀ █   █ █   █ █       █",
    "▀▀▀▀  ▀▀▀    ▀▀▀   ▀▀▀  ▄▄▄▄▀ ▀   ▀  ▀▀▀  ▀       ▀",
)


def _box_rows(text: str) -> list[str]:
    return [ANSI.sub("", ln) for ln in text.splitlines() if any(c in ln for c in BOX_CHARS)]


def _assert_square(rows: list[str]) -> None:
    assert rows, "no box rows rendered"
    widths = {len(r) for r in rows}
    assert len(widths) == 1, f"box rows differ in width: {sorted(widths)}\n" + "\n".join(rows)


def _assert_wordmark_is_product(rows: list[str]) -> None:
    """The box must contain the product wordmark, not the SLO brand mark."""
    art = "\n".join(rows)
    assert tui.WORDMARK == "sloughGPT"
    # No legacy SLO/ANSI-Regular glyph survives anywhere in the banner.
    assert "███████╗" not in art, "old SLO glyph present — banner still shows the brand mark"
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


class TestWordmark:
    def test_art_spells_the_product_name(self):
        """Glyph-level regression: the drawn wordmark must match, row for row."""
        block = max(len(row) for row in EXPECTED_ART)
        pad = (62 - block) // 2
        assert list(tui.art_rows(62)) == [" " * pad + row for row in EXPECTED_ART]

    def test_narrow_terminal_falls_back_to_the_product_text(self):
        rows = tui.art_rows(20)
        assert rows == [" " * 5 + "sloughGPT"]

    def test_fallback_is_the_product_name_not_the_brand(self):
        assert tui.art_rows(10) == ["sloughGPT"]

    def test_art_fits_the_interactive_box_interior(self):
        assert max(len(r) for r in tui.art_rows(62)) <= 62

    def test_art_packs_two_pixel_rows_per_terminal_row(self):
        """Terminal cells are ~1:2, so one character per pixel doubles the
        height and the letters break into slivers (the 'unreadable' report)."""
        assert all(set(row) <= set(" ▀▄█") for row in tui.WORDMARK_ART), (
            "art must be drawn with half/full blocks"
        )
        assert len(tui.WORDMARK_ART) * 2 == tui._GLYPH_ROWS


class TestHelpHeaderBanner:
    def test_rows_are_uniform_and_wordmark_is_the_product(self, capsys):
        from apps.cli.src.cli import _show_welcome_banner

        _show_welcome_banner()
        out = capsys.readouterr().out
        rows = _box_rows(out)
        _assert_square(rows)
        _assert_wordmark_is_product(rows)

    def test_wide_terminal_draws_the_block_wordmark(self, capsys, monkeypatch):
        from apps.cli.src.cli import _show_welcome_banner

        set_terminal(monkeypatch, 80, 40)
        _show_welcome_banner()
        out = ANSI.sub("", capsys.readouterr().out)
        for row in EXPECTED_ART:
            assert row in out, f"help header is missing wordmark row: {row!r}"

    def test_narrow_terminal_still_names_the_product(self, capsys, monkeypatch):
        from apps.cli.src.cli import _show_welcome_banner

        set_terminal(monkeypatch, 60, 40)
        _show_welcome_banner()
        out = ANSI.sub("", capsys.readouterr().out)
        _assert_square(_box_rows(out))
        assert "sloughGPT" in out


class TestTreeStamp:
    """Which checkout answers — the banner must say it without overflowing."""

    def test_banner_names_the_running_tree(self, capsys, monkeypatch):
        from apps.cli.src.cli import _show_welcome_banner

        set_terminal(monkeypatch, 80, 40)
        _show_welcome_banner()
        out = ANSI.sub("", capsys.readouterr().out)
        line = next((ln for ln in out.splitlines() if ln.strip().startswith("tree")), "")
        assert line, "banner is missing the tree stamp"
        assert "current" in line or "behind" in line

    @pytest.mark.parametrize("cols", [60, 80, 120])
    def test_tree_stamp_never_wraps(self, capsys, monkeypatch, cols):
        """A wrapped status line reads as a broken banner."""
        from apps.cli.src.cli import _show_welcome_banner

        set_terminal(monkeypatch, cols, 40)
        _show_welcome_banner()
        out = ANSI.sub("", capsys.readouterr().out)
        line = next(ln for ln in out.splitlines() if ln.strip().startswith("tree"))
        assert len(line) <= cols


class TestWelcomeBanner:
    def test_rows_are_uniform_and_wordmark_is_the_product(self, capsys):
        show_welcome_banner("sloughgpt v0.1.0 (Linux x86_64, Python 3.12.14)")
        rows = _box_rows(capsys.readouterr().out)
        _assert_square(rows)
        _assert_wordmark_is_product(rows)

    def test_banner_draws_the_block_art(self, capsys):
        show_welcome_banner("sloughgpt v0.1.0")
        out = ANSI.sub("", capsys.readouterr().out)
        for row in EXPECTED_ART:
            assert row in out, f"wordmark row missing from banner: {row!r}"
        assert "██████" not in out, "old ANSI-Regular SLO glyph still drawn"

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
        set_terminal(monkeypatch, cols, 24)
        show_welcome_banner("sloughgpt v0.1.0")
        rows = _box_rows(capsys.readouterr().out)
        _assert_square(rows)
        assert max(len(r) for r in rows) <= cols, f"box wider than {cols} cols"
