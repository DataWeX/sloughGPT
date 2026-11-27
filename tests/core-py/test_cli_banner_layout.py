"""Banner layout regressions — boxes must stay square and the wordmark right.

Two defects fixed 2026-11-27:

* ``--help`` header (``apps/cli/src/cli.py::_show_welcome_banner``) rendered
  "TLO" — the T glyph (``████████╗ / ╚══██╔══╝``) where S belongs — and its
  art rows were 39–40 chars inside a 38-char box interior, so the right
  border was pushed out of alignment (the "looks weird / whack" report).
* The interactive welcome banner (``core/helpers.py::show_welcome_banner``)
  spelled "BLOCH", ran art rows 2 chars short of the box, and its version
  line overflowed to 94 chars against a 64-char box because the padding was
  hardcoded for a shorter string.

These tests assert the invariants, not the exact glyphs: every box row is
one width, the wordmark is SLO (no T/B first glyph), and a long version
string truncates instead of breaking the box.
"""

import re

import pytest

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
