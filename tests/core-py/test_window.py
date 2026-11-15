"""Tests for the window frame: geometry, chrome, and the cube.

These read the chrome layer's own framebuffer rather than rendering through a
terminal harness. The rendered bytes are the *engine's* business and are already
covered in ``test_graphics``; what matters here is which cell each part of the
frame claims, and the layer is where that is decided.

The cube assertions are pinned by exact cell. Two off-by-ones lived in
``draw_cube`` -- the front wall rule was written one cell too wide, and the
receding diagonals overwrote the corners they were supposed to meet -- and both
produced a frame that looked entirely plausible on screen. Only reading the exact
cell catches them.
"""

from __future__ import annotations

from domain.shell._internal.graphics import (
    BoxStyle,
    Framebuffer,
    GraphicsEngine,
)
from domain.shell._internal.window import Chrome, Rect, Window

ROWS, COLS = 24, 80


# ── helpers ──────────────────────────────────────────────────────────────


def make_engine(rows: int = ROWS, cols: int = COLS) -> GraphicsEngine:
    """An engine sized but never opened.

    ``open()`` emits alt-screen and cursor escapes and probes the real terminal,
    neither of which belongs in a unit test. The buffers are all the renderer
    needs, so allocate them directly.
    """
    eng = GraphicsEngine()
    eng._rows, eng._cols = rows, cols
    eng._front_buffer = Framebuffer(rows, cols)
    eng._back_buffer = Framebuffer(rows, cols)
    return eng


def make_window(
    chrome: Chrome | None = None,
    *,
    title: str = "pane",
    w: int = COLS,
    h: int = ROWS - 3,
    x: int = 0,
    y: int = 2,
    box: BoxStyle = BoxStyle.SINGLE,
) -> Window:
    win = Window("w", Rect(x, y, w, h), title=title, box=box, chrome=chrome)
    win.attach(make_engine())
    return win


# The cube is the default (Chrome.depth = 2), so any test asserting *flat*
# geometry has to say so. Omitting it silently changes what is under test,
# which is how a flat-frame assertion starts passing against a cube.
FLAT = Chrome(depth=0)


def text(win: Window, row: int) -> str:
    """One row of the window's chrome layer, window-local coordinates."""
    fb = win.chrome_layer.framebuffer
    return "".join(fb.get(row, c).char for c in range(fb.cols))


def cell(win: Window, row: int, col: int):
    return win.chrome_layer.framebuffer.get(row, col)


# ── Rect ─────────────────────────────────────────────────────────────────


class TestRect:
    def test_empty_when_no_area(self):
        assert Rect(0, 0, 0, 5).empty
        assert Rect(0, 0, 5, 0).empty

    def test_not_empty_with_area(self):
        assert not Rect(0, 0, 1, 1).empty

    def test_inner_insets_every_side(self):
        assert Rect(10, 20, 10, 6).inner(1) == Rect(11, 21, 8, 4)

    def test_contains_is_half_open(self):
        r = Rect(2, 3, 4, 5)
        assert r.contains(3, 2)
        # One past the far edge is outside on both axes: a 4-wide rect spans
        # cols 2..5, so col 6 is not in it.
        assert not r.contains(8, 2)
        assert not r.contains(3, 6)


# ── Chrome ───────────────────────────────────────────────────────────────


class TestChrome:
    def test_default_depth_is_a_cube(self):
        """The window is a cube by default; 0 is the opt-out.

        A default of 0 would put the burden on every caller, so the only
        windows rendered as cubes would be the ones someone remembered to ask
        about -- charts, overlays and dialogs would stay rectangles and the TUI
        would read as two different products. Flat callers pin ``depth=0``.
        """
        assert Chrome().depth == 2
        assert Chrome(depth=0).depth == 0

    def test_as_dict_carries_depth(self):
        assert Chrome(depth=3).as_dict()["depth"] == 3

    def test_as_dict_carries_the_diagonal_glyphs(self):
        d = Chrome().as_dict()
        assert d["diag_down"] == "\\"
        assert d["diag_up"] == "/"

    def test_is_frozen(self):
        import dataclasses

        import pytest

        with pytest.raises(dataclasses.FrozenInstanceError):
            Chrome().depth = 3  # type: ignore[misc]


# ── flat frame geometry ──────────────────────────────────────────────────


class TestFlatFrame:
    """Flat-frame geometry. Every window here is explicitly ``FLAT``.

    The cube is the default, so these would otherwise assert flat expectations
    against a cube and fail for a reason that has nothing to do with them.
    """

    def test_inset_is_one_for_a_boxed_window(self):
        assert make_window(FLAT)._inset == 1

    def test_inset_is_zero_for_a_borderless_window(self):
        win = make_window(FLAT, box=BoxStyle.NONE)
        assert win._inset == 0

    def test_content_rect_sits_inside_the_border(self):
        win = make_window(FLAT)
        inner = win.inner_rect()
        assert inner.x == win.rect.x + 2  # border + 1 padding
        assert inner.y == win.rect.y + 3  # border + header band + 1 padding

    def test_padding_is_squeezed_rather_than_eating_the_body(self):
        # A window with room for exactly one content row must still have one.
        win = make_window(FLAT, title="t", w=20, h=5, y=0)
        assert win.inner_rect().h >= 1

    def test_header_band_is_dropped_when_it_would_leave_no_body(self):
        win = make_window(FLAT, title="t", w=20, h=3, y=0)
        assert win._header_h == 0

    def test_write_is_clipped_to_the_content_area(self):
        win = make_window(FLAT, w=30, h=10)
        assert win.write(0, 0, "hello") == 5
        # One row past the body writes nothing and says so by returning 0.
        assert win.write(win.inner_rect().h, 0, "nope") == 0
        # Text longer than the body is truncated to the body's width.
        assert win.write(0, 0, "x" * 999) == win.inner_rect().w


# ── the cube ─────────────────────────────────────────────────────────────


class TestCubeFits:
    def _depth(self, depth: int) -> Chrome:
        return Chrome(
            rule="─",
            rule_after="─",
            marker="▌",
            frame=238,
            frame_focused=244,
            depth=depth,
        )

    def test_roomy_window_takes_the_cube(self):
        win = make_window(self._depth(2), h=ROWS - 3)
        assert win._cube_fits
        assert win._inset == 2

    def test_window_too_short_falls_back_to_flat(self):
        # h = 2*depth + 3, one row under the 2*depth + 4 a wall needs.
        win = make_window(self._depth(2), h=7)
        assert not win._cube_fits
        assert win._inset == 1

    def test_window_too_narrow_falls_back_to_flat(self):
        win = make_window(self._depth(3), w=8, h=ROWS - 3)
        assert not win._cube_fits
        assert win._inset == 1

    def test_fallback_draws_no_perspective(self):
        win = make_window(self._depth(2), h=7)
        assert "\\" not in text(win, 0)
        assert "/" not in text(win, 0)

    def test_depth_zero_is_flat(self):
        win = make_window(Chrome(depth=0), h=ROWS - 3)
        assert not win._cube_fits
        assert win._inset == 1


class TestCubeGeometry:
    def _cube(self, depth: int, **kw) -> Window:
        chrome = Chrome(
            rule="─",
            rule_after="─",
            marker="▌",
            frame=238,
            frame_focused=244,
            depth=depth,
        )
        win = make_window(chrome, h=2 * depth + 6, y=2, **kw)
        win.draw_cube(active=True)
        return win

    def test_front_wall_corners_land_on_the_exact_cells(self):
        for depth in (1, 2, 3):
            win = self._cube(depth, w=44)
            d = depth
            assert cell(win, d, d).char == "┏"
            assert cell(win, d, 44 - d - 1).char == "┓"
            assert cell(win, win.rect.h - 1 - d, d).char == "┗"
            assert cell(win, win.rect.h - 1 - d, 44 - d - 1).char == "┛"

    def test_diagonal_glyphs_come_from_the_chrome_not_the_literals(self):
        """The receding edges are configurable, and both orientations are wired.

        Geometry is identical either way -- only the glyph changes -- so this
        pins that the pair is actually read from ``Chrome`` and that swapping it
        cannot silently leave one side on the old glyph.
        """
        for depth in (1, 2, 3):
            chrome = Chrome(
                rule="─",
                rule_after="─",
                frame=238,
                frame_focused=244,
                depth=depth,
                diag_down="╲",
                diag_up="╱",
            )
            win = make_window(chrome, h=2 * depth + 6, y=2, w=44)
            win.draw_cube(active=True)
            fb = win.chrome_layer.framebuffer
            for row in range(depth):
                runs = {}
                for col in range(fb.cols):
                    ch = fb.get(row, col).char
                    if ch in "╲╱":
                        runs.setdefault(ch, []).append(col)
                assert runs.get("╲"), f"depth={depth} row={row}: no ╲ run"
                assert runs.get("╱"), f"depth={depth} row={row}: no ╱ run"
                # And nothing is left on the ASCII pair.
                for col in range(fb.cols):
                    assert fb.get(row, col).char not in "\\/", (
                        f"depth={depth} row={row} col={col}: ASCII diagonal "
                        f"survived the Unicode swap"
                    )

    def test_default_diagonals_are_ascii_for_glyph_coverage(self):
        """The default is ASCII because U+2571/U+2572 are not universal.

        Measured over the 68 monospace faces installed here, 14 lack the Unicode
        diagonals -- every Nimbus Mono and Roboto Mono weight. A default that
        renders tofu in one terminal font in five is worse than a default that
        renders a dotted edge in all of them, so this pins the *conservative*
        choice and leaves the continuous edge opt-in.
        """
        chrome = Chrome()
        assert chrome.diag_down == "\\"
        assert chrome.diag_up == "/"
        assert "╲" not in chrome.diag_down and "╱" not in chrome.diag_up

    def test_both_wall_rails_run_the_full_height(self):
        win = self._cube(3, w=44)
        d, last = 3, win.rect.h - 1 - 3
        for row in range(d + 1, last):
            assert cell(win, row, d).char == "┃", f"left rail broken at row {row}"
            assert cell(win, row, 44 - d - 1).char == "┃", f"right rail at row {row}"

    def test_ceiling_and_floor_are_mirrored_left_to_right(self):
        # A room drawn one column wider on one side than the other is a drawing
        # mistake, not perspective. This caught the right-hand diagonal being
        # placed by `W - d - lc`, which agrees with the true mirror only on the
        # ceiling's back row and is one column short on every row after it.
        for depth in (1, 2, 3):
            win = self._cube(depth, w=44)
            fb = win.chrome_layer.framebuffer
            bands = list(range(depth)) + list(range(win.rect.h - 1 - depth, win.rect.h))
            for row in bands:
                inked = [c for c in range(fb.cols) if fb.get(row, c).char.strip()]
                mirrored = [fb.cols - 1 - c for c in inked]
                assert mirrored == list(reversed(inked)), (
                    f"depth={depth} row={row}: ink {inked} has no mirror partner {mirrored}"
                )

    def test_receding_edges_taper_by_one_cell_per_row(self):
        # The defining property of a receding trapezoid, and the one thing that
        # separates a taper from a slab. Without this, replacing the shrinking
        # run with a constant-length one still passes every other check here:
        # the corners land, the rails run, the room stays mirror-symmetric, and
        # the diagonals are all still there. It just reads as a thick step
        # rather than as distance.
        for depth in (2, 3):
            win = self._cube(depth, w=44)
            fb = win.chrome_layer.framebuffer

            def run(row: int, glyph: str) -> list[int]:
                """Columns holding *glyph* on *row*."""
                return [c for c in range(fb.cols) if fb.get(row, c).char == glyph]

            # The taper is only *visible* below the back edge. The back edge is
            # written over the ceiling's first row and clips that row's run, so
            # the rows that show an unbroken taper are 1..depth-1, where the run
            # shrinks by exactly one cell per row. The floor's nearest row is the
            # full-width front edge for the same reason.
            # The far row's run is `depth - 1` cells; the row above it is clipped back to
            # a single cell by the back edge (ceiling) or the front edge (floor).
            # So the visible runs read `depth-1, 1` for depth 3 and `1` for
            # depth 2 -- not a monotone taper, which is why an earlier version of
            # this assertion was wrong about its own renderer.
            def visible(row: int) -> int:
                return len(run(row, "\\"))

            ceiling = [visible(r) for r in range(1, depth)]
            floor = [visible(win.rect.h - 1 - r) for r in range(1, depth)]
            want = list(range(depth - 1, 0, -1))[:-1] + [1] if depth > 1 else []
            assert ceiling == want, f"depth={depth} ceiling runs {ceiling} != {want}"
            assert floor == ceiling, f"depth={depth} floor runs {floor} != ceiling {ceiling}"
            # And the edges step outward toward the wall, one column per row: the left
            # diagonal starts one column further left on each row down, the right
            # one further right. The floor has one row fewer than the ceiling --
            # its nearest row is the full-width front edge, not a diagonal.
            for rows, want, where in (
                # Ceiling, far row first: the left edge steps left toward the wall,
                # ending exactly on the wall's corner column.
                (range(depth), [2 * depth - 1 - i for i in range(depth)], "ceiling"),
                # Floor, far row first: the left edge steps *right* toward the
                # wall, and its nearest row is the front edge, not a diagonal.
                (
                    range(win.rect.h - depth, win.rect.h - 1),
                    list(range(depth, 2 * depth - 1)),
                    "floor",
                ),
            ):
                left = [min(run(r, "\\")) for r in rows]
                right = [min(run(r, "/")) for r in rows]
                assert left == want, (
                    f"depth={depth} {where} left edge recedes as {left}, expected {want}"
                )
                # The right edge is the mirror of the left about the window's own width, so
                # state it that way instead of a direction: reading rows far-first
                # the ceiling's right edge steps right and the floor's steps left,
                # and a sorted() assertion silently passes on either.
                # Both sides taper by the same amount on the same rows. Comparing
                # the two runs' *start columns* would be wrong: the left
                # diagonal leans one way and the right one the other, so their
                # runs mirror as sets and not start-for-start. Whole-row set
                # mirroring is pinned by the separate test above.
                for row in rows:
                    lrun, rrun = len(run(row, "\\")), len(run(row, "/"))
                    assert lrun == rrun, (
                        f"depth={depth} {where} row {row}: left run {lrun} cells, right run {rrun}"
                    )
                assert all(ls < rs for ls, rs in zip(left, right)), (
                    f"depth={depth} {where} diagonals crossed: {left} / {right}"
                )

    def test_front_face_wins_against_every_far_surface(self):
        """The order of the draw blocks *is* the depth, so pin the outcome.

        Four surfaces meet at the wall's corners: the receding diagonals, the
        ceiling's back edge, the floor's front edge, and the wall itself. There is
        no per-cell z, so the last writer wins and only the wall is allowed to.

        This exists because that ordering used to be an accident of statement
        order. Reordering the blocks for readability broke it, and it fails
        quietly: a `\\` where `┏` belongs is still a plausible receding edge, so
        the frame keeps rendering and the front face is simply gone. Asserting
        the corner *glyph* catches it; asserting "something is inked there" does
        not.
        """
        for depth in (1, 2, 3):
            win = self._cube(depth, w=44)
            fb = win.chrome_layer.framebuffer
            corners = (
                (depth, depth),
                (depth, 44 - depth - 1),
                (win.rect.h - 1 - depth, depth),
                (win.rect.h - 1 - depth, 44 - depth - 1),
            )
            for row, col in corners:
                got = fb.get(row, col).char
                assert got in "┏┓┗┛", (
                    f"depth={depth} corner ({row},{col}) is {got!r}, not a wall "
                    f"corner -- a far surface was painted after the front face"
                )
            # And the two edges that cap the receding surfaces belong to those
            # surfaces, not the wall: a heavy wall rule running the full width
            # would mean the ceiling had been drawn over by the front face.
            assert fb.get(0, 2 * depth).char == "─", "ceiling back edge missing"
            assert fb.get(win.rect.h - 1, 0).char == "─", "floor front edge missing"

    def test_side_walls_are_left_blank(self):
        # The margins between the window edge and the front wall read as empty
        # space. A glyph there would compete with the text.
        win = self._cube(3, w=44)
        d, last = 3, win.rect.h - 1 - 3
        for row in range(d, last + 1):
            for col in list(range(d)) + list(range(44 - d, 44)):
                assert cell(win, row, col).char == " ", f"stray at ({row},{col})"

    def test_ceiling_and_floor_recede_in_diagonals(self):
        win = self._cube(3, w=44)
        d = 3
        ceiling = "".join(text(win, r) for r in range(d))
        floor = "".join(text(win, win.rect.h - 1 - r) for r in range(d))
        for band in (ceiling, floor):
            assert "\\" in band, f"no ceiling/floor diagonals in {band!r}"
            assert "/" in band

    def test_content_stays_inside_the_front_wall(self):
        win = self._cube(3, w=44)
        d, last = 3, win.rect.h - 1 - 3
        inner = win.inner_rect()
        local = Rect(inner.x - win.rect.x, inner.y - win.rect.y, inner.w, inner.h)
        assert local.x >= d
        assert local.y >= d
        assert local.x + local.w <= 44 - d
        assert local.y + local.h <= last + 1

    def test_content_writes_land_on_the_wall(self):
        win = self._cube(2, w=44)
        assert win.write(0, 0, "hello") == 5
        # The title rides on the wall's top edge -- row `depth` -- not on the top
        # of the whole block, where it would sit out on the ceiling.
        assert "pane" in text(win, 2)
        assert "pane" not in text(win, 0)

    def test_cube_survives_a_resize(self):
        # `resize` and `redraw_all` both redraw chrome; if each picked its own
        # renderer a cube window came back from a resize drawn flat.
        win = self._cube(2, w=44)
        assert "┏" in text(win, 2)
        win.resize(Rect(0, 2, 44, 10))
        assert "┏" in text(win, 2), "resize reverted the cube to a flat frame"

    def test_untitled_cube_has_no_header_band(self):
        win = self._cube(2, w=44, title="")
        assert win._header_h == 0
        assert cell(win, 2, 2).char == "┏"
