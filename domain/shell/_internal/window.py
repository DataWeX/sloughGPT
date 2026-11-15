"""Windows and tabs — a containment model over the graphics engine.

Why this exists
---------------
``graphics.py`` knows how to composite ``Layer`` s and diff them to the
terminal, but it has no notion of a *window*: a bordered, titled region with
its own drawing surface that can be nested in another and hidden as a unit.
The TUI grew that by hand — six layers named ``console_bg``/``console``/
``output_bg``/``output``/``status``/``input``, each pinned to a ``Pane``
region by ``_place_layers`` — so every new pane cost a field on ``TuiRepl``, a
resize branch, and a case in ``_get_layer``.

This module supplies the model.  ``Window`` is the unit, ``TabBar`` groups
windows into switchable tabs, ``WindowManager`` owns the root and the strip.

Design notes
------------
**Reuse, not replacement.**  ``Layer`` draws, ``BoxStyle`` supplies border
glyphs, ``TabManager`` owns focus navigation, ``PaneLayout`` solves geometry.
A ``Window`` owns *Layers* and says where to put them, so adding one is
``add_window(...)`` rather than three new places in the TUI.

**Chrome and content are separate layers.**  Border and title redraw when the
window moves or the active tab changes; the body redraws when its contents
change.  A content redraw therefore does not re-emit the border.

**Visibility is the tab mechanism.**  A tab does not composite differently — it
sets ``window.visible``.  The engine already skips invisible layers, so
switching tabs costs one boolean.

**Background, not a transparency flag.**  A layer declares its ``background``
and calls ``fill_background()``; compositing skips a cell only when it is
blank *and* still ``Color.DEFAULT``.  So ``DEFAULT`` means transparent and a
real colour means a solid panel, with no flag to keep in sync.

**Sole screen ownership.**  ``GraphicsEngine`` is the only thing that writes
terminal bytes.  Nothing here emits any.
"""

from __future__ import annotations

from dataclasses import dataclass

from .graphics import Attr, BoxStyle, Color, GraphicsEngine, Layer

# ── geometry ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Rect:
    """An immutable rectangle in terminal cells."""

    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0

    @property
    def empty(self) -> bool:
        return self.w <= 0 or self.h <= 0

    def inner(self, border: int = 1) -> Rect:
        """The content rect, inset by *border* cells on every side."""
        return Rect(self.x + border, self.y + border, self.w - 2 * border, self.h - 2 * border)

    def contains(self, row: int, col: int) -> bool:
        return self.y <= row < self.y + self.h and self.x <= col < self.x + self.w


# ── chrome ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Chrome:
    """How a window's frame is drawn: which glyphs, in which colours.

    Kept as data rather than literals inside ``redraw_chrome`` so that changing
    the look is a configuration change, not a code change -- and so alternative
    looks can be rendered side by side and judged against each other instead of
    argued about in the abstract.

    The default accent is the product's own brand violet, not cyan. The web app
    sets ``--primary`` to ``#7c52c4``, and ``rgb_to_ansi256(124, 82, 196)``
    lands on index 98; a TUI that accents in cyan while the product accents in
    violet reads as two different applications.
    """

    rule: str = "━"
    """Glyph for the header band. Heavier reads as more prominent; ``─`` is
    quieter and leaves the title to carry the emphasis."""

    accent: int = 98
    """Reserved for the *selected* tab, the focus marker, and content that means
    something. 98 is the brand violet.

    Accent spent on chrome is wasted. Measured against k9s, a frame whose rails
    and rules are drawn in the accent puts 45% of all its inked cells on borders
    and under 1% on content -- the colour then marks the frame and nothing else,
    so there is no accent left to draw attention with when it is needed. The
    rule is: accent means something, or it is not the accent."""

    frame: int = 238
    """Rails, rules and separators -- a recessive grey.

    The frame is what the eye glides over on the way to the content, so it should
    be the quietest thing on screen. 238 is the standard 256-colour grey ramp
    entry; k9s uses the same family for its own rules."""

    frame_focused: int = 244
    """The frame when its window has focus. Brighter grey, deliberately *not* the
    accent: focus is already signalled by the marker glyph, so brightening the
    frame a step is enough. Two channels would mean the accent could be spent on
    content instead."""

    accent_dim: int = 141
    """Accent when de-emphasised -- an unfocused tab label. Lighter than `accent`
    reads as *less* prominent; making it darker is the more common instinct and
    is wrong."""

    band_bg: int = Color.DEFAULT
    """Fill painted behind the title on the band. ``Color.DEFAULT`` leaves the
    band unfilled, so the emphasis comes from the rule and the title alone."""

    marker: str = ""
    """Glyph drawn immediately before the title in the header.

    This is the small-OS look: a solid tick at the leading edge of the title bar,
    the way a window manager marks the focused pane. It costs no row, needs no
    fill, and marks focus in the same place every time, so the eye learns one
    position instead of re-reading a colour. Empty falls back to centring the
    title in the rule."""

    rule_after: str = "─"
    """Glyph for the part of the header row that follows the title. A thin rule
    there keeps the bar readable without shouting over the content."""

    depth: int = 2
    """Cells of perspective on each side. 0 is a flat frame; 2 or 3 draws the
    window as a solid block -- a rectangular front wall with a ceiling, floor and
    side walls receding away from the viewer.

    Default is 2, and that is a product decision rather than a style preference:
    the window *is* a cube now. A default of 0 would mean every window builder
    outside this module -- charts, overlays, dialogs -- drew flat, and the look
    would split into two families. That is the same failure the accent had when
    ``frame`` and ``accent`` were one field.

    2 rather than 3, because the cost is measured: a depth-3 room spends 6 rows
    and 6 columns of chrome on perspective and a small pane falls back to flat
    anyway. 2 fits a normal pane, reads as depth, and leaves the ink budget
    closer to the flat frame. Depth 1 collapses the ceiling to a single row of
    dots, which reads as a mistake rather than as recession.

    This is the one place where "3D" is worth the cells. A flat frame is a
    rectangle, and a rectangle reads as a document. Giving it a visible front face
    with surfaces falling away behind makes it read as a *volume* the reader is
    looking into, and that is what makes a log pane feel like a screen with state
    in it rather than a box of text.

    The front wall stays a true rectangle. Text columns are not skewed, so every
    character still lands on a cell grid and `write` keeps its alignment -- which
    is the whole reason this is drawn as a room and not as a full isometric
    projection. A cube whose text is diagonal cannot be selected, scrolled, or
    diffed, and the diff engine is not going to grow shear support for a frame.

    Costs ``2 * depth`` columns and ``2 * depth`` rows of real estate. Windows too
    small for a usable front wall fall back to the flat frame, so the default is
    safe to apply unconditionally. Callers that *want* flat -- notably the
    comparison variants in ``.splitwork/design_variants.py``, whose whole purpose
    is to be a flat baseline -- must pin ``depth=0`` explicitly."""

    diag_down: str = "\\"
    """Glyph for the receding edges that fall to the right. Default ASCII.

    There is a real choice here and it is not a style preference, it is glyph
    coverage. The Unicode box-drawing diagonals ``╲`` U+2572 and ``╱`` U+2571 are
    designed to join across cell boundaries, so they render a *continuous*
    receding edge; ASCII ``\\`` and ``/`` leave a gap in every cell and read as a
    dotted line. Measured over the 68 monospace faces installed here, 54 carry
    U+2571/U+2572 and **14 do not** -- every Nimbus Mono and every Roboto Mono
    weight. So the continuous-looking edge is unavailable in roughly one terminal
    font in five, where it degrades to tofu, and tofu on a frame is far worse than
    a dotted line is subtle.

    That measurement is why the default is ASCII. Set both fields to the Unicode
    pair to take the continuous edge when you know the font has it.
    """

    diag_up: str = "/"
    """Glyph for the receding edges that fall to the left. See :attr:`diag_down`."""

    def as_dict(self) -> dict[str, object]:
        return {
            "rule": self.rule,
            "accent": self.accent,
            "frame": self.frame,
            "frame_focused": self.frame_focused,
            "accent_dim": self.accent_dim,
            "band_bg": self.band_bg,
            "marker": self.marker,
            "rule_after": self.rule_after,
            "depth": self.depth,
            "diag_down": self.diag_down,
            "diag_up": self.diag_up,
        }


# ── window ──────────────────────────────────────────────────────────


class Window:
    """A titled panel with its own drawing surface.

    Args:
        name: unique identifier.
        rect: where the window sits, in terminal cells.
        title: text for the header band. Empty draws a plain border instead.
        box: border glyph set for the rails and bottom edge. The header band
            replaces the top border, so a window with a title spends no extra
            row on chrome.
        z: layer order. Children are drawn at ``z + 1``, content over chrome.
        background: colour painted across the window's whole rectangle before
            drawing.  ``Color.DEFAULT`` leaves it transparent and whatever is
            behind shows through; a real colour makes a solid panel, so a
            space inside a line of text is content that covers the backdrop.

            Separating two panes by lifting one a step off the background is
            cheaper than a border between them: it costs no glyphs, no cells,
            and nothing for the eye to trace. Mature TUIs keep a small ladder of
            near-blacks and alternate up it. Lifting is only visible on a
            terminal that can show the colour, so pair it with a rule or a border
            rather than relying on it alone.
        padding: cells of breathing room between the frame and the content.
            Text hard against a border is the single cheapest way to make a TUI
            look cramped.
        chrome: glyphs and colours for the frame. Defaults to the brand look.
    """

    def __init__(
        self,
        name: str,
        rect: Rect | None = None,
        title: str = "",
        box: BoxStyle = BoxStyle.ROUNDED,
        *,
        z: int = 10,
        background: int = Color.DEFAULT,
        padding: int = 1,
        chrome: Chrome | None = None,
    ) -> None:
        self.name = name
        self.rect = rect or Rect()
        self.title = title
        self.box = box
        self.z = z
        self.background = background
        self.padding = max(0, padding)
        self.chrome = chrome if chrome is not None else Chrome()
        self.children: list[Window] = []
        self.visible = True
        self.focused = False

        self._chrome: Layer | None = None
        self._content: Layer | None = None
        self._engine: GraphicsEngine | None = None

    # ── wiring ───────────────────────────────────────────────────────
    def attach(self, engine: GraphicsEngine) -> Window:
        """Create this window's layers on *engine*, attach any children, place."""
        self._engine = engine
        self._chrome = engine.create_layer(f"{self.name}.chrome", z=self.z)
        self._chrome.background = self.background
        self._content = engine.create_layer(f"{self.name}.content", z=self.z + 1)
        self._content.background = self.background
        self.resize(self.rect)
        for child in self.children:
            child.attach(engine)
        return self

    @property
    def chrome_layer(self) -> Layer | None:
        """The frame's drawing surface, for callers that need the raw layer.

        Named apart from :attr:`chrome` because that is the *style* -- the
        glyphs and colours -- and the two were briefly the same name, which
        shadowed the style with the layer and made every constructor fail.
        """
        return self._chrome

    @property
    def content(self) -> Layer | None:
        """The drawing surface for the body, already positioned."""
        return self._content

    @property
    def _inset(self) -> int:
        """Cells of frame on each side, including the cube's perspective.

        A cube's front wall sits ``depth`` cells in from the window's edge, so the
        content rect has to account for that or every line of text would land on
        the ceiling and the side walls. Falls back to a flat frame when the window
        is too small to have a usable front wall -- otherwise a short window would
        inset by more than its own size and clip all of its text.
        """
        base = 1 if self.box != BoxStyle.NONE else 0
        d = self.chrome.depth
        if d <= 0 or not self._cube_fits:
            return base
        return d

    @property
    def _cube_fits(self) -> bool:
        """Whether there is room for a front wall worth having.

        Needs ``depth + 2`` on each axis at minimum, and a little more to be
        legible. Below that the perspective is a decorative smudge that eats the
        content, so the window draws flat instead.
        """
        d = self.chrome.depth
        if d <= 0:
            return False
        return self.rect.w >= 2 * d + 6 and self.rect.h >= 2 * d + 4

    @property
    def _header_h(self) -> int:
        """Rows spent on the header band. Zero when there is nothing to label."""
        if not (self.title and self._inset):
            return 0
        # A band that leaves no room for a body is not a label, it is a lid. A
        # three-row window draws as a plain border instead: an empty box with a
        # title on it tells the reader less than a small window that has text in
        # it. Spending the row on content is the better trade at that size.
        if self.rect.h <= 2 * self._inset + 1:
            return 0
        return 1

    # ── geometry ─────────────────────────────────────────────────────
    def resize(self, rect: Rect) -> Window:
        self.rect = rect
        if self._chrome is not None:
            self._chrome.resize(max(rect.h, 0), max(rect.w, 0))
            self._chrome.set_position(rect.x, rect.y)
        if self._content is not None:
            inner = self.inner_rect()
            self._content.resize(max(inner.h, 0), max(inner.w, 0))
            self._content.set_position(inner.x, inner.y)
        self._draw_frame()
        return self

    def _draw_frame(self, active: bool = False) -> Window:
        """Draw whichever frame this window's chrome asks for.

        The one place that chooses between the flat frame and the cube. It has to
        be one place: `resize` and :meth:`WindowManager.redraw_all` both redraw
        chrome, and when each picked its own renderer a cube window came back from
        a resize drawn flat -- correct in isolation, wrong next to every other path
        that draws it in 3D.
        """
        if self.chrome.depth > 0:
            return self.draw_cube(active=active)
        return self.redraw_chrome(active=active)

    def inner_rect(self) -> Rect:
        """The drawable area, in screen coordinates.

        Inset by the frame on every side plus :attr:`padding`, and pushed down
        by the header band when there is one.
        """
        r = self.rect
        b = self._inset
        head = self._header_h
        # Padding must never be able to consume the whole body. A five-row
        # window with a header band and a cell of padding has zero content rows,
        # and every `write` into it is silently clipped -- the window renders as
        # an empty box and the caller gets no error to explain why. Squeeze the
        # padding instead, down to none, so a window that has room for a row
        # always has one.
        p = min(self.padding, max(0, (r.h - (2 * b + head) - 1) // 2))
        return Rect(
            r.x + b + p,
            r.y + b + head + p,
            r.w - 2 * (b + p),
            r.h - (b + head + p) - (b + p),
        )

    # ── drawing ──────────────────────────────────────────────────────
    def draw_cube(self, active: bool = False) -> Window:
        """Draw the window as a solid block: front wall plus receding surfaces.

        A room, not an isometric projection. The front wall is an axis-aligned
        rectangle so text stays on the cell grid, and the ceiling, floor and side
        walls fall away from it using diagonal glyphs.

        The geometry is a trapezoid receding to a vanishing region above and
        below. Going down the ceiling, the left edge steps one column left per
        row and the right edge steps one column right, so the back edge is
        ``depth`` cells narrower on each side than the front wall. The floor
        mirrors it. The side walls are what is left between the front wall and
        those diagonals -- left blank, because a wall with nothing on it should
        read as empty space and anything drawn there would compete with the text.

        Weight carries the depth: the front wall is drawn heavy and the receding
        edges light. That is what separates "the panel you are reading" from
        "the space behind it" without spending another colour.
        """
        if self._chrome is None:
            return self
        r = self.rect
        if r.empty or not self._cube_fits:
            return self.redraw_chrome(active=active)

        c = self.chrome
        d = c.depth
        focused = self.focused or active
        fg = c.frame_focused if focused else c.frame
        marker_fg = c.accent if focused else c.accent_dim

        self._chrome.fill_background()
        W, H = r.w, r.h

        # Geometry of the front wall: an axis-aligned rectangle `d` cells in from
        # the window's edge. Heavy, because it is the plane the reader is looking
        # at, and the only one text ever lands on.
        top, bot = d, H - d - 1
        left, right = d, W - d - 1

        # ── painter's order: far surfaces first, near surfaces last ───
        #
        # There is no per-cell z here, so the last thing written to a cell is what
        # shows, and the order of these blocks *is* the depth. Two orderings are
        # load-bearing, and they are not the same one:
        #
        #   1. The cap edges (below) must land *after* the diagonals. They are
        #      what clips the diagonal runs, and a run that reaches the back edge
        #      un-clipped stops tapering -- the room turns into a slab. This one
        #      is geometrically unavoidable: the cap and the diagonal want the
        #      same cell and only one can win.
        #
        #   2. The front wall must land *after* both. In principle the diagonals
        #      never reach the corners now that each run is shortened as it
        #      approaches the wall, so this is defence in depth rather than the
        #      thing holding the frame together. It was previously the *only*
        #      thing holding it together -- the runs used to reach the corners and
        #      the wall won by being written later -- and that ordering was an
        #      accident of statement order, so reordering the blocks for
        #      readability silently ate the front face while the frame kept
        #      rendering plausibly.
        for i in range(d):
            # A run `span` long starting at column `lc` is mirrored about the
            # window's vertical axis by starting at `W - lc - span`: its last
            # column then sits exactly as far from the right edge as `lc` does
            # from the left. The shorter form, `W - d - lc`, only agrees with
            # that when `span == d`, so it was quietly correct on the ceiling's
            # back row and one column short on every row after it -- a room whose
            # right wall recedes to a different place than its left, which reads
            # as a drawing mistake rather than as perspective.
            span = d - i  # one shorter each row down: see below
            lc = 2 * d - 1 - i
            rc = W - lc - span
            down = self.chrome.diag_down * span
            up = self.chrome.diag_up * span
            self._chrome.write(i, lc, down, fg=fg)
            self._chrome.write(i, rc, up, fg=fg)
            # Floor: the same convergence, read from the bottom up, so both
            # surfaces narrow toward the same back edge instead of the floor
            # splaying outward past the wall's right corner.
            fr = H - 1 - i
            self._chrome.write(fr, lc, down, fg=fg)
            self._chrome.write(fr, rc, up, fg=fg)

        # ── depth 2: the two horizontal edges that cap the receding surfaces ──
        #
        # The ceiling's back edge closes the top of the room and the floor's front
        # edge closes the bottom. Without them the ceiling reads as an open lid
        # rather than a surface. They sit between the diagonals and the wall in
        # the painter's order, and *that* middle position is load-bearing: each
        # one shares its row with a diagonal run and clips it, which is what makes
        # the run taper instead of reaching all the way back.
        self._chrome.write(0, 2 * d, "─" * (W - 4 * d), fg=fg)
        self._chrome.write(H - 1, 0, "─" * W, fg=fg)

        # ── depth 3: the front wall, nearest the viewer, painted last ──
        self._chrome.write(top, left, "┏" + "━" * (right - left - 1) + "┓", fg=fg)
        for row in range(top + 1, bot):
            self._chrome.write(row, left, "┃", fg=fg)
            self._chrome.write(row, right, "┃", fg=fg)
        self._chrome.write(bot, left, "┗" + "━" * (right - left - 1) + "┛", fg=fg)

        if self.title:
            # The band sits *on* the front wall's top edge, replacing the stretch
            # between the corners. The rule is written first and the title over it,
            # so the corners stay put -- punching the title out of a wall that then
            # redraws its own corners on top would leave a gap.
            self._chrome.write(top, left + 1, "━" * (right - left - 1), fg=fg)
            self._draw_header(
                r,
                focused,
                row=top,
                col=left + 1,
                width=right - left - 1,
                marker_fg=marker_fg,
            )
        return self

    def clear_body(self) -> Window:
        """Repaint the body with the window's background, leaving chrome alone.

        The default first step of any body redraw: fill across the full width,
        then write text over it.  Because the fill gives every cell a real
        background colour, the blanks cover what is behind this window instead
        of letting it show through -- including the spaces *inside* a line,
        which are content.

        Use :meth:`clear` only when the chrome must go too (closing a window);
        clearing chrome without redrawing it would leave the border erased.
        """
        if self._content is not None:
            self._content.fill_background()
        return self

    def clear(self) -> Window:
        """Erase both layers back to transparent. Safe to call every frame."""
        for layer in (self._chrome, self._content):
            if layer is not None:
                layer.framebuffer.clear()
        return self

    def redraw_chrome(self, active: bool = False) -> Window:
        """Redraw the frame: header band, rails, and bottom edge.

        The header band *replaces* the top border rather than interrupting it.
        Centring a title inside the top rule was the old look and it read badly
        for two reasons: it broke the line the eye follows around the frame, and
        it cost nothing to remove but gave back a row. A filled band left-aligns
        the label, survives a long title, and gives the window a clear top.

        Focus is signalled on three channels -- band colour, title weight, and
        border glyph set -- so it does not depend on colour alone to be legible.
        """
        if self._chrome is None:
            return self
        r = self.rect
        if r.empty:
            self._chrome.framebuffer.clear()
            return self

        focused = self.focused or active
        c = self.chrome
        # The frame is grey, never the accent. Focus is already marked by the
        # header's marker glyph, so the rail only brightens a step -- which keeps
        # the accent free to mean something.
        rail = c.frame_focused if focused else c.frame

        # Background first, across the full width, so the panel is solid
        # rather than patterned even where the border glyphs are not.
        self._chrome.fill_background()

        if self.box == BoxStyle.NONE:
            if self.title:
                self._draw_header(r, focused)
            return self

        head = self._header_h
        # Rails run from the band (or the top corner) down to the bottom edge.
        v = "│" if self.box in (BoxStyle.SINGLE, BoxStyle.ROUNDED) else "┃"
        if self.box == BoxStyle.ASCII:
            v = "|"
        for row in range(head, r.h - 1):
            self._chrome.write(row, 0, v, fg=rail)
            self._chrome.write(row, r.w - 1, v, fg=rail)
        # Rounded feet look right under a heavy header rule; only the top
        # corners would clash, and the header has already replaced them.
        round_ = self.box == BoxStyle.ROUNDED
        tl, tr, bl, br = (
            ("╰", "╯", "╰", "╯")
            if round_
            else ("└", "┘", "└", "┘")
            if self.box == BoxStyle.SINGLE
            else ("+", "+", "+", "+")
            if self.box == BoxStyle.ASCII
            else ("╚", "╝", "╚", "╝")
            if self.box == BoxStyle.DOUBLE
            else ("┗", "┛", "┗", "┛")
        )
        bottom = r.h - 1
        if bottom >= head:
            self._chrome.write(bottom, 0, bl, fg=rail)
            self._chrome.write(bottom, r.w - 1, br, fg=rail)
            if r.w > 2:
                self._chrome.write(bottom, 1, "─" * (r.w - 2), fg=rail)

        if self.title:
            self._draw_header(r, focused)
        else:
            # Untitled windows keep a top border so the frame closes.
            tl, tr = ("╭", "╮") if round_ else ("┌", "┐")
            if self.box == BoxStyle.ASCII:
                tl, tr = "+", "+"
            elif self.box == BoxStyle.DOUBLE:
                tl, tr = "╔", "╗"
            elif self.box == BoxStyle.HEAVY:
                tl, tr = "┏", "┓"
            self._chrome.write(0, 0, tl, fg=rail)
            self._chrome.write(0, r.w - 1, tr, fg=rail)
            if r.w > 2:
                self._chrome.write(0, 1, "─" * (r.w - 2), fg=rail)
        del tl, tr  # only meaningful for the untitled branch above
        return self

    def _draw_header(
        self,
        r: Rect,
        focused: bool,
        *,
        row: int = 0,
        col: int = 0,
        width: int | None = None,
        marker_fg: int | None = None,
    ) -> None:
        """Draw the header rule and punch the title out of it.

        The rule is a *glyph*, not just a background colour. A band made only of
        coloured spaces looks right on one terminal and vanishes on the next --
        and in any plain-text dump of the frame it is invisible, which means the
        design cannot be reviewed without a live terminal. Drawing ``━`` and
        then writing the title over it with an opaque background gives the same
        appearance and survives monochrome.

        ``row``/``col``/``width`` place the band somewhere other than the top edge
        of ``r``. A cube's title belongs on its *front wall*, inset by the depth,
        not on the top of the whole block where it would sit on the ceiling.
        """
        c = self.chrome
        rule_fg = c.frame_focused if focused else c.frame
        # Only the marker takes the accent. The rule that runs out from the title
        # is chrome like any other, so colouring it accent is exactly the
        # "accent spent on the frame" mistake the split exists to prevent.
        if marker_fg is None:
            marker_fg = c.accent if focused else c.accent_dim
        width = r.w if width is None else width
        label = self.title
        # Reserve room for the marker as well as the rule run either side.
        lead = 1 + len(c.marker) if c.marker else 1
        room = width - 2 * lead
        if room <= 0:
            self._chrome.write(row, col, c.rule * width, fg=rule_fg)
            return
        if len(label) > room:
            label = label[: room - 1] + "…"

        if c.marker:
            # Small-OS layout: marker, title, then a thin rule out to the edge.
            # No leading rule -- the tick *is* the left edge of the bar, which is
            # what makes focus land in one predictable place.
            self._chrome.write(row, col, c.marker, fg=marker_fg)
            end = lead + len(label) + 1
            self._chrome.write(
                row,
                col + lead,
                f" {label} ",
                fg=rule_fg,
                bg=c.band_bg,
                attr=Attr.BOLD if focused else Attr.NONE,
            )
            if end < width:
                self._chrome.write(row, col + end, c.rule_after * (width - end), fg=rule_fg)
            return

        # Otherwise: rule, title punched out of it, rule. Two runs rather than a
        # full-width write, because with `band_bg` left at DEFAULT the spaces
        # around the label are transparent and the rule would show through the
        # word.
        self._chrome.write(row, col, c.rule * lead, fg=rule_fg)
        end = lead + len(label) + 2
        self._chrome.write(
            row,
            col + lead,
            f" {label} ",
            fg=Color.BRIGHT_WHITE,
            bg=c.band_bg,
            attr=Attr.BOLD if focused else Attr.NONE,
        )
        if end < width:
            self._chrome.write(row, col + end, c.rule * (width - end), fg=rule_fg)

    def write(
        self,
        row: int,
        col: int,
        text: str,
        fg: int = Color.DEFAULT,
        attr: int = Attr.NONE,
    ) -> int:
        """Write into the body, clipped to the content area."""
        if self._content is None:
            return 0
        inner = self.inner_rect()
        if not (0 <= row < inner.h):
            return 0
        room = inner.w - col
        if room <= 0:
            return 0
        return self._content.write(row, col, text[:room], fg=fg, attr=attr)

    # ── containment ──────────────────────────────────────────────────
    def add(self, child: Window) -> Window:
        """Add *child*, sized to sit inside this window's content area."""
        inner = self.inner_rect()
        self.children.append(child)
        if self._engine is not None:
            child.attach(self._engine)
            child.resize(Rect(inner.x, inner.y, inner.w, inner.h))
        return child

    def set_visible(self, visible: bool) -> Window:
        """Show or hide this window and everything inside it."""
        self.visible = visible
        for layer in (self._chrome, self._content):
            if layer is not None:
                layer.visible = visible
        for child in self.children:
            child.set_visible(visible)
        return self

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Window {self.name} {self.rect.w}x{self.rect.h} @ {self.rect.x},{self.rect.y}>"


# ── tabs ────────────────────────────────────────────────────────────


class TabBar:
    """A strip of tab labels wrapping a group of windows.

    Exactly one tab is active.  Activating one shows its window and hides the
    others -- visibility is the whole mechanism, because the engine already
    skips invisible layers.

    Tabs keep their declared order; the active one is marked by a brighter label
    and an accent segment on the rule. Reordering the strip would make the active
    tab jump under the cursor, so the highlight is the only thing that changes on
    a switch.
    """

    def __init__(
        self,
        engine: GraphicsEngine,
        rect: Rect,
        *,
        background: int = Color.DEFAULT,
        chrome: Chrome | None = None,
    ) -> None:
        self.rect = rect
        self.background = background
        self.chrome = chrome if chrome is not None else Chrome()
        self._layer = engine.create_layer("tabbar", z=100)
        # The strip fills its own background like a window body. Left
        # transparent, the backdrop pattern reads straight through the gaps
        # between labels and the strip looks like it is standing in a window.
        self._layer.background = background
        self._tabs: list[tuple[str, Window]] = []
        self._active = 0
        # `create_layer` hands back a full-screen layer, and a strip that never
        # shrinks would fill its background over every window beneath it.
        self.resize(rect)

    # ── membership ───────────────────────────────────────────────────
    def add(self, label: str, window: Window) -> TabBar:
        self._tabs.append((label, window))
        self._sync()
        self.redraw()
        return self

    @property
    def labels(self) -> list[str]:
        return [label for label, _ in self._tabs]

    @property
    def active(self) -> int:
        return self._active

    @property
    def active_window(self) -> Window | None:
        return self._tabs[self._active][1] if self._tabs else None

    # ── switching ────────────────────────────────────────────────────
    def activate(self, index: int) -> TabBar:
        if self._tabs:
            self._active = index % len(self._tabs)
            self._sync()
            self.redraw()
        return self

    def next_tab(self) -> TabBar:
        return self.activate(self._active + 1)

    def _sync(self) -> None:
        for i, (_, win) in enumerate(self._tabs):
            win.set_visible(i == self._active)
            win.focused = i == self._active

    # ── drawing ──────────────────────────────────────────────────────
    def resize(self, rect: Rect) -> TabBar:
        self.rect = rect
        self._layer.resize(max(rect.h, 0), max(rect.w, 0))
        self._layer.set_position(rect.x, rect.y)
        return self.redraw()

    def redraw(self) -> TabBar:
        """Labels above a rule, with the active tab accenting the rule beneath it.

        No box around the strip. It used to draw one, and a two-row box whose
        labels sat on row 1 left row 0 as an orphaned rule while the labels
        collided into the bottom border -- the most obviously broken thing on
        screen.

        The active tab is marked twice: brighter and bold in the label row, and
        a heavy accent segment under it in the rule row. Both markers are
        glyphs, not just colour, so the selection survives a monochrome terminal
        and can be reviewed in a plain-text dump.

        Tabs keep their declared order. Moving the active tab to the front is a
        common trick, and it makes the selection jump under the cursor, which is
        worse than the untidiness it hides.
        """
        r = self.rect
        if r.empty:
            self._layer.framebuffer.clear()
            return self
        self._layer.fill_background()

        c = self.chrome
        # Pass 1: rule across the bottom row, and find where each label lands.
        rule = r.h - 1
        self._layer.write(rule, 0, "─" * r.w, fg=c.frame, bg=Color.BLACK)
        col = 2
        active_span = None
        for i, (label, _) in enumerate(self._tabs):
            text = f" {label} "
            if col + len(text) > r.w - 1:
                break
            active = i == self._active
            self._layer.write(
                0,
                col,
                text,
                fg=Color.BRIGHT_WHITE if active else Color.BRIGHT_BLACK,
                bg=Color.BLACK,
                attr=Attr.BOLD if active else Attr.NONE,
            )
            if active:
                active_span = (col, len(text))
            col += len(text) + 1

        # Pass 2: the accent segment, so it lands on the rule and not the label.
        # Always the heavy rule, whatever the window header uses. A thin variant
        # would leave the active tab marked by colour alone, which is the one
        # thing this strip must not do -- unreadable in a dump, invisible on a
        # monochrome terminal.
        if active_span is not None and rule != 0:
            start_col, span = active_span
            self._layer.write(rule, start_col, "━" * span, fg=c.accent, bg=Color.BLACK)
        return self


# ── manager ─────────────────────────────────────────────────────────


class WindowManager:
    """Owns the root window and the tab strip, and keeps layers in step.

    The manager is the only thing that knows the screen layout. It mutates
    layer geometry and visibility; it never writes terminal bytes -- the
    engine's diff does that.
    """

    def __init__(self, engine: GraphicsEngine) -> None:
        self.engine = engine
        self.root: Window | None = None
        self.tabbar: TabBar | None = None

    def set_root(self, window: Window) -> Window:
        """Adopt *window* as the screen and attach it."""
        self.root = window.attach(self.engine)
        return self.root

    def set_tabs(
        self,
        rect: Rect,
        *,
        background: int = Color.DEFAULT,
        chrome: Chrome | None = None,
    ) -> TabBar:
        self.tabbar = TabBar(self.engine, rect, background=background, chrome=chrome)
        return self.tabbar

    def add_window(self, name: str, rect: Rect, title: str = "", **kw) -> Window:
        """Add a top-level window. Requires :meth:`set_root` first."""
        if self.root is None:
            raise RuntimeError("set_root() first")
        window = Window(name, rect, title, **kw)
        self.root.children.append(window)
        window.attach(self.engine)
        return window

    def layout(self, rect: Rect) -> WindowManager:
        """Resize the root to fill *rect* and lay the strip along its top."""
        if self.root is not None:
            self.root.resize(rect)
        if self.tabbar is not None:
            # Two rows: labels, then the rule the active tab accents.
            self.tabbar.resize(Rect(rect.x, rect.y, rect.w, 2))
        return self.redraw_all()

    def redraw_all(self) -> WindowManager:
        """Redraw every window's chrome and re-apply the tab selection."""
        if self.root is not None:
            for win in self.root.children or [self.root]:
                win._draw_frame(active=win is self.active_window)
        if self.tabbar is not None:
            self.tabbar._sync()
            self.tabbar.redraw()
        return self

    @property
    def active_window(self) -> Window | None:
        return self.tabbar.active_window if self.tabbar else None
