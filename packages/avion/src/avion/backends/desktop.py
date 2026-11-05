"""Desktop backend — mouse/keyboard/screen control for desktop apps.

Actions need pyautogui (and Pillow for screenshots); observation does not.
Coordinates are screen pixels, and all third-party imports are lazy.

Observation is the native Linux accessibility bus (AT-SPI), not pixels: the
tree goes out in the same {role, name, children} shape the browser backends
use, so the agent's prompt and goal verification work on the desktop too.
Because it walks the bus instead of the screen it needs no pyautogui and no
OCR — pass require_actions=False to observe on a machine that has neither.
On Debian/Ubuntu the install is `apt install python3-gi gir1.2-atspi-2.0`;
the `desktop` extra names the same thing, but a pip build of PyGObject wants
meson/cairo headers that boxes like this one do not have.

Only the selected window is walked, and what it exposes lands in transcripts:
titles, labels, field contents. Chromium and Electron appear as an empty frame
unless launched with --force-renderer-accessibility.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from typing import Any

# The role vocabulary, the classifier and the tree packing are shared with the
# browser backends; what is local to this file is how to read them off the bus.
from avion.backends import axtree

_ACTIONABLE = axtree.ACTIONABLE
_CONTAINERS = axtree.CONTAINERS
_NAME_CAP = axtree.NAME_CAP
_READ_WORDS = axtree.READ_WORDS
_TEXT_CAP = axtree.TEXT_CAP
_fit = axtree.fit
_norm_role = axtree.norm_role
_signal_count = axtree.signal_count
pos_of = axtree.pos_of
role_words = axtree.role_words


def _pyautogui():
    try:
        import pyautogui

        return pyautogui
    except ImportError as e:
        raise ImportError(
            "pyautogui is not installed. Install it with: pip install pyautogui"
        ) from e


def _atspi():
    """AT-SPI 2.0 bindings, or ImportError naming the install line.

    A missing typelib raises ValueError from require_version, not ImportError;
    both become one actionable message.
    """
    try:
        import gi

        gi.require_version("Atspi", "2.0")
        from gi.repository import Atspi

        return Atspi
    except (ImportError, ValueError) as e:
        raise ImportError(
            "desktop observation needs PyGObject and the AT-SPI typelib: "
            "apt install python3-gi gir1.2-atspi-2.0"
        ) from e


def _atspi_or_none():
    """Observation degrades to no tree; it never raises at a caller."""
    try:
        return _atspi()
    except ImportError:
        return None


# ── Roles ────────────────────────────────────────────────────────────────
#
# AT-SPI has ~95 role names and the model speaks eleven. The vocabulary, the
# word-based classifier and the tree-packing helpers all live in
# avion/backends/axtree.py, which the browser backends use too — one place to
# change what "a thing the agent can see" means.
#
# Lossy where the bus has no equivalent: check/radio collapse to "button" (the
# action path is coordinates, so only "this is clickable" survives anyway),
# and there is no "searchbox", so a GTK search entry is a "textbox".


# ── Walk bounds ──────────────────────────────────────────────────────────
#
# Bounds exist because the bus is slow (one roundtrip per attribute) and mostly
# noise: a live probe of this machine returned 24 applications whose shells were
# unnamed panels reporting unallocated extents.
_MAX_TEXT_READS = 8
_CHILD_CAP = 12
_WIDE_CHILDREN = 20
# A cancelled await does not stop its thread, so the walk bounds its own time
# rather than trusting the caller's step_timeout.
_WALK_DEADLINE_S = 1.5
_INT32_MAX = 2**31 - 1
_WATCHED_STATES = ("active", "focused", "showing", "editable", "expandable")


@dataclass
class _Node:
    """One accessibility node, detached from the bus.

    Pruning, ranking and packing read this and never an AT-SPI object, which is
    what makes the whole algorithm testable without a display.
    """

    role: str
    name: str
    text: str
    extents: tuple[int, int, int, int] | None
    states: frozenset[str]
    words: frozenset[str] = frozenset()
    children: list[_Node] = field(default_factory=list)

    @property
    def signal(self) -> bool:
        return bool(self.name) or bool(self.text) or self.role in _ACTIONABLE


def _safe(get, default):
    """An application that is dying raises on any read; observing it must not."""
    try:
        value = get()
    except Exception:
        return default
    return default if value is None else value


def _rect_of(Atspi, acc) -> tuple[int, int, int, int] | None:
    """Screen extents, or None when the bus is lying about them.

    Unallocated nodes report INT_MIN coordinates or 0x0 boxes. Keeping those
    would hand the model a click target that is on no screen.
    """
    try:
        r = acc.get_extents(Atspi.CoordType.SCREEN)
        x, y, w, h = (int(r.x), int(r.y), int(r.width), int(r.height))
    except Exception:
        return None
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        return None
    if max(abs(x), abs(y), abs(w), abs(h)) >= _INT32_MAX:
        return None
    return (x, y, w, h)


def _states_of(Atspi, acc) -> frozenset[str]:
    try:
        states = acc.get_state_set()
    except Exception:
        return frozenset()
    found = set()
    for name in _WATCHED_STATES:
        try:
            if states.contains(getattr(Atspi.StateType, name.upper())):
                found.add(name)
        except Exception:
            continue
    return frozenset(found)


def _text_of(acc) -> str:
    count = _safe(acc.get_character_count, 0)
    if not isinstance(count, int) or count <= 0:
        return ""
    return _safe(lambda: acc.get_text(0, min(count, _TEXT_CAP)), "").strip()


@dataclass
class _Budget:
    deadline: float
    node_cap: int
    nodes: int = 0
    texts: int = 0


def _collect(Atspi, acc, depth: int, budget: _Budget, max_depth: int) -> _Node | None:
    """Read one subtree into _Nodes, bounded by depth, node count, time."""
    if budget.nodes >= budget.node_cap or depth > max_depth:
        return None
    if time.monotonic() > budget.deadline:
        return None
    role_raw = _safe(acc.get_role_name, "").strip().lower()
    if not role_raw:
        return None
    budget.nodes += 1
    name = _safe(acc.get_name, "").strip()[:_NAME_CAP]
    words = role_words(role_raw)
    node = _Node(
        role=_norm_role(role_raw),
        name=name,
        text="",
        extents=_rect_of(Atspi, acc),
        states=_states_of(Atspi, acc),
        words=words,
    )
    if not name and words & _READ_WORDS and budget.texts < _MAX_TEXT_READS:
        budget.texts += 1
        node.text = _text_of(acc)
    child_count = max(_safe(acc.get_child_count, 0), 0)
    take = min(child_count, _CHILD_CAP) if child_count > _WIDE_CHILDREN else child_count
    for i in range(take):
        child = _safe(lambda: acc.get_child_at_index(i), None)
        if child is None:
            continue
        kept = _collect(Atspi, child, depth + 1, budget, max_depth)
        if kept is not None:
            node.children.append(kept)
    return node


def _prune(node: _Node) -> _Node | None:
    """Keep a node if it carries signal, or if anything under it does.

    One rule, applied bottom-up: everything else is scaffolding.
    """
    survivors = [kept for kept in (_prune(c) for c in node.children) if kept is not None]
    node.children = survivors
    return node if (node.signal or survivors) else None


def _collapse(node: _Node) -> _Node:
    """Drop nameless single-child wrappers, so a button is not buried under
    four panels. The same test at every depth.
    """
    node.children = [_collapse(c) for c in node.children]
    while (
        len(node.children) == 1
        and not node.name
        and not node.text
        and node.role not in _ACTIONABLE | _CONTAINERS
    ):
        node = node.children[0]
    return node


def _rank(node: _Node) -> None:
    """Best-first children, so a budget cut loses the weakest tail.

    ai/slo.py's _tree_text slices the rendered JSON and ai/choice.py walks in
    document order: both keep whatever comes first, so ranking here is what
    makes a small prompt budget mean "the important controls" instead of
    "whatever the bus enumerated first".
    """
    node.children.sort(key=_score, reverse=True)
    for child in node.children:
        _rank(child)


def _score(node: _Node) -> int:
    score = 0
    if node.role in _ACTIONABLE:
        score += 6 if node.extents else 4
    if node.role in _CONTAINERS:
        score += 3
    if node.name:
        score += 1
    if node.text:
        score += 1
    return score


def _pos(extents: tuple[int, int, int, int] | None) -> str:
    return "" if not extents else pos_of(*extents)


def _to_dict(node: _Node) -> dict[str, Any]:
    """The shortest form that still says what and where.

    axtree.pack decides which keys that is, so the desktop tree and the browser
    tree read alike. Every actionable node carries "pos" because the desktop
    action path is mouse_click{x,y} — an observation that cannot say where is
    not usable by a coordinate agent.
    """
    return axtree.pack(
        node.role,
        name=node.name,
        text=node.text,
        pos=_pos(node.extents) if node.role in _ACTIONABLE else "",
        children=[_to_dict(c) for c in node.children],
    )


@dataclass
class _Window:
    """A top-level window, holding its bus object so it can be walked."""

    app: str
    name: str
    extents: tuple[int, int, int, int] | None
    states: frozenset[str]
    acc: Any

    @property
    def area(self) -> int:
        return 0 if not self.extents else self.extents[2] * self.extents[3]


def _windows(Atspi) -> list[_Window]:
    """Every application's top-level windows: two levels deep, no text reads."""
    out: list[_Window] = []
    try:
        desktop = Atspi.get_desktop(0)
        app_count = max(int(desktop.get_child_count()), 0)
    except Exception:
        return out
    for i in range(app_count):
        app = _safe(lambda: desktop.get_child_at_index(i), None)
        if app is None:
            continue
        app_name = _safe(app.get_name, "").strip()
        frame_count = max(_safe(app.get_child_count, 0), 0)
        for j in range(frame_count):
            acc = _safe(lambda: app.get_child_at_index(j), None)
            if acc is None:
                continue
            if _norm_role(_safe(acc.get_role_name, "").strip().lower()) not in _CONTAINERS:
                continue
            out.append(
                _Window(
                    app=app_name,
                    name=_safe(acc.get_name, "").strip()[:_NAME_CAP],
                    extents=_rect_of(Atspi, acc),
                    states=_states_of(Atspi, acc),
                    acc=acc,
                )
            )
    return out


class DesktopBackend:
    """OS-level automation: no DOM, coordinates only — but not blind.

    Implements the InteractionBackend protocol (mouse_*, keyboard_*) so
    Mouse/Keyboard controllers and InteractionChain drive it directly, and
    answers the observation side of the Backend protocol — get_accessibility_tree,
    get_url, get_title — from AT-SPI.

    Observation needs no pyautogui, so require_actions=False gives a machine
    that cannot install it a desktop run that can see and verify. It still
    cannot act: seeing is not reaching.
    """

    name = "desktop"

    def __init__(
        self,
        *,
        require_actions: bool = True,
        tree_char_budget: int = 1400,
        max_nodes: int = 40,
        max_depth: int = 12,
    ) -> None:
        """tree_char_budget bounds the JSON that reaches the prompt.

        With max_prompt_chars=2400 the tree section gets roughly 1600 of them
        (ai/slo.py:_split_budget), and only text that survives this budget can
        match a text_present goal — a narrower window than what is on screen.
        """
        self._started = False
        self._require_actions = require_actions
        self._tree_char_budget = tree_char_budget
        self._max_nodes = max_nodes
        self._max_depth = max_depth
        self._bus_lock = threading.Lock()
        self._ready = False

    async def start(self) -> None:
        if self._require_actions:
            _pyautogui()
        self._started = True

    async def stop(self) -> None:
        self._started = False

    # ── Observation ─────────────────────────────────────────────────────

    async def get_accessibility_tree(self) -> dict[str, Any] | None:
        """The selected window as {role, name, children}; None if unreadable.

        None, not an empty dict: "cannot see" must not read as "an empty
        desktop", and the prompt renders it as "(no accessibility tree)".
        """
        Atspi = _atspi_or_none()
        if Atspi is None:
            return None
        return await asyncio.to_thread(self._bus, self._tree_sync, Atspi)

    async def get_url(self) -> str:
        """Always "". The desktop has no URL, and url_contains must fail loudly.

        A title in this channel would let url_contains "chat" pass on a window
        named "Chat" — an unverified goal reading as a pass. The title is
        reachable as text instead, through the tree and flatten_tree_text.
        """
        return ""

    async def get_title(self) -> str:
        """The selected window's name, or ""."""
        Atspi = _atspi_or_none()
        if Atspi is None:
            return ""
        return await asyncio.to_thread(self._bus, self._title_sync, Atspi)

    def _bus(self, work, Atspi):
        """One bus walk at a time; a busy walk means no tree this step.

        Without this, a slow walk plus a step timeout could stack threads, and
        each would press the same live session bus.
        """
        if not self._bus_lock.acquire(blocking=False):
            return None
        try:
            return work(Atspi)
        finally:
            self._bus_lock.release()

    def _init(self, Atspi) -> None:
        if not self._ready:
            Atspi.init()
            self._ready = True

    def _tree_sync(self, Atspi) -> dict[str, Any] | None:
        self._init(Atspi)
        roots: list[_Node] = []
        for window in self._select(Atspi):
            budget = _Budget(deadline=time.monotonic() + _WALK_DEADLINE_S, node_cap=self._max_nodes)
            node = _collect(Atspi, window.acc, 0, budget, self._max_depth)
            if node is None:
                continue
            pruned = _prune(node)
            if pruned is None:
                continue
            kept = _collapse(pruned)
            _rank(kept)
            roots.append(kept)
        if not roots:
            return None
        # One window is the tree itself; wrapping it adds a level the model has
        # to parse for no information. Only a genuine second window is a child.
        tree = (
            _to_dict(roots[0])
            if len(roots) == 1
            else {"role": "window", "children": [_to_dict(r) for r in roots]}
        )
        tree = _fit(tree, self._tree_char_budget)
        # One name and no controls is a caption, not an observation.
        return tree if _signal_count(tree) >= 2 else None

    def _title_sync(self, Atspi) -> str:
        self._init(Atspi)
        chosen = self._select(Atspi)
        return chosen[0].name if chosen else ""

    def _select(self, Atspi) -> list[_Window]:
        """Which windows to read, in order of confidence: active, then sized,
        then any.

        Three tiers because STATE_ACTIVE on a frame is compositor-dependent —
        set on this box's elementary terminal under Wayland, but not a promise.
        Reading only the selection also stops the walk from mining every
        application's text on the machine.
        """
        windows = _windows(Atspi)
        for tier in (
            [w for w in windows if "active" in w.states],
            [w for w in windows if w.extents],
            windows,
        ):
            if not tier:
                continue
            ranked = sorted(tier, key=lambda w: (w.area, len(w.name)), reverse=True)
            return ranked[:2]
        return []

    # ── Actions: pyautogui, coordinates only ────────────────────────────

    async def mouse_move(self, x: float, y: float) -> None:
        pg = _pyautogui()
        await asyncio.to_thread(pg.moveTo, x, y)

    async def mouse_down(self, button: str = "left") -> None:
        pg = _pyautogui()
        await asyncio.to_thread(pg.mouseDown, button=button)

    async def mouse_up(self, button: str = "left") -> None:
        pg = _pyautogui()
        await asyncio.to_thread(pg.mouseUp, button=button)

    async def mouse_click(self, x: float, y: float, button: str = "left") -> None:
        pg = _pyautogui()
        await asyncio.to_thread(pg.click, x, y, button=button)

    async def mouse_double_click(self, x: float, y: float) -> None:
        pg = _pyautogui()
        await asyncio.to_thread(pg.doubleClick, x, y)

    async def mouse_scroll(self, x: float, y: float, delta_x: int = 0, delta_y: int = 0) -> None:
        pg = _pyautogui()
        await asyncio.to_thread(pg.moveTo, x, y)
        if delta_y:
            await asyncio.to_thread(pg.scroll, delta_y)
        if delta_x:
            await asyncio.to_thread(pg.hscroll, delta_x)

    async def keyboard_down(self, key: str) -> None:
        pg = _pyautogui()
        await asyncio.to_thread(pg.keyDown, key)

    async def keyboard_up(self, key: str) -> None:
        pg = _pyautogui()
        await asyncio.to_thread(pg.keyUp, key)

    async def keyboard_press(self, key: str, hold_ms: float = 0) -> None:
        pg = _pyautogui()
        if "+" in key:  # "Control+a" style combos
            parts = [p.strip() for p in key.split("+")]
            await asyncio.to_thread(pg.hotkey, *parts)
        else:
            await asyncio.to_thread(pg.press, key)

    async def keyboard_type(self, text: str, delay_ms: float = 0) -> None:
        pg = _pyautogui()
        await asyncio.to_thread(pg.typewrite, text, interval=delay_ms / 1000 if delay_ms else 0)

    async def get_viewport_size(self) -> tuple[int, int]:
        pg = _pyautogui()
        size = await asyncio.to_thread(pg.size)
        return (size.width, size.height)

    async def get_screenshot_as_bytes(self) -> bytes:
        import io

        pg = _pyautogui()
        img = await asyncio.to_thread(pg.screenshot)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    async def screenshot(self, path: str | None = None) -> bytes:
        data = await self.get_screenshot_as_bytes()
        if path:
            with open(path, "wb") as f:
                f.write(data)
        return data

    async def wait_for_timeout(self, ms: float) -> None:
        await asyncio.sleep(ms / 1000)

    async def locate_text(self, text: str) -> list[dict[str, Any]]:
        """Locate text via OCR when pytesseract is available, else []."""
        try:
            import pytesseract  # type: ignore[import]
        except ImportError:
            return []
        import io

        from PIL import Image  # type: ignore[import]

        data = await self.get_screenshot_as_bytes()
        img = Image.open(io.BytesIO(data))
        words = await asyncio.to_thread(
            pytesseract.image_to_data, img, output_type=pytesseract.Output.DICT
        )
        out = []
        for i, word in enumerate(words.get("text", [])):
            if word and text.lower() in word.lower():
                out.append(
                    {
                        "text": word,
                        "x": words["left"][i],
                        "y": words["top"][i],
                        "width": words["width"][i],
                        "height": words["height"][i],
                    }
                )
        return out
