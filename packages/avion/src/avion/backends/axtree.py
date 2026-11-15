"""The observation specification: what a tree node is and how it is chosen.

One place for what every backend used to carry its own copy of:

* the eleven-word role vocabulary the model speaks, and the classifier that
  maps a role name into it — by the *words* in the name, so a new role from a
  new toolkit needs no new code. AT-SPI spells roles "push button", CDP spells
  them "pushButton"/"RootWebArea"; both are just words.
* the {role, name, children} (+ "pos") shape that ai/choice.py's walk() and
  ai/slo.py's renderer read, and pack() — the only writer of a node, so the
  desktop walk and the CDP walk cannot drift apart on which keys to send.
  Nothing else in avion consumes a tree, so this shape is the contract, and it
  is the only one.

`fetch_ax_tree` exists because `page.accessibility` was removed in Playwright
1.62 and avion's CDP backend used to hand back the raw
{"nodes": [...]} with childIds, which no walker can read. Both now go through
Chrome DevTools Protocol's Accessibility.getFullAXTree and out here normalized.

Chromium only: the CDP domains do not exist in Firefox or WebKit, so those
backends get None and the prompt says "(no accessibility tree)" — which is
honest, unlike an empty page.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Awaitable, Callable

# ── The vocabulary ───────────────────────────────────────────────────────
#
# ai/choice.py matches on the actionable words; ai/slo.py renders whatever we
# emit. Order encodes specificity: "combo box" is a combobox, any other box
# with a control word in it is a button, "menu bar" is neither.
VOCABULARY = frozenset(
    {
        "button",
        "link",
        "tab",
        "menuitem",
        "option",
        "textbox",
        "combobox",
        "window",
        "dialog",
        "text",
        "generic",
    }
)
ACTIONABLE = frozenset({"button", "link", "tab", "menuitem", "option", "textbox", "combobox"})
CONTAINERS = frozenset({"window", "dialog"})
TEXT_WORDS = frozenset(
    {
        "text",
        "static",
        "label",
        "paragraph",
        "terminal",
        "log",
        "caption",
        "heading",
        "document",
        "content",
        "formula",
        "section",
    }
)
# Roles worth a text read: content, and a field's current value.
READ_WORDS = TEXT_WORDS | {
    "textbox",
    "searchbox",
    "entry",
    "editbar",
    "password",
    "editor",
    "input",
    "search",
}

_CONTROL_WORDS = frozenset({"button", "check", "radio", "toggle"})
# A field, spelled either as one word (ARIA's "textbox") or by what it holds
# (AT-SPI's "entry", "password text"). "input"/"search" cover the elements
# Chrome names after the tag. Matched before "item" and "tab", so a field is
# never read as a choice or as a page tab.
_FIELD_WORDS = READ_WORDS - TEXT_WORDS
# A tab you click, as opposed to the container that holds the tabs.
_TAB_GROUP_WORDS = frozenset({"list", "panel", "group", "bar", "box"})

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

# How much text reaches the model, in characters.
NAME_CAP = 80
TEXT_CAP = 120


def role_words(raw: str) -> frozenset[str]:
    """The words in a role name, whatever dialect spelled it with."""
    return frozenset(_CAMEL_BOUNDARY.sub(" ", str(raw or "")).replace("-", " ").lower().split())


def norm_role(raw: str) -> str:
    """Classify a role name by the words in it. Never returns outside VOCABULARY."""
    words = role_words(raw)
    if "combo" in words:
        return "combobox"
    # Before the control words: "check menu item" is a menu entry that happens
    # to be checkable, and menu-ness is what the model acts on.
    if "menuitem" in words or {"menu", "item"} <= words:
        return "menuitem"
    if words & _FIELD_WORDS:
        return "textbox"
    if "item" in words or "option" in words:
        return "option"
    if "tab" in words and not words & _TAB_GROUP_WORDS:
        return "tab"
    if words & _CONTROL_WORDS:
        return "button"
    if "link" in words:
        return "link"
    if "document" in words:
        return "text"
    if words & {"frame", "window"}:
        return "window"
    if words & {"dialog", "alert"}:
        return "dialog"
    if words & TEXT_WORDS:
        return "text"
    return "generic"


def pos_of(x: int, y: int, w: int, h: int) -> str:
    return f"{x},{y},{w}x{h}"


def parse_pos(pos: Any) -> tuple[int, int, int, int] | None:
    """The box a "pos" string describes, or None when it does not describe one.

    One writer and one reader for the format: a tree node's coordinates are the
    only way a coordinate-only policy can act on what it observed.
    """
    if not isinstance(pos, str) or "," not in pos:
        return None
    try:
        x, y, size = pos.split(",", 2)
        width, height = size.split("x", 1)
        return int(x), int(y), int(width), int(height)
    except ValueError:
        return None


def pack(
    role: str,
    *,
    name: str = "",
    text: str = "",
    pos: str = "",
    children: list[dict[str, Any]] | None = None,
    omitted: int = 0,
) -> dict[str, Any]:
    """One observation node, built the same way by every backend.

    Keys a model cannot use are not sent. A role is dropped only from text that
    has no label and nothing under it, where the key repeats what its position
    in the tree already said; a node with children always keeps its role, so a
    group reads as a group whichever toolkit walked it. "pos" is the caller's
    decision because only a control's coordinates are actionable.
    """
    out: dict[str, Any] = {}
    if role != "text" or name or children:
        out["role"] = role
    if name:
        out["name"] = name
    if text and text != name:
        out["text"] = text
    if pos:
        out["pos"] = pos
    if omitted:
        out["_omitted"] = omitted
    elif children:
        out["children"] = children
    return out


# ── Shared tree utilities ────────────────────────────────────────────────


def weakest(tree: dict[str, Any]) -> tuple[list[dict[str, Any]], int] | None:
    """(the children list, index) of the last node in pre-order, or None."""
    root = tree.get("children")
    if not root:
        return None
    last = (root, len(root) - 1)

    def visit(children: list[dict[str, Any]]) -> None:
        nonlocal last
        for i, node in enumerate(children):
            last = (children, i)
            visit(node.get("children") or [])

    visit(root)
    return last


def _trim(tree: dict[str, Any], passes: int) -> int:
    """Drop whole subtrees, weakest tail first, for `passes` rounds."""
    dropped = 0
    for _ in range(passes):
        target = weakest(tree)
        if target is None:
            break
        siblings, index = target
        siblings.pop(index)
        dropped += 1
    return dropped


def cap_nodes(tree: dict[str, Any], max_nodes: int) -> dict[str, Any]:
    """Cut the tree to max_nodes by dropping whole subtrees, never text."""
    over = count_nodes(tree) - max_nodes
    if over > 0:
        dropped = _trim(tree, over)
        tree["_trimmed"] = tree.get("_trimmed", 0) + dropped
    return tree


def fit(tree: dict[str, Any], char_budget: int) -> dict[str, Any]:
    """Drop the weakest subtrees until the JSON fits, never truncating text.

    ai/slo.py's _tree_text slices the rendered JSON, which would cut this tree
    mid-structure; removing whole nodes keeps it parseable. Children are
    best-first by the time we get here, so the last node in pre-order is the
    weakest subtree available.
    """
    if char_budget <= 0:
        return tree
    dropped = 0
    while len(json.dumps(tree, default=str)) > char_budget:
        target = weakest(tree)
        if target is None:
            break
        siblings, index = target
        siblings.pop(index)
        dropped += 1
    if dropped:
        tree["_trimmed"] = tree.get("_trimmed", 0) + dropped
    return tree


def count_nodes(tree: dict[str, Any]) -> int:
    stack, count = [tree], 0
    while stack:
        node = stack.pop()
        count += 1
        stack.extend(node.get("children") or [])
    return count


def signal_count(tree: dict[str, Any]) -> int:
    """Nodes that say something: a label, content, or a control."""
    stack = [tree]
    count = 0
    while stack:
        node = stack.pop()
        if node.get("name") or node.get("text") or node.get("role") in ACTIONABLE:
            count += 1
        stack.extend(node.get("children") or [])
    return count


def score(node: dict[str, Any]) -> int:
    """How much this node is worth a place in a small prompt."""
    value = 0
    role = node.get("role")
    if role in ACTIONABLE:
        value += 6 if node.get("pos") else 4
    if role in CONTAINERS:
        value += 3
    if node.get("name"):
        value += 1
    if node.get("text"):
        value += 1
    return value


def rank(tree: dict[str, Any]) -> None:
    """Best-first children, so a budget cut loses the weakest tail.

    ai/slo.py's _tree_text slices the rendered JSON and ai/choice.py walks in
    document order: both keep whatever comes first, so ranking here is what
    makes a small prompt budget mean "the important controls" instead of
    "whatever the page enumerated first". Children are sorted in place — a leaf
    keeps no empty "children" key, which would cost budget and say nothing.
    """
    children = tree.get("children")
    if not children:
        return
    children.sort(key=score, reverse=True)
    for child in children:
        rank(child)


# ── CDP Accessibility.getFullAXTree ──────────────────────────────────────

Send = Callable[..., Awaitable[dict[str, Any]]]

_HOPS = 64


def _field(node: dict[str, Any], key: str) -> str:
    """AX nodes wrap every value: {"type": ..., "value": ...}.

    The wrapped value is a string for the roles we keep and a node reference
    for the ones we do not, so anything non-string reads as empty rather than
    as a Python dict in the prompt.
    """
    raw = (node.get(key) or {}).get("value")
    return raw if isinstance(raw, str) else ""


def _kept_parent(
    node: dict[str, Any],
    by_id: dict[str, dict[str, Any]],
    memo: dict[str, dict[str, Any] | None],
) -> dict[str, Any] | None:
    """The nearest non-ignored ancestor, or None when the chain runs out.

    Chrome marks most of the layout tree ignored — a real button often sits
    under several ignored generics. Dropping those without re-parenting their
    children would lose the button, so the tree is rebuilt over these edges.
    """
    chain: list[str] = []
    seen: set[str] = set()
    pid = node.get("parentId")
    found: dict[str, Any] | None = None
    while pid is not None and len(chain) < _HOPS:
        key = str(pid)
        if key in memo:
            found = memo[key]
            break
        parent = by_id.get(key)
        if parent is None:
            break
        if not parent.get("ignored"):
            found = parent
            break
        if key in seen:
            break
        seen.add(key)
        chain.append(key)
        pid = parent.get("parentId")
    for key in chain:
        memo[key] = found
    return found


def _convert(
    node: dict[str, Any],
    children: list[dict[str, Any]],
    depth: int,
    max_depth: int,
) -> dict[str, Any] | None:
    """One AX node into the observation shape, or None if it says nothing.

    The same rules the desktop walk applies, because both end in pack(): a node
    stays when it carries a label, content, a control, or an actionable
    descendant.
    """
    role = norm_role(_field(node, "role"))
    name = _field(node, "name")[:NAME_CAP].strip()
    value = _field(node, "value")[:TEXT_CAP].strip()
    cut = depth >= max_depth and bool(node.get("childIds"))
    out = pack(
        role,
        name=name,
        text=value,
        children=[] if cut else children,
        omitted=len(node["childIds"]) if cut else 0,
    )
    if role in ACTIONABLE and node.get("backendDOMNodeId") is not None:
        out["_dom"] = node["backendDOMNodeId"]
    if not (name or value or role in ACTIONABLE or role in CONTAINERS or children):
        return None
    return out


def _scaffold(node: dict[str, Any]) -> bool:
    """Nothing here is clickable, so this node only carries text."""
    if node.get("role") in ACTIONABLE:
        return False
    return all(_scaffold(c) for c in node.get("children") or [])


def _drop_repeats(node: dict[str, Any]) -> None:
    """Cut text that only says its parent again.

    Chrome emits an InlineTextBox per rendered line under every StaticText, and
    a heading or section whose name is computed from its own contents repeats
    that text at every level. Only scaffolding can be a repeat, so a control
    labelled like its window is never dropped.
    """
    parent_text = (node.get("name") or node.get("text") or "").strip().lower()
    kept = []
    for child in node.get("children") or []:
        own = (child.get("name") or child.get("text") or "").strip().lower()
        if parent_text and own and own in parent_text and _scaffold(child):
            continue
        kept.append(child)
    if "children" in node:
        if kept:
            node["children"] = kept
        else:
            del node["children"]  # a leaf that lost every child stays a leaf
    for child in kept:
        _drop_repeats(child)


def _targets(tree: dict[str, Any]) -> list[tuple[dict[str, Any], int]]:
    """Nodes wanting a box, with their DOM ids, in the tree's own order."""
    found: list[tuple[dict[str, Any], int]] = []

    def visit(node: dict[str, Any]) -> None:
        dom = node.pop("_dom", None)
        if dom is not None:
            found.append((node, dom))
        for child in node.get("children") or []:
            visit(child)

    visit(tree)
    return found


def normalize_ax_nodes(
    nodes: list[dict[str, Any]],
    *,
    max_nodes: int = 40,
    max_depth: int = 24,
) -> tuple[dict[str, Any] | None, list[tuple[dict[str, Any], int]]]:
    """CDP AX nodes -> {role, name, children} + the nodes that want a box.

    A caller that can afford only a few DOM round-trips spends the ones this
    returns on its strongest controls, because the tree is ranked before they
    are collected.
    """
    by_id = {str(n["nodeId"]): n for n in nodes if n.get("nodeId") is not None}
    kept = [n for n in by_id.values() if not n.get("ignored")]
    if not kept:
        return None, []

    memo: dict[str, dict[str, Any] | None] = {}
    kids: dict[int, list[dict[str, Any]]] = {}
    roots: list[dict[str, Any]] = []
    for node in kept:
        parent = _kept_parent(node, by_id, memo)
        if parent is None:
            roots.append(node)
        else:
            kids.setdefault(id(parent), []).append(node)

    def build(node: dict[str, Any], depth: int) -> dict[str, Any] | None:
        children = [
            made
            for made in (build(c, depth + 1) for c in kids.get(id(node), []))
            if made is not None
        ]
        return _convert(node, children, depth, max_depth)

    built = [made for made in (build(r, 0) for r in roots) if made is not None]
    if not built:
        return None, []
    tree = built[0] if len(built) == 1 else {"role": "generic", "children": built}
    _drop_repeats(tree)
    rank(tree)
    cap_nodes(tree, max_nodes)
    return tree, _targets(tree)


async def _fill_boxes(
    send: Send, targets: list[tuple[dict[str, Any], int]], limit: int, deadline: float
) -> None:
    """Coordinates for the top-ranked controls, best-effort and time-bounded.

    One round-trip per node, so a page of a hundred buttons could cost more
    than the step it is observing: hence the limit and the deadline. A node
    that gets no box simply has no "pos" — the tree says what it does not know.
    """
    ends = time.monotonic() + deadline
    for node, dom_id in targets[:limit]:
        if time.monotonic() >= ends:
            return
        try:
            reply = await send("DOM.getBoxModel", {"backendNodeId": dom_id})
        except Exception:  # a dead target is a missing position, not a failed step
            continue
        box = (reply or {}).get("model") or {}
        border = (box.get("border") or box.get("content") or [])[:8]
        if len(border) < 8:
            continue
        xs, ys = border[0::2], border[1::2]
        width, height = max(xs) - min(xs), max(ys) - min(ys)
        if width > 0 and height > 0:
            node["pos"] = pos_of(int(min(xs)), int(min(ys)), int(width), int(height))


async def fetch_ax_tree(
    send: Send,
    *,
    char_budget: int = 1400,
    max_nodes: int = 40,
    max_depth: int = 24,
    box_limit: int = 16,
    box_deadline: float = 0.8,
) -> dict[str, Any] | None:
    """The live page as the tree avion speaks, or None when there is no signal.

    `send` is any CDP sender: Playwright's CDPSession.send, or CdpBackend._send.
    Fewer than two nodes saying anything returns None rather than a bare root —
    "(no accessibility tree)" is a truer prompt than one title.
    """
    for enable in ("DOM.enable", "Accessibility.enable"):
        try:
            await send(enable)
        except Exception:
            pass  # enable only gates events; the tree reads fine without it
    result = await send("Accessibility.getFullAXTree")
    tree, targets = normalize_ax_nodes(
        (result or {}).get("nodes") or [], max_nodes=max_nodes, max_depth=max_depth
    )
    if tree is None or signal_count(tree) < 2:
        return None
    await _fill_boxes(send, targets, box_limit, box_deadline)
    return fit(tree, char_budget)
