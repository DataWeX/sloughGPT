"""Injected element-picker overlay for flow mapping.

What it does
------------
Injects a self-contained JS overlay into the live page via ``backend.evaluate``.
The overlay lets an agent (or a human) hover to highlight and click to pick an
element. On pick it computes the *best* stable selector for that element, in
priority order:

    test-id  >  role[accessible-name]  >  label  >  stable CSS  >  xpath

...and resolves it back into a ready-to-reuse :class:`ElementLocator`, which is
also recorded into the flow as a ``SELECT`` event so the run can be replayed or
visualised later (see ``avion.flow``).

Why the resolution happens in-page
----------------------------------
The picker runs entirely inside the page and returns a JSON payload, so it works
headless over CDP with no human and no separate UI server. The JS is a single
constant (``_PICKER_JS``) with no external dependency — avion stays pure stdlib.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from avion.core.element import ElementLocator, Selector, SelectorStrategy
from avion.events.models import EventType

# Poll expression used by ElementPicker.pick() to read the picked payload.
# The leading sentinel comment makes it unambiguously distinct from the
# overlay-injection call, so a backend (or a test fake) can tell the two
# evaluate() calls apart without substring guessing.
_POLL_EXPR = "/*avion-picker-poll*/ window.__avionPicked || null"

# The injected overlay. Self-contained: builds a shadow-DOM highlight layer,
# tracks the element under the cursor, and on click computes the best selector.
# Returns a JSON string: {strategy, value, tag, text, rect, candidates:[...]}.
_PICKER_JS = r"""
(() => {
  if (window.__avionPicker) { window.__avionPicker.destroy(); }
  const NS = "http://www.w3.org/2000/svg";
  const layer = document.createElementNS(NS, "svg");
  layer.setAttribute("style", "position:fixed;inset:0;width:100vw;height:100vh;" +
    "pointer-events:none;z-index:2147483647;");
  const box = document.createElementNS(NS, "rect");
  box.setAttribute("fill", "rgba(124,58,237,0.15)");
  box.setAttribute("stroke", "#7c3aed");
  box.setAttribute("stroke-width", "2");
  box.setAttribute("stroke-dasharray", "6 3");
  layer.appendChild(box);
  (document.body || document.documentElement).appendChild(layer);

  const cssEscape = (s) => (window.CSS && CSS.escape) ? CSS.escape(s) : String(s).replace(/[^\w-]/g, "\\$&");

  // Stable CSS: prefer #id when it looks auto-generated-free, else a short tag path.
  const stableCss = (el) => {
    const id = el.getAttribute("id");
    if (id && !/^[0-9]/.test(id) && !/[:.\d]/.test(id)) return "#" + cssEscape(id);
    const testid = el.getAttribute("data-testid") || el.getAttribute("data-test-id");
    if (testid) return `[data-testid="${testid}"]`;
    const parts = [];
    let node = el;
    while (node && node.nodeType === 1 && parts.length < 4) {
      let seg = node.tagName.toLowerCase();
      const pid = node.getAttribute("id");
      if (pid) { parts.unshift("#" + cssEscape(pid)); break; }
      const parent = node.parentElement;
      if (parent) {
        const sibs = Array.from(parent.children).filter((c) => c.tagName === node.tagName);
        if (sibs.length > 1) seg += ":nth-of-type(" + (sibs.indexOf(node) + 1) + ")";
      }
      parts.unshift(seg);
      node = parent;
    }
    return parts.join(" > ");
  };

  const accessibleName = (el) =>
    el.getAttribute("aria-label") || (el.getAttribute("aria-labelledby")
      ? (document.getElementById(el.getAttribute("aria-labelledby")) || {}).textContent : "")
    || el.innerText || el.value || "";

  const candidates = (el) => {
    const out = [];
    const testid = el.getAttribute("data-testid") || el.getAttribute("data-test-id");
    if (testid) out.push({ strategy: "test_id", value: testid });
    const role = el.getAttribute("role") || ({ A: "link", BUTTON: "button", INPUT: el.type === "submit" ? "button" : "textbox",
      SELECT: "combobox", TEXTAREA: "textbox", IMG: "img", NAV: "navigation", MAIN: "main" })[el.tagName];
    if (role) {
      const name = accessibleName(el).trim();
      out.push({ strategy: "role", value: name ? role + "[" + name + "]" : role });
    }
    const label = el.labels && el.labels[0] ? el.labels[0].innerText.trim() : "";
    if (label) out.push({ strategy: "label", value: label });
    out.push({ strategy: "css", value: stableCss(el) });
    return out;
  };

  const onMove = (e) => {
    const t = e.target;
    if (!t || t === layer || layer.contains(t)) return;
    const r = t.getBoundingClientRect();
    box.setAttribute("x", r.x); box.setAttribute("y", r.y);
    box.setAttribute("width", r.width); box.setAttribute("height", r.height);
  };

  const onClick = (e) => {
    e.preventDefault(); e.stopPropagation();
    const t = e.target;
    const r = t.getBoundingClientRect();
    const cands = candidates(t);
    const payload = {
      strategy: cands[0].strategy, value: cands[0].value,
      tag: t.tagName.toLowerCase(),
      text: (accessibleName(t) || "").trim().slice(0, 120),
      rect: { x: r.x, y: r.y, width: r.width, height: r.height },
      candidates: cands,
    };
    window.__avionPicked = payload;
    destroy();
    return payload;
  };

  const destroy = () => { layer.remove(); document.removeEventListener("mousemove", onMove, true);
    document.removeEventListener("click", onClick, true); window.__avionPicker = null; };
  window.__avionPicker = { destroy };
  document.addEventListener("mousemove", onMove, true);
  document.addEventListener("click", onClick, true);
  return "picker_active";
})();
"""


def _locator_from_payload(payload: dict[str, Any]) -> ElementLocator:
    """Build an ElementLocator from the in-page picker payload (best-first).

    The payload's ``candidates`` are already ordered by stability, so we keep
    that order in the locator chain — resolution tries them in turn.
    """
    selectors: list[Selector] = []
    for cand in payload.get("candidates", []):
        strategy = cand.get("strategy", "css")
        value = cand.get("value", "")
        if not value:
            continue
        try:
            selectors.append(Selector(SelectorStrategy(strategy), value))
        except ValueError:
            continue  # skip unknown strategy rather than fail the whole pick
    if not selectors:  # last resort: the raw css the JS computed
        selectors.append(Selector(SelectorStrategy.CSS, payload.get("value", "")))
    return ElementLocator(
        selectors=tuple(selectors),
        description=f"picked <{payload.get('tag', '?')}> {payload.get('text', '')!r}".strip(),
    )


@dataclass
class PickedElement:
    """A picked element: its best locator plus the raw resolution detail."""

    locator: ElementLocator
    tag: str = ""
    text: str = ""
    rect: dict[str, float] = field(default_factory=dict)
    candidates: list[dict[str, str]] = field(default_factory=list)

    def to_flow_data(self) -> dict[str, Any]:
        """The payload recorded into the flow as a SELECT event."""
        return {
            "tag": self.tag,
            "text": self.text,
            "rect": self.rect,
            "best": self.locator.describe(),
            "candidates": self.candidates,
        }


class ElementPicker:
    """Drives the injected overlay over a backend and records picks into a flow.

    Usage::

        picker = ElementPicker(backend, recorder=recorder)
        picked = await picker.pick(timeout=10)
        await picker.act(picked)   # click the picked element, records a CLICK
    """

    def __init__(self, backend: Any, recorder: Any = None) -> None:
        self._backend = backend
        self._recorder = recorder

    async def _shot(self) -> bytes | None:
        """Best-effort page screenshot at pick time (the trace-viewer image).

        A capture failure (backend without a page, CDP hiccup) must never
        break picking — the event is still recorded, just without its image.
        """
        try:
            return await self._backend.screenshot()
        except Exception:
            return None

    async def activate(self) -> bool:
        """Inject the overlay and arm it for picking."""
        result = await self._backend.evaluate(_PICKER_JS)
        return result == "picker_active"

    async def pick(self, timeout: float = 30.0) -> PickedElement | None:
        """Activate, then wait for a pick. Returns None on timeout.

        Polls ``window.__avionPicked`` (set by the overlay's click handler).
        """
        import asyncio

        await self.activate()
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            payload = await self._backend.evaluate(_POLL_EXPR)
            if payload:
                picked = PickedElement(
                    locator=_locator_from_payload(payload),
                    tag=payload.get("tag", ""),
                    text=payload.get("text", ""),
                    rect=payload.get("rect", {}),
                    candidates=payload.get("candidates", []),
                )
                # Guard on `is not None`, NOT truthiness: EventRecorder defines
                # __len__ (event count), so a fresh recorder is falsey and a
                # truthiness guard silently skips recording the first pick.
                if self._recorder is not None:
                    self._recorder.record(
                        EventType.SELECT,
                        name=f"pick <{picked.tag}>",
                        data=picked.to_flow_data(),
                        screenshot=await self._shot(),
                    )
                return picked
            await asyncio.sleep(0.1)
        return None

    async def act(self, picked: PickedElement) -> None:
        """Click the picked element through its best selector, recording a CLICK.

        Uses the backend's own click path so the flow stays backend-agnostic.
        """
        element = await self._backend.find_element(picked.locator)
        if element is None:
            if self._recorder is not None:
                self._recorder.record(
                    EventType.CLICK,
                    name=f"click <{picked.tag}>",
                    success=False,
                    error="picked element no longer resolvable",
                    screenshot=await self._shot(),
                )
            return
        await self._backend.click(element)
        if self._recorder is not None:
            self._recorder.record(
                EventType.CLICK,
                name=f"click <{picked.tag}>",
                data={"best": picked.locator.describe()},
                screenshot=await self._shot(),
            )
