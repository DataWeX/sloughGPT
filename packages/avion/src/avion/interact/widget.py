"""Live-state widget — the on-screen twin of avion's identify block.

What it does
------------
A floating side panel injected into the live page showing what avion sees
RIGHT NOW: the match set with its original-set ordinals, which matches are
hidden (and therefore skipped), which were clicked, the last action, and
whether a recording is running. It is a READ-ONLY projection of state the
CLI already holds — no server, no endpoint, no page-side re-probing. The
page never re-runs matching: a single source of truth cannot drift.

Why it is invisible to avion
----------------------------
The panel lives in a SHADOW ROOT under an unknown ``<avion-widget>`` host
marked ``aria-hidden``/``presentation``:

- ``querySelectorAll('body *')`` (both probes) cannot select into a shadow
  tree, so widget elements are never returned by ``find_elements``;
- the TEXT predicate reads light-DOM ``textContent``, which does not
  include shadow content — verified against live Chromium: a marker string
  placed only in the shadow → 0 matches, near-miss reports "no element on
  the page contains".

Zero light-DOM text, host ``pointer-events: none``, everything built with
``createElement`` + ``textContent`` (no ``innerHTML`` anywhere — frame data
is app text and must never be parsed as markup): observationally a no-op.

Why push, not poll
------------------
Frames are pushed from the action sites (``_cmd_click`` & friends in
``avion.cli``) through ``backend.evaluate`` — the same injection path
``avion.interact.picker`` uses. ``WidgetBridge`` keeps the last frame so a
capture can hide the panel and restore it afterwards. Every call is
best-effort: one failure (navigating target, backend gone) disables the
bridge and the verb proceeds untouched — the widget can never fail a run.
"""

from __future__ import annotations

import json
from typing import Any

# Leading sentinel comment: unambiguously distinct from picker/probe
# evaluates, so a backend (or a test fake) can tell widget traffic apart
# without substring guessing.
_SENTINEL = "/*avion-widget*/"

# The injection body: builds the host + shadow panel and assigns
# window.__avionWidget = {render, hide, restore, _last}. Statements, not an
# IIFE — callers splice it into `if (!window.__avionWidget) { ... }`.
# Only ever parsed once per page (guarded by the caller); idempotent.
_ENSURE_JS = r"""
const host = document.createElement("avion-widget");
host.setAttribute("aria-hidden", "true");
host.setAttribute("role", "presentation");
host.style.cssText = "position:fixed;top:50%;right:12px;transform:translateY(-50%);" +
  "z-index:2147483647;pointer-events:none;";
const sr = host.attachShadow({ mode: "open" });
const style = document.createElement("style");
style.textContent = `
  :host { all: initial; }
  .card { all: initial; box-sizing: border-box; display: block; width: 340px;
    max-height: 60vh; overflow: auto; padding: 10px 12px;
    border: 1px solid rgba(167,139,250,.45); border-radius: 10px;
    background: rgba(17,12,28,.94); box-shadow: 0 8px 28px rgba(0,0,0,.45);
    font: 11px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace;
    color: #e9e4f7; pointer-events: none; }
  .hd { display: flex; gap: 8px; align-items: center; }
  .brand { color: #a78bfa; font-weight: 700; letter-spacing: .12em; }
  .rec { color: #f87171; font-weight: 700; }
  .ts { color: #8b83a3; margin-left: auto; }
  .act { color: #c4b5fd; margin: 6px 0; word-break: break-all; }
  .m { display: flex; gap: 6px; align-items: baseline; }
  .n { color: #a78bfa; min-width: 24px; flex: none; }
  .d { color: #ddd6f2; overflow: hidden; text-overflow: ellipsis;
    white-space: nowrap; flex: 1; }
  .ok { color: #34d399; font-weight: 700; flex: none; }
  .hid { color: #fbbf24; flex: none; }
  .off { color: #6b6480; flex: none; }
  .sum { color: #8b83a3; margin-top: 6px; }
  .err { color: #f87171; }
`;
const card = document.createElement("div");
card.className = "card";
sr.append(style, card);
(document.documentElement || document.body).appendChild(host);
window.__avionWidget = {
  _last: null,
  _card: card,
  _host: host,
  render(frame) {
    this._last = frame;
    this._host.style.display = "";
    this._paint();
  },
  hide() {
    this._host.style.display = "none";
  },
  restore() {
    this._host.style.display = "";
    if (this._last) this._paint();
  },
  _paint() {
    const f = this._last || {};
    const c = this._card;
    c.textContent = "";
    const line = (cls, text) => {
      const el = document.createElement("div");
      el.className = cls;
      el.textContent = text;
      return el;
    };
    const hd = document.createElement("div");
    hd.className = "hd";
    const brand = document.createElement("span");
    brand.className = "brand";
    brand.textContent = "AVION";
    hd.append(brand);
    if (f.recording) {
      const rec = document.createElement("span");
      rec.className = "rec";
      rec.textContent = "\u25cf REC";
      hd.append(rec);
    }
    const ts = document.createElement("span");
    ts.className = "ts";
    ts.textContent = f.ts || "";
    hd.append(ts);
    c.append(hd);
    const actText = [
      f.action || "",
      f.by || "",
      f.value != null ? JSON.stringify(String(f.value)) : "",
      f.text != null ? JSON.stringify(String(f.text)) : "",
      f.url || "",
      f.nth != null ? "nth " + f.nth : "",
    ]
      .filter(Boolean)
      .join(" ");
    c.append(line("act", actText));
    if (f.error) {
      c.append(line("err", "\u2717 " + f.error));
    }
    let clicked = 0;
    let hidden = 0;
    for (const m of f.matches || []) {
      const row = document.createElement("div");
      row.className = "m";
      const n = document.createElement("span");
      n.className = "n";
      n.textContent = "#" + m.n;
      const d = document.createElement("span");
      d.className = "d";
      d.textContent = m.desc || "";
      d.title = m.desc || "";
      row.append(n, d);
      if (m.clicked) {
        const s = document.createElement("span");
        s.className = "ok";
        s.textContent = "\u2713";
        row.append(s);
        clicked++;
      } else if (m.hidden) {
        const s = document.createElement("span");
        s.className = "hid";
        s.textContent = "hidden";
        row.append(s);
        hidden++;
      } else {
        const s = document.createElement("span");
        s.className = "off";
        s.textContent = "not clicked";
        row.append(s);
      }
      c.append(row);
    }
    if (f.matches && f.matches.length) {
      let sum = f.matches.length + " matches \u00b7 " + clicked + " clicked";
      if (hidden) {
        sum += " \u00b7 " + hidden + " hidden";
      }
      if (f.note) {
        sum += " \u00b7 " + f.note;
      }
      c.append(line("sum", sum));
    }
  },
};
window.__avionWidget._paint();
"""


def _render_expr(frame: dict[str, Any]) -> str:
    """Inject-if-missing, then render ``frame`` — one self-healing evaluate.

    Re-runs the ensure on every push, so a full navigation (which destroys
    the host) is repaired by the NEXT action's push with no listener of our
    own; SPA route changes keep the host natively.
    """
    return (
        f"{_SENTINEL} (() => {{ if (!window.__avionWidget) {{ {_ENSURE_JS} }}"
        f" window.__avionWidget.render({json.dumps(frame, default=str)});"
        " return 'widget_rendered'; })()"
    )


# Hide/restore deliberately do NOT ensure: a capture of a page the widget
# was never pushed to must stay widget-free, not inject-then-hide.
_HIDE_EXPR = (
    f"{_SENTINEL} (() => {{ const w = window.__avionWidget;"
    " if (w) w.hide(); return 'widget_hidden'; })()"
)
_RESTORE_EXPR = (
    f"{_SENTINEL} (() => {{ const w = window.__avionWidget;"
    " if (w) w.restore(); return 'widget_restored'; })()"
)


class WidgetBridge:
    """Push state frames to the in-page widget; never fail the verb.

    The bridge is the ONLY writer: callers hand it JSON-able dicts, it
    serialises them into the render expression. On the first evaluate
    failure it disables itself (the page may have navigated, the backend
    may be gone) — subsequent calls are cheap no-ops.
    """

    def __init__(self, backend: Any) -> None:
        self._backend = backend
        self._enabled = True
        self._last: dict[str, Any] | None = None

    @property
    def last(self) -> dict[str, Any] | None:
        """The most recent frame pushed (or None) — test/inspection hook."""
        return self._last

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def push(self, frame: dict[str, Any]) -> bool:
        """Render ``frame`` in the panel; inject the panel if missing."""
        self._last = frame
        return await self._eval(_render_expr(frame))

    async def hide(self) -> bool:
        """Hide the panel around a capture (no-op when never injected)."""
        return await self._eval(_HIDE_EXPR)

    async def restore(self) -> bool:
        """Bring the panel back after a capture."""
        return await self._eval(_RESTORE_EXPR)

    async def _eval(self, expr: str) -> bool:
        if not self._enabled:
            return False
        try:
            await self._backend.evaluate(expr)
            return True
        except Exception:
            # One strike: the widget is debug sugar. A failure must never
            # surface as a failed verb (and CdpBackend.evaluate already
            # logged the JS error to the console store on its way up).
            self._enabled = False
            return False
