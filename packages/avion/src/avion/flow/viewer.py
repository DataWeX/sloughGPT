"""Flow viewer — renders a FlowGraph as a self-contained trace-viewer HTML page.

What it does
------------
Turns the recorded flow (the same data ``EventRecorder.save`` exports) into ONE
standalone HTML file with the Playwright-Trace-Viewer layout:

    ┌────────────┬──────────────────────┬───────────────────────┐
    │ step list  │      flow map        │   step detail         │
    │ (timeline) │   (SVG nodes+edges)  │   (side panel:        │
    │            │                      │    instructions,      │
    │            │                      │    selector, data)    │
    └────────────┴──────────────────────┴───────────────────────┘

Clicking a step — in the list OR on the map — drives the right-hand detail
panel. Everything is pre-rendered and toggled with ~15 lines of inline JS:
no CDN, no framework, no server. Opens from ``file://`` anywhere.

Why a generated file, not a web app page
----------------------------------------
avion is pure stdlib with no UI layer and no web server; a generated page keeps
it that way (check-if-exists: ``core/reporter.py`` renders task results to
terminal/markdown — a different domain; nothing in the repo renders event flows).
An apps/web integration can consume the same JSON later; this file is the
headless-testable model of the view.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid a circular import with the package __init__
    from avion.flow import FlowGraph, FlowStep

# Display constants (kept here so the layout is one readable block).
_NODE_W = 240
_NODE_H = 46
_NODE_GAP = 34
_SEGMENT_GAP = 64
_SVG_W = 640
_SVG_X = _SVG_W // 2
_MAX_LABEL = 28  # fits "[123] " prefix + text inside _NODE_W at font-size 11
_MAX_STEPS = 500  # safety cap; a flow bigger than this is a log, not a map


def _count(n: int, word: str) -> str:
    """'1 step' / '2 steps' — pluralise instead of shipping '1 picks'."""
    return f"{n} {word}" + ("" if n == 1 else "s")


def _label(step: FlowStep) -> str:
    text = step.label.replace("\n", " ").strip()
    if len(text) > _MAX_LABEL:
        text = text[: _MAX_LABEL - 1] + "\u2026"
    return html.escape(text)


def _data_rows(step: FlowStep) -> str:
    """Key/value rows for the detail panel, escaping every user string."""
    rows: list[str] = []
    data = step.data or {}
    for key in ("best", "tag", "text"):
        if key in data:
            rows.append(
                f'<tr><th>{html.escape(str(key))}</th>'
                f"<td>{html.escape(str(data[key]))}</td></tr>"
            )
    if "rect" in data and isinstance(data["rect"], dict):
        r = data["rect"]
        rows.append(
            "<tr><th>rect</th><td>"
            + html.escape(
                ", ".join(f"{k}={r.get(k, '?')}" for k in ("x", "y", "width", "height"))
            )
            + "</td></tr>"
        )
    if "candidates" in data:
        items = "".join(
            f"<li><code>{html.escape(str(c.get('strategy', '?')))}</code> "
            f"{html.escape(str(c.get('value', '')))}</li>"
            for c in data.get("candidates", [])
            if isinstance(c, dict)
        )
        if items:
            rows.append(f"<tr><th>candidates</th><td><ul>{items}</ul></td></tr>")
    # Everything not surfaced above, as a JSON fallback row.
    shown = {"best", "tag", "text", "rect", "candidates"}
    rest = {k: v for k, v in data.items() if k not in shown}
    if rest:
        payload = json.dumps(rest, default=str, ensure_ascii=False)
        payload = payload.replace("</", "<\\/")  # never let data close a script/tag
        rows.append(
            f"<tr><th>data</th><td><code>{html.escape(payload)}</code></td></tr>"
        )
    return "".join(rows)


def _detail_panel(index: int, step: FlowStep) -> str:
    """One pre-rendered detail card (hidden unless selected)."""
    status = (
        '<span class="ok">passed</span>'
        if step.success
        else '<span class="fail">failed</span>'
    )
    err = (
        f'<p class="error">{html.escape(step.error)}</p>' if step.error else ""
    )
    dur = (
        f'<span class="dur">{step.duration_ms:.0f} ms</span>'
        if step.duration_ms
        else ""
    )
    kind = " \u2014 picked element" if step.is_pick else ""
    rows = _data_rows(step)
    return f"""
    <section class="detail" id="d{index}"{' hidden' if index else ''}>
      <h2>Step {step.seq}: {html.escape(step.type)}{kind}</h2>
      <p class="instr">{
        html.escape(f"\u2192 {step.label}" if not step.is_pick
                    else f"\u2192 pick: {step.data.get('best') or step.label}")
      }</p>
      <p class="meta">{status}{dur}</p>
      {err}
      {f'<table class="data">{rows}</table>' if rows else ''}
    </section>"""


def _map_svg(graph: FlowGraph) -> str:
    """The centre flow map: nodes, edges, segment separators — all inline SVG.

    Two phases: compute layout (indices by identity, not dataclass equality),
    then emit the tag with the real height — no placeholder patching, so a
    step label containing braces can never be corrupted.
    """
    # Identity map: equal-valued steps (same seq/data) must not collide.
    idx_by_id = {id(s): i for i, s in enumerate(graph.steps)}

    nodes: list[tuple[int, int, FlowStep]] = []  # (index, y_center, step)
    edges: list[tuple[int, int]] = []  # node-index pairs to connect
    seg_labels: list[tuple[int, str]] = []  # (y, text) separators
    y = _NODE_H // 2 + 6
    prev = -1
    segments = graph.segments()
    for si, seg in enumerate(segments):
        if si > 0:
            y += _SEGMENT_GAP - _NODE_GAP
            # name the boundary after the navigation that opened the page,
            # not a generic "page boundary"
            target = " ".join(seg[0].label.split())[:40] if seg else ""
            seg_labels.append((y - _SEGMENT_GAP // 2 - 6, f"page: {target}"))
            y += _SEGMENT_GAP // 2
        for step in seg:
            i = idx_by_id[id(step)]
            if prev >= 0:
                edges.append((prev, i))
            prev = i
            nodes.append((i, y, step))
            y += _NODE_H + _NODE_GAP
    height = max(y, 120)

    parts: list[str] = [
        f'<svg viewBox="0 0 {_SVG_W} {height}" xmlns="http://www.w3.org/2000/svg">',
        '<defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#6b7280"/></marker></defs>',
    ]
    y_map = {i: cy for i, cy, _ in nodes}
    for a, b in edges:
        y1, y2 = y_map[a] + _NODE_H // 2, y_map[b] - _NODE_H // 2
        parts.append(
            f'<line x1="{_SVG_X}" y1="{y1}" x2="{_SVG_X}" y2="{y2}" '
            'stroke="#6b7280" stroke-width="1.5" marker-end="url(#ar)"/>'
        )
    for ly, text in seg_labels:
        parts.append(
            f'<text x="{_SVG_X}" y="{ly}" text-anchor="middle" font-size="10" '
            f'fill="#9ca3af" font-family="monospace">\u2014 {html.escape(text)} \u2014</text>'
        )

    for i, cy, step in nodes:
        cls = "fail" if not step.success else ("pick" if step.is_pick else "step")
        stroke = "#dc2626" if not step.success else ("#7c3aed" if step.is_pick else "#374151")
        fill = "#fee2e2" if not step.success else ("#f5f3ff" if step.is_pick else "#ffffff")
        cy0 = cy - _NODE_H // 2
        if step.is_pick:
            shape = (
                f'<polygon points="{_SVG_X},{cy0} {_SVG_X + _NODE_W // 2},{cy} '
                f'{_SVG_X},{cy0 + _NODE_H} {_SVG_X - _NODE_W // 2},{cy}" '
                f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
            )
        else:
            shape = (
                f'<rect x="{_SVG_X - _NODE_W // 2}" y="{cy0}" width="{_NODE_W}" '
                f'height="{_NODE_H}" rx="8" fill="{fill}" stroke="{stroke}" '
                'stroke-width="1.5"/>'
            )
        parts.append(
            f'<g class="node {cls}" id="n{i}" onclick="selectStep({i})" '
            f'onkeydown="if(event.key===\'Enter\'||event.key===\' \')'
            f'{{event.preventDefault();selectStep({i})}}" '
            f'role="button" tabindex="0" aria-label="Step {step.seq}: '
            f'{_label(step)}" style="cursor:pointer">'
            f"{shape}"
            f'<text x="{_SVG_X}" y="{cy + 4}" text-anchor="middle" font-size="11" '
            f'font-family="system-ui,sans-serif" fill="#111827">'
            f"[{step.seq}] {_label(step)}</text></g>"
        )
    parts.append("</svg>")
    return "".join(parts)


def render_html(graph: FlowGraph) -> str:
    """Render the flow as one standalone trace-viewer HTML page."""
    steps = graph.steps[:_MAX_STEPS]
    # Build the map from the SAME capped steps the list renders: otherwise a
    # >500-step flow emits map nodes whose selectStep() has no detail panel.
    view = type(graph)(steps=list(steps), session_name=graph.session_name)
    title = html.escape(graph.session_name or "avion flow")

    list_items = "".join(
        f'<li id="li{i}" onclick="selectStep({i})" '
        f'onkeydown="if(event.key===\'Enter\'||event.key===\' \')'
        f'{{event.preventDefault();selectStep({i})}}" tabindex="0"'
        f' class="{"" if s.success else "bad"}">'
        f'<span class="seq">{s.seq}</span> '
        f'<span class="type">{html.escape(s.type)}</span> '
        f"<span>{_label(s)}</span></li>"
        for i, s in enumerate(steps)
    )
    details = "".join(_detail_panel(i, s) for i, s in enumerate(steps))
    counts = (
        f"{_count(len(graph.steps), 'step')} \u00b7 "
        f"{_count(len(graph.picks), 'pick')}"
        + (f" \u00b7 {len(graph.failed)} failed" if graph.failed else "")
    )
    trunc = (
        f'<p class="empty">showing first {_MAX_STEPS} of '
        f"{len(graph.steps)} steps</p>"
        if len(graph.steps) > _MAX_STEPS
        else ""
    )
    empty = (
        '<p class="empty">No steps recorded yet \u2014 run a flow, then reload.</p>'
        if not steps
        else ""
    )

    return f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>{title} \u2014 flow trace</title>
<style>
  :root {{ font-family: system-ui, sans-serif; color: #111827; }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: #f9fafb; }}
  header {{ padding: 10px 16px; background: #111827; color: #f9fafb; }}
  header h1 {{ margin: 0; font-size: 15px; font-weight: 600; }}
  header .counts {{ opacity: .7; font-size: 12px; }}
  main {{ display: grid; grid-template-columns: 260px 1fr 340px;
          height: calc(100vh - 44px); }}
  aside, section {{ overflow: auto; }}
  .timeline {{ background: #fff; border-right: 1px solid #e5e7eb; padding: 8px; }}
  .timeline ul {{ list-style: none; margin: 0; padding: 0; }}
  .timeline li {{ display: flex; gap: 6px; align-items: baseline; padding: 6px 8px;
                  border-radius: 6px; cursor: pointer; font-size: 12.5px; }}
  .timeline li:hover {{ background: #f3f4f6; }}
  .timeline li.active {{ background: #ede9fe; outline: 1px solid #7c3aed; }}
  .timeline li.bad .type {{ color: #dc2626; }}
  .timeline li:focus-visible, .map .node:focus-visible {{
    outline: 2px solid #7c3aed; outline-offset: 2px; }}
  @media (max-width: 1100px) {{
    main {{ grid-template-columns: 210px 1fr 290px; }}
  }}
  .seq {{ font-family: monospace; color: #6b7280; min-width: 2ch; }}
  .type {{ font-weight: 600; color: #7c3aed; }}
  .map {{ background: #f9fafb; padding: 12px; display: flex;
          justify-content: center; }}
  .map svg {{ max-width: 100%; height: auto; }}
  .node.step:hover rect, .node.pick:hover polygon {{ stroke-width: 2.5; }}
  .map .active rect, .map .active polygon {{ stroke: #7c3aed; stroke-width: 2.5; }}
  .details {{ background: #fff; border-left: 1px solid #e5e7eb; padding: 14px; }}
  .detail h2 {{ margin: 0 0 6px; font-size: 14px; }}
  .instr {{ font-family: ui-monospace, monospace; font-size: 12.5px;
            background: #f3f4f6; padding: 8px; border-radius: 6px;
            white-space: pre-wrap; word-break: break-word; }}
  .meta {{ font-size: 12px; }}
  .ok {{ color: #059669; }} .fail {{ color: #dc2626; font-weight: 600; }}
  .dur {{ color: #6b7280; margin-left: 8px; }}
  .error {{ color: #b91c1c; font-size: 12.5px; background: #fef2f2;
            padding: 8px; border-radius: 6px; }}
  table.data {{ width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 8px; }}
  table.data th {{ text-align: left; color: #6b7280; font-weight: 600;
                   padding: 4px 8px 4px 0; vertical-align: top; white-space: nowrap; }}
  table.data td {{ padding: 4px 0; word-break: break-word; }}
  table.data ul {{ margin: 0; padding-left: 14px; }}
  .empty {{ color: #6b7280; font-size: 13px; }}
</style>
<header><h1>{title}</h1><span class="counts">{counts}</span></header>
<main>
  <aside class="timeline"><ul>{list_items}</ul>{empty}{trunc}</aside>
  <div class="map">{_map_svg(view) if steps else ''}</div>
  <section class="details">{details}
    <p class="empty"{' hidden' if steps else ''}>Nothing selected.</p>
  </section>
</main>
<script>
function selectStep(i) {{
  document.querySelectorAll('.detail').forEach(d => d.hidden = true);
  const d = document.getElementById('d' + i); if (d) d.hidden = false;
  document.querySelectorAll('.timeline li').forEach(el => {{
    el.classList.remove('active'); el.removeAttribute('aria-current'); }});
  const li = document.getElementById('li' + i);
  if (li) {{ li.classList.add('active'); li.setAttribute('aria-current', 'true');
             li.scrollIntoView({{block: 'nearest'}}); }}
  document.querySelectorAll('.map .node').forEach(el => {{
    el.classList.remove('active'); el.removeAttribute('aria-current'); }});
  const n = document.getElementById('n' + i);
  if (n) {{ n.classList.add('active'); n.setAttribute('aria-current', 'true'); }}
}}
if (document.getElementById('d0')) selectStep(0);
</script>
</html>
"""


def save_html(graph: FlowGraph, path: str) -> Path:
    """Write the viewer page to ``path`` (parent dirs created)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(graph), encoding="utf-8")
    return out
