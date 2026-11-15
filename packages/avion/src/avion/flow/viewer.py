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

Recording mode (``render_html(graph, recording=…)``)
-----------------------------------------------------
When a recording artifact is passed in, the same page gains the **player**:
sampled frames inline as data URIs, a scrubber (``<input type=range>``) and
pre-rendered box overlays toggled by ``#boxtoggle``. Boxes come from
``boxes.jsonl`` geometry rendered as %-positioned divs AT BUILD TIME —
selecting a step shows its nearest frame (``SF[i]``, precomputed on the same
shared clock) and that frame's overlays. Pixels stay clean: nothing is burned
into the frames, so the viewer, the export and a future re-derivation all
draw the same labels from derived data.

Why a generated file, not a web app page
----------------------------------------
avion is pure stdlib with no UI layer and no web server; a generated page keeps
it that way (check-if-exists: ``core/reporter.py`` renders task results to
terminal/markdown — a different domain; nothing in the repo renders event flows).
An apps/web integration can consume the same JSON later; this file is the
headless-testable model of the view.
"""

from __future__ import annotations

import base64
import html
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # avoid a circular import with the package __init__
    from avion.flow import FlowGraph, FlowStep
    from avion.recording import RecordingArtifact

# Display constants (kept here so the layout is one readable block).
_NODE_W = 240
_NODE_H = 46
_NODE_GAP = 34
_SEGMENT_GAP = 64
_SVG_W = 640
_SVG_X = _SVG_W // 2
_MAX_LABEL = 28  # fits "[123] " prefix + text inside _NODE_W at font-size 11
_MAX_STEPS = 500  # safety cap; a flow bigger than this is a log, not a map
_MAX_FRAMES = 400  # inline data URIs are heavy; a longer take is a video, not a page


# ── recording model ─────────────────────────────────────────────────────


@dataclass
class FrameView:
    """One frame as the player renders it: inline data URI + its overlays.

    ``boxes`` are FRAME-space (already crop-mapped and clamped by
    ``boxes.jsonl``); ``width``/``height`` are the frame's pixel size and the
    denominator of every %-positioned overlay.
    """

    index: int
    src: str  # data:<mime>;base64,…
    t_ns: int  # shared RecordingClock stamp — the step↔frame alignment axis
    width: int
    height: int
    boxes: list[dict[str, Any]] = field(default_factory=list)

    @property
    def has_geometry(self) -> bool:
        return self.width > 0 and self.height > 0


@dataclass
class RecordingData:
    """Everything the player half needs, loaded from an artifact dir."""

    frames: list[FrameView] = field(default_factory=list)
    fps: float = 0.0
    target: dict[str, Any] = field(default_factory=dict)
    dropped: int = 0
    truncated: int = 0  # frames beyond _MAX_FRAMES (stated, never hidden)

    @property
    def box_count(self) -> int:
        return sum(len(f.boxes) for f in self.frames)

    @classmethod
    def from_artifact(
        cls, artifact: RecordingArtifact, *, max_frames: int = _MAX_FRAMES
    ) -> RecordingData:
        """Read manifest + frames + boxes.jsonl into render-ready form.

        Missing boxes.jsonl yields overlays-less frames (the ``view``
        command rebuilds boxes first — this loader never writes).
        """
        boxes_by_index: dict[int, list[dict[str, Any]]] = {}
        for line in artifact.read_boxes():
            if isinstance(line, dict) and isinstance(line.get("boxes"), list):
                boxes_by_index[int(line.get("index", -1))] = list(line["boxes"])
        frames: list[FrameView] = []
        total = len(artifact.frames)
        for info in artifact.frames[:max_frames]:
            try:
                payload = artifact.frame_path(info).read_bytes()
            except OSError:
                continue  # a lost frame skips; the index around it survives
            src = f"data:{info.mime};base64,{base64.b64encode(payload).decode('ascii')}"
            frames.append(
                FrameView(
                    index=info.index,
                    src=src,
                    t_ns=info.t_ns,
                    width=info.width,
                    height=info.height,
                    boxes=boxes_by_index.get(info.index, []),
                )
            )
        manifest = artifact.manifest
        target = manifest.get("target") or {"kind": "full"}
        return cls(
            frames=frames,
            fps=float(manifest.get("fps", 0.0) or 0.0),
            target=dict(target),
            dropped=int(manifest.get("dropped_frames", 0) or 0),
            truncated=max(0, total - max_frames),
        )


def _step_frame_map(steps: list[FlowStep], rec: RecordingData) -> list[int]:
    """Precompute step → frame index on the SHARED clock.

    Nearest frame at/before the event's ``_t_ns`` (a subtraction on one
    axis); steps without a stamp (recorded outside a session) hold the
    previous mapping, and a step after the last frame lands on the last
    frame. When NO step carries a stamp (mixed/legacy stream), indices
    distribute proportionally so the scrubber still tracks the timeline.
    """
    if not rec.frames:
        return []
    last = len(rec.frames) - 1
    stamped = [s for s in steps if isinstance((s.data or {}).get("_t_ns"), (int, float))]
    out: list[int] = []
    current = 0
    for i, step in enumerate(steps):
        t = (step.data or {}).get("_t_ns")
        if isinstance(t, (int, float)):
            want = int(t)
            for fi, fr in enumerate(rec.frames):
                if fr.t_ns <= want:
                    current = fi
                else:
                    break
            current = min(current, last)
        elif not stamped:
            current = min(last, int(i * len(rec.frames) / max(1, len(steps))))
        out.append(current)
    return out


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
                f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(data[key]))}</td></tr>"
            )
    if "rect" in data and isinstance(data["rect"], dict):
        r = data["rect"]
        rows.append(
            "<tr><th>rect</th><td>"
            + html.escape(", ".join(f"{k}={r.get(k, '?')}" for k in ("x", "y", "width", "height")))
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
    # Everything not surfaced above, as a JSON fallback row — EXCEPT
    # screenshot pixels (rendered as an <img>, never as base64 text) and
    # "_"-prefixed plumbing keys such as ``_t_ns`` (the viewer's own
    # alignment stamp; noise to a human reading the step detail).
    shown = {"best", "tag", "text", "rect", "candidates"}
    rest = {
        k: v
        for k, v in data.items()
        if k not in shown and k != "screenshot" and not str(k).startswith("_")
    }
    if rest:
        payload = json.dumps(rest, default=str, ensure_ascii=False)
        payload = payload.replace("</", "<\\/")  # never let data close a script/tag
        rows.append(f"<tr><th>data</th><td><code>{html.escape(payload)}</code></td></tr>")
    return "".join(rows)


def _detail_panel(index: int, step: FlowStep) -> str:
    """One pre-rendered detail card (hidden unless selected)."""
    status = (
        '<span class="ok">passed</span>' if step.success else '<span class="fail">failed</span>'
    )
    err = f'<p class="error">{html.escape(step.error)}</p>' if step.error else ""
    dur = f'<span class="dur">{step.duration_ms:.0f} ms</span>' if step.duration_ms else ""
    kind = " \u2014 picked element" if step.is_pick else ""
    rows = _data_rows(step)
    # Per-step screenshot (the trace-viewer hallmark). Bytes from a live
    # recorder; a base64 string after a JSON round-trip. Inline data URI —
    # the page must stay one self-contained file:// document.
    raw = step.data.get("screenshot")
    if isinstance(raw, (bytes, bytearray)) and raw:
        src = base64.b64encode(bytes(raw)).decode("ascii")
    elif isinstance(raw, str) and raw:
        src = raw
    else:
        src = ""
    img = (
        f'<img class="shot" alt="page state at step {step.seq}" src="data:image/png;base64,{src}">'
        if src
        else ""
    )
    return f"""
    <section class="detail" id="d{index}"{" hidden" if index else ""}>
      <h2>Step {step.seq}: {html.escape(step.type)}{kind}</h2>
      <p class="instr">{
        html.escape(
            f"\u2192 {step.label}"
            if not step.is_pick
            else f"\u2192 pick: {step.data.get('best') or step.label}"
        )
    }</p>
      <p class="meta">{status}{dur}</p>
      {err}
      {img}
      {f'<table class="data">{rows}</table>' if rows else ""}
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
            f"onkeydown=\"if(event.key==='Enter'||event.key===' ')"
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


def _box_divs(fr: FrameView) -> str:
    """Pre-rendered %-positioned overlays for one frame (hidden unless active).

    Geometry is frame-space already (boxes.jsonl contract), so percentages
    are ``x/width`` — clamped here a second time because a viewer that
    draws outside its frame lies about the recording.
    """
    if not fr.has_geometry:
        return ""
    out: list[str] = []
    for box in fr.boxes:
        try:
            x, y = float(box.get("x", 0.0)), float(box.get("y", 0.0))
            w, h = float(box.get("width", 0.0)), float(box.get("height", 0.0))
        except (TypeError, ValueError):
            continue
        if w <= 0 or h <= 0:
            continue
        left = 100.0 * max(0.0, min(x, fr.width)) / fr.width
        top = 100.0 * max(0.0, min(y, fr.height)) / fr.height
        bw = 100.0 * max(0.0, min(x + w, fr.width) - max(x, 0.0)) / fr.width
        bh = 100.0 * max(0.0, min(y + h, fr.height) - max(y, 0.0)) / fr.height
        if bw <= 0 or bh <= 0:
            continue
        label = html.escape(str(box.get("label", ""))[:40])
        out.append(
            f'<div class="box" style="left:{left:.2f}%;top:{top:.2f}%;'
            f'width:{bw:.2f}%;height:{bh:.2f}%" role="img" '
            f'aria-label="{label}"><span>{label}</span></div>'
        )
    return "".join(out)


def _player_html(rec: RecordingData, step_frames: list[int]) -> str:
    """The player section: one frame <img>, hidden per-frame overlay layers,
    a scrubber range input and the box-visibility toggle.

    Frame pixels ship once as a JS data-URI array (the <img> src swaps on
    scrub); overlays are pre-rendered in the DOM so toggling them is a
    ``hidden`` flip, not a layout pass.
    """
    if not rec.frames:
        return ""
    first = rec.frames[0]
    overlays: list[str] = []
    box_frames = 0
    for pos, fr in enumerate(rec.frames):
        inner = _box_divs(fr)
        if inner:
            box_frames += 1
        overlays.append(
            f'<div class="ovl" id="ovl{pos}"{"" if pos == 0 else " hidden"}>{inner}</div>'
        )
    frames_json = json.dumps([f.src for f in rec.frames])
    sf_json = json.dumps(step_frames)
    fps = f"{rec.fps:g}" if rec.fps else "?"
    meta_bits = [f"frame 1/{len(rec.frames)}", f"{fps} fps"]
    if box_frames:
        meta_bits.append(f"{_count(rec.box_count, 'box')}")
    if rec.dropped:
        meta_bits.append(f"{rec.dropped} dropped")
    note = (
        f'<p class="empty">showing first {len(rec.frames)} of '
        f"{len(rec.frames) + rec.truncated} frames</p>"
        if rec.truncated
        else ""
    )
    return f"""<section class="player" id="player" aria-label="Recording player">
  <div class="stage" id="stage">
    <img id="frame" alt="recorded frame 1" src="{first.src}">
    {"".join(overlays)}
  </div>
  <div class="controls">
    <label class="tgl" for="boxtoggle"><input type="checkbox" id="boxtoggle"
      checked> box overlays</label>
    <input type="range" id="scrub" min="0" max="{len(rec.frames) - 1}" value="0"
      aria-label="Scrub recorded frames" oninput="showFrame(+this.value)">
    <span class="fmeta" id="fmeta">{html.escape(" \u00b7 ".join(meta_bits))}</span>
  </div>
  {note}
  <script>
  const FRAMES = {frames_json};
  const SF = {sf_json};
  function showFrame(i) {{
    if (!FRAMES.length) return;
    i = Math.max(0, Math.min(FRAMES.length - 1, i | 0));
    const img = document.getElementById('frame');
    if (img) {{ img.src = FRAMES[i]; img.alt = 'recorded frame ' + (i + 1); }}
    document.querySelectorAll('.ovl').forEach(o => {{ o.hidden = true; }});
    const ov = document.getElementById('ovl' + i);
    if (ov) ov.hidden = false;
    const sc = document.getElementById('scrub');
    if (sc && +sc.value !== i) sc.value = i;
    const fm = document.getElementById('fmeta');
    if (fm) {{
      fm.textContent = fm.textContent.replace(/frame \\d+\\/\\d+/,
        'frame ' + (i + 1) + '/' + FRAMES.length);
    }}
  }}
  const _bt = document.getElementById('boxtoggle');
  if (_bt) _bt.addEventListener('change', () => {{
    document.getElementById('player').classList.toggle('noboxes', !_bt.checked);
  }});
  </script>
</section>"""


def render_html(graph: FlowGraph, recording: RecordingData | None = None) -> str:
    """Render the flow as one standalone trace-viewer HTML page.

    ``recording`` (a :class:`RecordingData` loaded from an artifact dir)
    adds the player half: frames, scrubber and box overlays.
    """
    steps = graph.steps[:_MAX_STEPS]
    # Build the map from the SAME capped steps the list renders: otherwise a
    # >500-step flow emits map nodes whose selectStep() has no detail panel.
    view = type(graph)(steps=list(steps), session_name=graph.session_name)
    title = html.escape(graph.session_name or "avion flow")
    rec = recording if recording is not None and recording.frames else None
    step_frames = _step_frame_map(steps, rec) if rec else []
    player = _player_html(rec, step_frames) if rec else ""

    list_items = "".join(
        f'<li id="li{i}" onclick="selectStep({i})" '
        f"onkeydown=\"if(event.key==='Enter'||event.key===' ')"
        f'{{event.preventDefault();selectStep({i})}}" tabindex="0"'
        f' class="{"" if s.success else "bad"}">'
        f'<span class="seq">{s.seq}</span> '
        f'<span class="type">{html.escape(s.type)}</span> '
        f"<span>{_label(s)}</span></li>"
        for i, s in enumerate(steps)
    )
    details = "".join(_detail_panel(i, s) for i, s in enumerate(steps))
    counts = f"{_count(len(graph.steps), 'step')} \u00b7 {_count(len(graph.picks), 'pick')}" + (
        f" \u00b7 {len(graph.failed)} failed" if graph.failed else ""
    )
    if rec:
        counts += f" \u00b7 {_count(len(rec.frames), 'frame')}" + (
            f" \u00b7 {_count(rec.box_count, 'box')}" if rec.box_count else ""
        )
    trunc = (
        f'<p class="empty">showing first {_MAX_STEPS} of {len(graph.steps)} steps</p>'
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
  main.with-player {{ grid-template-rows: auto minmax(0, 1fr); }}
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
  .detail img.shot {{ max-width: 100%; border: 1px solid #e5e7eb;
         border-radius: 6px; margin-top: 8px; }}
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
  .player {{ grid-column: 1 / -1; background: #111827; color: #f9fafb;
             padding: 10px 14px; border-bottom: 1px solid #374151; }}
  .player .stage {{ position: relative; display: inline-block; max-width: 100%;
             line-height: 0; }}
  .player .stage img {{ max-height: 40vh; max-width: 100%; width: auto;
             border: 1px solid #374151; border-radius: 6px; background: #fff; }}
  .player .ovl {{ position: absolute; inset: 0; }}
  .player.noboxes .ovl .box {{ display: none; }}
  .player .box {{ position: absolute; border: 2px solid #a78bfa;
             background: rgba(167, 139, 250, .18); border-radius: 3px;
             box-sizing: border-box; }}
  .player .box span {{ position: absolute; top: -16px; left: -2px;
             font: 10px/1.4 ui-monospace, monospace; color: #111827;
             background: #a78bfa; padding: 0 4px; border-radius: 3px 3px 0 0;
             white-space: nowrap; max-width: 160px; overflow: hidden;
             text-overflow: ellipsis; }}
  .player .controls {{ display: flex; gap: 14px; align-items: center;
             margin-top: 8px; font-size: 12.5px; }}
  .player .controls input[type=range] {{ flex: 1; accent-color: #a78bfa; }}
  .player .tgl {{ display: inline-flex; gap: 6px; align-items: center;
             cursor: pointer; white-space: nowrap; }}
  .player .fmeta {{ opacity: .75; font-family: ui-monospace, monospace;
             white-space: nowrap; }}
  .player .empty {{ color: #9ca3af; }}
</style>
<header><h1>{title}</h1><span class="counts">{counts}</span></header>
<main{" class=with-player" if rec else ""}>
  {player}
  <aside class="timeline"><ul>{list_items}</ul>{empty}{trunc}</aside>
  <div class="map">{_map_svg(view) if steps else ""}</div>
  <section class="details">{details}
    <p class="empty"{" hidden" if steps else ""}>Nothing selected.</p>
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
  if (typeof showFrame === 'function' && typeof SF !== 'undefined' && i < SF.length)
    showFrame(SF[i]);
}}
if (document.getElementById('d0')) selectStep(0);
</script>
</html>
"""


def save_html(graph: FlowGraph, path: str, recording: RecordingData | None = None) -> Path:
    """Write the viewer page to ``path`` (parent dirs created)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(graph, recording=recording), encoding="utf-8")
    return out
