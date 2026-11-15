"""Box derivation — STRUCTURE first, vision tier-2 behind a capability probe.

Trust order for overlay geometry (what the viewer draws over a frame):

1. **dom-sweep** — one ``evaluate`` at record start collects interactive
   elements' ``getBoundingClientRect`` rects with their accessible labels
   (role/name/testid) from the live DOM: CDP structure, no vision at all.
2. **event rects** — rects resolved WHILE recording: picker SELECT rects
   (already in event data, source f7d5b59cd) and locator rects resolved
   through ``resolve_rect`` (element bounding_box when a backend populates
   it, in-page ``getBoundingClientRect`` for CSS selectors otherwise).
3. **vision/ocr** — tier-2 only: ``ImageAnalyzer().capabilities().ocr``.
   pytesseract/cv2 are NOT installed here, so this degrades to a stated
   capability reason in the manifest — it must never crash a recording.

Every box is PAGE-space until ``to_frame_box`` maps it into the capture
target's crop and clamps it inside frame bounds — boxes.jsonl may only
contain geometry the viewer can actually draw over the frame it names.

Stdlib + avion imports only; Pillow/pytesseract are touched lazily inside
functions (never at import time).
"""

from __future__ import annotations

import json
from typing import Any

from avion.recording.artifact import RecordingArtifact

# Collects the page's actionable elements once at record start. Returns
# [{x, y, width, height, label}] in viewport coordinates (the same space
# Page.captureScreenshot screenshots live in). Self-contained, no deps.
SWEEP_JS = r"""
(() => {
  const out = [];
  const seen = new Set();
  const nodes = document.querySelectorAll(
    'a,button,input,select,textarea,[role=button],[role=link],[role=tab],[onclick],[data-testid]'
  );
  const labelOf = (el) => {
    const direct = el.getAttribute('aria-label') || el.getAttribute('data-testid') ||
      (el.innerText || '').trim() || el.getAttribute('name') ||
      el.getAttribute('placeholder') || '';
    return String(direct).replace(/\s+/g, ' ').trim().slice(0, 40);
  };
  for (const el of nodes) {
    if (seen.has(el)) continue;
    seen.add(el);
    const r = el.getBoundingClientRect();
    if (r.width < 4 || r.height < 4) continue;
    out.push({
      x: r.x, y: r.y, width: r.width, height: r.height,
      label: labelOf(el) || el.tagName.toLowerCase(),
    });
    if (out.length >= 100) break;
  }
  return out;
})()
"""

_MAX_SWEEP = 100


async def dom_sweep(backend: Any) -> list[dict[str, Any]]:
    """Structure boxes from the live DOM. Best-effort: a backend without
    evaluate, a rejected context, or a hostile page returns [] — the
    recording continues without sweep boxes, never with an exception."""
    try:
        raw = await backend.evaluate(SWEEP_JS)
    except Exception:
        return []
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw[:_MAX_SWEEP]:
        if not isinstance(item, dict):
            continue
        try:
            box = {
                "x": float(item.get("x", 0.0)),
                "y": float(item.get("y", 0.0)),
                "width": float(item.get("width", 0.0)),
                "height": float(item.get("height", 0.0)),
            }
        except (TypeError, ValueError):
            continue
        if box["width"] <= 0 or box["height"] <= 0:
            continue
        label = str(item.get("label", ""))[:40]
        out.append({**box, "label": label})
    return out


async def resolve_rect(backend: Any, locator: Any) -> dict[str, float] | None:
    """Page-space rect for a locator — the structure tier's per-event path.

    1. backend.find_element + Element.bounding_box (Playwright populates it);
    2. in-page getBoundingClientRect for each CSS selector (the CDP path —
       this is also what ``--target element`` captures once at record start,
       the picker's rect + stable selector pair from f7d5b59cd);
    3. anything unresolvable → None (degrade, never raise).
    """
    from avion.core.element import SelectorStrategy

    try:
        el = await backend.find_element(locator)
        bb = getattr(el, "bounding_box", None) if el is not None else None
        if isinstance(bb, dict) and float(bb.get("width", 0) or 0) > 0:
            return {
                "x": float(bb.get("x", 0.0)),
                "y": float(bb.get("y", 0.0)),
                "width": float(bb.get("width", 0.0)),
                "height": float(bb.get("height", 0.0)),
            }
    except Exception:
        pass

    selectors = getattr(locator, "selectors", ()) or ()
    for sel in selectors:
        if getattr(sel, "strategy", None) is not SelectorStrategy.CSS:
            continue
        js = (
            "(() => { const el = document.querySelector("
            + json.dumps(str(sel.value))
            + "); if (!el) return null; const r = el.getBoundingClientRect();"
            " return {x: r.x, y: r.y, width: r.width, height: r.height}; })()"
        )
        try:
            rect = await backend.evaluate(js)
        except Exception:
            continue
        if not isinstance(rect, dict):
            continue
        try:
            w, h = float(rect.get("width", 0.0)), float(rect.get("height", 0.0))
        except (TypeError, ValueError):
            continue
        if w > 0 and h > 0:
            return {
                "x": float(rect.get("x", 0.0)),
                "y": float(rect.get("y", 0.0)),
                "width": w,
                "height": h,
            }
    return None


def clamp_box(box: dict[str, Any], *, width: float, height: float) -> dict[str, float]:
    """Clip a box into [0, width] × [0, height]; size may shrink to 0."""
    try:
        x0, y0 = float(box.get("x", 0.0)), float(box.get("y", 0.0))
        x1, y1 = x0 + float(box.get("width", 0.0)), y0 + float(box.get("height", 0.0))
    except (TypeError, ValueError):
        return {"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0}
    cx0, cy0 = max(0.0, min(x0, width)), max(0.0, min(y0, height))
    cx1, cy1 = max(0.0, min(x1, width)), max(0.0, min(y1, height))
    return {"x": cx0, "y": cy0, "width": max(0.0, cx1 - cx0), "height": max(0.0, cy1 - cy0)}


def to_frame_box(
    box: dict[str, Any],
    *,
    crop: dict[str, float] | None,
    width: float,
    height: float,
) -> dict[str, float] | None:
    """Page-space box → frame-space box (crop offset + clamp).

    None when the frame has no known geometry or nothing of the box remains
    inside the crop — geometry the viewer could not draw is never written.
    """
    if width <= 0 or height <= 0:
        return None
    shifted = dict(box)
    if crop:
        shifted["x"] = float(box.get("x", 0.0)) - float(crop.get("x", 0.0))
        shifted["y"] = float(box.get("y", 0.0)) - float(crop.get("y", 0.0))
    clamped = clamp_box(shifted, width=width, height=height)
    if clamped["width"] < 1 or clamped["height"] < 1:
        return None
    return clamped


def vision_boxes(frame_bytes: bytes) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Tier-2 OCR boxes off one frame — capability-gated, never fatal.

    Returns (boxes, capability_status). In an environment without
    pytesseract the boxes are [] and the status carries the probe's reason
    ("missing pytesseract"), which lands in manifest.capabilities.vision_ocr.
    """
    from avion.vision.detector import ImageAnalyzer

    analyzer = ImageAnalyzer()
    caps = analyzer.capabilities()
    ocr = caps.ocr
    if not ocr.available:
        return [], ocr.to_dict()
    try:
        words = analyzer.ocr_text(frame_bytes)
        boxes: list[dict[str, Any]] = []
        for text, bb in words:
            box = to_frame_box(
                {"x": bb.x, "y": bb.y, "width": bb.width, "height": bb.height},
                crop=None,
                width=10**9,
                height=10**9,
            )
            if box is None:
                continue
            boxes.append({**box, "label": str(text)[:40], "source": "ocr"})
        return boxes[:200], ocr.to_dict()
    except Exception as exc:  # a probe that raises must not sink a recording
        return [], {"available": False, "reason": f"ocr failed: {exc}"}


def _event_t_ns(event: dict[str, Any]) -> int | None:
    value = (event.get("data") or {}).get("_t_ns")
    return int(value) if isinstance(value, (int, float)) else None


def _event_rect(event: dict[str, Any]) -> dict[str, float] | None:
    rect = (event.get("data") or {}).get("rect")
    if not isinstance(rect, dict):
        return None
    try:
        if float(rect.get("width", 0)) > 0 and float(rect.get("height", 0)) > 0:
            return {k: float(rect.get(k, 0.0)) for k in ("x", "y", "width", "height")}
    except (TypeError, ValueError):
        return None
    return None


def derive_boxes(
    artifact: RecordingArtifact,
    events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build boxes.jsonl lines: one per frame that has at least one box.

    Sources merge into per-frame lists in trust order (sweep, events, OCR);
    every box is mapped through the capture target's crop and clamped into
    its frame's bounds. Returns (lines, vision_capability_status).
    """
    manifest = artifact.manifest
    crop = artifact.crop
    frames = artifact.frames
    per_frame: dict[int, list[dict[str, Any]]] = {fi.index: [] for fi in frames}
    if not frames:
        return [], {"available": False, "reason": "no frames captured"}

    # 1. DOM sweep — page segment: from record start until the NEXT navigate
    #    (after that the swept page is gone; keeping its boxes would lie).
    sweep_t_ns = int(manifest.get("sweep_t_ns", 0) or 0)
    nav_t_ns: int | None = None
    for ev in events:
        if ev.get("type") == "navigate":
            t = _event_t_ns(ev)
            if t is not None and t > sweep_t_ns:
                nav_t_ns = t
                break
    segment = [
        fi for fi in frames if fi.t_ns >= sweep_t_ns and (nav_t_ns is None or fi.t_ns < nav_t_ns)
    ]

    def add_segment(box: dict[str, Any], label: str, source: str) -> None:
        for fi in segment:
            mapped = to_frame_box(box, crop=crop, width=fi.width, height=fi.height)
            if mapped is not None:
                per_frame[fi.index].append({**mapped, "label": label, "source": source})

    for box in manifest.get("sweep", []):
        if isinstance(box, dict):
            add_segment(box, str(box.get("label", ""))[:40] or "element", "sweep")

    # 2. Event rects — nearest frame at/before the event (same clock, so the
    #    attachment is a subtraction, not a guess).
    for ev in events:
        rect = _event_rect(ev)
        if rect is None:
            continue
        t = _event_t_ns(ev)
        idx = frames[0].index
        if t is not None:
            idx = (
                max(fi.index for fi in frames if fi.t_ns <= t)
                if any(fi.t_ns <= t for fi in frames)
                else frames[0].index
            )
        elif isinstance(ev.get("seq"), int):
            seq = int(ev["seq"])
            idx = frames[min(seq, len(frames) - 1)].index
        source = "pick" if ev.get("type") == "select" else "event"
        label = str(ev.get("name", ""))[:40] or str(ev.get("type", ""))
        mapped = to_frame_box(rect, crop=crop, width=frames[idx].width, height=frames[idx].height)
        if mapped is not None:
            per_frame[idx].append({**mapped, "label": label, "source": source})

    # 3. Vision tier-2 — off the first frame of the sweep segment; degrades
    #    to a capability reason when OCR is not installed (it is not here).
    vision_status: dict[str, Any] = {"available": False, "reason": "no frames captured"}
    if segment:
        first = segment[0]
        try:
            payload = artifact.frame_path(first).read_bytes()
        except OSError:
            payload = b""
        if payload:
            vision_boxes_list, vision_status = vision_boxes(payload)
            for box in vision_boxes_list:
                mapped = to_frame_box(box, crop=crop, width=first.width, height=first.height)
                if mapped is not None:
                    per_frame[first.index].append(
                        {**mapped, "label": box.get("label", ""), "source": "ocr"}
                    )

    lines: list[dict[str, Any]] = []
    for fi in frames:
        boxes = per_frame.get(fi.index) or []
        if not boxes:
            continue
        lines.append(
            {
                "index": fi.index,
                "file": fi.file,
                "t_ns": fi.t_ns,
                "width": fi.width,
                "height": fi.height,
                "boxes": boxes,
            }
        )
    return lines, vision_status


def load_frame_boxes(artifact: RecordingArtifact, frame_index: int) -> list[dict[str, Any]]:
    """boxes.jsonl entry for one frame index ([] when it has none).

    The export path draws per-frame; this is its indexed read (same
    tolerance as ``read_boxes``: a torn/missing line is "no boxes here").
    """
    for line in artifact.read_boxes():
        if isinstance(line, dict) and int(line.get("index", -1)) == frame_index:
            boxes = line.get("boxes")
            if isinstance(boxes, list):
                return [b for b in boxes if isinstance(b, dict)]
    return []


def rebuild_boxes(
    artifact: RecordingArtifact,
    *,
    dropped_frames: int | None = None,
    completed: bool = True,
) -> dict[str, Any]:
    """Regenerate boxes.jsonl + close the manifest (the ``avion stop`` path).

    Also what ``view`` runs when boxes.jsonl is missing — boxes are derived,
    so any reader can rebuild them from manifest + events without touching
    the recorded frames.
    """
    payload = artifact.read_events()
    events = payload.get("events", [])
    lines, vision = derive_boxes(artifact, events)
    artifact.write_boxes(lines)
    box_count = sum(len(line.get("boxes", [])) for line in lines)
    dropped = dropped_frames
    if dropped is None:
        dropped = int(artifact.manifest.get("dropped_frames", 0) or 0)
    summary = artifact.finalize(
        event_count=len(events),
        box_count=box_count,
        dropped_frames=dropped,
        vision_ocr=vision,
        completed=completed,
    )
    summary["box_lines"] = len(lines)
    return summary
