"""avion.recording — record a browser session into a viewer-ready artifact.

FEATURE: recording-viewer — one recording run produces::

    <out>/
      manifest.json   target, clock, capabilities, sweep, frame index
      events.json     EventRecorder output (+ _t_ns shared-clock stamps)
      boxes.jsonl     derived overlay geometry — one line per boxed frame
      frames/*.jpg    sampled at 2–5 fps (JPEG via Pillow, PNG fallback)

Design rules that live in this package (do not dilute them):

* **ONE clock** — ``RecordingClock`` stamps frames and event ``_t_ns``;
  ``Event.timestamp`` stays wall-clock for humans and may step.
* **Boxes derived, never burned** — pixels stay clean; the viewer/export
  draw overlays from ``boxes.jsonl`` at render time; ``rebuild_boxes()``
  regenerates it from manifest + events without touching a frame.
* **Structure first** — DOM sweep + event rects; OCR/vision tier-2 behind
  the ``Capability`` probe (degrades to a stated reason, never a crash).
* **Sync core, explicit seam** — the sampler is a plain thread; the CLI
  bridges screenshots onto the loop with ``run_coroutine_threadsafe``.
* **Stdlib-only at import** — Pillow/pytesseract/imageio-ffmpeg are probed
  lazily inside functions (same rule as avion.vision).
"""

from __future__ import annotations

from avion.recording.artifact import (
    BOXES_NAME,
    EVENTS_NAME,
    FRAME_DIR,
    MANIFEST_NAME,
    FrameInfo,
    RecordingArtifact,
    frame_capabilities,
    new_recording_id,
    png_size,
    prepare_frame,
)
from avion.recording.boxes import (
    clamp_box,
    derive_boxes,
    dom_sweep,
    rebuild_boxes,
    resolve_rect,
    to_frame_box,
    vision_boxes,
)
from avion.recording.clock import RecordingClock
from avion.recording.export import _ENCODER_MISSING, export_mp4, export_summary
from avion.recording.sampler import MAX_FPS, MIN_FPS, FrameSampler, clamp_fps
from avion.recording.session import RecordingSession, build_locator, parse_rect

__all__ = [
    "BOXES_NAME",
    "EVENTS_NAME",
    "FRAME_DIR",
    "MANIFEST_NAME",
    "MAX_FPS",
    "MIN_FPS",
    "FrameInfo",
    "FrameSampler",
    "RecordingArtifact",
    "RecordingClock",
    "RecordingSession",
    "build_locator",
    "clamp_box",
    "clamp_fps",
    "derive_boxes",
    "dom_sweep",
    "export_mp4",
    "export_summary",
    "frame_capabilities",
    "new_recording_id",
    "parse_rect",
    "png_size",
    "prepare_frame",
    "rebuild_boxes",
    "resolve_rect",
    "to_frame_box",
    "vision_boxes",
    "_ENCODER_MISSING",
]
