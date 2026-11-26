"""avion.recording — record a browser session into a viewer-ready artifact.

(No FEATURE: tag on purpose — this package is part of the avion CLI, whose
files carry no feature-flag markers, and registering flags lives outside
this package's ownership.)

One recording run produces::

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
from avion.recording.compare import (
    COMPARISON_NAME,
    KEEP_RUNS,
    RUN_META,
    RunComparison,
    compare_runs,
    diff_shot,
    diff_steps,
    git_sha,
    list_runs,
    new_run_id,
    prune_runs,
    render_summary,
    runs_root,
    shot_name,
    slugify,
)
from avion.recording.export import _ENCODER_MISSING, export_mp4, export_summary
from avion.recording.sampler import MAX_FPS, MIN_FPS, FrameSampler, clamp_fps
from avion.recording.session import RecordingSession, build_locator, parse_rect

__all__ = [
    "BOXES_NAME",
    "COMPARISON_NAME",
    "EVENTS_NAME",
    "FRAME_DIR",
    "KEEP_RUNS",
    "MANIFEST_NAME",
    "MAX_FPS",
    "MIN_FPS",
    "RUN_META",
    "FrameInfo",
    "FrameSampler",
    "RecordingArtifact",
    "RecordingClock",
    "RecordingSession",
    "RunComparison",
    "build_locator",
    "clamp_box",
    "clamp_fps",
    "compare_runs",
    "derive_boxes",
    "diff_shot",
    "diff_steps",
    "dom_sweep",
    "export_mp4",
    "export_summary",
    "frame_capabilities",
    "git_sha",
    "list_runs",
    "new_recording_id",
    "new_run_id",
    "parse_rect",
    "png_size",
    "prepare_frame",
    "prune_runs",
    "rebuild_boxes",
    "render_summary",
    "resolve_rect",
    "runs_root",
    "shot_name",
    "slugify",
    "to_frame_box",
    "vision_boxes",
    "_ENCODER_MISSING",
]
