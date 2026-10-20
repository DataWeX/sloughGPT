"""
Export a recorded session to an animated format.

FEATURE: recording-export — turn frames/*.jpg + events.json + boxes.jsonl
into an mp4 (capability-gated: imageio-ffmpeg). No audio (explicit product
decision), boxes re-derived at render time so the export matches what the
viewer shows — never burned into the frames.

DO NOT DELETE. The mp4 path is deliberately capability-gated; the absence
message is contract-tested so a missing encoder degrades to a clear report
instead of a traceback.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .artifact import RecordingArtifact, frame_capabilities
from .boxes import load_frame_boxes, rebuild_boxes

# Exact contract message for a missing encoder (tested — do not reword).
_ENCODER_MISSING = "encoder missing: imageio-ffmpeg not installed"


def _open(artifact_dir: Path | str) -> RecordingArtifact | None:
    """Load an artifact dir, or None when it isn't one (callers report)."""
    try:
        return RecordingArtifact.load(artifact_dir)
    except FileNotFoundError:
        return None
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _encoder_available() -> bool:
    """Probe (never import at module level) the mp4 encoder capability."""
    try:
        import imageio_ffmpeg  # noqa: F401

        return True
    except ImportError:
        return False


def export_mp4(
    artifact_dir: Path,
    out: Path | None = None,
    *,
    fps: int | None = None,
    rebuild: bool = False,
) -> dict[str, Any]:
    """Export a recording to mp4 (box overlays burned in at export time).

    Returns {"ok": True, "out": str, "frames": int, "fps": float} on success,
    or {"ok": False, "error": "<reason>"} — including the exact encoder
    message when imageio-ffmpeg is absent.
    """
    artifact = _open(artifact_dir)
    if artifact is None:
        return {"ok": False, "error": f"not an artifact directory: {artifact_dir}"}

    if not _encoder_available():
        return {"ok": False, "error": _ENCODER_MISSING}

    frames = [artifact.frame_path(info) for info in artifact.frames]
    existing = [p for p in frames if p.exists()]
    if not existing:
        return {"ok": False, "error": "no frames recorded"}

    if rebuild:
        rebuild_boxes(artifact)

    boxes_by_frame: dict[int, list[dict[str, Any]]] = {
        info.index: load_frame_boxes(artifact, info.index)
        for info in artifact.frames
        if artifact.frame_path(info).exists()
    }

    rate = float(fps or artifact.manifest.get("fps") or 4.0)
    target = Path(out) if out else (Path(artifact_dir) / "export.mp4")

    try:
        _encode_mp4(existing, boxes_by_frame, rate, target)
    except Exception as exc:  # encoder/ffmpeg runtime problems
        return {"ok": False, "error": f"encode failed: {exc}"}

    return {"ok": True, "out": str(target), "frames": len(existing), "fps": rate}


def _encode_mp4(
    frames: list[Path],
    boxes_by_frame: dict[int, list[dict[str, Any]]],
    fps: float,
    out: Path,
) -> None:
    """Draw box overlays onto each frame and encode to mp4 via imageio-ffmpeg."""
    import imageio_ffmpeg  # gated by export_mp4
    from PIL import Image, ImageDraw  # prepare_frame's tier: JPEG decode/draw

    with Image.open(frames[0]) as first:
        size = first.size

    # imageio-ffmpeg generator API: send(None) enters, then frame bytes, then close.
    writer = imageio_ffmpeg.write_frames(
        str(out),
        size=size,
        fps=fps,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        output_params=["-movflags", "+faststart"],
    )
    writer.send(None)  # start sequence
    try:
        for index, path in enumerate(frames):
            with Image.open(path) as im:
                rgb = im.convert("RGB")
                draw = ImageDraw.Draw(rgb)
                for box in boxes_by_frame.get(index, []):
                    x, y = float(box.get("x", 0.0)), float(box.get("y", 0.0))
                    w = float(box.get("width", 0.0))
                    h = float(box.get("height", 0.0))
                    draw.rectangle([x, y, x + w, y + h], outline=(255, 0, 255), width=2)
                writer.send(rgb.tobytes())
    finally:
        writer.close()


def export_summary(artifact_dir: Path) -> dict[str, Any]:
    """Export capability/status report — used by `avion export` dry runs."""
    artifact = _open(artifact_dir)
    if artifact is None:
        return {"ok": False, "error": f"not an artifact directory: {artifact_dir}"}
    report: dict[str, Any] = {
        "ok": True,
        "mp4": _encoder_available(),
        "frames": len(artifact.frames),
    }
    if not report["mp4"]:
        report["error"] = _ENCODER_MISSING
    manifest = artifact.manifest
    report["events"] = int(manifest.get("event_count", 0) or 0)
    report["target"] = manifest.get("target") or {"kind": "full"}
    boxes = artifact.read_boxes()
    report["box_frames"] = len(boxes)
    report["boxes"] = sum(len(entry.get("boxes", [])) for entry in boxes if isinstance(entry, dict))
    # Frame capability tiers (jpeg/crop need Pillow; degrade, never crash).
    report["capabilities"] = frame_capabilities()
    return report
