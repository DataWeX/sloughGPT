"""Recording artifact — the on-disk layout a recording session produces.

    recording/<id>/
      manifest.json    id, target, clock, capabilities, frame index, DOM sweep
      events.json      EventRecorder.save() output (also a valid avion run flow)
      boxes.jsonl      per-frame overlay geometry — derived, never burned in
      frames/*.jpg     sampled frames (JPEG via Pillow; PNG passthrough when
                       the encoder/crop tier is unavailable)

WHY boxes.jsonl IS SEPARATE: the recorded pixels stay clean and the labels
stay toggleable and re-detectable. The viewer draws overlays from
boxes.jsonl AT RENDER TIME; re-running ``avion stop`` regenerates the file
from manifest + events without touching a single frame.

Stdlib-only at import time — Pillow (if present at all) is probed and used
lazily inside functions, so importing avion.recording never pulls a heavy
dependency (the Capability probe pattern from avion.vision.detector).
"""

from __future__ import annotations

import io
import json
import secrets
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FRAME_DIR = "frames"
MANIFEST_NAME = "manifest.json"
EVENTS_NAME = "events.json"
BOXES_NAME = "boxes.jsonl"
MANIFEST_VERSION = 1
_DEFAULT_QUALITY = 85  # JPEG quality — 2–5 fps at ~800x600 stays ~30-80KB/frame


def new_recording_id() -> str:
    """Sortable-unique id: wall time plus a short random tail."""
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"


def frame_capabilities() -> dict[str, Any]:
    """Probe (never import at module level) the frame-storage tiers.

    jpeg and crop both need Pillow — when it's missing the manifest says so
    and prepare_frame degrades to PNG passthrough / full frames.
    """
    from avion.vision.detector import _capability  # reuse the probe pattern

    jpeg = _capability("jpeg", "PIL")
    crop = _capability("crop", "PIL")
    return {
        "jpeg": jpeg.to_dict(),
        "crop": crop.to_dict(),
    }


def _pil():
    """Pillow's Image module, or None when not installed (never raises)."""
    try:
        from PIL import Image  # type: ignore[import]

        return Image
    except ImportError:
        return None


def png_size(data: bytes) -> tuple[int, int]:
    """(width, height) from a PNG IHDR header — stdlib, no decoder needed."""
    if len(data) >= 24 and data[12:16] == b"IHDR":
        w, h = struct.unpack(">II", data[16:24])
        return int(w), int(h)
    return 0, 0


def prepare_frame(
    raw: bytes,
    crop: dict[str, float] | None = None,
    *,
    quality: int = _DEFAULT_QUALITY,
) -> tuple[bytes, str, int, int]:
    """Turn one captured frame into storable bytes: (data, ext, width, height).

    * Pillow present → JPEG ``frames/*.jpg`` (the artifact contract); a
      ``crop`` rect crops the frame to the capture target first.
    * Pillow missing → degrade, never crash: raw bytes pass through with
      their true extension (PNG/JPEG sniffed from magic), and a crop target
      falls back to the full frame — recorded dishonesty-free, the missing
      capability is reported in the manifest.
    """
    Image = _pil()
    if Image is None:
        if raw[:8] == b"\x89PNG\r\n\x1a\n":
            w, h = png_size(raw)
            return raw, ".png", w, h
        if raw[:2] == b"\xff\xd8":
            return raw, ".jpg", 0, 0
        return raw, ".bin", 0, 0

    img = Image.open(io.BytesIO(raw))
    if crop:
        left, top = int(round(crop.get("x", 0.0))), int(round(crop.get("y", 0.0)))
        width = max(1, int(round(crop.get("width", 1.0))))
        height = max(1, int(round(crop.get("height", 1.0))))
        img = img.crop((left, top, left + width, top + height))
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    w, h = img.size
    return buf.getvalue(), ".jpg", int(w), int(h)


@dataclass
class FrameInfo:
    """One sampled frame's index entry (also what manifest.json stores)."""

    index: int
    file: str  # relative to the recording dir, e.g. frames/000003.jpg
    t_ns: int  # shared monotonic clock (RecordingClock) — alignment axis
    t_wall: float  # wall clock — display only, see clock desync note
    width: int = 0
    height: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "file": self.file,
            "t_ns": self.t_ns,
            "t_wall": self.t_wall,
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FrameInfo:
        return cls(
            index=int(d.get("index", 0)),
            file=str(d.get("file", "")),
            t_ns=int(d.get("t_ns", 0)),
            t_wall=float(d.get("t_wall", 0.0)),
            width=int(d.get("width", 0)),
            height=int(d.get("height", 0)),
        )

    @property
    def mime(self) -> str:
        if self.file.endswith(".jpg") or self.file.endswith(".jpeg"):
            return "image/jpeg"
        if self.file.endswith(".png"):
            return "image/png"
        return "application/octet-stream"


def _atomic_write(path: Path, payload: str) -> None:
    """tmp + replace — a reader never sees a half-written artifact file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)


@dataclass
class RecordingArtifact:
    """Create/read one recording directory; sole owner of its layout."""

    root: Path
    manifest: dict[str, Any] = field(default_factory=dict)
    frames: list[FrameInfo] = field(default_factory=list)

    # ── creation ───────────────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        root: Path | str,
        *,
        recording_id: str,
        fps: float,
        target: dict[str, Any],
        clock: Any,
        sweep: list[dict[str, Any]] | None = None,
        sweep_t_ns: int = 0,
        capabilities: dict[str, Any] | None = None,
    ) -> RecordingArtifact:
        root = Path(root)
        (root / FRAME_DIR).mkdir(parents=True, exist_ok=True)
        manifest: dict[str, Any] = {
            "id": recording_id,
            "version": MANIFEST_VERSION,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "fps": float(fps),
            "target": dict(target),
            "clock": {
                "source": "time.monotonic_ns",
                "started_wall": getattr(clock, "started_wall", time.time()),
                "note": clock.note() if hasattr(clock, "note") else "",
            },
            "capabilities": dict(capabilities or {}),
            "sweep": list(sweep or []),
            "sweep_t_ns": int(sweep_t_ns),
            "frames": [],
            "frame_count": 0,
            "event_count": 0,
            "box_count": 0,
            "dropped_frames": 0,
            "completed": False,
        }
        art = cls(root=root, manifest=manifest)
        art.flush_manifest()
        return art

    # ── writing (record time) ──────────────────────────────────────────

    @property
    def frames_dir(self) -> Path:
        return self.root / FRAME_DIR

    @property
    def manifest_path(self) -> Path:
        return self.root / MANIFEST_NAME

    @property
    def events_path(self) -> Path:
        return self.root / EVENTS_NAME

    @property
    def boxes_path(self) -> Path:
        return self.root / BOXES_NAME

    @property
    def target(self) -> dict[str, Any]:
        return self.manifest.setdefault("target", {"kind": "full"})

    @property
    def crop(self) -> dict[str, float] | None:
        crop = self.target.get("crop")
        return dict(crop) if isinstance(crop, dict) and crop else None

    def write_frame(self, raw: bytes, t_ns: int, t_wall: float) -> FrameInfo:
        """Encode/crop one captured frame and append its index entry.

        The manifest is flushed here too (atomic, ≤5 fps by the sampler
        contract): the frame index must survive the record process dying —
        ``avion stop`` recovers by loading the ON-DISK manifest, so an index
        that only exists in memory would orphan every frame a killed run
        already wrote.
        """
        data, ext, w, h = prepare_frame(raw, self.crop)
        index = len(self.frames)
        rel = f"{FRAME_DIR}/{index:06d}{ext}"
        (self.root / rel).write_bytes(data)
        info = FrameInfo(
            index=index, file=rel, t_ns=int(t_ns), t_wall=float(t_wall), width=w, height=h
        )
        self.frames.append(info)
        self.manifest["frames"].append(info.to_dict())
        self.manifest["frame_count"] = len(self.frames)
        self.flush_manifest()
        return info

    def write_events(self, payload: dict[str, Any]) -> None:
        """Persist the recorder view (rewritten incrementally while recording)."""
        _atomic_write(self.events_path, json.dumps(payload, indent=2, default=str))

    def write_boxes(self, lines: list[dict[str, Any]]) -> None:
        """Replace boxes.jsonl — derived data, safe to regenerate any time."""
        body = "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines)
        _atomic_write(self.boxes_path, body)

    def flush_manifest(self) -> None:
        _atomic_write(self.manifest_path, json.dumps(self.manifest, indent=2, default=str))

    def finalize(
        self,
        *,
        event_count: int,
        box_count: int,
        dropped_frames: int = 0,
        vision_ocr: dict[str, Any] | None = None,
        completed: bool = True,
    ) -> dict[str, Any]:
        """Stamp the closing numbers into the manifest (used by record AND stop)."""
        self.manifest["frame_count"] = len(self.frames)
        self.manifest["event_count"] = int(event_count)
        self.manifest["box_count"] = int(box_count)
        self.manifest["dropped_frames"] = int(dropped_frames)
        self.manifest["completed"] = bool(completed)
        if vision_ocr is not None:
            self.manifest.setdefault("capabilities", {})["vision_ocr"] = vision_ocr
        self.flush_manifest()
        return dict(self.manifest)

    # ── reading (stop/view time) ───────────────────────────────────────

    @classmethod
    def load(cls, root: Path | str) -> RecordingArtifact:
        """Open an existing recording; raises FileNotFoundError with a hint."""
        root = Path(root)
        path = root / MANIFEST_NAME
        if not path.exists():
            raise FileNotFoundError(
                f"no {MANIFEST_NAME} in {root} — not a recording directory "
                "(make one with `avion record`)"
            )
        manifest = json.loads(path.read_text(encoding="utf-8"))
        frames = [FrameInfo.from_dict(d) for d in manifest.get("frames", [])]
        return cls(root=root, manifest=manifest, frames=frames)

    def read_events(self) -> dict[str, Any]:
        """events.json payload; a recording whose process died before the first
        command simply has none yet — empty flow, not an error."""
        if not self.events_path.exists():
            return {"session": "", "events": []}
        try:
            data = json.loads(self.events_path.read_text(encoding="utf-8"))
        except ValueError:
            return {"session": "", "events": []}
        if not isinstance(data, dict):
            return {"session": "", "events": []}
        data.setdefault("events", [])
        return data

    def read_boxes(self) -> list[dict[str, Any]]:
        """boxes.jsonl lines; tolerates a missing/torn file (returns [])."""
        if not self.boxes_path.exists():
            return []
        out: list[dict[str, Any]] = []
        for line in self.boxes_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except ValueError:
                continue  # torn tail — same tolerance as the journal
            if isinstance(parsed, dict):
                out.append(parsed)
        return out

    def frame_path(self, info: FrameInfo) -> Path:
        return self.root / info.file
