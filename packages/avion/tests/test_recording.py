"""Recording tests — one clock, derived boxes, sampler cadence, session lifecycle.

Covers avion.recording end to end:

* RecordingClock — the ONE monotonic axis frames and events align on.
* RecordingArtifact — round-trip layout, tolerant readers, frame tiers.
* box derivation — structure first (sweep/event rects), vision tier-2 degrades
  to a stated capability reason, every box clamped inside frame bounds.
* FrameSampler — 2–5 fps on its own thread, ring → disk, error isolation.
* RecordingSession — begin()/finalize() lifecycle, capture target resolved
  ONCE, teardown never blocked by an in-flight capture.
* export — capability-gated mp4 (exact missing-encoder contract message).
"""

from __future__ import annotations

import asyncio
import json
import struct
import sys
import time
import zlib
from pathlib import Path

import pytest
from avion.core.element import SelectorStrategy
from avion.events.models import EventType
from avion.events.recorder import EventRecorder
from avion.recording.artifact import RecordingArtifact, png_size, prepare_frame
from avion.recording.boxes import (
    clamp_box,
    derive_boxes,
    load_frame_boxes,
    rebuild_boxes,
    to_frame_box,
    vision_boxes,
)
from avion.recording.clock import RecordingClock
from avion.recording.export import _ENCODER_MISSING, export_mp4, export_summary
from avion.recording.sampler import FrameSampler, clamp_fps
from avion.recording.session import RecordingSession, build_locator, parse_rect

# ── helpers ─────────────────────────────────────────────────────────────────


def make_png(width: int = 8, height: int = 6) -> bytes:
    """Minimal valid RGB PNG — stdlib only (zlib + struct).

    Works with and without Pillow installed: the JPEG tier re-encodes it to
    true dimensions, the PNG-passthrough tier reads them from IHDR — both
    paths report the same width/height, so geometry assertions hold either way.
    """

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    row = b"\x00" + bytes((120, 80, 200)) * width  # filter 0 + RGB pixels
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(row * height))
        + chunk(b"IEND", b"")
    )


def build_artifact(
    root: Path,
    *,
    t_ns: tuple[int, ...] = (10, 20, 30),
    sweep: list | None = None,
    sweep_t_ns: int = 0,
    fps: float = 3.0,
    target: dict | None = None,
    events: list | None = None,
    frame_size: tuple[int, int] = (64, 48),
) -> RecordingArtifact:
    """A small on-disk artifact: frames stamped on explicit clock values."""
    clock = RecordingClock()
    art = RecordingArtifact.create(
        root,
        recording_id="test-rec",
        fps=fps,
        target=target or {"kind": "full"},
        clock=clock,
        sweep=(
            sweep
            if sweep is not None
            else [{"x": 5, "y": 5, "width": 30, "height": 12, "label": "Go"}]
        ),
        sweep_t_ns=sweep_t_ns,
    )
    png = make_png(*frame_size)
    for t in t_ns:
        art.write_frame(png, t_ns=t, t_wall=1_700_000_000.0 + t / 1e9)
    if events is not None:
        art.write_events({"session": "test", "events": events})
    return art


@pytest.fixture(autouse=True)
def _no_ocr(monkeypatch):
    """Force the vision tier-2 probe to degrade regardless of host installs."""
    monkeypatch.setitem(sys.modules, "pytesseract", None)


def _assert_in_bounds(line: dict) -> None:
    """boxes.jsonl contract: only geometry the viewer can actually draw."""
    width, height = line["width"], line["height"]
    assert width > 0 and height > 0
    for box in line["boxes"]:
        assert box["x"] >= 0 and box["y"] >= 0
        assert box["x"] + box["width"] <= width + 1e-6
        assert box["y"] + box["height"] <= height + 1e-6


# ── clock ───────────────────────────────────────────────────────────────────


class TestRecordingClock:
    def test_now_ns_is_monotonic_since_arm(self):
        clock = RecordingClock()
        first = clock.now_ns()
        second = clock.now_ns()
        assert 0 <= first <= second

    def test_arm_is_idempotent(self):
        ticks = [1000]
        clock = RecordingClock(monotonic=lambda: ticks[0], wall=lambda: 42.0)
        assert clock.origin_ns == 1000
        ticks[0] = 2500
        clock.arm()  # re-arming would rewind every stamp already taken
        assert clock.origin_ns == 1000
        assert clock.now_ns() == 1500

    def test_note_states_the_alignment_contract(self):
        note = RecordingClock().note()
        assert "monotonic" in note
        assert "_t_ns" in note
        assert "timestamp" in note  # the desync risk is stated, not implied


# ── artifact ────────────────────────────────────────────────────────────────


class TestArtifact:
    def test_round_trip(self, tmp_path):
        art = build_artifact(tmp_path / "rec")
        art.write_boxes([{"index": 0, "boxes": [{"x": 1, "y": 2, "width": 3, "height": 4}]}])
        summary = art.finalize(event_count=0, box_count=1, dropped_frames=2)

        loaded = RecordingArtifact.load(art.root)

        assert summary["frame_count"] == 3
        assert summary["dropped_frames"] == 2
        assert len(loaded.frames) == 3
        assert loaded.manifest["id"] == "test-rec"
        assert loaded.manifest["dropped_frames"] == 2
        assert loaded.frames[0].file == "frames/000000.jpg" or loaded.frames[0].file.endswith(
            ".png"
        )
        assert loaded.frame_path(loaded.frames[0]).exists()
        assert len(loaded.read_boxes()) == 1

    def test_load_missing_dir_raises_with_hint(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="avion record"):
            RecordingArtifact.load(tmp_path / "nope")

    def test_frame_index_survives_without_finalize(self, tmp_path):
        """The `avion stop` recovery path: a killed record process must not
        orphan the frames it already wrote — write_frame persists its index."""
        build_artifact(tmp_path / "rec")  # no finalize() call anywhere

        loaded = RecordingArtifact.load(tmp_path / "rec")

        assert len(loaded.frames) == 3  # recoverable from the on-disk manifest

    def test_png_size_reads_ihdr(self):
        assert png_size(make_png(7, 5)) == (7, 5)
        assert png_size(b"not a png at all") == (0, 0)

    def test_prepare_frame_jpeg_when_pillow_present(self):
        data, ext, width, height = prepare_frame(make_png(10, 8))
        assert ext == ".jpg"
        assert (width, height) == (10, 8)
        assert data[:2] == b"\xff\xd8"  # real JPEG bytes, not a rename

    def test_prepare_frame_crop_shrinks_to_target(self):
        data, ext, width, height = prepare_frame(
            make_png(20, 20), {"x": 4, "y": 4, "width": 6, "height": 6}
        )
        assert ext == ".jpg"
        assert (width, height) == (6, 6)

    def test_prepare_frame_degrades_to_png_passthrough(self, monkeypatch):
        """Pillow missing → raw bytes survive with their true size, not a crash."""
        monkeypatch.setattr("avion.recording.artifact._pil", lambda: None)
        raw = make_png(9, 3)
        data, ext, width, height = prepare_frame(raw, {"x": 0, "y": 0, "width": 4, "height": 2})
        assert (data, ext) == (raw, ".png")  # crop degrades too — full frame stored
        assert (width, height) == (9, 3)

    def test_read_events_tolerates_missing_and_torn(self, tmp_path):
        art = build_artifact(tmp_path / "rec", t_ns=())
        assert art.read_events() == {"session": "", "events": []}
        art.events_path.write_text("{not json", encoding="utf-8")
        assert art.read_events() == {"session": "", "events": []}
        art.events_path.write_text("[1, 2, 3]", encoding="utf-8")
        assert art.read_events() == {"session": "", "events": []}

    def test_read_boxes_tolerates_torn_tail(self, tmp_path):
        art = build_artifact(tmp_path / "rec", t_ns=())
        good = json.dumps({"index": 0, "boxes": []})
        art.boxes_path.write_text(good + "\n" + '{"index": 1, "bo', encoding="utf-8")
        lines = art.read_boxes()
        assert [line["index"] for line in lines] == [0]

    def test_write_frame_appends_index_entries(self, tmp_path):
        art = build_artifact(tmp_path / "rec", t_ns=(5, 6))
        assert [f.index for f in art.frames] == [0, 1]
        assert [f.t_ns for f in art.frames] == [5, 6]
        assert art.manifest["frame_count"] == 2
        assert len(art.manifest["frames"]) == 2
        assert art.manifest["frames"][1]["t_ns"] == 6


# ── boxes ───────────────────────────────────────────────────────────────────


class TestBoxGeometry:
    def test_clamp_box_clips_to_frame(self):
        assert clamp_box({"x": -10, "y": -5, "width": 30, "height": 20}, width=10, height=10) == {
            "x": 0.0,
            "y": 0.0,
            "width": 10.0,
            "height": 10.0,
        }
        assert clamp_box({"x": 2, "y": 3, "width": 4, "height": 5}, width=10, height=10) == {
            "x": 2.0,
            "y": 3.0,
            "width": 4.0,
            "height": 5.0,
        }

    def test_to_frame_box_maps_picker_rect_into_crop(self):
        """The picker's page rect lands at the same spot inside a cropped frame."""
        box = to_frame_box(
            {"x": 100, "y": 50, "width": 20, "height": 10},
            crop={"x": 90, "y": 40, "width": 30, "height": 25},
            width=30,
            height=25,
        )
        assert box == {"x": 10.0, "y": 10.0, "width": 20.0, "height": 10.0}

    def test_to_frame_box_rejects_unusable_geometry(self):
        assert (
            to_frame_box({"x": 1, "y": 1, "width": 5, "height": 5}, crop=None, width=0, height=9)
            is None
        )  # a frame with no known geometry gets no boxes
        assert (
            to_frame_box(
                {"x": 500, "y": 500, "width": 5, "height": 5}, crop=None, width=10, height=10
            )
            is None
        )  # nothing of the box remains inside the frame


class TestDeriveBoxes:
    def test_all_geometry_within_frame_bounds(self, tmp_path):
        art = build_artifact(
            tmp_path / "rec",
            sweep=[{"x": -10, "y": 5, "width": 500, "height": 12, "label": "wide"}],
        )
        lines, vision = derive_boxes(art, [])
        assert lines
        for line in lines:
            _assert_in_bounds(line)
        assert vision["available"] is False  # no pytesseract here

    def test_sweep_segment_cut_at_navigate(self, tmp_path):
        """After the next navigate the swept page is gone — its boxes must go too."""
        art = build_artifact(
            tmp_path / "rec",
            t_ns=(10, 20, 50),
            sweep_t_ns=5,
            sweep=[{"x": 5, "y": 5, "width": 30, "height": 12, "label": "Go"}],
        )
        events = [{"type": "navigate", "name": "next", "data": {"_t_ns": 30}}]
        lines, _vision = derive_boxes(art, events)
        by_index = {line["index"]: line for line in lines}
        assert 0 in by_index and 1 in by_index  # frames at 10, 20 < navigate at 30
        assert 2 not in by_index  # frame at 50 is past the navigate
        for line in lines:
            _assert_in_bounds(line)

    def test_select_event_rect_lands_on_nearest_frame(self, tmp_path):
        art = build_artifact(
            tmp_path / "rec",
            t_ns=(10, 20, 30),
            sweep_t_ns=5,
            sweep=[],  # event rects only
        )
        events = [
            {
                "type": "select",
                "name": "#go",
                "data": {"_t_ns": 25, "rect": {"x": 8, "y": 8, "width": 16, "height": 8}},
            }
        ]
        lines, _vision = derive_boxes(art, events)
        assert [line["index"] for line in lines] == [1]  # nearest frame at/before 25
        box = lines[0]["boxes"][0]
        assert box["source"] == "pick"
        assert (box["x"], box["y"], box["width"], box["height"]) == (8.0, 8.0, 16.0, 8.0)
        _assert_in_bounds(lines[0])

    def test_event_without_rect_is_not_a_box(self, tmp_path):
        art = build_artifact(tmp_path / "rec", t_ns=(10,), sweep_t_ns=5, sweep=[])
        events = [{"type": "click", "name": "x", "data": {"_t_ns": 10}}]
        lines, _vision = derive_boxes(art, events)
        assert lines == []

    def test_no_frames_reports_reason_not_crash(self, tmp_path):
        art = build_artifact(tmp_path / "rec", t_ns=())
        lines, vision = derive_boxes(art, [])
        assert lines == []
        assert "no frames" in vision["reason"]

    def test_vision_boxes_degrade_with_stated_reason(self):
        boxes, status = vision_boxes(make_png(4, 4))
        assert boxes == []
        assert status["available"] is False
        assert "pytesseract" in status["reason"]  # exact tier-2 degradation

    def test_rebuild_and_load_frame_boxes(self, tmp_path):
        art = build_artifact(tmp_path / "rec")
        summary = rebuild_boxes(art, dropped_frames=7)
        assert summary["completed"] is True
        assert summary["dropped_frames"] == 7
        assert art.boxes_path.exists()
        first = load_frame_boxes(art, 0)
        assert first and first[0]["source"] == "sweep"
        assert load_frame_boxes(art, 99) == []


# ── sampler ─────────────────────────────────────────────────────────────────


class TestFrameSampler:
    def test_clamp_fps_bounds_the_contract(self):
        assert clamp_fps(0) == 2.0
        assert clamp_fps(99) == 5.0
        assert clamp_fps(3.5) == 3.5
        assert clamp_fps("x") == 3.0

    def test_double_start_is_an_error(self, tmp_path):
        art = build_artifact(tmp_path / "rec", t_ns=())
        sampler = FrameSampler(
            capture=lambda: make_png(), artifact=art, clock=RecordingClock(), fps=5.0
        )
        sampler.start()
        try:
            with pytest.raises(RuntimeError, match="already started"):
                sampler.start()
        finally:
            sampler.stop()

    def test_capture_errors_are_counted_not_fatal(self, tmp_path):
        art = build_artifact(tmp_path / "rec", t_ns=())

        def boom() -> bytes:
            raise RuntimeError("page gone")

        sampler = FrameSampler(capture=boom, artifact=art, clock=RecordingClock(), fps=5.0)
        sampler.start()
        time.sleep(0.35)
        stats = sampler.stop()

        assert stats["errors"] >= 1  # the recording survived its own failure
        assert stats["written"] == 0
        assert not sampler.running

    def test_ring_drops_oldest_when_drain_lags(self, tmp_path, monkeypatch):
        art = build_artifact(tmp_path / "rec", t_ns=())
        sampler = FrameSampler(
            capture=lambda: make_png(),
            artifact=art,
            clock=RecordingClock(),
            fps=5.0,
            ring=1,
        )
        monkeypatch.setattr(sampler, "_drain", lambda: None)  # simulate a stalled disk
        sampler.start()
        time.sleep(0.8)
        stats = sampler.stop()

        assert stats["dropped"] >= 1  # bounded backlog: oldest pending frame loses

    def test_written_frames_share_the_clock(self, tmp_path):
        art = build_artifact(tmp_path / "rec", t_ns=())
        clock = RecordingClock()
        sampler = FrameSampler(capture=lambda: make_png(), artifact=art, clock=clock, fps=5.0)
        sampler.start()
        time.sleep(0.6)
        stats = sampler.stop()

        assert stats["written"] >= 2
        stamps = [f.t_ns for f in art.frames]
        assert stamps == sorted(stamps)  # one axis, never rewound
        assert all(art.frame_path(f).exists() for f in art.frames)


# ── session ─────────────────────────────────────────────────────────────────


class FakeBackend:
    """Async double for the recording session: sweep + screenshots, no browser."""

    def __init__(self, png: bytes | None = None, sweep: list | None = None) -> None:
        self.png = png or make_png(64, 48)
        self.sweep = (
            sweep
            if sweep is not None
            else [{"x": 5, "y": 5, "width": 30, "height": 12, "label": "Go"}]
        )
        self.navigations: list[str] = []
        self.screenshot_calls = 0

    async def navigate(self, url: str) -> None:
        self.navigations.append(url)

    async def wait_for_timeout(self, ms: float) -> None:
        return None

    async def evaluate(self, expression: str):
        if "querySelectorAll" in expression:  # the sweep expression
            return self.sweep
        return None

    async def screenshot(self) -> bytes:
        self.screenshot_calls += 1
        return self.png


class HangingBackend(FakeBackend):
    """A page that stops answering screenshots — teardown must not wait on it."""

    async def screenshot(self) -> bytes:
        self.screenshot_calls += 1
        await asyncio.sleep(30)
        return self.png


class TestRecordingSession:
    async def test_begin_finalize_lifecycle(self, tmp_path):
        backend = FakeBackend()
        recorder = EventRecorder(session_name="lifecycle")
        session = RecordingSession(tmp_path / "rec", fps=5.0)

        artifact = await session.begin(backend, recorder, url="http://local/page")

        assert backend.navigations == ["http://local/page"]
        assert artifact.manifest["sweep"]  # structure tier collected
        nav = recorder.of_type(EventType.NAVIGATE)[0]
        # The open is stamped BEFORE the sweep so it starts the segment
        # rather than cutting it (derive_boxes keeps navs strictly after).
        assert nav.data["_t_ns"] <= artifact.manifest["sweep_t_ns"]

        await asyncio.sleep(0.6)  # let the sampler thread take frames
        recorder.record(EventType.CLICK, name="#go", data=session.stamp({"rect": {"x": 1, "y": 2}}))
        summary = session.finalize(recorder)

        assert summary["ok"] is True
        assert summary["frames_written"] >= 2
        assert summary["event_count"] == 2
        assert summary["box_count"] >= 1
        assert artifact.boxes_path.exists()
        assert artifact.events_path.exists()
        payload = json.loads(artifact.events_path.read_text(encoding="utf-8"))
        assert all("_t_ns" in (ev.get("data") or {}) for ev in payload["events"])
        manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
        assert manifest["completed"] is True
        assert manifest["frame_count"] == summary["frames_written"]

    async def test_finalize_aborts_inflight_capture(self, tmp_path):
        """Teardown with a capture in flight returns promptly (the _closing gate).

        Without the gate, finalize() joins a sampler thread parked on a
        screenshot the loop can no longer service — up to _CAPTURE_TIMEOUT
        of stall per teardown.
        """
        backend = HangingBackend()
        session = RecordingSession(tmp_path / "rec", fps=5.0)
        await session.begin(backend)
        await asyncio.sleep(0.25)  # sampler is now waiting on a screenshot

        started = time.monotonic()
        summary = session.finalize()
        elapsed = time.monotonic() - started

        assert summary["ok"] is True
        assert elapsed < 2.0  # one capture slice, not the 15s timeout

    def test_finalize_before_begin_reports_error(self, tmp_path):
        summary = RecordingSession(tmp_path / "rec").finalize()
        assert summary == {"ok": False, "error": "session never began"}

    def test_unknown_target_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="unknown capture target"):
            RecordingSession(tmp_path / "rec", target="page")

    def test_rect_target_needs_rect(self, tmp_path):
        with pytest.raises(ValueError, match="needs --rect"):
            RecordingSession(tmp_path / "rec", target="rect")

    def test_element_target_needs_selector(self, tmp_path):
        with pytest.raises(ValueError, match="needs --selector"):
            RecordingSession(tmp_path / "rec", target="element")

    def test_parse_rect(self):
        assert parse_rect("10,20,30,40") == {"x": 10.0, "y": 20.0, "width": 30.0, "height": 40.0}
        assert parse_rect("1, 2, 3, 4")["width"] == 3.0
        assert parse_rect("1,2,3") is None
        assert parse_rect("a,b,c,d") is None
        assert parse_rect("0,0,0,10") is None  # zero-area crop is not a target

    def test_build_locator_strategies(self):
        assert build_locator("#a", "css").selectors[0].strategy is SelectorStrategy.CSS
        assert build_locator("Go", "text").selectors[0].strategy is SelectorStrategy.TEXT
        assert build_locator("save", "testid").selectors[0].strategy is SelectorStrategy.TEST_ID
        assert build_locator("x", "bogus") is None

    def test_stamp_carries_shared_clock(self, tmp_path):
        session = RecordingSession(tmp_path / "rec")
        data = session.stamp({"value": "hi"})
        assert data["value"] == "hi"
        assert isinstance(data["_t_ns"], int)
        assert session.stamp()["_t_ns"] >= data["_t_ns"]

    def test_persist_oserror_is_survivable(self, tmp_path, monkeypatch):
        """A full disk during events.json rewrite must not kill the driving loop."""
        art = build_artifact(tmp_path / "rec", t_ns=())
        session = RecordingSession(tmp_path / "other")
        session.artifact = art
        monkeypatch.setattr(art, "write_events", self._raise_oserror)
        session.persist(EventRecorder(session_name="x"))  # must not raise

    @staticmethod
    def _raise_oserror(_payload):
        raise OSError("disk full")


# ── export ──────────────────────────────────────────────────────────────────


class TestExport:
    def test_missing_encoder_is_the_exact_contract_message(self, tmp_path, monkeypatch):
        art = build_artifact(tmp_path / "rec")
        monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)
        report = export_summary(art.root)
        assert report["ok"] is True  # the artifact is fine; only mp4 is gated
        assert report["mp4"] is False
        assert report["error"] == _ENCODER_MISSING
        result = export_mp4(art.root)
        assert result == {"ok": False, "error": _ENCODER_MISSING}

    def test_open_rejects_non_artifacts(self, tmp_path):
        art = build_artifact(tmp_path / "rec", t_ns=())
        art.manifest_path.write_text("not json", encoding="utf-8")
        result = export_mp4(art.root)
        assert result["ok"] is False
        assert "not an artifact directory" in result["error"]
        assert export_summary(tmp_path / "missing")["ok"] is False

    def test_success_path_encodes_existing_frames(self, tmp_path, monkeypatch):
        art = build_artifact(tmp_path / "rec")
        seen: dict = {}

        def fake_encode(frames, boxes_by_frame, fps, out):
            seen["frames"] = len(frames)
            seen["fps"] = fps
            Path(out).write_bytes(b"fake-mp4")

        monkeypatch.setattr("avion.recording.export._encoder_available", lambda: True)
        monkeypatch.setattr("avion.recording.export._encode_mp4", fake_encode)
        result = export_mp4(art.root, fps=4)

        assert result["ok"] is True
        assert result["frames"] == 3 and result["fps"] == 4.0
        assert seen["frames"] == 3 and seen["fps"] == 4.0
        assert Path(result["out"]).exists()

    def test_export_without_frames_reports_reason(self, tmp_path, monkeypatch):
        art = build_artifact(tmp_path / "rec", t_ns=())
        monkeypatch.setattr("avion.recording.export._encoder_available", lambda: True)
        result = export_mp4(art.root)
        assert result == {"ok": False, "error": "no frames recorded"}
