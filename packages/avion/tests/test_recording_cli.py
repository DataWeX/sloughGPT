"""Recording-CLI tests — parser shape, session-aware dispatch, exit codes.

Covers the `avion record/stop/view/export` command layer: parser arguments,
`_dispatch` stamping + rect nesting (the overlay-sugar contract with box
derivation), and the degrade-don't-crash exit codes for missing artifacts
and the capability-gated mp4 encoder.
"""

from __future__ import annotations

import json
import sys

from avion.cli import _dispatch, build_parser, main
from avion.events.models import EventType
from avion.events.recorder import EventRecorder
from avion.recording import RecordingSession
from test_recording import build_artifact, make_png


class FakeElement:
    """Playwright-style element: bounding_box populated by the backend."""

    def __init__(self, box: dict) -> None:
        self.bounding_box = box


class FakeBackend:
    """Minimal driving backend for _dispatch: find/click/fill with a box."""

    def __init__(self, box: dict | None = None) -> None:
        self.box = box or {"x": 10, "y": 20, "width": 30, "height": 40}
        self.clicked: list[str] = []
        self.filled: list[tuple[str, str]] = []

    async def find_element(self, locator):
        return FakeElement(self.box)

    async def click(self, element) -> None:
        self.clicked.append("clicked")

    async def fill(self, element, value: str, *, force: bool = False) -> None:
        self.filled.append(("filled", value))

    async def wait_for_text(self, text: str, *, timeout: float = 10.0) -> bool:
        return True  # wait succeeds immediately in the fake page

    async def screenshot(self) -> bytes:
        return make_png()


# ── parser ──────────────────────────────────────────────────────────────────


class TestParsers:
    def test_record_defaults(self):
        args = build_parser().parse_args(["record", "http://x/"])
        assert args.cmd == "record"
        assert args.url == "http://x/"
        assert args.out is None
        assert args.fps == 3.0
        assert args.target == "full"
        assert args.rect is None and args.selector is None
        assert args.by == "css"

    def test_record_full_options(self):
        args = build_parser().parse_args(
            ["record", "--out", "rec/", "--fps", "5", "--target", "element", "--selector", "#app"]
        )
        assert args.url is None  # the page URL is optional
        assert (args.out, args.fps, args.target, args.selector) == ("rec/", 5.0, "element", "#app")

    def test_stop_view_export_dirs(self):
        assert build_parser().parse_args(["stop", "rec/"]).dir == "rec/"
        view = build_parser().parse_args(["view", "rec/", "--out", "v.html"])
        assert (view.dir, view.out) == ("rec/", "v.html")
        export = build_parser().parse_args(["export", "rec/", "--mp4"])
        assert export.dir == "rec/" and export.mp4 == ""  # present, default path
        explicit = build_parser().parse_args(["export", "rec/", "--mp4", "out.mp4"])
        assert explicit.mp4 == "out.mp4"
        dry = build_parser().parse_args(["export", "rec/"])
        assert dry.mp4 is None and dry.rebuild is False and dry.json is False


# ── session-aware dispatch ──────────────────────────────────────────────────


class TestDispatchRecording:
    async def test_click_data_nests_rect_and_carries_stamp(self, tmp_path):
        session = RecordingSession(tmp_path / "rec")
        recorder = EventRecorder(session_name="d")
        backend = FakeBackend(box={"x": 10, "y": 20, "width": 30, "height": 40})

        code = await _dispatch(backend, ["click", "#go"], recorder=recorder, session=session)

        assert code == 0
        click = recorder.of_type(EventType.CLICK)[0]
        # Rect nested under data["rect"] (boxes._event_rect's contract) plus
        # the shared-clock stamp — this is the overlay-sugar contract.
        assert click.data["rect"] == {"x": 10.0, "y": 20.0, "width": 30.0, "height": 40.0}
        assert isinstance(click.data["_t_ns"], int)
        found = recorder.of_type(EventType.ELEMENT_FOUND)[0]
        assert isinstance(found.data["_t_ns"], int)

    async def test_fill_data_carries_value_and_rect(self, tmp_path):
        session = RecordingSession(tmp_path / "rec")
        recorder = EventRecorder(session_name="d")
        backend = FakeBackend()

        code = await _dispatch(backend, ["fill", "#q", "hello"], recorder=recorder, session=session)

        assert code == 0
        fill = recorder.of_type(EventType.FILL)[0]
        assert fill.data["value"] == "hello"
        assert fill.data["rect"] == {"x": 10.0, "y": 20.0, "width": 30.0, "height": 40.0}
        assert isinstance(fill.data["_t_ns"], int)

    async def test_wait_discovery_carries_shared_clock_stamp(self, tmp_path):
        """The wait branch must stamp too — E2E caught it unstamped once.

        Every recorded event joins the ONE monotonic axis; an unstamped
        element_found would force the viewer's step→frame map onto the
        no-stamp fallback (hold previous) instead of a subtraction.
        """
        session = RecordingSession(tmp_path / "rec")
        recorder = EventRecorder(session_name="d")

        code = await _dispatch(FakeBackend(), ["wait", "Hello"], recorder=recorder, session=session)

        assert code == 0
        found = recorder.of_type(EventType.ELEMENT_FOUND)[0]
        assert isinstance(found.data["_t_ns"], int)

        # No session → passthrough, same as the other verbs.
        plain = EventRecorder(session_name="d")
        await _dispatch(FakeBackend(), ["wait", "Hello"], recorder=plain)
        assert "_t_ns" not in plain.of_type(EventType.ELEMENT_FOUND)[0].data

    async def test_no_session_means_no_rect_or_stamp(self):
        recorder = EventRecorder(session_name="d")
        backend = FakeBackend()

        await _dispatch(backend, ["click", "#go"], recorder=recorder)

        click = recorder.of_type(EventType.CLICK)[0]
        assert "rect" not in click.data and "_t_ns" not in click.data

    async def test_unresolvable_rect_degrades_not_fails(self, tmp_path):
        """A verb must succeed even when the overlay geometry never resolves."""
        session = RecordingSession(tmp_path / "rec")
        recorder = EventRecorder(session_name="d")

        class NoBoxBackend(FakeBackend):
            async def find_element(self, locator):
                return None  # element itself missing → verb fails, no event

        code = await _dispatch(
            NoBoxBackend(), ["click", "#gone"], recorder=recorder, session=session
        )

        assert code == 1
        assert recorder.of_type(EventType.CLICK) == []  # failures never replay


# ── command exit codes ──────────────────────────────────────────────────────


class TestCommandExitCodes:
    def test_stop_missing_dir_is_rc1(self, tmp_path, capsys):
        assert main(["stop", str(tmp_path / "missing")]) == 1
        assert "not a recording directory" in capsys.readouterr().err

    def test_view_missing_dir_is_rc1(self, tmp_path, capsys):
        assert main(["view", str(tmp_path / "missing")]) == 1
        assert "not a recording directory" in capsys.readouterr().err

    def test_export_missing_dir_is_rc1(self, tmp_path, capsys):
        assert main(["export", str(tmp_path / "missing")]) == 1
        assert "not an artifact directory" in capsys.readouterr().err

    def test_export_mp4_missing_encoder_exact_message(self, tmp_path, monkeypatch, capsys):
        art = build_artifact(tmp_path / "rec")
        monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)

        assert main(["export", str(art.root), "--mp4"]) == 1

        # Exact contract message, verbatim, no prefix — the reason IS the report.
        assert capsys.readouterr().err.strip() == "encoder missing: imageio-ffmpeg not installed"

    def test_export_report_is_parseable_json(self, tmp_path, monkeypatch, capsys):
        art = build_artifact(tmp_path / "rec")
        monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)

        assert main(["export", str(art.root), "--json"]) == 0

        report = json.loads(capsys.readouterr().out)
        assert report["ok"] is True
        assert report["mp4"] is False
        assert report["frames"] == 3
        assert "encoder missing" in report["error"]

    def test_stop_and_view_recover_an_artifact(self, tmp_path, capsys):
        art = build_artifact(tmp_path / "rec")
        assert main(["stop", str(art.root)]) == 0
        assert "stopped:" in capsys.readouterr().out
        assert main(["view", str(art.root)]) == 0
        html = (art.root / "viewer.html").read_text(encoding="utf-8")
        assert 'id="player"' in html
        assert 'id="scrub"' in html
        assert 'id="boxtoggle"' in html
        assert "const SF =" in html
        assert 'class="box"' in html  # sweep overlay drawn from boxes.jsonl
