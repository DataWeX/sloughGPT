"""Recording-viewer tests — player, scrubber, overlays, step→frame map.

Covers the recording half of `avion/flow/viewer.py`: RecordingData loads an
artifact into render-ready frames (inline data URIs + boxes by index), the
player section renders one frame <img> with pre-hidden overlay layers and a
scrubber, and `_step_frame_map` aligns steps to frames on the shared clock.
"""

from __future__ import annotations

from avion.flow import FlowStep, build_flow
from avion.flow.viewer import (
    FrameView,
    RecordingData,
    _data_rows,
    _step_frame_map,
    render_html,
)
from test_recording import build_artifact


def _graph():
    return build_flow(
        [
            {"seq": 0, "type": "navigate", "name": "go /", "data": {"_t_ns": 5}},
            {"seq": 1, "type": "click", "name": "click save", "data": {"_t_ns": 15}},
        ],
        session_name="rec_run",
    )


def _recording(n: int = 3, boxes: dict[int, list[dict]] | None = None) -> RecordingData:
    frames = [
        FrameView(
            index=i,
            src=f"data:image/png;base64,AAAA{i}",
            t_ns=10 * (i + 1),
            width=320,
            height=240,
            boxes=(boxes or {}).get(i, []),
        )
        for i in range(n)
    ]
    return RecordingData(frames=frames, fps=3.0)


# ── artifact loading ────────────────────────────────────────────────────────


class TestRecordingData:
    def test_from_artifact_inlines_frames_and_boxes_by_index(self, tmp_path):
        art = build_artifact(
            tmp_path / "rec",
            sweep=[{"x": 5, "y": 5, "width": 30, "height": 12, "label": "Go"}],
        )
        art.write_boxes(
            [
                {
                    "index": 1,
                    "file": art.frames[1].file,
                    "t_ns": art.frames[1].t_ns,
                    "width": 64,
                    "height": 48,
                    "boxes": [{"x": 5, "y": 5, "width": 30, "height": 12, "label": "Go"}],
                }
            ]
        )

        rec = RecordingData.from_artifact(art)

        assert len(rec.frames) == 3
        assert rec.frames[0].src.startswith("data:image/") and ";base64," in rec.frames[0].src
        assert rec.frames[0].boxes == []  # boxes attach by frame index, not order
        assert rec.frames[1].boxes[0]["label"] == "Go"
        assert rec.box_count == 1
        assert rec.fps == 3.0 and rec.dropped == 0 and rec.truncated == 0

    def test_missing_boxes_file_yields_overlayless_frames(self, tmp_path):
        art = build_artifact(tmp_path / "rec")  # never wrote boxes.jsonl

        rec = RecordingData.from_artifact(art)

        assert len(rec.frames) == 3 and rec.box_count == 0  # loader never writes

    def test_truncation_is_stated_not_hidden(self, tmp_path):
        art = build_artifact(tmp_path / "rec")

        rec = RecordingData.from_artifact(art, max_frames=2)

        assert len(rec.frames) == 2
        assert rec.truncated == 1

    def test_lost_frame_skips_without_killing_the_player(self, tmp_path):
        art = build_artifact(tmp_path / "rec")
        art.frame_path(art.frames[0]).unlink()

        rec = RecordingData.from_artifact(art)

        assert [f.index for f in rec.frames] == [1, 2]  # the index around it survives


# ── player rendering ────────────────────────────────────────────────────────


class TestPlayerHtml:
    def test_one_frame_img_and_js_arrays(self):
        html = render_html(_graph(), recording=_recording())
        assert html.count('<img id="frame"') == 1  # src swaps on scrub
        assert "const FRAMES = " in html
        assert "const SF = " in html

    def test_overlay_layers_hidden_except_active(self):
        html = render_html(_graph(), recording=_recording(3, {0: [], 1: []}))
        assert 'class="ovl" id="ovl0">' in html  # first frame active
        assert 'class="ovl" id="ovl1" hidden' in html
        assert 'class="ovl" id="ovl2" hidden' in html

    def test_box_percentages_within_frame(self):
        box = {"x": 80, "y": 60, "width": 160, "height": 120, "label": "Go"}
        html = render_html(_graph(), recording=_recording(2, {0: [box]}))
        assert 'style="left:25.00%;top:25.00%;width:50.00%;height:50.00%"' in html

    def test_box_geometry_escaped_and_clamped(self):
        box = {"x": -50, "y": 10, "width": 9999, "height": 12, "label": "<script>alert(1)</script>"}
        html = render_html(_graph(), recording=_recording(1, {0: [box]}))
        assert "left:0.00%" in html and "width:100.00%" in html  # clamped to frame
        assert "<script>alert(1)</script>" not in html  # label escaped
        assert "&lt;script&gt;" in html

    def test_scrubber_range_and_toggle(self):
        html = render_html(_graph(), recording=_recording(4))
        assert 'id="scrub" min="0" max="3" value="0"' in html
        assert 'id="boxtoggle"' in html
        assert ".player.noboxes .ovl .box" in html  # the toggle's CSS contract

    def test_truncation_note_rendered(self):
        rec = RecordingData(frames=_recording(2).frames, fps=3.0, dropped=1, truncated=5)
        html = render_html(_graph(), recording=rec)
        assert "showing first 2 of 7 frames" in html
        assert "1 dropped" in html

    def test_step_selection_calls_show_frame_via_sf(self):
        html = render_html(_graph(), recording=_recording())
        assert "showFrame(SF[i]);" in html  # selectStep drives the player

    def test_no_recording_renders_without_player(self):
        html = render_html(_graph())
        assert 'id="player"' not in html
        assert 'id="scrub"' not in html


# ── step → frame map ────────────────────────────────────────────────────────


class TestStepFrameMap:
    @staticmethod
    def _steps(*stamps):
        return [
            FlowStep(seq=i, type="click", name=f"s{i}", data=({"_t_ns": t} if t else {}))
            for i, t in enumerate(stamps)
        ]

    def test_nearest_frame_at_or_before_the_event(self):
        steps = self._steps(10, 20, 30)  # frames at t_ns 10, 20, 30
        assert _step_frame_map(steps, _recording()) == [0, 1, 2]
        # strictly at-or-before: an event between frames stays on the earlier one
        assert _step_frame_map(self._steps(15), _recording()) == [0]

    def test_event_after_last_frame_lands_on_last(self):
        assert _step_frame_map(self._steps(999), _recording()) == [2]

    def test_unstamped_steps_hold_the_previous_mapping(self):
        steps = self._steps(15, None, 40)
        assert _step_frame_map(steps, _recording()) == [0, 0, 2]

    def test_no_stamps_at_all_distributes_proportionally(self):
        steps = self._steps(None, None, None, None)
        assert _step_frame_map(steps, _recording(2)) == [0, 0, 1, 1]

    def test_no_frames_is_empty(self):
        assert _step_frame_map(self._steps(1), RecordingData()) == []


# ── detail panel data rows ──────────────────────────────────────────────────


class TestDataRows:
    def test_plumbing_keys_and_pixels_never_reach_the_detail_panel(self):
        step = FlowStep(
            seq=0,
            type="click",
            name="x",
            data={"_t_ns": 123456789, "screenshot": "b64pixels", "note": "kept"},
        )
        rows = _data_rows(step)
        assert "kept" in rows
        assert "123456789" not in rows  # alignment stamp is viewer plumbing
        assert "b64pixels" not in rows  # pixels render as <img>, never text

    def test_rect_row_shown_for_overlay_events(self):
        step = FlowStep(
            seq=0,
            type="click",
            name="x",
            data={"rect": {"x": 1, "y": 2, "width": 3, "height": 4}},
        )
        rows = _data_rows(step)
        assert "rect" in rows and "x=1" in rows and "height=4" in rows
