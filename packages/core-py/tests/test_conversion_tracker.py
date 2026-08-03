"""Tests for ConversionTracker — per-model download + SLNC conversion progress."""

import pytest

from domains.infrastructure.conversion_tracker import (
    ConversionStage,
    ConversionStatus,
    ConversionTracker,
    get_tracker,
)


class TestConversionStage:
    def test_values(self):
        assert ConversionStage.IDLE.value == "idle"
        assert ConversionStage.DOWNLOADING.value == "downloading"
        assert ConversionStage.CONVERTING.value == "converting"
        assert ConversionStage.PROTECTING.value == "protecting"
        assert ConversionStage.LOADING.value == "loading"
        assert ConversionStage.READY.value == "ready"
        assert ConversionStage.ERROR.value == "error"

    def test_is_string_enum(self):
        assert isinstance(ConversionStage.IDLE, str)
        assert str(ConversionStage.READY) == "ConversionStage.READY"


class TestConversionStatus:
    def test_defaults(self):
        s = ConversionStatus(model_id="gpt2")
        assert s.stage == ConversionStage.IDLE
        assert s.progress == 0.0
        assert s.error is None
        assert s.message == ""

    def test_to_dict(self):
        s = ConversionStatus(model_id="gpt2", stage=ConversionStage.CONVERTING,
                             progress=0.5, message="half")
        d = s.to_dict()
        assert d["model_id"] == "gpt2"
        assert d["stage"] == "converting"
        assert d["progress"] == pytest.approx(0.5)
        assert d["message"] == "half"
        assert d["error"] is None
        assert "elapsed_s" in d

    def test_to_dict_progress_rounded(self):
        s = ConversionStatus(model_id="gpt2", progress=0.333333)
        assert s.to_dict()["progress"] == pytest.approx(0.33)


class TestConversionTrackerStart:
    def test_start_default_stage(self):
        t = ConversionTracker()
        s = t.start("gpt2")
        assert isinstance(s, ConversionStatus)
        assert s.model_id == "gpt2"
        assert s.stage == ConversionStage.IDLE
        assert s.message == "Preparing..."

    def test_start_explicit_stage(self):
        t = ConversionTracker()
        s = t.start("gpt2", stage=ConversionStage.DOWNLOADING)
        assert s.stage == ConversionStage.DOWNLOADING
        assert s.message == "Downloading model weights..."

    def test_start_custom_message(self):
        t = ConversionTracker()
        s = t.start("gpt2", stage=ConversionStage.CONVERTING, message="custom")
        assert s.message == "custom"

    def test_start_tracks_in_get(self):
        t = ConversionTracker()
        t.start("gpt2", stage=ConversionStage.LOADING)
        d = t.get("gpt2")
        assert d is not None
        assert d["stage"] == "loading"

    def test_start_tracks_in_get_all(self):
        t = ConversionTracker()
        t.start("a")
        t.start("b")
        assert len(t.get_all()) == 2

    def test_restart_overwrites(self):
        t = ConversionTracker()
        t.start("gpt2", stage=ConversionStage.DOWNLOADING)
        t.start("gpt2", stage=ConversionStage.LOADING)
        assert len(t.get_all()) == 1
        assert t.get("gpt2")["stage"] == "loading"


class TestConversionTrackerUpdate:
    def test_update_stage_and_progress(self):
        t = ConversionTracker()
        t.start("gpt2", stage=ConversionStage.DOWNLOADING)
        s = t.update("gpt2", stage=ConversionStage.CONVERTING, progress=0.5)
        assert s.stage == ConversionStage.CONVERTING
        assert s.progress == pytest.approx(0.5)
        assert s.message == "Converting to optimized format (.slnc)..."

    def test_update_clamps_progress(self):
        t = ConversionTracker()
        t.start("gpt2")
        assert t.update("gpt2", progress=2.0).progress == 1.0
        assert t.update("gpt2", progress=-1.0).progress == 0.0

    def test_update_message_kept_when_no_stage(self):
        t = ConversionTracker()
        t.start("gpt2", message="original")
        s = t.update("gpt2", progress=0.9)
        assert s.message == "original"

    def test_update_explicit_message_with_stage(self):
        t = ConversionTracker()
        t.start("gpt2")
        s = t.update("gpt2", stage=ConversionStage.DOWNLOADING, message="fetching weights")
        assert s.message == "fetching weights"

    def test_update_auto_starts_unknown(self):
        t = ConversionTracker()
        s = t.update("unknown", stage=ConversionStage.DOWNLOADING, progress=0.1)
        assert s.model_id == "unknown"
        assert s.stage == ConversionStage.DOWNLOADING

    def test_update_sets_elapsed(self):
        t = ConversionTracker()
        t.start("gpt2")
        s = t.update("gpt2", progress=0.5)
        assert s.elapsed_s >= 0.0


class TestConversionTrackerFinish:
    def test_finish_sets_ready(self):
        t = ConversionTracker()
        t.start("gpt2", stage=ConversionStage.CONVERTING)
        s = t.finish("gpt2")
        assert s.stage == ConversionStage.READY
        assert s.progress == 1.0
        assert s.message == "Ready"
        assert t.get("gpt2")["stage"] == "ready"

    def test_finish_unknown_returns_none(self):
        t = ConversionTracker()
        assert t.finish("never-started") is None


class TestConversionTrackerFail:
    def test_fail_sets_error(self):
        t = ConversionTracker()
        t.start("gpt2")
        s = t.fail("gpt2", "download timeout")
        assert s.stage == ConversionStage.ERROR
        assert s.error == "download timeout"
        assert s.message == "Error: download timeout"

    def test_fail_excluded_from_active(self):
        t = ConversionTracker()
        t.start("gpt2", stage=ConversionStage.DOWNLOADING)
        t.fail("gpt2", "boom")
        assert t.get_active() == []

    def test_fail_unknown_returns_none(self):
        t = ConversionTracker()
        assert t.fail("never-started", "boom") is None


class TestConversionTrackerGet:
    def test_get_missing_returns_none(self):
        t = ConversionTracker()
        assert t.get("missing") is None

    def test_get_active_filters(self):
        t = ConversionTracker()
        t.start("in-progress", stage=ConversionStage.CONVERTING)
        t.start("done", stage=ConversionStage.READY)
        t.start("failed", stage=ConversionStage.ERROR)
        active = t.get_active()
        assert [a["model_id"] for a in active] == ["in-progress"]

    def test_get_active_includes_all_mid_stages(self):
        t = ConversionTracker()
        for stage in (ConversionStage.IDLE, ConversionStage.DOWNLOADING,
                      ConversionStage.CONVERTING, ConversionStage.PROTECTING,
                      ConversionStage.LOADING):
            t.start(stage.value, stage=stage)
        assert len(t.get_active()) == 5


class TestConversionTrackerClear:
    def test_clear_one(self):
        t = ConversionTracker()
        t.start("a")
        t.start("b")
        t.clear("a")
        assert t.get("a") is None
        assert t.get("b") is not None

    def test_clear_all(self):
        t = ConversionTracker()
        t.start("a")
        t.start("b")
        t.clear()
        assert t.get_all() == []

    def test_clear_missing_is_safe(self):
        t = ConversionTracker()
        t.clear("missing")
        assert t.get_all() == []


class TestDefaultMessages:
    def test_all_stages_have_messages(self):
        for stage in ConversionStage:
            msg = ConversionTracker._default_message(stage)
            assert msg != ""

    def test_unknown_stage_empty(self):
        assert ConversionTracker._default_message("nope") == ""


class TestGetTrackerSingleton:
    def test_returns_tracker(self):
        t = get_tracker()
        assert isinstance(t, ConversionTracker)

    def test_singleton_identity(self):
        assert get_tracker() is get_tracker()
