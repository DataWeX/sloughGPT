"""Tests for ConversionTracker — SLNC conversion lifecycle tracking."""
from __future__ import annotations

import pytest

from domains.infrastructure.conversion_tracker import ConversionStage, ConversionTracker


@pytest.fixture()
def tracker() -> ConversionTracker:
    return ConversionTracker()


class TestStart:
    def test_start_creates_status(self, tracker: ConversionTracker):
        status = tracker.start("model-1")
        assert status is not None
        assert status.stage == ConversionStage.IDLE

    def test_start_with_stage(self, tracker: ConversionTracker):
        status = tracker.start("model-1", stage=ConversionStage.DOWNLOADING)
        assert status.stage == ConversionStage.DOWNLOADING

    def test_start_with_message(self, tracker: ConversionTracker):
        status = tracker.start("model-1", message="starting up")
        assert status.message == "starting up"


class TestUpdate:
    def test_update_stage(self, tracker: ConversionTracker):
        tracker.start("model-1")
        updated = tracker.update("model-1", stage=ConversionStage.CONVERTING)
        assert updated is not None
        assert updated.stage == ConversionStage.CONVERTING

    def test_update_progress(self, tracker: ConversionTracker):
        tracker.start("model-1")
        updated = tracker.update("model-1", progress=0.5)
        assert updated is not None
        assert updated.progress == 0.5

    def test_update_message(self, tracker: ConversionTracker):
        tracker.start("model-1")
        updated = tracker.update("model-1", message="halfway there")
        assert updated is not None
        assert updated.message == "halfway there"

    def test_update_nonexistent_returns_none(self, tracker: ConversionTracker):
        assert tracker.update("nonexistent") is None


class TestFinish:
    def test_finish_sets_ready(self, tracker: ConversionTracker):
        tracker.start("model-1")
        result = tracker.finish("model-1")
        assert result is not None
        assert result.stage == ConversionStage.READY

    def test_finish_nonexistent_returns_none(self, tracker: ConversionTracker):
        assert tracker.finish("nonexistent") is None


class TestFail:
    def test_fail_sets_error(self, tracker: ConversionTracker):
        tracker.start("model-1")
        result = tracker.fail("model-1", "disk full")
        assert result is not None
        assert result.stage == ConversionStage.ERROR
        assert result.error == "disk full"

    def test_fail_nonexistent_returns_none(self, tracker: ConversionTracker):
        assert tracker.fail("nonexistent", "err") is None


class TestGetAndList:
    def test_get_returns_dict(self, tracker: ConversionTracker):
        tracker.start("model-1")
        d = tracker.get("model-1")
        assert d is not None
        assert "stage" in d

    def test_get_nonexistent_returns_none(self, tracker: ConversionTracker):
        assert tracker.get("nonexistent") is None

    def test_get_all(self, tracker: ConversionTracker):
        tracker.start("m1")
        tracker.start("m2")
        all_status = tracker.get_all()
        assert len(all_status) == 2

    def test_get_active(self, tracker: ConversionTracker):
        tracker.start("m1")
        tracker.start("m2")
        tracker.finish("m2")
        active = tracker.get_active()
        assert len(active) == 1


class TestClear:
    def test_clear_single(self, tracker: ConversionTracker):
        tracker.start("m1")
        tracker.clear("m1")
        assert tracker.get("m1") is None

    def test_clear_all(self, tracker: ConversionTracker):
        tracker.start("m1")
        tracker.start("m2")
        tracker.clear()
        assert tracker.get_all() == []


class TestDefaultMessage:
    def test_default_messages_exist(self):
        for stage in ConversionStage:
            msg = ConversionTracker._default_message(stage)
            assert isinstance(msg, str)
