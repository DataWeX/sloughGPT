"""Checkpoint lookup must cover every directory saves actually write to.

Final job saves land in models/<stem>_trained.soul (outside auto-training/),
and job records hand that path style back for "Load for chat".
"""

from __future__ import annotations

import json
from pathlib import Path

from domain.training._internal import checkpoints as cp


def _point_dirs(tmp_path: Path, monkeypatch):
    auto = tmp_path / "auto-training"
    auto.mkdir()
    turbo = tmp_path / "turbo-trained"
    turbo.mkdir()
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(cp, "CHECKPOINTS_DIR", auto)
    monkeypatch.setattr(cp, "TURBO_DIR", turbo)
    monkeypatch.setattr(cp, "TRAINED_DIR", models)
    return auto, turbo, models


class TestFindCheckpoint:
    def test_resolves_final_save_from_job_record_path(self, tmp_path, monkeypatch):
        _, _, models = _point_dirs(tmp_path, monkeypatch)
        target = models / "journey_select_trained.soul"
        target.write_bytes(b"x" * 5000)
        assert cp.find_checkpoint("models/journey_select_trained.soul") == target.resolve()
        assert cp.find_checkpoint("journey_select_trained.soul") == target.resolve()

    def test_directory_components_cannot_escape_search_roots(self, tmp_path, monkeypatch):
        _, _, _ = _point_dirs(tmp_path, monkeypatch)
        (tmp_path / "secret.soul").write_bytes(b"x" * 5000)
        assert cp.find_checkpoint("../secret.soul") is None

    def test_missing_or_empty_name_returns_none(self, tmp_path, monkeypatch):
        _point_dirs(tmp_path, monkeypatch)
        assert cp.find_checkpoint("") is None
        assert cp.find_checkpoint("no_such_checkpoint") is None


class TestScanAllCheckpoints:
    def test_final_trained_saves_are_listed(self, tmp_path, monkeypatch):
        _, _, models = _point_dirs(tmp_path, monkeypatch)
        soul = models / "journey_select_trained.soul"
        soul.write_bytes(b"\x00" * 5000)
        (models / "journey_select_trained.soul.meta.json").write_text(
            json.dumps({"soul_name": "journey_select"}),
            encoding="utf-8",
        )

        rows = cp._scan_all_checkpoints()
        match = [r for r in rows if r["name"] == "journey_select_trained.soul"]
        assert match, "final trained save missing from checkpoint list"
        assert match[0]["source"] == "trained"
