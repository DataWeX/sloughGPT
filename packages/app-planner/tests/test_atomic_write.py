"""Crash-safety battery for planner store writes (card faa1cfa7).

Complements the MogDB battery (``packages/mogdb/tests/test_durability.py``):
planner stores are single-file JSON/JSONL, so their crash windows are
truncate-in-place and torn publish — pinned here against
``app_planner.atomic`` and the store methods that must use it.

The invariant under test: **a failed/crashed publish leaves the previous
file byte-identical, and corrupt input lines are reported, never silently
dropped.**
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest
from app_planner import atomic, kanban, slot_chain
from app_planner import store as store_mod
from app_planner.kanban import Board, _JSONBackend
from app_planner.store import PlannerStore


class _CrashPublish:
    """os stand-in whose os.replace always crashes; everything else real."""

    def __init__(self, real: object) -> None:
        object.__setattr__(self, "_real", real)

    def __getattr__(self, name: str):  # pragma: no cover - passthrough
        return getattr(self._real, name)

    @staticmethod
    def replace(src, dst):  # noqa: ANN001 - mirrors os.replace
        raise OSError("simulated crash at publish")


class _SyncSpy:
    """os stand-in that records fsync calls instead of performing them."""

    def __init__(self, real: object) -> None:
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "syncs", 0)

    def __getattr__(self, name: str):  # pragma: no cover - passthrough
        if name == "fsync":
            raise AttributeError(name)  # force call sites to use the spy below
        return getattr(self._real, name)

    def fsync(self, fd: int) -> None:  # noqa: ANN001
        object.__setattr__(self, "syncs", self.syncs + 1)


def _boom(*_args, **_kwargs):  # noqa: ANN002, ANN003
    raise OSError("simulated crash before publish")


class TestAtomicWriteText:
    def test_replaces_content_and_leaves_no_tmp(self, tmp_path: Path) -> None:
        p = tmp_path / "board.jsonl"
        atomic.atomic_write_text(p, "one\n")
        atomic.atomic_write_text(p, "two\n")
        assert p.read_text() == "two\n"
        assert list(tmp_path.glob("*.tmp")) == []

    def test_failed_publish_keeps_original_and_cleans_tmp(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        p = tmp_path / "board.jsonl"
        atomic.atomic_write_text(p, "orig\n")
        monkeypatch.setattr(atomic, "os", _CrashPublish(os))
        with pytest.raises(OSError):
            atomic.atomic_write_text(p, "new\n")
        monkeypatch.undo()
        assert p.read_text() == "orig\n", "failed publish must not touch the original"
        assert list(tmp_path.iterdir()) == [p], "tmp litter left behind"

    def test_fsyncs_data_and_directory(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        spy = _SyncSpy(os)
        monkeypatch.setattr(atomic, "os", spy)
        atomic.atomic_write_text(tmp_path / "x.json", "data")
        assert spy.syncs >= 2, f"expected file fsync + dir fsync, got {spy.syncs}"


class TestNotesStore:
    def test_failed_write_leaves_notes_byte_identical(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        s = PlannerStore(board_dir=tmp_path / "b", notes_dir=tmp_path / "n")
        s.create_note(title="keep me")
        notes_file = tmp_path / "n" / "notes.journal.jsonl"
        before = notes_file.read_bytes()
        monkeypatch.setattr(store_mod, "atomic_write_text", _boom)
        with pytest.raises(OSError):
            s.create_note(title="doomed")
        assert notes_file.read_bytes() == before

    def test_corrupt_line_reported_not_silently_dropped(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        s = PlannerStore(board_dir=tmp_path / "b", notes_dir=tmp_path / "n")
        s.create_note(title="good")
        notes_file = tmp_path / "n" / "notes.journal.jsonl"
        with open(notes_file, "ab") as f:
            f.write(b'{"torn')
        with caplog.at_level(logging.WARNING, logger="app_planner.store"):
            notes = s.list_notes()
        assert [n.title for n in notes] == ["good"]
        assert any("corrupt line" in r.message for r in caplog.records), (
            "corrupt notes line was skipped without a report"
        )

    def test_board_write_failure_leaves_board_intact(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        s = PlannerStore(board_dir=tmp_path / "b", notes_dir=tmp_path / "n")
        board_file = tmp_path / "b" / "board.jsonl"
        board_file.write_text('{"meta": true}\n', encoding="utf-8")
        monkeypatch.setattr(store_mod, "atomic_write_text", _boom)
        with pytest.raises(OSError):
            s._atomic_write('{"meta": false}\n')
        assert board_file.read_text() == '{"meta": true}\n'


class TestJSONBackend:
    def test_failed_publish_keeps_board(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # KanbanStore delegates every board save to _JSONBackend.save.
        backend = _JSONBackend(tmp_path)
        backend.save(Board())
        board_file = tmp_path / "board.json"
        before = board_file.read_bytes()
        monkeypatch.setattr(kanban, "atomic_write_text", _boom)
        with pytest.raises(OSError):
            backend.save(Board())
        assert board_file.read_bytes() == before


class TestAppendLines:
    def test_appends_complete_lines_across_calls(self, tmp_path: Path) -> None:
        p = tmp_path / "hist.jsonl"
        atomic.append_lines(p, ['{"n": 1}\n', '{"n": 2}\n'])
        atomic.append_lines(p, ['{"n": 3}\n'])
        assert p.read_text() == '{"n": 1}\n{"n": 2}\n{"n": 3}\n'

    def test_empty_batch_is_a_noop(self, tmp_path: Path) -> None:
        p = tmp_path / "hist.jsonl"
        atomic.append_lines(p, [])
        assert not p.exists()


class TestSlotChainWriteBoard:
    def test_failed_publish_keeps_board(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "board.jsonl"
        original = '{"a": 1}\n'
        path.write_text(original, encoding="utf-8")
        monkeypatch.setattr(slot_chain, "atomic_write_text", _boom)
        with pytest.raises(OSError):
            slot_chain.write_board(path, ['{"a": 1}'], {0: {"a": 2}}, {0})
        assert path.read_text() == original
        assert list(tmp_path.glob("*.tmp")) == []
