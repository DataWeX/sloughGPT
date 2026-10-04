"""Tests for the boot finalizers and the startup-history deadlock fix.

Covers kanban card 223001e4 (infra review 9/11/12 follow-ups):

1. ``get_optimization_suggestions`` used to self-deadlock on a plain
   ``threading.Lock`` — and because ``health.startup_history`` is an async
   handler, that froze the entire uvicorn event loop. It must return.
2. ``StartupOrchestrator._finalize_startup`` is the single wiring point for
   the three boot finalizers that each previously had a start-side call and
   zero end-side callers: history commit, terminal summary, webhook event.
3. Card d1f544fb: persistence swapped from a JSON file to a MogDB store —
   records survive restart, the legacy JSON migrates once (original kept as
   ``.bak``), and a missing mogdb degrades to in-memory-only instead of
   failing boot.
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_DIR = REPO_ROOT / "apps/api/server"
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from infrastructure.startup_history import StartupHistory  # noqa: E402


def _seed_records(history: StartupHistory, count: int = 3) -> None:
    for _ in range(count):
        history.start_startup()
        history.record_stage("critical", 5.0)
        history.record_hook("db_pool", 0.5)
        history.finish_startup(success=True)


def test_optimization_suggestions_does_not_deadlock(tmp_path, monkeypatch) -> None:
    """9-P0: nested acquisition of the history lock must not hang."""
    monkeypatch.setenv("SLO_STARTUP_HISTORY_PATH", str(tmp_path / "no_legacy.json"))
    history = StartupHistory(db_path=tmp_path / "history_mogdb")
    _seed_records(history, count=3)  # >= 2 records to pass the guard at line 387

    results: list = []
    errors: list[BaseException] = []

    def call() -> None:
        try:
            results.append(history.get_optimization_suggestions())
        except BaseException as exc:  # noqa: BLE001 - must surface, not vanish
            errors.append(exc)

    worker = threading.Thread(target=call, daemon=True)
    worker.start()
    worker.join(timeout=3.0)

    assert not worker.is_alive(), "get_optimization_suggestions self-deadlocked on the history lock"
    assert not errors, f"unexpected error: {errors!r}"
    assert isinstance(results[0], list)


def test_history_record_is_committed_and_persisted(tmp_path, monkeypatch) -> None:
    """d1f544fb: a finished startup must land in records and in the store."""
    monkeypatch.setenv("SLO_STARTUP_HISTORY_PATH", str(tmp_path / "no_legacy.json"))
    db = tmp_path / "history_mogdb"
    history = StartupHistory(db_path=db)
    _seed_records(history, count=1)

    assert len(history.get_records()) == 1
    assert db.exists() and any(db.iterdir()), "finish_startup must persist to the store"
    # A second instance sees the committed record
    reloaded = StartupHistory(db_path=db)
    assert len(reloaded.get_records()) == 1


def test_legacy_json_migrates_once_then_backs_up(tmp_path, monkeypatch) -> None:
    """d1f544fb: legacy records import once; the original survives as .bak."""
    legacy = tmp_path / "startup_history.json"
    legacy.write_text(
        json.dumps(
            [
                {
                    "timestamp": 111.0,
                    "total_duration": 9.9,
                    "stage_durations": {"critical": 3.0},
                    "hook_durations": {},
                    "model_load_duration": 0,
                    "success": True,
                    "error": None,
                }
            ]
        )
    )
    monkeypatch.setenv("SLO_STARTUP_HISTORY_PATH", str(legacy))
    db = tmp_path / "history_mogdb"

    first = StartupHistory(db_path=db)
    assert any(r["timestamp"] == 111.0 for r in first.get_records(50))
    backup = legacy.parent / (legacy.name + ".bak")
    assert backup.exists(), "original must be preserved as .bak, never deleted"
    assert not legacy.exists()

    # Idempotent: a second boot does not duplicate the migrated record
    second = StartupHistory(db_path=db)
    assert sum(1 for r in second.get_records(50) if r["timestamp"] == 111.0) == 1


def test_degrades_to_memory_without_mogdb(tmp_path, monkeypatch) -> None:
    """d1f544fb: no mogdb => in-memory only, one warning, boot never fails."""
    import infrastructure.startup_history as sh

    def _unavailable():
        raise ImportError("mogdb deliberately unavailable")

    monkeypatch.setattr(sh, "_import_mogdb", _unavailable)
    history = StartupHistory(db_path=tmp_path / "history_mogdb")
    _seed_records(history, count=1)

    assert len(history.get_records()) == 1  # memory path still works
    assert not (tmp_path / "history_mogdb").exists()  # nothing persisted, no crash


def _make_orchestrator():
    from infrastructure.startup import StartupOrchestrator

    return StartupOrchestrator.__new__(StartupOrchestrator)


def _run_finalize(loader) -> object:
    """Drive _finalize_startup on a running loop and await its webhook task."""
    orch = _make_orchestrator()

    async def drive() -> None:
        with (
            patch("infrastructure.startup_terminal.get_terminal_viz") as mock_viz_factory,
            patch("infrastructure.startup_webhooks.get_webhook_manager") as mock_webhooks_factory,
        ):
            manager = MagicMock()
            manager.emit = AsyncMock()
            mock_webhooks_factory.return_value = manager
            viz = MagicMock()
            mock_viz_factory.return_value = viz

            orch._finalize_startup(loader)
            await orch._startup_complete_task

            return manager, viz

    return asyncio.run(drive())


def test_finalize_success_commits_all_three_finalizers() -> None:
    """End of boot must fire finish_history + viz.finish + startup.complete."""
    loader = MagicMock()
    loader.get_status.return_value = {"errors": {}}

    manager, viz = _run_finalize(loader)

    loader.finish_history.assert_called_once_with(success=True, error=None)
    viz.finish.assert_called_once_with(success=True)
    assert manager.emit.await_count == 1
    event, payload = manager.emit.await_args.args
    assert event.value == "startup.complete"
    assert payload["success"] is True


def test_finalize_with_hook_errors_emits_failed() -> None:
    """A boot with hook errors records failure and emits startup.failed."""
    loader = MagicMock()
    loader.get_status.return_value = {
        "errors": {"db_pool": "timeout after 5.0s"},
    }

    manager, viz = _run_finalize(loader)

    loader.finish_history.assert_called_once_with(
        success=False, error="db_pool: timeout after 5.0s"
    )
    viz.finish.assert_called_once_with(success=False)
    event, payload = manager.emit.await_args.args
    assert event.value == "startup.failed"
    assert payload["error"] == "db_pool: timeout after 5.0s"


def test_finalize_fatal_error_overrides_success() -> None:
    """A raise inside _phase_ready must still finalize — as a failure."""
    from unittest.mock import patch as _patch

    loader = MagicMock()
    loader.get_status.return_value = {"errors": {}}
    orch = _make_orchestrator()

    async def drive() -> tuple:
        with (
            _patch("infrastructure.startup_terminal.get_terminal_viz") as viz_factory,
            _patch("infrastructure.startup_webhooks.get_webhook_manager") as wh_factory,
        ):
            manager = MagicMock()
            manager.emit = AsyncMock()
            wh_factory.return_value = manager
            viz_factory.return_value = MagicMock()

            orch._finalize_startup(loader, fatal="phase_ready exploded")
            await orch._startup_complete_task
            return manager

    manager = asyncio.run(drive())

    loader.finish_history.assert_called_once_with(success=False, error="phase_ready exploded")
    event, _ = manager.emit.await_args.args
    assert event.value == "startup.failed"
