"""db_pool legacy adoption — per-pair decision memo.

The legacy/new dir check ran on every get_db() call: legacy iterdir()
stats plus a repeated "Both ... contain data" warning per miss (measured
11 lines/10s against a live server). The decision is one-time per process,
so it must be computed once per (new, legacy) pair.
"""

from __future__ import annotations

import logging
from pathlib import Path

from apps.api.server.infrastructure import db_pool


def _both_contain_records(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if "contain data" in r.getMessage()]


def test_adopt_decision_and_warning_happen_once_per_pair(tmp_path: Path, caplog):
    legacy = tmp_path / "legacy" / "docstore"
    new = tmp_path / "new" / "docstore"
    legacy.mkdir(parents=True)
    new.mkdir(parents=True)
    (legacy / "old.json").write_text("{}")
    (new / "current.json").write_text("{}")

    db_pool._ADOPT_MEMO.clear()
    try:
        with caplog.at_level(logging.WARNING):
            first = db_pool._adopt_legacy_dir(new, legacy)
            warnings_after_first = _both_contain_records(caplog)
            second = db_pool._adopt_legacy_dir(new, legacy)
            warnings_after_second = _both_contain_records(caplog)

        assert first == str(new)
        assert second == first
        assert len(warnings_after_first) == 1
        assert len(warnings_after_second) == 1  # memo hit: no re-check, no re-warn
    finally:
        db_pool._ADOPT_MEMO.clear()
