"""
Guard: every non-empty status in repo-tracked note stores must resolve
through the shared vocabulary (canonical + legacy aliases). Null/empty
statuses are legacy absence — sync leaves those cards unchanged (or
repairs them explicitly), so they are skipped here.

Card 3ccc38b7 — kanban sync hardening.
"""

import json
from pathlib import Path

from app_planner import config

REPO = Path(__file__).resolve().parents[3]
JOURNALS = [
    REPO / ".dev-notes" / "notes.journal.jsonl",
    REPO / ".dev-notes" / "store" / "notes.journal.jsonl",
]


def _offending(path: Path) -> list[str]:
    bad: list[str] = []
    for raw in path.read_text().splitlines():
        if not raw.strip():
            continue
        try:
            rec = json.loads(raw)
        except json.JSONDecodeError:
            continue
        data = rec.get("data") if isinstance(rec.get("data"), dict) else rec
        status = data.get("status")
        if status in (None, ""):
            continue
        if not isinstance(status, str) or not config.is_known_status(status):
            bad.append(f"{path.name}:{str(data.get('id', '?'))[:40]} status={status!r}")
    return bad


def test_repo_journals_use_known_statuses():
    offenders: list[str] = []
    for journal in JOURNALS:
        if journal.exists():
            offenders += _offending(journal)
    assert not offenders, "statuses outside canonical+legacy vocabulary:\n" + "\n".join(offenders)


def test_guard_detects_bogus_status(tmp_path):
    p = tmp_path / "n.jsonl"
    p.write_text(json.dumps({"id": "x", "status": "partial"}) + "\n")
    assert _offending(p)


def test_canonical_vocabulary_is_doing_not_wip():
    assert "doing" in config.STATUSES
    assert "wip" not in config.STATUSES
    assert config.COLUMN_TO_STATUS["in_progress"] == "doing"
    assert config.is_known_status("wip")  # legacy stays accepted read-side
    assert config.is_known_status("in_progress")
    assert not config.is_known_status("partial")
