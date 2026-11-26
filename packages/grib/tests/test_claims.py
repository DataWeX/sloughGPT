"""Lane claims — worktree/session presence registry (card 20261126_102530).

Sessions coordinate by `git status` heuristics today, which fails when a
checkout is clean-but-owned or a parallel session reverts edits (2026-11-26
incidents). Claims are append-only events in the canonical store; `list`
folds them per session with heartbeat-style expiry.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path

import pytest
from grib.cli import main as cli_main
from grib.store import PlannerStore, reset_store


@pytest.fixture(autouse=True)
def _reset():
    reset_store()
    yield
    reset_store()


def _run(store_dir, *parts, env=None):
    buf = io.StringIO()
    old = {}
    if env:
        for k, v in env.items():
            old[k] = os.environ.get(k)
            os.environ[k] = v
    try:
        with contextlib.redirect_stdout(buf):
            try:
                code = cli_main(
                    ["--board-dir", str(store_dir), "--notes-dir", str(store_dir), *parts]
                )
            except SystemExit as e:
                code = e.code or 1
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return code, buf.getvalue()


def _events(store_dir: Path) -> list[dict]:
    path = store_dir / "claims.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _seed_card(store_dir: Path, title: str) -> str:
    store = PlannerStore(board_dir=store_dir, notes_dir=store_dir)
    return store.add_card(title, column="wip").id


# ── claim add ────────────────────────────────────────────────────────────


def test_claim_add_writes_event(tmp_path):
    code, out = _run(tmp_path, "claim", "add", "--card", "20261126_102530_x", "--session", "sess-a")
    assert code == 0, out
    events = _events(tmp_path)
    assert len(events) == 1
    ev = events[0]
    assert ev["event"] == "claim"
    assert ev["session"] == "sess-a"
    assert ev["card_id"] == "20261126_102530_x"
    # git detection: either real values or explicit nulls — never missing keys
    assert "worktree" in ev and "branch" in ev and "ts" in ev and "ttl_min" in ev


def test_claim_add_without_git_is_graceful(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # tmp_path is outside any git worktree
    code, out = _run(tmp_path, "claim", "add", "--session", "sess-nogit")
    assert code == 0, out
    ev = _events(tmp_path)[0]
    assert ev["session"] == "sess-nogit"
    assert ev["worktree"] is None and ev["branch"] is None


def test_claim_add_fills_assignee_when_empty(tmp_path):
    card_id = _seed_card(tmp_path, "Lane A")
    code, out = _run(tmp_path, "claim", "add", "--card", card_id, "--session", "sess-a")
    assert code == 0, out
    card = PlannerStore(board_dir=tmp_path, notes_dir=tmp_path).get_card(card_id)
    assert card is not None and card.assignee == "sess-a"


def test_claim_add_never_steals_assignee(tmp_path):
    card_id = _seed_card(tmp_path, "Lane A")
    _run(tmp_path, "claim", "add", "--card", card_id, "--session", "sess-a")
    code, out = _run(tmp_path, "claim", "add", "--card", card_id, "--session", "sess-b")
    assert code == 0, out
    card = PlannerStore(board_dir=tmp_path, notes_dir=tmp_path).get_card(card_id)
    assert card is not None and card.assignee == "sess-a"


# ── claim list ───────────────────────────────────────────────────────────


def test_claim_list_shows_active(tmp_path):
    _run(tmp_path, "claim", "add", "--card", "20261126_102530_x", "--session", "sess-a")
    code, out = _run(tmp_path, "claim", "list")
    assert code == 0, out
    assert "sess-a" in out
    assert "20261126_102530_x" in out
    assert "active" in out


def test_claim_list_active_flag_hides_expired(tmp_path, monkeypatch):
    # Write an event whose heartbeat is older than the TTL directly.
    from grib.claims import claim_path, fold

    path = claim_path(tmp_path)
    path.write_text(
        json.dumps(
            {
                "event": "claim",
                "session": "sess-old",
                "card_id": "20261126_102530_x",
                "worktree": None,
                "branch": None,
                "ts": 0.0,  # 1970 — far past any TTL
                "ttl_min": 120,
            }
        )
        + "\n"
    )
    claims = fold(path, now=1_000_000.0)
    assert claims[0]["status"] == "expired"

    code, out = _run(tmp_path, "claim", "list", "--active")
    assert code == 0, out
    assert "sess-old" not in out
    code, out = _run(tmp_path, "claim", "list")
    assert code == 0, out
    assert "sess-old" in out and "expired" in out


def test_claim_list_empty(tmp_path):
    code, out = _run(tmp_path, "claim", "list")
    assert code == 0, out
    assert "no claims" in out.lower() or out.strip() == ""


# ── fold semantics ───────────────────────────────────────────────────────


def test_fold_latest_event_wins_per_session(tmp_path):
    from grib.claims import claim_path, fold

    path = claim_path(tmp_path)
    rows = [
        {"event": "claim", "session": "s1", "card_id": "card-1", "ts": 10.0, "ttl_min": 120},
        {"event": "claim", "session": "s1", "card_id": "card-2", "ts": 20.0, "ttl_min": 120},
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    claims = fold(path, now=30.0)
    assert len(claims) == 1
    assert claims[0]["card_id"] == "card-2"
    assert claims[0]["status"] == "active"


def test_fold_release_then_reclaim(tmp_path):
    from grib.claims import claim_path, fold

    path = claim_path(tmp_path)
    rows = [
        {"event": "claim", "session": "s1", "card_id": "card-1", "ts": 10.0, "ttl_min": 120},
        {"event": "release", "session": "s1", "ts": 20.0},
        {"event": "claim", "session": "s2", "card_id": "card-1", "ts": 30.0, "ttl_min": 120},
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    claims = {c["session"]: c for c in fold(path, now=40.0)}
    assert claims["s1"]["status"] == "released"
    assert claims["s2"]["status"] == "active"
    assert claims["s2"]["card_id"] == "card-1"


# ── claim release ────────────────────────────────────────────────────────


def test_claim_release_marks_released(tmp_path):
    _run(tmp_path, "claim", "add", "--card", "20261126_102530_x", "--session", "sess-a")
    code, out = _run(tmp_path, "claim", "release", "--session", "sess-a")
    assert code == 0, out
    code, out = _run(tmp_path, "claim", "list", "--active")
    assert code == 0, out
    assert "sess-a" not in out
    code, out = _run(tmp_path, "claim", "list")
    assert "sess-a" in out and "released" in out


def test_claim_release_defaults_to_env_session(tmp_path):
    _run(
        tmp_path,
        "claim",
        "add",
        "--card",
        "20261126_102530_x",
        env={"OPENCODE_SESSION_ID": "ses-env"},
    )
    code, out = _run(tmp_path, "claim", "release", env={"OPENCODE_SESSION_ID": "ses-env"})
    assert code == 0, out
    code, out = _run(tmp_path, "claim", "list")
    assert "ses-env" in out and "released" in out
