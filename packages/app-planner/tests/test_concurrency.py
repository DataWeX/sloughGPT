"""Concurrency tests — the OCC lost-update guard for board.jsonl / notes.

Regression net for the "vanishing card" bug (2026-10-04): every write funnel
used to rebuild full file content from a read snapshot with no generation
check, so a concurrent writer's line could be silently erased. These tests
hammer the store from multiple PROCESSES (the real failure mode — separate
CLI invocations) and force conflicts deterministically to prove the retry
loop re-reads and rebuilds.
"""

from __future__ import annotations

import multiprocessing as mp
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from app_planner.store import BoardWriteConflict, PlannerStore, reset_store

WORKERS = 8
PER_WORKER = 5


@pytest.fixture(autouse=True)
def _reset():
    reset_store()
    yield
    reset_store()


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def store(tmp_dir):
    return PlannerStore(board_dir=tmp_dir / "board", notes_dir=tmp_dir / "notes")


# ── process workers (module level: picklable / fork-inherited) ──────────


def _worker(payload: tuple[Any, ...]) -> str:
    """Dispatch by kind so one pool can mix adders, noters, and syncers."""
    kind = payload[0]
    board_dir = Path(payload[1])
    notes_dir = Path(payload[2])
    store = PlannerStore(board_dir=board_dir, notes_dir=notes_dir)
    if kind == "add":
        _tag, n = payload[3], payload[4]
        for i in range(n):
            store.add_card(f"{_tag}-{i}", column="todo")
    elif kind == "note":
        _tag, n = payload[3], payload[4]
        for i in range(n):
            store.create_note(f"{_tag}-{i}", status="open")
    elif kind == "sync":
        _runs = payload[3]
        for _ in range(_runs):
            store.sync()
    else:  # pragma: no cover — guards payload typos
        raise ValueError(f"unknown worker kind: {kind}")
    return kind


def _run(payloads: list[tuple[Any, ...]], workers: int) -> None:
    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        list(ex.map(_worker, payloads))


def _board_payloads(tmp_dir: Path, kind: str, workers: int, n: int) -> list[tuple[Any, ...]]:
    return [
        (kind, str(tmp_dir / "board"), str(tmp_dir / "notes"), f"w{w}", n)
        for w in range(workers)
    ]


# ── hammers ─────────────────────────────────────────────────────────────


def test_parallel_add_card_hammer(tmp_dir: Path) -> None:
    """8 processes × 5 adds: every card must survive on disk (40, no dups)."""
    _run(_board_payloads(tmp_dir, "add", WORKERS, PER_WORKER), workers=WORKERS)
    store = PlannerStore(board_dir=tmp_dir / "board", notes_dir=tmp_dir / "notes")
    cards = store.load_board().cards
    assert len(cards) == WORKERS * PER_WORKER
    titles = [c.title for c in cards]
    assert len(set(titles)) == WORKERS * PER_WORKER
    # file shape intact: one header + one line per card
    lines = store._read_board_lines()
    assert sum(1 for obj in lines if obj.get("id")) == WORKERS * PER_WORKER


def test_parallel_notes_hammer(tmp_dir: Path) -> None:
    """8 processes × 5 create_note: the notes journal keeps all 20 notes."""
    _run(_board_payloads(tmp_dir, "note", WORKERS, PER_WORKER), workers=WORKERS)
    store = PlannerStore(board_dir=tmp_dir / "board", notes_dir=tmp_dir / "notes")
    assert len(store._read_notes()) == WORKERS * PER_WORKER


def test_sync_does_not_eat_concurrent_cards(tmp_dir: Path) -> None:
    """The real-world incident: sync (per-card adds + full-file reorder via
    compute_chains) running while other sessions add cards. Exactly 34 cards
    must remain — 30 from six concurrent adders + 4 materialized notes."""
    board = tmp_dir / "board"
    notes = tmp_dir / "notes"
    seeder = PlannerStore(board_dir=board, notes_dir=notes)
    for i in range(4):
        seeder.create_note(f"sync-note-{i}", status="todo")

    adders = [
        ("add", str(board), str(notes), f"add{w}", 5) for w in range(6)
    ]
    syncers = [("sync", str(board), str(notes), 3)]
    _run(adders + syncers, workers=7)

    store = PlannerStore(board_dir=board, notes_dir=notes)
    titles = [c.title for c in store.load_board().cards]
    assert len(titles) == 34
    assert sum(1 for t in titles if t.startswith("add")) == 30
    assert sum(1 for t in titles if t.startswith("sync-note-")) == 4
    assert len(set(titles)) == 34


# ── deterministic conflict paths ────────────────────────────────────────


def test_forced_conflict_retries(store: PlannerStore) -> None:
    """A validation miss must trigger rebuild + re-commit, not a lost add."""
    calls = {"n": 0}
    orig = store._validate_token

    def flaky(path: Path, expect: str) -> bool:
        calls["n"] += 1
        if calls["n"] == 1:
            return False  # simulate: file changed since our read
        return orig(path, expect)

    store._validate_token = flaky
    card = store.add_card("occ-survivor")
    assert store.get_card(card.id) is not None
    assert calls["n"] >= 2  # the funnel really re-ran the commit


def test_conflict_exhaustion_raises(store: PlannerStore, monkeypatch) -> None:
    """Bounded retries end LOUDLY (BoardWriteConflict), never a silent drop."""
    import app_planner.store as store_mod

    monkeypatch.setattr(store_mod, "_OCC_ATTEMPTS", 3)
    monkeypatch.setattr(store_mod, "_OCC_BACKOFF_S", 0.001)
    store._validate_token = lambda path, expect: False
    with pytest.raises(BoardWriteConflict, match="unresolved after 3 OCC attempts"):
        store.add_card("doomed")


def test_notes_conflict_retries(store: PlannerStore) -> None:
    """Notes journal writes run through the same OCC loop as the board."""
    calls = {"n": 0}
    orig = store._validate_token

    def flaky(path: Path, expect: str) -> bool:
        calls["n"] += 1
        if calls["n"] == 1:
            return False
        return orig(path, expect)

    store._validate_token = flaky
    note = store.create_note("occ-note")
    assert store.get_note(note.id) is not None
    assert calls["n"] >= 2
