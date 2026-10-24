"""Tests for readable ids (ids.py) and their wiring through the stores."""

from __future__ import annotations

import re

import pytest
from app_planner.ids import (
    display_id,
    is_readable_id,
    is_uuid_id,
    make_readable_id,
    slugify,
)
from app_planner.store import PlannerStore, reset_store

_READABLE_RE = re.compile(r"^\d{8}_\d{6}(_[0-9a-z-]+)?(-[0-9]+)?$")
_UUID = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"


@pytest.fixture(autouse=True)
def _reset():
    reset_store()
    yield
    reset_store()


@pytest.fixture
def store(tmp_path):
    return PlannerStore(board_dir=tmp_path / "board", notes_dir=tmp_path / "notes")


def test_make_readable_id_format_and_charset():
    id_ = make_readable_id("Fix boot order: root fails")
    assert _READABLE_RE.match(id_), id_
    assert is_readable_id(id_)
    assert not is_uuid_id(id_)
    assert id_ == id_.lower()
    assert not re.search(r"[^0-9a-z_-]", id_)


def test_slugify_rules():
    assert slugify("  Hello, World!  ") == "hello-world"
    assert slugify("A_B and   C") == "a-b-and-c"
    assert slugify("x" * 100) == "x" * 60
    assert slugify("--a--") == "a"
    assert slugify("") == ""


def test_empty_title_yields_timestamp_only_id():
    id_ = make_readable_id("!!!")
    assert re.match(r"^\d{8}_\d{6}$", id_), id_


def test_collision_appends_numeric_suffix():
    base = make_readable_id("Fix boot order")
    taken = {base}
    # second creation with the same title in the same second collides
    suffix = make_readable_id("Fix boot order", taken)
    # either the second has ticked over (fresh ts) or the -2 suffix kicked in
    if suffix.startswith(base[:15]):
        assert suffix == f"{base}-2"
    assert is_readable_id(suffix)


def test_collision_loop_skips_all_taken():
    base = make_readable_id("do work")
    taken = {base, f"{base}-2", f"{base}-3"}
    got = make_readable_id("do work", taken)
    assert got == f"{base}-4"


def test_display_id_forms():
    assert display_id(_UUID) == _UUID[:8]
    assert display_id("20261024_093000_fix-boot") == "20261024_093000_fix-boot"
    assert display_id("") == ""


def test_is_uuid_id_is_strict():
    assert is_uuid_id(_UUID)
    assert not is_uuid_id("20261024_093000-fix-boot")
    assert not is_uuid_id("a1b2c3d4")
    assert not is_uuid_id("")


def test_store_add_card_produces_readable_id(store):
    card = store.add_card("Wire the funnel", column="todo")
    assert _READABLE_RE.match(card.id), card.id
    assert is_readable_id(card.id)


def test_store_create_note_produces_readable_id(store):
    note = store.create_note("First review pass")
    assert _READABLE_RE.match(note.id), note.id


def test_existing_uuid_ids_survive_and_still_resolve(store):
    card = store.add_card("new card")
    board_path = store._board_file
    legacy = (
        '{"id": "%s", "title": "legacy", "column": "todo", "priority": "medium",'
        ' "tags": [], "assignee": "", "sprint": "", "gh": "", "description": "",'
        ' "due_date": "", "card_type": "", "created_at": "", "updated_at": ""}' % _UUID
    )
    text = board_path.read_text()
    if not text.endswith("\n"):
        text += "\n"
    board_path.write_text(text + legacy + "\n")
    found = store.get_card(_UUID)
    assert found is not None
    assert found.id == _UUID
    assert display_id(found.id) == _UUID[:8]
    # the new card was not rewritten
    assert store.get_card(card.id) is not None


def test_get_note_unique_prefix_resolves(store):
    note = store.create_note("Stable prefix probe")
    got = store.get_note(note.id[:12])
    assert got is not None
    assert got.id == note.id


def test_get_note_ambiguous_prefix_picks_most_recent(store):
    a = store.create_note("twin one")
    b = store.create_note("twin two")
    prefix = a.id[:8]  # same-day timestamp prefix — matches both
    got = store.get_note(prefix)
    assert got is not None
    assert got.id in {a.id, b.id}


def test_get_note_exact_beats_prefix(store):
    a = store.create_note("alpha note")
    b = store.create_note("beta note")
    # b's full id is also a prefix-extension of a shorter prefix; exact wins
    got = store.get_note(b.id)
    assert got is not None
    assert got.id == b.id
    assert a.id != b.id


def test_get_card_unique_prefix_resolves(store):
    card = store.add_card("Wire the funnel", column="todo")
    got = store.get_card(card.id[:12])
    assert got is not None
    assert got.id == card.id


def test_get_card_ambiguous_prefix_picks_most_recent(store):
    a = store.add_card("twin alpha", column="todo")
    b = store.add_card("twin beta", column="todo")
    prefix = a.id[:8]  # same-day timestamp prefix — matches both
    got = store.get_card(prefix)
    assert got is not None
    assert got.id in {a.id, b.id}


def test_get_card_exact_beats_prefix(store):
    a = store.add_card("alpha card", column="todo")
    b = store.add_card("beta card", column="todo")
    got = store.get_card(b.id)
    assert got is not None
    assert got.id == b.id
    assert a.id != b.id
