"""Note <-> card identity: the card is the ticket, the note points at it.

Model (user ruling 2026-11-27): the CARD is the ticket of record — durable id,
history, comments. The NOTE is the working doc whose id may be re-derived when it
is edited. So the link between them must be STORED on the note and never derived:

* ``card_id`` is written once, when sync first links a note to a card, and
  survives later note edits (the note id may churn; the pointer must not).
* sync joins on ``card_id``. Title is display data. Joining on title is what let a
  rename fork a second card or merge two notes onto one card (measured 2026-11-27:
  31 cards whose id did not match their note, 6 titles held by two live notes).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from grib.store import PlannerStore, reset_store


@pytest.fixture(autouse=True)
def _reset():
    reset_store()
    yield
    reset_store()


@pytest.fixture
def store(tmp_path: Path) -> PlannerStore:
    return PlannerStore(board_dir=tmp_path / "board", notes_dir=tmp_path / "notes")


def _cards(store: PlannerStore):
    return store.load_board().cards


class TestSyncLinksNoteToCard:
    def test_sync_writes_the_card_id_onto_the_note(self, store: PlannerStore):
        note = store.create_note(title="Wire the relay")
        store.sync()

        cards = _cards(store)
        assert len(cards) == 1
        assert store.get_note(note.id).card_id == cards[0].id

    def test_second_sync_reuses_the_linked_card(self, store: PlannerStore):
        note = store.create_note(title="Wire the relay")
        store.sync()
        first_id = _cards(store)[0].id

        store.sync()
        assert len(_cards(store)) == 1
        assert _cards(store)[0].id == first_id
        assert store.get_note(note.id).card_id == first_id


class TestEditsKeepTheLink:
    def test_editing_the_title_keeps_the_card_and_the_link(self, store: PlannerStore):
        """A rename must not fork a second card, which the title join allowed."""
        note = store.create_note(title="Original title")
        store.sync()
        card_id = _cards(store)[0].id

        store.update_note(note.id, title="Completely different title")
        store.sync()

        cards = _cards(store)
        assert len(cards) == 1, "rename forked a card"
        assert cards[0].id == card_id
        assert stores_note(store, note.id).card_id == card_id

    def test_editing_the_body_keeps_the_link(self, store: PlannerStore):
        note = store.create_note(title="Ticket", body="first")
        store.sync()
        card_id = _cards(store)[0].id

        store.update_note(note.id, body="second")
        store.update_note(note.id, body="third")
        store.sync()

        assert [c.id for c in _cards(store)] == [card_id]
        assert stores_note(store, note.id).card_id == card_id

    def test_two_notes_renamed_to_one_title_stay_two_cards(self, store: PlannerStore):
        """Joining on title merged two real tickets onto one card."""
        a = store.create_note(title="Alpha")
        b = store.create_note(title="Beta")
        store.sync()
        assert len(_cards(store)) == 2

        store.update_note(b.id, title="Alpha")
        store.sync()

        assert len(_cards(store)) == 2, "title collision merged two tickets"
        assert stores_note(store, a.id).card_id != stores_note(store, b.id).card_id

    def test_legacy_notes_sharing_a_title_share_one_card(self, store: PlannerStore, caplog):
        """Never split a ticket in two to resolve an ambiguity — say so instead.

        Two unlinked notes with one title are one task whose note was renamed;
        forking a second card would divide the work between two records.
        """
        a = store.create_note(title="Same title")
        b = store.create_note(title="Same title")
        store.add_card("Same title")

        with caplog.at_level("WARNING"):
            store.sync()

        assert len(_cards(store)) == 1, "ambiguity forked a second ticket"
        linked = [n for n in (a, b) if stores_note(store, n.id).card_id]
        assert len(linked) == 1, "exactly one note drives the card"
        assert stores_note(store, linked[0].id).card_id == _cards(store)[0].id
        assert any("merge them by hand" in r.getMessage() for r in caplog.records)


class TestLegacyBootstrap:
    def test_note_links_to_the_existing_card_with_its_title(self, store: PlannerStore):
        """Notes created before card_id existed must adopt, not duplicate."""
        note = store.create_note(title="Legacy ticket")
        existing = store.add_card("Legacy ticket")

        store.sync()

        assert len(_cards(store)) == 1
        assert stores_note(store, note.id).card_id == existing.id

    def test_bootstrap_runs_once(self, store: PlannerStore):
        store.create_note(title="Legacy ticket")
        store.add_card("Legacy ticket")
        store.sync()
        store.sync()
        assert len(_cards(store)) == 1


class TestAddCardExplicitId:
    def test_add_card_takes_an_explicit_id(self, store: PlannerStore):
        """Sync must be able to name the card it creates, not mint a stranger."""
        card = store.add_card("Ticket", id="20260101_000000_ticket")
        assert card.id == "20260101_000000_ticket"
        assert _cards(store)[0].id == "20260101_000000_ticket"

    def test_add_card_still_mints_without_an_id(self, store: PlannerStore):
        card = store.add_card("Ticket")
        assert card.id.startswith("20") and card.id.endswith("ticket")


class TestBoardHeaderIsNotACard:
    def test_schema_row_does_not_load_as_a_card(self, store: PlannerStore):
        """board.jsonl carries a header row; counting it inflated stats by one."""
        store.add_card("Real card")
        store.sync()
        lines = store._board_file.read_text().splitlines()
        assert any("planner/1" in ln for ln in lines), "expected a schema header row"

        cards = _cards(store)
        assert len(cards) == 1
        assert cards[0].id and cards[0].title


def stores_note(store: PlannerStore, note_id: str):
    note = store.get_note(note_id)
    assert note is not None
    return note
