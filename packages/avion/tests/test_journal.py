"""Journal tests — durability, no-clobber, seq ordering, torn tails."""

from __future__ import annotations

import threading

from avion.events.journal import EventJournal
from avion.events.models import Event, EventType


def _event(name: str = "e") -> Event:
    return Event(type=EventType.CLICK, name=name)


class TestSeqOrdering:
    def test_seq_starts_at_one(self, tmp_path):
        with EventJournal(tmp_path / "events.jsonl") as journal:
            journal.append(_event("a"))
            journal.append(_event("b"))
            assert [e.seq for e in journal.events()] == [1, 2]

    def test_reopen_continues_seq(self, tmp_path):
        path = tmp_path / "events.jsonl"
        with EventJournal(path) as journal:
            for i in range(3):
                journal.append(_event(str(i)))
        with EventJournal(path) as journal:
            assert journal.append(_event("after-reopen")).seq == 4
            assert [e.seq for e in journal.events()] == [1, 2, 3, 4]

    def test_seq_never_restarts_across_many_reopens(self, tmp_path):
        path = tmp_path / "events.jsonl"
        seqs: list[int] = []
        for _ in range(5):
            with EventJournal(path) as journal:
                seqs.append(journal.append(_event()).seq)
        assert seqs == [1, 2, 3, 4, 5]

    def test_concurrent_appends_get_unique_seqs(self, tmp_path):
        journal = EventJournal(tmp_path / "events.jsonl")

        def worker(n: int) -> None:
            for i in range(50):
                journal.append(_event(f"{n}-{i}"))

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        seqs = [e.seq for e in journal.events()]
        assert sorted(seqs) == list(range(1, 201))
        journal.close()


class TestDurability:
    def test_append_is_visible_to_a_fresh_reader_immediately(self, tmp_path):
        path = tmp_path / "events.jsonl"
        journal = EventJournal(path)
        journal.append(_event("durable"))
        # A separate handle, no close(), no cooperation from the writer:
        # append() must have flushed before it returned.
        reader = EventJournal(path)
        assert [e.name for e in reader.events()] == ["durable"]

    def test_open_alone_writes_nothing(self, tmp_path):
        path = tmp_path / "events.jsonl"
        EventJournal(path)  # lazy: no file until the first append
        assert not path.exists()

    def test_missing_parent_dirs_are_created(self, tmp_path):
        path = tmp_path / "nested" / "deeper" / "events.jsonl"
        with EventJournal(path) as journal:
            journal.append(_event())
        assert path.exists()


class TestNoClobber:
    def test_reopen_never_truncates_history(self, tmp_path):
        path = tmp_path / "events.jsonl"
        with EventJournal(path) as journal:
            for i in range(3):
                journal.append(_event(f"old-{i}"))
        with EventJournal(path) as journal:
            journal.append(_event("new"))
        names = [e.name for e in EventJournal(path).events()]
        assert names == ["old-0", "old-1", "old-2", "new"]

    def test_clear_is_not_a_journal_operation(self, tmp_path):
        # The journal has no truncate/clear API at all — append-only.
        path = tmp_path / "events.jsonl"
        with EventJournal(path) as journal:
            journal.append(_event())
            assert not hasattr(journal, "clear")
            assert not hasattr(journal, "truncate")

    def test_empty_existing_file_is_kept(self, tmp_path):
        path = tmp_path / "events.jsonl"
        path.write_bytes(b"")
        with EventJournal(path) as journal:
            assert journal.append(_event()).seq == 1


class TestTornTail:
    def test_fragment_is_isolated_kept_and_its_seq_never_reused(self, tmp_path):
        path = tmp_path / "events.jsonl"
        with EventJournal(path) as journal:
            journal.append(_event("good"))
        # Simulated crash mid-write: partial line, no trailing newline.
        with open(path, "a") as fh:
            fh.write('{"seq": 7, "type": "cl')

        with EventJournal(path) as journal:
            assert journal.append(_event("next")).seq == 8
            assert [e.name for e in journal.events()] == ["good", "next"]
        # The fragment is preserved (deleting history is forbidden), just
        # isolated on its own line so later appends stay parseable.
        assert '{"seq": 7, "type": "cl' in path.read_text()

    def test_complete_final_line_without_newline_still_reads(self, tmp_path):
        path = tmp_path / "events.jsonl"
        with EventJournal(path) as journal:
            journal.append(_event("one"))
        # Truncate the trailing newline only (crash before the byte hit disk).
        raw = path.read_bytes()
        path.write_bytes(raw.rstrip(b"\n"))
        with EventJournal(path) as journal:
            assert [e.name for e in journal.events()] == ["one"]
            assert journal.append(_event("two")).seq == 2
        assert [e.name for e in EventJournal(path).events()] == ["one", "two"]

    def test_unparseable_lines_are_skipped_not_raised(self, tmp_path):
        path = tmp_path / "events.jsonl"
        with EventJournal(path) as journal:
            journal.append(_event("kept"))
        with open(path, "a") as fh:
            fh.write("not json at all\n")
        assert [e.name for e in EventJournal(path).events()] == ["kept"]
