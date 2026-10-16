"""Mind-state substrate: schema, append, query, robustness (card a670d95c)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from domain.cognition._internal.consciousness.mind_state import (
    RECORD_TYPES,
    MindLog,
    MindRecord,
    query_mind,
)


class TestLatestRecord:
    """latest(): the freshest record by O(1) tail read (card c41a421a)."""

    def test_empty_returns_none(self, tmp_path: Path):
        assert MindLog(tmp_path).latest() is None

    def test_returns_most_recent_append(self, tmp_path: Path):
        log = MindLog(tmp_path)
        first = log.append(MindRecord(type="episode", payload={"n": 1}, provenance="experience"))
        third = log.append(
            MindRecord(type="belief_delta", payload={"n": 3}, provenance="experience")
        )
        latest = log.latest()
        assert latest is not None
        assert latest.id == third.id
        assert latest.type == "belief_delta"
        assert latest.payload == {"n": 3}
        assert latest.id != first.id

    def test_skips_torn_tail(self, tmp_path: Path):
        log = MindLog(tmp_path)
        good = log.append(MindRecord(type="dream", payload={}, provenance="dream"))
        with log.path.open("ab") as f:
            f.write(b'{"id": "torn-')  # crash mid-append: no newline, invalid JSON
        latest = log.latest()
        assert latest is not None
        assert latest.id == good.id

    def test_all_garbage_returns_none(self, tmp_path: Path):
        log = MindLog(tmp_path)
        log.path.parent.mkdir(parents=True, exist_ok=True)
        log.path.write_bytes(b"not json\nalso not json\n")
        assert log.latest() is None

    def test_overlong_last_record_escalates_to_full_read(self, tmp_path: Path):
        # A record larger than the 64 KiB tail window: the window lands
        # mid-record, its candidates all fail, one full read retrieves it.
        log = MindLog(tmp_path)
        big = log.append(
            MindRecord(type="episode", payload={"text": "x" * 70_000}, provenance="experience")
        )
        latest = log.latest()
        assert latest is not None
        assert latest.id == big.id

    def test_standalone_query_mind_unchanged(self, tmp_path: Path):
        # latest() is a method of the substrate, not a new entry point
        log = MindLog(tmp_path)
        log.append(MindRecord(type="episode", payload={}, provenance="experience"))
        assert len(query_mind(tmp_path, limit=10)) == 1


class TestMindRecord:
    """Schema: typed, timed, attributed, edged."""

    def test_roundtrip(self):
        r = MindRecord(type="episode", payload={"a": 1}, provenance="experience")
        r2 = MindRecord.from_dict(r.to_dict())
        assert r2.id == r.id
        assert r2.t == r.t
        assert r2.type == "episode"
        assert r2.payload == {"a": 1}
        assert json.loads(json.dumps(r.to_dict())) == r.to_dict()

    def test_unknown_type_rejected(self):
        with pytest.raises(ValueError):
            MindRecord(type="mood", payload={}, provenance="experience")

    def test_defaults(self):
        r = MindRecord(type="dream", payload={}, provenance="dream")
        assert r.edges == []
        assert r.id
        assert r.t > 0

    def test_vocabulary_is_honest(self):
        """Only types the engine actually writes are in the schema."""
        assert RECORD_TYPES == ("episode", "dream", "reflection", "belief_delta")


class TestMindLogQuery:
    """query_mind: the query mechanism over structured temporal data."""

    def test_append_and_query(self, tmp_path: Path):
        log = MindLog(tmp_path)
        log.append(MindRecord(type="episode", payload={"n": 1}, provenance="experience"))
        log.append(MindRecord(type="dream", payload={"n": 2}, provenance="dream"))
        got = log.query()
        assert [r.type for r in got] == ["episode", "dream"]

    def test_filter_by_type_and_time(self, tmp_path: Path):
        log = MindLog(tmp_path)
        a = log.append(MindRecord(type="episode", payload={}, provenance="experience", t=100.0))
        b = log.append(MindRecord(type="dream", payload={}, provenance="dream", t=200.0))
        assert [r.id for r in log.query(record_type="dream")] == [b.id]
        assert [r.id for r in log.query(since=150.0)] == [b.id]
        assert [r.id for r in log.query(until=150.0)] == [a.id]

    def test_limit_returns_newest_in_chronological_order(self, tmp_path: Path):
        log = MindLog(tmp_path)
        for i in range(5):
            log.append(
                MindRecord(type="episode", payload={"i": i}, provenance="experience", t=float(i))
            )
        got = log.query(limit=2)
        assert [r.payload["i"] for r in got] == [3, 4]

    def test_torn_line_skipped(self, tmp_path: Path):
        """A crashed append must never poison a query."""
        log = MindLog(tmp_path)
        log.append(MindRecord(type="episode", payload={}, provenance="experience"))
        with log.path.open("a", encoding="utf-8") as f:
            f.write('{"type": "episode", "trunca')
        log.append(MindRecord(type="dream", payload={}, provenance="dream"))
        got = log.query()
        assert [r.type for r in got] == ["episode", "dream"]

    def test_foreign_line_skipped(self, tmp_path: Path):
        log = MindLog(tmp_path)
        with log.path.open("a", encoding="utf-8") as f:
            f.write('{"type": "mood", "payload": {}}\n')  # unknown type
        assert log.query() == []

    def test_missing_file_is_empty(self, tmp_path: Path):
        assert query_mind(tmp_path) == []
        assert query_mind(tmp_path, record_type="dream") == []
