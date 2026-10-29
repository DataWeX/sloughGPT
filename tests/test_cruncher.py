"""Tests for the cruncher pipeline — clean, snapshot, adapter, streaming."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from domain.training._internal.cruncher import (
    Cruncher,
    DedupIndex,
    SnapshotError,
    SnapshotStore,
    conversation_records,
    estimate_jaccard,
    quality_ok,
    resolve_latest,
    resolve_snapshot,
    scrub_pii,
    text_fingerprint,
    trainer_data_path,
)
from domain.training._internal.dataset_manifest import ManifestError

GOOD_TEXT = (
    "The mitochondria is the powerhouse of the cell. It converts nutrients "
    "into ATP through oxidative phosphorylation, a process studied extensively "
    "in molecular biology since the mid twentieth century."
)

GOOD_TEXT_2 = (
    "Photosynthesis converts sunlight into chemical energy stored in glucose. "
    "Chloroplasts in plant cells contain chlorophyll, the pigment that absorbs "
    "light most efficiently in the blue and red portions of the spectrum."
)


# ── scrub_pii ──────────────────────────────────────────────────────────


class TestScrubPII:
    def test_email(self):
        out, n = scrub_pii("contact me at user@example.com please")
        assert "<EMAIL>" in out
        assert "user@example.com" not in out
        assert n == 1

    def test_phone(self):
        out, n = scrub_pii("call 555-123-4567 today")
        assert "<PHONE>" in out
        assert n == 1

    def test_ssn_beats_phone(self):
        # SSN pattern must win over the phone pattern for 123-45-6789.
        out, n = scrub_pii("ssn 123-45-6789")
        assert "<SSN>" in out
        assert n == 1

    def test_ipv4(self):
        out, n = scrub_pii("ping 192.168.1.1 now")
        assert "<IPV4>" in out
        assert n == 1

    def test_clean_text_untouched(self):
        out, n = scrub_pii(GOOD_TEXT)
        assert out == GOOD_TEXT
        assert n == 0


# ── quality gate ───────────────────────────────────────────────────────


class TestQuality:
    def test_good_text_passes(self):
        assert quality_ok(GOOD_TEXT)

    def test_too_short_dropped(self):
        assert not quality_ok("hi")

    def test_repetition_dropped(self):
        spam = "buy now " * 200
        assert not quality_ok(spam)

    def test_non_alpha_blob_dropped(self):
        blob = "a1b2c3d4e5 " * 50  # hex/base64-ish dump
        assert not quality_ok(blob)


# ── MinHash dedup ──────────────────────────────────────────────────────


class TestDedup:
    def test_identical_is_duplicate(self):
        idx = DedupIndex()
        sig = text_fingerprint(GOOD_TEXT)
        idx.add(sig)
        assert idx.is_near_duplicate(text_fingerprint(GOOD_TEXT))

    def test_different_not_duplicate(self):
        idx = DedupIndex()
        idx.add(text_fingerprint(GOOD_TEXT))
        assert not idx.is_near_duplicate(text_fingerprint(GOOD_TEXT_2))

    def test_near_duplicate_detected(self):
        # A one-word edit keeps nearly all 5-grams -> jaccard > 0.85.
        base = GOOD_TEXT
        variant = GOOD_TEXT.replace("twentieth", "twenty-first")
        sig_a, sig_b = text_fingerprint(base), text_fingerprint(variant)
        assert estimate_jaccard(sig_a, sig_b) >= 0.85
        idx = DedupIndex()
        idx.add(sig_a)
        assert idx.is_near_duplicate(sig_b)

    def test_fingerprint_deterministic(self):
        assert np.array_equal(text_fingerprint(GOOD_TEXT), text_fingerprint(GOOD_TEXT))

    def test_state_roundtrip(self):
        idx = DedupIndex()
        idx.add(text_fingerprint(GOOD_TEXT))
        idx.add(text_fingerprint(GOOD_TEXT_2))
        restored = DedupIndex.from_state(idx.to_state())
        assert len(restored) == 2
        assert restored.is_near_duplicate(text_fingerprint(GOOD_TEXT))
        assert not restored.is_near_duplicate(
            text_fingerprint(
                "Quantum entanglement enables secure key distribution between "
                "distant parties without revealing the key itself to observers."
            )
        )


# ── snapshot store ─────────────────────────────────────────────────────


class TestSnapshotStore:
    def test_seal_and_load(self, tmp_path: Path):
        store = SnapshotStore(tmp_path)
        ref = store.seal("demo", [GOOD_TEXT, GOOD_TEXT_2], sources=["unit-test"])
        assert ref.version == 1
        assert ref.train_path.read_text(encoding="utf-8") == GOOD_TEXT + "\n" + GOOD_TEXT_2 + "\n"
        manifest = store.load_manifest(ref)
        assert manifest["dataset_id"] == "demo"
        assert manifest["version"] == 1
        assert manifest["domain"] == "general"
        assert manifest["splits"]["train"] == "train.txt"
        assert manifest["mix"] == {}

    def test_versions_bump_and_immutability(self, tmp_path: Path):
        store = SnapshotStore(tmp_path)
        r1 = store.seal("demo", [GOOD_TEXT], sources=["a"])
        v1_bytes = r1.train_path.read_bytes()
        r2 = store.seal("demo", [GOOD_TEXT_2], sources=["b"])
        assert r2.version == 2
        assert store.versions("demo") == [1, 2]
        # v1 untouched by the v2 seal.
        assert r1.train_path.read_bytes() == v1_bytes

    def test_empty_seal_rejected(self, tmp_path: Path):
        store = SnapshotStore(tmp_path)
        with pytest.raises(SnapshotError, match="no records"):
            store.seal("demo", [], sources=["x"])

    def test_latest(self, tmp_path: Path):
        store = SnapshotStore(tmp_path)
        assert store.latest("demo") is None
        store.seal("demo", [GOOD_TEXT], sources=["a"])
        ref2 = store.seal("demo", [GOOD_TEXT_2], sources=["b"])
        assert store.latest("demo").version == ref2.version == 2


# ── adapter ────────────────────────────────────────────────────────────


class TestAdapter:
    def test_resolve_snapshot(self, tmp_path: Path):
        store = SnapshotStore(tmp_path)
        ref = store.seal("demo", [GOOD_TEXT], sources=["unit"])
        path, manifest = resolve_snapshot(ref)
        assert path == ref.train_path
        assert path.is_file()
        assert manifest["dataset_id"] == "demo"

    def test_resolve_latest(self, tmp_path: Path):
        store = SnapshotStore(tmp_path)
        store.seal("demo", [GOOD_TEXT], sources=["unit"])
        store.seal("demo", [GOOD_TEXT_2], sources=["unit"])
        path, manifest = resolve_latest("demo", store=store)
        assert manifest["version"] == 2

    def test_resolve_latest_missing_dataset(self, tmp_path: Path):
        store = SnapshotStore(tmp_path)
        with pytest.raises(ManifestError, match="no sealed snapshot"):
            resolve_latest("missing", store=store)

    def test_trainer_data_path_validates(self, tmp_path: Path):
        store = SnapshotStore(tmp_path)
        ref = store.seal("demo", [GOOD_TEXT], sources=["unit"])
        assert trainer_data_path(ref) == str(ref.manifest_path)


# ── pipeline (streaming) ───────────────────────────────────────────────


class TestCruncher:
    def test_stream_then_seal(self, tmp_path: Path):
        store = SnapshotStore(tmp_path)
        cruncher = Cruncher("live", store=store)
        result = cruncher.crunch_stream([GOOD_TEXT, GOOD_TEXT_2], source="unit")
        assert result.records_seen == 2
        assert result.records_kept == 2
        ref = cruncher.seal()
        assert ref.version == 1
        assert GOOD_TEXT in ref.train_path.read_text(encoding="utf-8")

    def test_stream_dedup_within_run(self, tmp_path: Path):
        cruncher = Cruncher("live", store=SnapshotStore(tmp_path))
        result = cruncher.crunch_stream([GOOD_TEXT, GOOD_TEXT, GOOD_TEXT_2])
        assert result.records_kept == 2
        assert result.dropped_duplicate == 1

    def test_cross_snapshot_dedup(self, tmp_path: Path):
        store = SnapshotStore(tmp_path)
        c1 = Cruncher("live", store=store)
        c1.crunch_stream([GOOD_TEXT])
        c1.seal()

        # New cruncher run: dedup state restored from sealed v1.
        c2 = Cruncher("live", store=store)
        result = c2.crunch_stream([GOOD_TEXT, GOOD_TEXT_2])
        assert result.dropped_duplicate == 1
        assert result.records_kept == 1
        ref = c2.seal()
        assert ref.version == 2

    def test_incremental_stream_dedup_against_history(self, tmp_path: Path):
        store = SnapshotStore(tmp_path)
        c1 = Cruncher("live", store=store)
        c1.crunch_stream([GOOD_TEXT])
        c1.seal()

        c2 = Cruncher("live", store=store)
        seen = []
        # Feed records one at a time — the live contract.
        for rec in [GOOD_TEXT_2, GOOD_TEXT]:
            out = c2.crunch_stream([rec])
            seen.append(out.records_kept)
        assert seen == [1, 1]  # second record kept-count did not increment
        assert c2.result.dropped_duplicate == 1

    def test_quality_dropped(self, tmp_path: Path):
        cruncher = Cruncher("live", store=SnapshotStore(tmp_path))
        result = cruncher.crunch_stream(["ok " + GOOD_TEXT, "buy now " * 100])
        assert result.dropped_quality == 1
        assert result.records_kept == 1

    def test_seal_empty_buffer_rejected(self, tmp_path: Path):
        cruncher = Cruncher("live", store=SnapshotStore(tmp_path))
        with pytest.raises(SnapshotError, match="buffer is empty"):
            cruncher.seal()

    def test_crunch_batch_convenience(self, tmp_path: Path):
        store = SnapshotStore(tmp_path)
        cruncher = Cruncher("batch", store=store)
        ref = cruncher.crunch([GOOD_TEXT, GOOD_TEXT_2], source="unit", sources=["unit"])
        assert ref.version == 1
        assert "unit" in store.load_manifest(ref)["sources"]

    def test_rejects_non_str_record(self, tmp_path: Path):
        cruncher = Cruncher("live", store=SnapshotStore(tmp_path))
        with pytest.raises(TypeError, match="expects str"):
            cruncher.crunch_stream([b"bytes"])  # type: ignore[list-item]

    def test_manifest_stats_recorded(self, tmp_path: Path):
        store = SnapshotStore(tmp_path)
        cruncher = Cruncher("live", store=store)
        cruncher.crunch_stream([GOOD_TEXT, GOOD_TEXT, "x" * 5])
        ref = cruncher.seal()
        manifest = store.load_manifest(ref)
        stats = manifest["cruncher"]["stats"]
        assert stats["records_seen"] == 3
        assert stats["records_kept"] == 1
        assert stats["dropped_duplicate"] == 1
        assert stats["dropped_quality"] == 1


# ── live corpus source ─────────────────────────────────────────────────


class TestConversationRecords:
    def test_yields_assistant_turns(self, tmp_path: Path):
        corpus = tmp_path / "corpus.jsonl"
        corpus.write_text(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": "hi"},
                        {"role": "assistant", "content": "hello there"},
                    ]
                }
            )
            + "\n"
            + json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": "bye"},
                        {"role": "assistant", "content": "goodbye friend"},
                    ]
                }
            )
            + "\n",
            encoding="utf-8",
        )
        records = list(conversation_records(corpus))
        assert records == ["hello there", "goodbye friend"]

    def test_skips_bad_lines(self, tmp_path: Path):
        corpus = tmp_path / "corpus.jsonl"
        corpus.write_text(
            "not json\n"
            + json.dumps({"messages": [{"role": "assistant", "content": "kept"}]})
            + "\n",
            encoding="utf-8",
        )
        assert list(conversation_records(corpus)) == ["kept"]

    def test_missing_corpus_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            list(conversation_records(tmp_path / "nope.jsonl"))

    def test_end_to_end_live_source(self, tmp_path: Path):
        corpus = tmp_path / "corpus.jsonl"
        corpus.write_text(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": "q"},
                        {"role": "assistant", "content": GOOD_TEXT},
                    ]
                }
            )
            + "\n",
            encoding="utf-8",
        )
        store = SnapshotStore(tmp_path / "crunched")
        cruncher = Cruncher("api-conversations", store=store)
        cruncher.crunch_stream(conversation_records(corpus), source="corpus.jsonl")
        ref = cruncher.seal()
        assert GOOD_TEXT in ref.train_path.read_text(encoding="utf-8")
        path, _ = resolve_snapshot(ref)
        assert path.is_file()
