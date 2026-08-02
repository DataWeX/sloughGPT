"""Tests for domains/training/dataset.py TrainingDataset."""

import pytest

from domains.training.dataset import TrainingDataset


SHORT_TEXT = "tiny"


def _long_text(words: int = 400) -> str:
    return " ".join(f"word{i}" for i in range(words))


class TestBuild:
    def test_short_text_no_chunks(self):
        ds = TrainingDataset(SHORT_TEXT, chunk_size=400)
        assert ds.chunks == []
        assert ds.num_chunks == 0
        assert ds.store.count_sync() == 0

    def test_empty_text_no_chunks(self):
        ds = TrainingDataset("", chunk_size=400)
        assert ds.chunks == []

    def test_whitespace_text_no_chunks(self):
        ds = TrainingDataset("   \n  ", chunk_size=400)
        assert ds.chunks == []

    def test_single_chunk_within_limit(self):
        text = "This is a sufficiently long sentence to form a single chunk."
        ds = TrainingDataset(text, chunk_size=400)
        assert ds.num_chunks == 1
        assert ds.chunks[0] == text.strip()

    def test_multi_chunk_overlap(self):
        text = _long_text()
        ds = TrainingDataset(text, chunk_size=400, chunk_overlap=50)
        assert ds.num_chunks == 9
        assert all(len(c) > 20 for c in ds.chunks)
        assert ds.store.count_sync() == ds.num_chunks

    def test_store_indexes_all_chunks(self):
        text = _long_text()
        ds = TrainingDataset(text, chunk_size=400)
        assert ds.store.count_sync() == ds.num_chunks

    def test_source_attributes(self):
        text = _long_text()
        ds = TrainingDataset(text, chunk_size=400)
        assert ds.source_text == text
        assert ds.chunk_size == 400
        assert ds.chunk_overlap == 50

    def test_num_chunks_and_source_length(self):
        text = _long_text()
        ds = TrainingDataset(text)
        assert ds.num_chunks == len(ds.chunks)
        assert ds.source_length == len(text)

    def test_repr(self):
        ds = TrainingDataset(_long_text())
        assert repr(ds) == f"TrainingDataset(chunks={ds.num_chunks}, source_len={ds.source_length})"


class TestTeacherContext:
    def test_empty_store_returns_empty(self):
        ds = TrainingDataset(SHORT_TEXT)
        assert ds.get_teacher_context("anything") == ""

    def test_retrieves_matching_chunk(self):
        text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
        ds = TrainingDataset(text, chunk_size=400)
        context = ds.get_teacher_context("alpha beta gamma delta", min_score=0.1)
        assert context != ""
        assert "alpha" in context

    def test_falls_back_to_top_result_when_below_min_score(self):
        text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
        ds = TrainingDataset(text, chunk_size=400)
        context = ds.get_teacher_context("alpha beta gamma delta", min_score=0.99)
        assert context != ""

    def test_joins_multiple_relevant(self):
        text = ("The quick brown fox jumps over the lazy dog and runs fast. "
                "Another very distinct sentence about completely unrelated things here. "
                "Yet one more entirely separate statement to make the chunk long enough. "
                "Four more words to ensure the chunk survives the filter. Five words.")
        ds = TrainingDataset(text, chunk_size=400)
        context = ds.get_teacher_context("The quick brown fox", min_score=0.05)
        assert context != ""
        assert isinstance(context, str)


class TestStudent:
    def test_get_student_text(self):
        text = _long_text()
        ds = TrainingDataset(text)
        assert ds.get_student_text() == text

    def test_pairs_split_on_sentences(self):
        text = "First sentence here. Second sentence here. Third sentence here."
        ds = TrainingDataset(text, chunk_size=400)
        pairs = ds.get_student_pairs()
        assert pairs
        assert pairs[0]["user_msg"] == "First sentence here."
        assert pairs[0]["assistant_msg"] == "Second sentence here. Third sentence here."

    def test_pairs_fallback_for_single_sentence(self):
        text = "A single very long sentence without any period to split it apart anywhere"
        ds = TrainingDataset(text, chunk_size=400)
        pairs = ds.get_student_pairs()
        assert pairs
        assert pairs[0]["user_msg"] == "Tell me about this topic."

    def test_pairs_filter_short_assistant(self):
        ds = TrainingDataset(SHORT_TEXT)
        assert ds.get_student_pairs() == []


class TestFactories:
    def test_from_file_success(self, tmp_path):
        p = tmp_path / "data.txt"
        p.write_text(_long_text(), encoding="utf-8")
        ds = TrainingDataset.from_file(str(p))
        assert ds.source_length == len(_long_text())

    def test_from_file_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            TrainingDataset.from_file(str(tmp_path / "nope.txt"))

    def test_from_file_too_short(self, tmp_path):
        p = tmp_path / "data.txt"
        p.write_text("too short", encoding="utf-8")
        with pytest.raises(ValueError):
            TrainingDataset.from_file(str(p))

    def test_from_text_success(self):
        ds = TrainingDataset.from_text(_long_text())
        assert ds.num_chunks > 0

    def test_from_text_too_short(self):
        with pytest.raises(ValueError):
            TrainingDataset.from_text("too short")
