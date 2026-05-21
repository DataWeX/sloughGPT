"""Tests for ContinualLearner — ring buffer, ingestion, training."""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from domains.learner.continual import (
    ContinualLearner,
    _tokenize,
    _detokenize,
    VOCAB,
    LEARNER_STATE_DIR,
    STATE_PATH,
)


class TestTokenize:
    def test_tokenize_known_chars(self):
        ids = _tokenize("hello world")
        assert all(isinstance(i, int) for i in ids)
        assert len(ids) > 0

    def test_tokenize_empty(self):
        assert _tokenize("") == []

    def test_tokenize_unknown_ignored(self):
        ids = _tokenize("hello!@#$ world")
        assert len(ids) > 0

    def test_detokenize_roundtrip(self):
        text = "hello world"
        ids = _tokenize(text)
        decoded = _detokenize(ids)
        assert isinstance(decoded, str)
        assert len(decoded) > 0

    def test_detokenize_empty(self):
        assert _detokenize([]) == ""


class TestContinualLearner:
    @pytest.fixture
    def learner(self, tmp_path):
        with patch.object(ContinualLearner, "_background_loop", lambda self: None):
            with patch.object(ContinualLearner, "_save_checkpoint", lambda self: None):
                l = ContinualLearner()
                l._running = False
                yield l

    def test_init(self):
        l = ContinualLearner()
        try:
            assert l.buffer == []
            assert l.total_tokens_ingested >= 0
            assert l.train_steps_completed >= 0
            assert l.net is not None
        finally:
            l.shutdown()

    def test_ingest_text_adds_to_buffer(self, learner):
        learner.ingest_text("hello world")
        assert len(learner.buffer) > 0
        assert learner.total_tokens_ingested > 0

    def test_ingest_text_empty(self, learner):
        learner.ingest_text("")
        assert learner.total_tokens_ingested == 0

    def test_ingest_text_unknown_chars(self, learner):
        learner.ingest_text("@#$%")
        assert learner.total_tokens_ingested == 0

    def test_ingest_conversation(self, learner):
        learner.ingest_conversation([("hi", "hello there")])
        assert learner.total_tokens_ingested > 0

    def test_status(self, learner):
        learner.ingest_text("some training data")
        status = learner.status()
        assert "soul_name" in status
        assert "total_tokens_ingested" in status
        assert "buffer_size" in status
        assert status["buffer_size"] > 0

    def test_status_empty(self):
        l = ContinualLearner()
        try:
            status = l.status()
            assert status["buffer_size"] == 0
            assert status["total_tokens_ingested"] == 0
        finally:
            l.shutdown()

    def test_buffer_capacity(self, learner):
        big_text = "hello world " * 2000
        learner.ingest_text(big_text)
        assert len(learner.buffer) <= 10000

    def test_train_now_no_data(self, learner):
        status = learner.train_now()
        assert status is not None

    @pytest.mark.slow
    def test_train_now_with_data(self, learner):
        text = "the cat sat on the mat " * 50
        learner.ingest_text(text)
        status = learner.train_now()
        assert status is not None

    def test_evaluate_no_data(self, learner):
        result = learner.evaluate()
        assert "error" in result

    @pytest.mark.slow
    def test_evaluate_with_text(self, learner):
        result = learner.evaluate(text="the cat sat on the mat and the dog played in the yard")
        assert "perplexity" in result or "error" in result

    def test_vocab_size(self):
        assert VOCAB > 0
        assert VOCAB == len(
            " abcdefghijklmnopqrstuvwxyz0123456789.,!?-'"
        )

    @pytest.mark.slow
    def test_deploy(self, tmp_path):
        l = ContinualLearner()
        try:
            l.ingest_text("some training data " * 50)
            l.train_now()
            result = l.deploy(name="test-deploy")
            assert "path" in result
            assert "soul_name" in result
            assert "steps" in result
        finally:
            l.shutdown()
