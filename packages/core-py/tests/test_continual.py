"""Tests for ContinualLearner — tokenization helpers, buffer management, training, eval.

Covers:
  - _tokenize / _detokenize round-trip
  - CHAR_SET, STOI, ITOS consistency
  - ContinualLearner.ingest_text (buffer, total_tokens_ingested)
  - ContinualLearner.ingest_conversation
  - ContinualLearner.status
  - ContinualLearner.evaluate (short text, empty buffer)
  - ContinualLearner.deploy
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from domains.learner.continual import (
    _tokenize,
    _detokenize,
    CHAR_SET,
    STOI,
    ITOS,
    VOCAB,
    ContinualLearner,
    TRAIN_SEQ_LEN,
    BUFFER_CAPACITY,
)
from domains.learner.knowledge import get_knowledge_ingestor, get_knowledge_memory


class TestTokenizeDetokenize:
    def test_tokenize_returns_list(self):
        ids = _tokenize("hello")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_tokenize_length(self):
        text = "abc"
        ids = _tokenize(text)
        assert len(ids) == 3

    def test_tokenize_unknown_char_filtered(self):
        # '~' is not in CHAR_SET, filtered out by 'if c in STOI'
        ids = _tokenize("~")
        assert ids == []

    def test_detokenize_returns_string(self):
        text = _detokenize([1, 2, 3])
        assert isinstance(text, str)

    def test_roundtrip(self):
        text = "hello world"
        ids = _tokenize(text)
        decoded = _detokenize(ids)
        assert decoded == text.lower()

    def test_roundtrip_numbers(self):
        text = "test 123"
        ids = _tokenize(text)
        decoded = _detokenize(ids)
        assert decoded == text.lower()


class TestCharConstants:
    def test_stoi_itos_consistency(self):
        for ch, idx in STOI.items():
            assert ITOS[idx] == ch

    def test_special_tokens_at_start(self):
        assert STOI.get("<BOS>") == 0 or "<BOS>" not in STOI
        assert STOI.get("<EOS>") == 1 or "<EOS>" not in STOI
        assert STOI.get("<PAD>") == 2 or "<PAD>" not in STOI
        assert STOI.get("<UNK>") == 3 or "<UNK>" not in STOI

    def test_vocab_matches_char_set(self):
        assert VOCAB == len(CHAR_SET)

    def test_space_in_vocab(self):
        assert " " in STOI


class TestContinualLearner:
    @patch("domains.learner.knowledge.get_knowledge_memory")
    @patch("domains.learner.knowledge.get_knowledge_ingestor")
    def test_init(self, mock_knowledge, mock_ingestor):
        mock_ingestor_inst = MagicMock()
        mock_ingestor.return_value = mock_ingestor_inst
        mock_knowledge_inst = MagicMock()
        mock_knowledge.return_value = mock_knowledge_inst

        learner = ContinualLearner(n_embed=32, n_layer=2, n_head=2)
        assert learner.n_embed == 32
        assert learner.n_layer == 2
        assert learner.total_tokens_ingested == 0
        assert learner.train_steps_completed == 0
        learner._running = False

    @patch("domains.learner.knowledge.get_knowledge_memory")
    @patch("domains.learner.knowledge.get_knowledge_ingestor")
    def test_ingest_text(self, mock_knowledge, mock_ingestor):
        mock_ingestor.return_value = MagicMock()
        mock_knowledge.return_value = MagicMock()

        learner = ContinualLearner(n_embed=32, n_layer=2, n_head=2)
        learner.ingest_text("hello world")
        assert learner.total_tokens_ingested == len("hello world")
        assert len(learner.buffer) == len("hello world")
        learner._running = False

    @patch("domains.learner.knowledge.get_knowledge_memory")
    @patch("domains.learner.knowledge.get_knowledge_ingestor")
    def test_ingest_text_empty(self, mock_knowledge, mock_ingestor):
        mock_ingestor.return_value = MagicMock()
        mock_knowledge.return_value = MagicMock()

        learner = ContinualLearner(n_embed=32, n_layer=2, n_head=2)
        learner.ingest_text("")
        assert learner.total_tokens_ingested == 0
        learner._running = False

    @patch("domains.learner.knowledge.get_knowledge_memory")
    @patch("domains.learner.knowledge.get_knowledge_ingestor")
    def test_ingest_text_buffer_capped(self, mock_knowledge, mock_ingestor):
        mock_ingestor.return_value = MagicMock()
        mock_knowledge.return_value = MagicMock()

        learner = ContinualLearner(n_embed=32, n_layer=2, n_head=2)
        # Ingest more than BUFFER_CAPACITY
        text = "a" * (BUFFER_CAPACITY + 100)
        learner.ingest_text(text)
        assert len(learner.buffer) <= BUFFER_CAPACITY
        learner._running = False

    @patch("domains.learner.knowledge.get_knowledge_memory")
    @patch("domains.learner.knowledge.get_knowledge_ingestor")
    def test_ingest_conversation(self, mock_knowledge, mock_ingestor):
        mock_ingestor.return_value = MagicMock()
        mock_knowledge.return_value = MagicMock()

        learner = ContinualLearner(n_embed=32, n_layer=2, n_head=2)
        learner.ingest_conversation([("hi", "hello"), ("bye", "goodbye")])
        assert learner.total_tokens_ingested > 0
        learner._running = False

    @patch("domains.learner.knowledge.get_knowledge_memory")
    @patch("domains.learner.knowledge.get_knowledge_ingestor")
    def test_status(self, mock_knowledge, mock_ingestor):
        mock_ingestor_inst = MagicMock()
        mock_ingestor.return_value = mock_ingestor_inst
        mock_knowledge_inst = MagicMock()
        mock_knowledge.return_value = mock_knowledge_inst
        mock_knowledge_inst.stats.return_value = {"total_facts": 0}
        mock_ingestor_inst.list_feeds.return_value = []
        mock_ingestor_inst.filter.get_stats.return_value = {}
        mock_ingestor_inst.filter.get_config.return_value = {}

        learner = ContinualLearner(n_embed=32, n_layer=2, n_head=2)
        status = learner.status()
        assert "soul_name" in status
        assert "buffer_size" in status
        assert "total_tokens_ingested" in status
        assert status["arch"] == "transformer"
        learner._running = False

    @patch("domains.learner.knowledge.get_knowledge_memory")
    @patch("domains.learner.knowledge.get_knowledge_ingestor")
    def test_evaluate_short_text(self, mock_knowledge, mock_ingestor):
        mock_ingestor.return_value = MagicMock()
        mock_knowledge.return_value = MagicMock()

        learner = ContinualLearner(n_embed=32, n_layer=2, n_head=2)
        result = learner.evaluate("short")
        assert result["error"] == "text too short"
        learner._running = False

    @patch("domains.learner.knowledge.get_knowledge_memory")
    @patch("domains.learner.knowledge.get_knowledge_ingestor")
    def test_evaluate_empty_buffer(self, mock_knowledge, mock_ingestor):
        mock_ingestor.return_value = MagicMock()
        mock_knowledge.return_value = MagicMock()

        learner = ContinualLearner(n_embed=32, n_layer=2, n_head=2)
        result = learner.evaluate()
        assert result["error"] == "buffer too small"
        learner._running = False

    @patch("domains.learner.knowledge.get_knowledge_memory")
    @patch("domains.learner.knowledge.get_knowledge_ingestor")
    def test_evaluate_with_text(self, mock_knowledge, mock_ingestor):
        mock_ingestor.return_value = MagicMock()
        mock_knowledge.return_value = MagicMock()

        learner = ContinualLearner(n_embed=32, n_layer=2, n_head=2)
        text = "the quick brown fox jumps over the lazy dog " * 10
        result = learner.evaluate(text)
        assert "loss" in result
        assert "perplexity" in result
        assert result["eval_tokens"] > 0
        learner._running = False

    @patch("domains.learner.knowledge.get_knowledge_memory")
    @patch("domains.learner.knowledge.get_knowledge_ingestor")
    def test_deploy(self, mock_knowledge, mock_ingestor, tmp_path):
        mock_ingestor.return_value = MagicMock()
        mock_knowledge.return_value = MagicMock()

        learner = ContinualLearner(n_embed=32, n_layer=2, n_head=2)
        import domains.learner.continual as mod
        old_home = mod.Path.home
        try:
            mod.Path.home = lambda: tmp_path
            result = learner.deploy("test-deploy")
            assert result["soul_name"] == "continual"
            assert result["arch"] == "transformer"
            assert result["file_size"] > 0
        finally:
            mod.Path.home = old_home
        learner._running = False
