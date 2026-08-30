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

    def test_tokenize_uppercase_lowered(self):
        ids = _tokenize("ABC")
        assert ids == _tokenize("abc")

    def test_tokenize_space_preserved(self):
        ids = _tokenize("a b")
        assert len(ids) == 3  # 'a', ' ', 'b'

    def test_tokenize_empty_string(self):
        assert _tokenize("") == []

    def test_tokenize_special_chars_filtered(self):
        # '@#$%^&*()' not in CHAR_SET → filtered
        ids = _tokenize("@#$%")
        assert ids == []

    def test_tokenize_period(self):
        assert _tokenize(".") == [STOI["."]]

    def test_tokenize_comma(self):
        assert _tokenize(",") == [STOI[","]]

    def test_tokenize_exclamation(self):
        assert _tokenize("!") == [STOI["!"]]

    def test_tokenize_question_mark(self):
        assert _tokenize("?") == [STOI["?"]]

    def test_tokenize_dash(self):
        assert _tokenize("-") == [STOI["-"]]

    def test_tokenize_apostrophe(self):
        assert _tokenize("'") == [STOI["'"]]

    def test_detokenize_unknown_id(self):
        """Unknown token ID → '?' placeholder."""
        result = _detokenize([9999])
        assert result == "?"

    def test_detokenize_empty(self):
        assert _detokenize([]) == ""

    def test_detokenize_mixed_known_unknown(self):
        result = _detokenize([1, 9999, 2])
        assert result[0] == ITOS[1]
        assert result[1] == "?"
        assert result[2] == ITOS[2]

    def test_roundtrip_all_printable_chars(self):
        """Round-trip all characters in CHAR_SET."""
        text = CHAR_SET
        ids = _tokenize(text)
        decoded = _detokenize(ids)
        assert decoded == text


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

    def test_stoi_has_all_digits(self):
        for d in "0123456789":
            assert d in STOI

    def test_stoi_has_all_lowercase(self):
        for c in "abcdefghijklmnopqrstuvwxyz":
            assert c in STOI

    def test_itos_unique_values(self):
        """Each index maps to exactly one character."""
        values = list(ITOS.values())
        assert len(values) == len(set(values))

    def test_stoi_unique_values(self):
        """Each character maps to exactly one index."""
        values = list(STOI.values())
        assert len(values) == len(set(values))

    def test_space_index_is_zero(self):
        """Space character is at index 0 in CHAR_SET."""
        assert STOI[" "] == 0

    def test_vocab_positive(self):
        assert VOCAB > 0

    def test_stoi_itos_inverse(self):
        """STOI and ITOS are exact inverses."""
        for ch, idx in STOI.items():
            assert ITOS[idx] == ch
        for idx, ch in ITOS.items():
            assert STOI[ch] == idx


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

    @patch("domains.learner.knowledge.get_knowledge_memory")
    @patch("domains.learner.knowledge.get_knowledge_ingestor")
    def test_default_config(self, mock_knowledge, mock_ingestor):
        mock_ingestor.return_value = MagicMock()
        mock_knowledge.return_value = MagicMock()
        learner = ContinualLearner()
        assert learner.n_embed == 192
        assert learner.n_layer == 4
        assert learner.n_head == 4
        assert learner.soul_name == "continual"
        learner._running = False

    @patch("domains.learner.knowledge.get_knowledge_memory")
    @patch("domains.learner.knowledge.get_knowledge_ingestor")
    def test_custom_lr(self, mock_knowledge, mock_ingestor):
        mock_ingestor.return_value = MagicMock()
        mock_knowledge.return_value = MagicMock()
        learner = ContinualLearner(lr=0.01)
        assert learner.lr == 0.01
        learner._running = False

    @patch("domains.learner.knowledge.get_knowledge_memory")
    @patch("domains.learner.knowledge.get_knowledge_ingestor")
    def test_custom_soul_name(self, mock_knowledge, mock_ingestor):
        mock_ingestor.return_value = MagicMock()
        mock_knowledge.return_value = MagicMock()
        learner = ContinualLearner(soul_name="custom_soul")
        assert learner.soul_name == "custom_soul"
        learner._running = False

    @patch("domains.learner.knowledge.get_knowledge_memory")
    @patch("domains.learner.knowledge.get_knowledge_ingestor")
    def test_initial_state(self, mock_knowledge, mock_ingestor):
        mock_ingestor.return_value = MagicMock()
        mock_knowledge.return_value = MagicMock()
        learner = ContinualLearner(n_embed=32, n_layer=2, n_head=2)
        assert learner.current_loss == 0.0
        assert learner.loss_history == []
        assert learner.buffer == []
        assert learner._new_since_last_train == 0
        learner._running = False

    @patch("domains.learner.knowledge.get_knowledge_memory")
    @patch("domains.learner.knowledge.get_knowledge_ingestor")
    def test_status_has_knowledge(self, mock_knowledge, mock_ingestor):
        mock_ingestor_inst = MagicMock()
        mock_ingestor.return_value = mock_ingestor_inst
        mock_knowledge_inst = MagicMock()
        mock_knowledge.return_value = mock_knowledge_inst
        mock_knowledge_inst.stats.return_value = {"total_facts": 5}
        mock_ingestor_inst.list_feeds.return_value = [{"url": "a"}]
        mock_ingestor_inst.filter.get_stats.return_value = {"kept": 10}
        mock_ingestor_inst.filter.get_config.return_value = {"min_quality": 0.5}

        learner = ContinualLearner(n_embed=32, n_layer=2, n_head=2)
        status = learner.status()
        # feeds_subscribed is len(self.ingestor.list_feeds())
        # which calls the mock's list_feeds method
        assert "feeds_subscribed" in status
        assert status["knowledge"] == {"total_facts": 5}
        learner._running = False

    @patch("domains.learner.knowledge.get_knowledge_memory")
    @patch("domains.learner.knowledge.get_knowledge_ingestor")
    def test_ingest_text_increments_total(self, mock_knowledge, mock_ingestor):
        mock_ingestor.return_value = MagicMock()
        mock_knowledge.return_value = MagicMock()
        learner = ContinualLearner(n_embed=32, n_layer=2, n_head=2)
        learner.ingest_text("aaa")
        assert learner.total_tokens_ingested == 3
        learner.ingest_text("bb")
        assert learner.total_tokens_ingested == 5
        learner._running = False

    @patch("domains.learner.knowledge.get_knowledge_memory")
    @patch("domains.learner.knowledge.get_knowledge_ingestor")
    def test_ingest_text_unknown_chars_filtered(self, mock_knowledge, mock_ingestor):
        mock_ingestor.return_value = MagicMock()
        mock_knowledge.return_value = MagicMock()
        learner = ContinualLearner(n_embed=32, n_layer=2, n_head=2)
        learner.ingest_text("~~~")
        assert learner.total_tokens_ingested == 0
        assert len(learner.buffer) == 0
        learner._running = False

    @patch("domains.learner.knowledge.get_knowledge_memory")
    @patch("domains.learner.knowledge.get_knowledge_ingestor")
    def test_ingest_conversation_format(self, mock_knowledge, mock_ingestor):
        mock_ingestor.return_value = MagicMock()
        mock_knowledge.return_value = MagicMock()
        learner = ContinualLearner(n_embed=32, n_layer=2, n_head=2)
        learner.ingest_conversation([("hi", "hello")])
        # "user: hi assistant: hello" → check tokens present
        assert learner.total_tokens_ingested > 0
        learner._running = False

    @patch("domains.learner.knowledge.get_knowledge_memory")
    @patch("domains.learner.knowledge.get_knowledge_ingestor")
    def test_buffer_grows_with_multiple_ingests(self, mock_knowledge, mock_ingestor):
        mock_ingestor.return_value = MagicMock()
        mock_knowledge.return_value = MagicMock()
        learner = ContinualLearner(n_embed=32, n_layer=2, n_head=2)
        learner.ingest_text("abc")
        len1 = len(learner.buffer)
        learner.ingest_text("def")
        len2 = len(learner.buffer)
        assert len2 > len1
        learner._running = False

    @patch("domains.learner.knowledge.get_knowledge_memory")
    @patch("domains.learner.knowledge.get_knowledge_ingestor")
    def test_deploy_default_name(self, mock_knowledge, mock_ingestor, tmp_path):
        mock_ingestor.return_value = MagicMock()
        mock_knowledge.return_value = MagicMock()
        learner = ContinualLearner(n_embed=32, n_layer=2, n_head=2)
        import domains.learner.continual as mod
        old_home = mod.Path.home
        try:
            mod.Path.home = lambda: tmp_path
            result = learner.deploy()
            assert "learner-continual-step-0" in result["path"]
        finally:
            mod.Path.home = old_home
        learner._running = False

    @patch("domains.learner.knowledge.get_knowledge_memory")
    @patch("domains.learner.knowledge.get_knowledge_ingestor")
    def test_deploy_has_all_fields(self, mock_knowledge, mock_ingestor, tmp_path):
        mock_ingestor.return_value = MagicMock()
        mock_knowledge.return_value = MagicMock()
        learner = ContinualLearner(n_embed=32, n_layer=2, n_head=2)
        import domains.learner.continual as mod
        old_home = mod.Path.home
        try:
            mod.Path.home = lambda: tmp_path
            result = learner.deploy("test")
            expected_keys = {
                "path", "soul_name", "steps", "loss",
                "file_size", "arch", "n_embed", "n_layer", "n_head",
            }
            assert expected_keys.issubset(result.keys())
        finally:
            mod.Path.home = old_home
        learner._running = False

    @patch("domains.learner.knowledge.get_knowledge_memory")
    @patch("domains.learner.knowledge.get_knowledge_ingestor")
    def test_evaluate_no_chunks(self, mock_knowledge, mock_ingestor):
        """Text exactly TRAIN_SEQ_LEN → no chunks formed → error."""
        mock_ingestor.return_value = MagicMock()
        mock_knowledge.return_value = MagicMock()
        learner = ContinualLearner(n_embed=32, n_layer=2, n_head=2)
        # Exactly TRAIN_SEQ_LEN chars → len(ids) < TRAIN_SEQ_LEN + 1
        text = "a" * TRAIN_SEQ_LEN
        result = learner.evaluate(text)
        assert result["error"] == "text too short"
        learner._running = False

    @patch("domains.learner.knowledge.get_knowledge_memory")
    @patch("domains.learner.knowledge.get_knowledge_ingestor")
    def test_evaluate_buffer_too_small(self, mock_knowledge, mock_ingestor):
        """Buffer with TRAIN_SEQ_LEN tokens → too small for eval."""
        mock_ingestor.return_value = MagicMock()
        mock_knowledge.return_value = MagicMock()
        learner = ContinualLearner(n_embed=32, n_layer=2, n_head=2)
        learner.ingest_text("a" * TRAIN_SEQ_LEN)
        result = learner.evaluate()
        assert result["error"] == "buffer too small"
        learner._running = False

    @patch("domains.learner.knowledge.get_knowledge_memory")
    @patch("domains.learner.knowledge.get_knowledge_ingestor")
    def test_train_step_insufficient_data(self, mock_knowledge, mock_ingestor):
        """_train_step with insufficient buffer → no training."""
        mock_ingestor.return_value = MagicMock()
        mock_knowledge.return_value = MagicMock()
        learner = ContinualLearner(n_embed=32, n_layer=2, n_head=2)
        learner.ingest_text("short")
        learner._train_step()
        assert learner.train_steps_completed == 0
        learner._running = False

    @patch("domains.learner.knowledge.get_knowledge_memory")
    @patch("domains.learner.knowledge.get_knowledge_ingestor")
    def test_status_buffer_size(self, mock_knowledge, mock_ingestor):
        mock_ingestor_inst = MagicMock()
        mock_ingestor.return_value = mock_ingestor_inst
        mock_knowledge_inst = MagicMock()
        mock_knowledge.return_value = mock_knowledge_inst
        mock_knowledge_inst.stats.return_value = {}
        mock_ingestor_inst.list_feeds.return_value = []
        mock_ingestor_inst.filter.get_stats.return_value = {}
        mock_ingestor_inst.filter.get_config.return_value = {}

        learner = ContinualLearner(n_embed=32, n_layer=2, n_head=2)
        learner.ingest_text("hello world")
        status = learner.status()
        assert status["buffer_size"] == len("hello world")
        learner._running = False
