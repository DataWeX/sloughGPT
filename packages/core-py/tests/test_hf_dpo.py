"""Tests for domains.feedback.hf_dpo — HFDPOTrainer pure logic."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from domains.feedback.hf_dpo import HFDPOTrainer, DPO_BETA, DEFAULT_LR, DEFAULT_EPOCHS


# ── Constants ──────────────────────────────────────────────────────────


class TestConstants:
    def test_dpo_beta(self):
        assert DPO_BETA == 0.1

    def test_default_lr(self):
        assert DEFAULT_LR == 1e-4

    def test_default_epochs(self):
        assert DEFAULT_EPOCHS == 2


# ── _is_trainable ─────────────────────────────────────────────────────


class TestIsTrainable:
    def test_none_returns_false(self):
        assert HFDPOTrainer._is_trainable(None) is False

    def test_no_parameters_returns_false(self):
        obj = MagicMock(spec=[])
        assert HFDPOTrainer._is_trainable(obj) is False

    def test_non_callable_parameters_returns_false(self):
        obj = MagicMock()
        obj.parameters = "not_callable"
        assert HFDPOTrainer._is_trainable(obj) is False

    def test_empty_parameters_returns_false(self):
        obj = MagicMock()
        obj.parameters.return_value = []
        obj.forward = MagicMock()
        assert HFDPOTrainer._is_trainable(obj) is False

    def test_parameters_no_forward_returns_false(self):
        obj = MagicMock()
        obj.parameters.return_value = [MagicMock()]
        del obj.forward
        assert HFDPOTrainer._is_trainable(obj) is False

    def test_callable_params_exception_returns_false(self):
        obj = MagicMock()
        obj.parameters.side_effect = RuntimeError("broken")
        assert HFDPOTrainer._is_trainable(obj) is False

    def test_valid_trainable_model(self):
        obj = MagicMock()
        obj.parameters.return_value = [MagicMock(), MagicMock()]
        obj.forward = MagicMock()
        assert HFDPOTrainer._is_trainable(obj) is True


# ── _encode ────────────────────────────────────────────────────────────


class TestEncode:
    def _make_trainer(self, tokenizer=None):
        model = MagicMock()
        return HFDPOTrainer(model=model, tokenizer=tokenizer)

    def test_with_encode_method(self):
        tok = MagicMock()
        tok.encode.return_value = [1, 2, 3]
        trainer = self._make_trainer(tokenizer=tok)
        ids = trainer._encode("abc")
        np.testing.assert_array_equal(ids, [1, 2, 3])
        assert ids.dtype == np.int64

    def test_with_tokenize_method(self):
        tok = MagicMock()
        del tok.encode
        tok.tokenize.return_value = [10, 20]
        trainer = self._make_trainer(tokenizer=tok)
        ids = trainer._encode("hi")
        np.testing.assert_array_equal(ids, [10, 20])

    def test_with_encode_as_ids_method(self):
        tok = MagicMock()
        del tok.encode
        del tok.tokenize
        tok.encode_as_ids.return_value = [7, 8, 9]
        trainer = self._make_trainer(tokenizer=tok)
        ids = trainer._encode("xyz")
        np.testing.assert_array_equal(ids, [7, 8, 9])

    def test_encode_returns_tuple(self):
        tok = MagicMock()
        tok.encode.return_value = (5, 6)
        trainer = self._make_trainer(tokenizer=tok)
        ids = trainer._encode("ab")
        np.testing.assert_array_equal(ids, [5, 6])

    def test_encode_returns_ndarray(self):
        tok = MagicMock()
        tok.encode.return_value = np.array([100, 200])
        trainer = self._make_trainer(tokenizer=tok)
        ids = trainer._encode("ab")
        np.testing.assert_array_equal(ids, [100, 200])

    def test_encode_returns_empty_falls_back(self):
        tok = MagicMock()
        tok.encode.return_value = []
        trainer = self._make_trainer(tokenizer=tok)
        ids = trainer._encode("ab")
        # Falls back to char-vocab: ord('a'), ord('b')
        np.testing.assert_array_equal(ids, [97, 98])

    def test_encode_exception_falls_back(self):
        tok = MagicMock()
        tok.encode.side_effect = RuntimeError("fail")
        trainer = self._make_trainer(tokenizer=tok)
        ids = trainer._encode("hi")
        np.testing.assert_array_equal(ids, [104, 105])

    def test_no_tokenizer_uses_char_vocab(self):
        trainer = self._make_trainer(tokenizer=None)
        ids = trainer._encode("Hi")
        # ord('H')=72, ord('i')=105; both < 128
        np.testing.assert_array_equal(ids, [72, 105])

    def test_non_ascii_chars_filtered(self):
        trainer = self._make_trainer(tokenizer=None)
        ids = trainer._encode("a\u00e9")
        # ord('a')=97 ok, ord('\u00e9')=233 >= 128 filtered
        np.testing.assert_array_equal(ids, [97])

    def test_empty_string(self):
        trainer = self._make_trainer(tokenizer=None)
        ids = trainer._encode("")
        assert len(ids) == 0


# ── _reject ────────────────────────────────────────────────────────────


class TestReject:
    def _make_trainer(self):
        model = MagicMock()
        return HFDPOTrainer(model=model, tokenizer=None)

    def test_reject_structure(self):
        trainer = self._make_trainer()
        result = trainer._reject("reason text", elapsed=1.5)
        assert result["status"] == "rejected"
        assert result["reason"] == "reason text"
        assert result["steps"] == 0
        assert result["avg_loss"] is None
        assert result["ppl_before"] is None
        assert result["ppl_after"] is None
        assert result["ppl_delta_pct"] is None
        assert result["pairs_trained"] == 0
        assert result["elapsed_seconds"] == 1.5

    def test_reject_zero_elapsed(self):
        trainer = self._make_trainer()
        result = trainer._reject("x", elapsed=0.0)
        assert result["elapsed_seconds"] == 0.0


# ── train rejection paths ─────────────────────────────────────────────


class TestTrainRejections:
    def _make_trainer(self, model=None):
        if model is None:
            model = MagicMock()
        return HFDPOTrainer(model=model, tokenizer=None)

    def test_too_few_pairs(self):
        trainer = self._make_trainer()
        result = trainer.train(pairs=[{"chosen": "a", "rejected": "b"}])
        assert result["status"] == "rejected"
        assert "at least 2" in result["reason"]

    def test_zero_pairs(self):
        trainer = self._make_trainer()
        result = trainer.train(pairs=[])
        assert result["status"] == "rejected"

    def test_not_trainable_model(self):
        # Model without parameters() or forward()
        model = MagicMock(spec=[])
        trainer = self._make_trainer(model=model)
        pairs = [
            {"chosen": "good1", "rejected": "bad1"},
            {"chosen": "good2", "rejected": "bad2"},
        ]
        result = trainer.train(pairs=pairs)
        assert result["status"] == "rejected"
        assert "trainable" in result["reason"]

    def test_max_pairs_limits(self):
        trainer = self._make_trainer()
        # Even though we pass 3 pairs, max_pairs=1 results in < 2 after slicing
        pairs = [
            {"chosen": "a", "rejected": "b"},
            {"chosen": "c", "rejected": "d"},
            {"chosen": "e", "rejected": "f"},
        ]
        result = trainer.train(pairs=pairs, max_pairs=1)
        assert result["status"] == "rejected"

    def test_pairs_none_calls_prepare(self):
        trainer = self._make_trainer()
        with patch.object(trainer, "prepare_dpo_pairs", return_value=[]) as mock_prep:
            result = trainer.train(pairs=None)
            mock_prep.assert_called_once_with(max_pairs=None)
            assert result["status"] == "rejected"


# ── prepare_dpo_pairs ─────────────────────────────────────────────────


class TestPrepareDPOPairs:
    def _make_trainer(self):
        model = MagicMock()
        return HFDPOTrainer(model=model, tokenizer=None)

    def test_empty_feedback(self):
        trainer = self._make_trainer()
        with patch("domains.feedback.hf_dpo.get_feedback_db") as mock_db:
            db = mock_db.return_value
            db.get_all_feedback.return_value = []
            pairs = trainer.prepare_dpo_pairs()
            assert pairs == []

    def test_pair_building_basic(self):
        trainer = self._make_trainer()
        with patch("domains.feedback.hf_dpo.get_feedback_db") as mock_db:
            db = mock_db.return_value
            db.get_all_feedback.side_effect = [
                # thumbs_up
                [{"content": "good answer", "conversation_id": "c1"}],
                # thumbs_down
                [{"content": "bad answer", "conversation_id": "c1"}],
            ]
            pairs = trainer.prepare_dpo_pairs()
            assert len(pairs) == 1
            assert pairs[0]["chosen"] == "good answer"
            assert pairs[0]["rejected"] == "bad answer"

    def test_same_conversation_preferred(self):
        trainer = self._make_trainer()
        with patch("domains.feedback.hf_dpo.get_feedback_db") as mock_db:
            db = mock_db.return_value
            db.get_all_feedback.side_effect = [
                # thumbs_up: one from same conv, one from different
                [
                    {"content": "other conv good", "conversation_id": "c2"},
                    {"content": "same conv good", "conversation_id": "c1"},
                ],
                # thumbs_down from c1
                [{"content": "bad answer", "conversation_id": "c1"}],
            ]
            pairs = trainer.prepare_dpo_pairs()
            assert pairs[0]["chosen"] == "same conv good"

    def test_skips_empty_content(self):
        trainer = self._make_trainer()
        with patch("domains.feedback.hf_dpo.get_feedback_db") as mock_db:
            db = mock_db.return_value
            db.get_all_feedback.side_effect = [
                [{"content": "good", "conversation_id": "c1"}],
                [{"content": "", "conversation_id": "c1"}],
            ]
            pairs = trainer.prepare_dpo_pairs()
            assert pairs == []

    def test_skips_identical_content(self):
        trainer = self._make_trainer()
        with patch("domains.feedback.hf_dpo.get_feedback_db") as mock_db:
            db = mock_db.return_value
            db.get_all_feedback.side_effect = [
                [{"content": "same text", "conversation_id": "c1"}],
                [{"content": "same text", "conversation_id": "c1"}],
            ]
            pairs = trainer.prepare_dpo_pairs()
            assert pairs == []

    def test_no_matching_chosen_skipped(self):
        trainer = self._make_trainer()
        with patch("domains.feedback.hf_dpo.get_feedback_db") as mock_db:
            db = mock_db.return_value
            db.get_all_feedback.side_effect = [
                # all chosen have empty content
                [{"content": "", "conversation_id": "c1"}],
                [{"content": "bad", "conversation_id": "c1"}],
            ]
            pairs = trainer.prepare_dpo_pairs()
            assert pairs == []

    def test_max_pairs_cap(self):
        trainer = self._make_trainer()
        with patch("domains.feedback.hf_dpo.get_feedback_db") as mock_db:
            db = mock_db.return_value
            db.get_all_feedback.side_effect = [
                [{"content": f"good{i}", "conversation_id": "c1"} for i in range(5)],
                [{"content": f"bad{i}", "conversation_id": "c1"} for i in range(5)],
            ]
            pairs = trainer.prepare_dpo_pairs(max_pairs=2)
            assert len(pairs) == 2


# ── Constructor ────────────────────────────────────────────────────────


class TestConstructor:
    def test_defaults(self):
        model = MagicMock()
        trainer = HFDPOTrainer(model=model, tokenizer=None)
        assert trainer.model is model
        assert trainer.tokenizer is None
        assert trainer.learning_rate == DEFAULT_LR
        assert trainer.beta == DPO_BETA

    def test_custom_params(self):
        model = MagicMock()
        tok = MagicMock()
        trainer = HFDPOTrainer(model=model, tokenizer=tok, learning_rate=0.01, beta=0.5)
        assert trainer.learning_rate == 0.01
        assert trainer.beta == 0.5
        assert trainer.tokenizer is tok
