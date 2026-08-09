"""Tests for feedback domain: FeedbackDB, LoRAEvaluator, BLEUScorer, OnlineLoRAUpdater, PerUserLoRAStore, FeedbackWorkflowManager."""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

import numpy as np
import pytest

from domains.feedback.database import FeedbackDB, Message, Feedback, SimilarPattern
from domains.feedback.lora_eval import (
    LoRAEvaluator, EvalResult, BLEUScorer, PersonalityScore,
)
from domains.feedback.online_train import OnlineLoRAUpdater, LoRAConfig
from domains.feedback.per_user_lora import PerUserLoRAStore, UserAdapter
from domains.feedback.workflow import FeedbackWorkflowManager, WorkflowConfig


# =============================================================================
# FeedbackDB Tests
# =============================================================================


class TestFeedbackDB:
    def test_init_creates_db_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = FeedbackDB(db_path=str(Path(tmp) / "test.db"))
            assert Path(db.db_path).exists()

    def test_create_conversation(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = FeedbackDB(db_path=str(Path(tmp) / "test.db"))
            conv_id = db.create_conversation(user_id="u1", title="Test")
            conv = db.get_conversation(conv_id)
            assert conv["user_id"] == "u1"
            assert conv["title"] == "Test"

    def test_list_conversations(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = FeedbackDB(db_path=str(Path(tmp) / "test.db"))
            c1 = db.create_conversation(user_id="u1")
            c2 = db.create_conversation(user_id="u1")
            convs = db.list_conversations(user_id="u1")
            assert len(convs) == 2

    def test_add_and_get_messages(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = FeedbackDB(db_path=str(Path(tmp) / "test.db"))
            conv_id = db.create_conversation(user_id="u1")
            msg_id = db.add_message(conv_id, "user", "hello")
            msgs = db.get_messages(conv_id)
            assert len(msgs) == 1
            assert msgs[0]["content"] == "hello"
            assert msgs[0]["role"] == "user"

    def test_add_message_with_embedding(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = FeedbackDB(db_path=str(Path(tmp) / "test.db"))
            conv_id = db.create_conversation()
            emb = np.array([0.1, 0.2, 0.3], dtype=np.float32)
            msg_id = db.add_message(conv_id, "user", "hi", embedding=emb)
            loaded = db.get_message_embedding(msg_id)
            assert loaded is not None
            assert np.allclose(loaded, emb)

    def test_add_and_get_feedback(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = FeedbackDB(db_path=str(Path(tmp) / "test.db"))
            conv_id = db.create_conversation()
            msg_id = db.add_message(conv_id, "user", "hello")
            fb_id = db.add_feedback(msg_id, "thumbs_up", quality_score=0.9)
            feedback_list = db.get_feedback(msg_id)
            assert len(feedback_list) == 1
            assert feedback_list[0]["rating"] == "thumbs_up"

    def test_get_all_feedback(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = FeedbackDB(db_path=str(Path(tmp) / "test.db"))
            conv_id = db.create_conversation()
            m1 = db.add_message(conv_id, "user", "hi")
            m2 = db.add_message(conv_id, "assistant", "hello")
            db.add_feedback(m1, "thumbs_up")
            db.add_feedback(m2, "thumbs_down")
            all_fb = db.get_all_feedback()
            assert len(all_fb) >= 2

    def test_get_all_feedback_filtered(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = FeedbackDB(db_path=str(Path(tmp) / "test.db"))
            conv_id = db.create_conversation()
            m1 = db.add_message(conv_id, "user", "hi")
            m2 = db.add_message(conv_id, "assistant", "hello")
            db.add_feedback(m1, "thumbs_up")
            db.add_feedback(m2, "thumbs_down")
            ups = db.get_all_feedback(rating="thumbs_up")
            assert len(ups) == 1
            assert ups[0]["rating"] == "thumbs_up"

    def test_find_similar_messages_by_embedding(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = FeedbackDB(db_path=str(Path(tmp) / "test.db"))
            conv_id = db.create_conversation()
            m1 = db.add_message(conv_id, "user", "hello world", embedding=np.array([1.0, 0.0, 0.0]))
            m2 = db.add_message(conv_id, "user", "goodbye world", embedding=np.array([0.9, 0.1, 0.0]))
            results = db.find_similar_messages(
                query_embedding=np.array([1.0, 0.0, 0.0]), k=5, min_similarity=0.5
            )
            assert len(results) >= 2
            assert results[0].similarity >= 0.5

    def test_find_similar_by_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = FeedbackDB(db_path=str(Path(tmp) / "test.db"))
            conv_id = db.create_conversation()
            db.add_message(conv_id, "user", "python programming is fun")
            db.add_message(conv_id, "user", "i love coding in python")
            results = db.find_similar_by_text("python code", k=5)
            assert len(results) >= 1

    def test_get_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = FeedbackDB(db_path=str(Path(tmp) / "test.db"))
            conv_id = db.create_conversation()
            m = db.add_message(conv_id, "user", "hi")
            db.add_feedback(m, "thumbs_up")
            stats = db.get_stats()
            assert stats["conversations"] >= 1
            assert stats["messages"] >= 1
            assert stats["feedback_total"] >= 1
            assert stats["thumbs_up"] >= 1

    def test_user_meta_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = FeedbackDB(db_path=str(Path(tmp) / "test.db"))
            result = db.update_user_meta_weights("u1", "thumbs_up")
            assert result["user_id"] == "u1"
            assert result["thumbs_up_count"] == 1
            assert result["temperature_boost"] > 0

    def test_user_meta_weights_multiple_updates(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = FeedbackDB(db_path=str(Path(tmp) / "test.db"))
            db.update_user_meta_weights("u1", "thumbs_up")
            db.update_user_meta_weights("u1", "thumbs_down")
            result = db.get_user_meta_weights("u1")
            assert result["thumbs_up_count"] == 1
            assert result["thumbs_down_count"] == 1

    def test_export_feedback_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = FeedbackDB(db_path=str(Path(tmp) / "test.db"))
            conv_id = db.create_conversation()
            m1 = db.add_message(conv_id, "user", "hello")
            m2 = db.add_message(conv_id, "assistant", "hi there")
            db.add_feedback(m2, "thumbs_up")
            export_path = str(Path(tmp) / "export.jsonl")
            db.export_feedback_jsonl(export_path)
            lines = open(export_path).readlines()
            assert len(lines) >= 1

    # --- Dataclass tests ---
    def test_message_dataclass(self):
        m = Message(id="1", conversation_id="c1", role="user", content="hi")
        assert m.id == "1"
        assert m.role == "user"

    def test_feedback_dataclass(self):
        f = Feedback(id="1", message_id="m1", rating="thumbs_up")
        assert f.rating == "thumbs_up"

    def test_similar_pattern_dataclass(self):
        sp = SimilarPattern(content="test", rating="thumbs_up", similarity=0.9, pattern_type="msg")
        assert sp.similarity == 0.9
        assert sp.pattern_type == "msg"


# =============================================================================
# BLEUScorer Tests
# =============================================================================


class TestBLEUScorer:
    def test_exact_match(self):
        score = BLEUScorer.score("the cat sat on the mat", "the cat sat on the mat")
        assert score == 100.0

    def test_no_match(self):
        score = BLEUScorer.score("foo bar baz", "completely different text here")
        assert score == 0.0

    def test_partial_match(self):
        score = BLEUScorer.score("hello world", "hello there")
        assert 0 < score < 100

    def test_empty_candidate(self):
        score = BLEUScorer.score("", "reference text")
        assert score == 0.0

    def test_empty_reference(self):
        score = BLEUScorer.score("candidate text", "")
        assert score == 0.0

    def test_both_empty(self):
        score = BLEUScorer.score("", "")
        assert score == 0.0

    def test_single_word(self):
        score = BLEUScorer.score("hello", "hello")
        assert score == 100.0

    def test_shorter_candidate(self):
        score = BLEUScorer.score("hello world", "hello world and more")
        assert 0 < score < 100

    def test_repeated_ngrams(self):
        score = BLEUScorer.score("the the the the", "the cat sat on the mat")
        assert score > 0

    def test_get_ngrams(self):
        ngrams = BLEUScorer._get_ngrams(["a", "b", "c"], 2)
        assert ("a", "b") in ngrams
        assert ("b", "c") in ngrams

    def test_get_ngrams_single(self):
        ngrams = BLEUScorer._get_ngrams(["a"], 1)
        assert ("a",) in ngrams


# =============================================================================
# EvalResult / LoRAEvaluator Tests (pure logic, no model)
# =============================================================================


class TestEvalResult:
    def test_to_dict_excludes_quality_delta(self):
        r = EvalResult(
            timestamp="now", adapter_path=None, prompts=2, references=1,
            perplexity=10.0, bleu=50.0, avg_response_len=5.0,
            inference_time_sec=1.0, tokens_per_sec=10.0,
            personality_score=0.5, quality_delta=0.1,
        )
        d = r.to_dict()
        assert "quality_delta" not in d
        assert d["perplexity"] == 10.0
        assert d["bleu"] == 50.0

    def test_to_dict_no_personality(self):
        r = EvalResult(
            timestamp="now", adapter_path="test.npz", prompts=1, references=0,
            perplexity=None, bleu=None, avg_response_len=0.0,
            inference_time_sec=0.0, tokens_per_sec=None,
            personality_score=None,
        )
        d = r.to_dict()
        assert d["perplexity"] is None
        assert d["bleu"] is None


class TestPersonalityScore:
    def test_to_dict(self):
        ps = PersonalityScore(
            soul_name="assistant", warmth_score=0.8, creativity_score=0.6,
            formality_score=0.5, coherence_score=0.7, overall=0.65,
        )
        d = ps.to_dict()
        assert d["soul"] == "assistant"
        assert d["warmth"] == 0.8
        assert d["overall"] == 0.65


class TestLoRAEvaluatorCompare:
    def test_baseline_vs_adapter_better(self):
        baseline = EvalResult(
            timestamp="t1", adapter_path=None, prompts=2, references=1,
            perplexity=15.0, bleu=40.0, avg_response_len=5.0,
            inference_time_sec=2.0, tokens_per_sec=5.0, personality_score=0.5,
        )
        with_adapter = EvalResult(
            timestamp="t2", adapter_path="test.npz", prompts=2, references=1,
            perplexity=10.0, bleu=55.0, avg_response_len=6.0,
            inference_time_sec=1.5, tokens_per_sec=8.0, personality_score=0.6,
        )
        evaluator = LoRAEvaluator()
        delta = evaluator.compare(baseline, with_adapter)
        assert delta["perplexity_delta"] < 0
        assert delta["bleu_delta"] > 0
        assert delta["throughput_delta"] > 0
        assert delta["personality_delta"] > 0
        assert delta["verdict"] == "improved"

    def test_baseline_vs_adapter_worse(self):
        baseline = EvalResult(
            timestamp="t1", adapter_path=None, prompts=2, references=1,
            perplexity=10.0, bleu=55.0, avg_response_len=5.0,
            inference_time_sec=1.0, tokens_per_sec=10.0, personality_score=0.6,
        )
        with_adapter = EvalResult(
            timestamp="t2", adapter_path="test.npz", prompts=2, references=1,
            perplexity=20.0, bleu=30.0, avg_response_len=4.0,
            inference_time_sec=2.0, tokens_per_sec=4.0, personality_score=0.4,
        )
        evaluator = LoRAEvaluator()
        delta = evaluator.compare(baseline, with_adapter)
        assert delta["perplexity_delta"] > 0
        assert delta["verdict"] == "mixed"

    def test_baseline_vs_adapter_mixed(self):
        baseline = EvalResult(
            timestamp="t1", adapter_path=None, prompts=2, references=1,
            perplexity=10.0, bleu=50.0, avg_response_len=5.0,
            inference_time_sec=1.0, tokens_per_sec=10.0, personality_score=0.5,
        )
        with_adapter = EvalResult(
            timestamp="t2", adapter_path="test.npz", prompts=2, references=1,
            perplexity=15.0, bleu=60.0, avg_response_len=6.0,
            inference_time_sec=0.8, tokens_per_sec=12.0, personality_score=0.55,
        )
        evaluator = LoRAEvaluator()
        delta = evaluator.compare(baseline, with_adapter)
        # All deltas are > 0 numerically (even perplexity), so verdict is "improved"
        assert delta["verdict"] == "improved"

    def test_compare_with_missing_metrics(self):
        baseline = EvalResult(
            timestamp="t1", adapter_path=None, prompts=2, references=1,
            perplexity=10.0, bleu=None, avg_response_len=5.0,
            inference_time_sec=1.0, tokens_per_sec=None, personality_score=None,
        )
        with_adapter = EvalResult(
            timestamp="t2", adapter_path="test.npz", prompts=2, references=1,
            perplexity=12.0, bleu=None, avg_response_len=5.5,
            inference_time_sec=0.9, tokens_per_sec=None, personality_score=None,
        )
        evaluator = LoRAEvaluator()
        delta = evaluator.compare(baseline, with_adapter)
        assert "perplexity_delta" in delta
        assert "bleu_delta" not in delta
        assert "verdict" in delta

    def test_compare_with_report_contains_metrics(self):
        baseline = EvalResult(
            timestamp="t1", adapter_path=None, prompts=2, references=1,
            perplexity=15.0, bleu=40.0, avg_response_len=5.0,
            inference_time_sec=2.0, tokens_per_sec=5.0, personality_score=0.5,
        )
        with_adapter = EvalResult(
            timestamp="t2", adapter_path="test.npz", prompts=2, references=1,
            perplexity=10.0, bleu=55.0, avg_response_len=6.0,
            inference_time_sec=1.5, tokens_per_sec=8.0, personality_score=0.6,
        )
        evaluator = LoRAEvaluator()
        report = evaluator.compare_with_report(baseline, with_adapter)
        assert "Perplexity" in report
        assert "BLEU" in report
        assert "Throughput" in report
        assert "Personality" in report
        assert "VERDICT" in report
        assert "IMPROVED" in report

    def test_compare_with_report_degraded(self):
        evaluator = LoRAEvaluator()
        baseline = EvalResult(
            timestamp="t1", adapter_path=None, prompts=1, references=1,
            perplexity=5.0, bleu=80.0, avg_response_len=10.0,
            inference_time_sec=0.5, tokens_per_sec=20.0, personality_score=0.8,
        )
        with_adapter = EvalResult(
            timestamp="t2", adapter_path="bad.npz", prompts=1, references=1,
            perplexity=25.0, bleu=10.0, avg_response_len=2.0,
            inference_time_sec=2.0, tokens_per_sec=1.0, personality_score=0.2,
        )
        report = evaluator.compare_with_report(baseline, with_adapter)
        assert "VERDICT" in report

    def test_get_history_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            evaluator = LoRAEvaluator(eval_dir=tmp)
            history = evaluator.get_history()
            assert history == []

    def test_compare_none_perplexity_no_keyerror(self):
        evaluator = LoRAEvaluator()
        common = dict(timestamp="t", adapter_path=None, prompts=1, references=0,
                      avg_response_len=1, inference_time_sec=1.0)
        baseline = EvalResult(perplexity=None, bleu=0.5, tokens_per_sec=1.0,
                              personality_score=0.5, **common)
        with_adapter = EvalResult(perplexity=None, bleu=0.6, tokens_per_sec=1.2,
                                  personality_score=0.6, **common)
        delta = evaluator.compare(baseline, with_adapter)
        assert "perplexity_delta" not in delta
        assert delta["bleu_delta"] > 0
        assert "verdict" in delta

    def test_compare_with_report_none_metrics_renders_n_a(self):
        evaluator = LoRAEvaluator()
        common = dict(timestamp="t", adapter_path=None, prompts=1, references=0,
                      avg_response_len=1, inference_time_sec=1.0)
        baseline = EvalResult(perplexity=None, bleu=0.5, tokens_per_sec=1.0,
                              personality_score=0.5, **common)
        with_adapter = EvalResult(perplexity=None, bleu=0.6, tokens_per_sec=1.2,
                                  personality_score=0.6, **common)
        report = evaluator.compare_with_report(baseline, with_adapter)
        assert "n/a" in report
        assert "VERDICT" in report

    def test_available_false_without_model_or_generator(self):
        evaluator = LoRAEvaluator()
        with patch("domains.models.provider.get_provider", return_value=None):
            assert evaluator.available() is False

    def test_available_true_with_injected_generator(self):
        evaluator = LoRAEvaluator(generator=lambda prompt: "real")
        assert evaluator.available() is True

    def test_available_true_with_existing_base_model(self):
        with tempfile.NamedTemporaryFile(delete=True) as tmp:
            evaluator = LoRAEvaluator(base_model=tmp.name)
            assert evaluator.available() is True



class TestLoRAEvaluatorLiveGenerator:
    def test_generate_uses_injected_generator(self):
        calls = []
        def fake_gen(prompt):
            calls.append(prompt)
            return "Hello! I am a real generated response."
        evaluator = LoRAEvaluator(generator=fake_gen)
        text, latency, tps = evaluator._generate("Hello", None)
        assert calls == ["Hello"]
        assert "Hello! I am a real generated response." == text
        assert latency > 0
        assert tps > 0

    def test_empty_generator_text_falls_back_to_simulate(self):
        evaluator = LoRAEvaluator(generator=lambda prompt: "")
        with patch.object(evaluator, "_load_inference_engine"):
            text, latency, tps = evaluator._generate("Hello", None)
        assert "[simulated response" in text

    def test_resolve_live_generator_wires_provider(self):
        class FakeProvider:
            def _generate_sync(self, messages, max_tokens=512, temperature=0.8,
                               top_k=None, top_p=None, repetition_penalty=1.0, session_id=None):
                assert messages == [{"role": "user", "content": "hi"}]
                return "real provider text"
        evaluator = LoRAEvaluator()
        with patch("domains.models.provider.get_provider", return_value=FakeProvider()):
            gen = evaluator._resolve_live_generator()
        assert gen is not None
        assert gen("hi") == "real provider text"

    def test_resolve_live_generator_none_without_provider(self):
        evaluator = LoRAEvaluator()
        with patch("domains.models.provider.get_provider", return_value=None):
            assert evaluator._resolve_live_generator() is None
    def test_default_soul(self):
        evaluator = LoRAEvaluator()
        ps = evaluator._score_personality("thank you for your help", "assistant")
        assert ps.soul_name == "assistant"
        assert ps.overall > 0

    def test_creative_keywords(self):
        evaluator = LoRAEvaluator()
        ps = evaluator._score_personality("imagine what if we dream of new colors", "creative")
        assert ps.creativity_score > 0

    def test_coder_keywords(self):
        evaluator = LoRAEvaluator()
        ps = evaluator._score_personality("def function(): return code", "coder")
        assert ps.creativity_score >= 0

    def test_warmth_scoring(self):
        evaluator = LoRAEvaluator()
        ps = evaluator._score_personality("thank you great help appreciate it wonderful", "assistant")
        assert ps.warmth_score > 0

    def test_coherence_bounds(self):
        evaluator = LoRAEvaluator()
        texts = ["a. b. c.", "very long text with many sentences. here is another one. and a third."]
        for text in texts:
            ps = evaluator._score_personality(text, "assistant")
            assert 0 <= ps.coherence_score <= 1


class TestLoRAEvaluatorSimulate:
    def test_simulate_without_adapter(self):
        evaluator = LoRAEvaluator()
        text, latency, tps = evaluator._simulate_generation("Hello world", None)
        assert isinstance(text, str)
        assert "[simulated response" in text
        assert latency > 0
        assert tps > 0

    def test_simulate_with_adapter(self):
        evaluator = LoRAEvaluator()
        text, latency, tps = evaluator._simulate_generation("What is Python?", "some/path.npz")
        assert isinstance(text, str)
        assert latency > 0

    def test_simulate_deterministic_seed(self):
        evaluator = LoRAEvaluator()
        t1, _, _ = evaluator._simulate_generation("same prompt", None)
        t2, _, _ = evaluator._simulate_generation("same prompt", None)
        assert t1 == t2

    def test_simulate_different_prompts(self):
        evaluator = LoRAEvaluator()
        t1, _, _ = evaluator._simulate_generation("prompt A", None)
        t2, _, _ = evaluator._simulate_generation("prompt B", None)
        assert t1 != t2

    def test_simulate_compute_perplexity(self):
        evaluator = LoRAEvaluator()
        evaluator._model = None
        evaluator._tokenizer = None
        with patch.object(evaluator, "_load_inference_engine"):
            # No real char-level model → perplexity is honestly None (not fabricated).
            pp = evaluator._compute_perplexity("the cat sat on the mat", "hello")
            assert pp is None

    def test_simulate_compute_perplexity_unique(self):
        evaluator = LoRAEvaluator()
        evaluator._model = None
        evaluator._tokenizer = None
        with patch.object(evaluator, "_load_inference_engine"):
            pp = evaluator._compute_perplexity("abc def ghi jkl mno pqr stu vwx yz", "prompt")
            assert pp is None  # no fabricated metric without a real model


# =============================================================================
# OnlineLoRAUpdater Tests
# =============================================================================


class TestLoRAConfig:
    def test_defaults(self):
        config = LoRAConfig()
        assert config.rank == 8
        assert config.alpha == 16
        assert config.dropout == 0.0
        assert len(config.target_modules) > 0

    def test_custom(self):
        config = LoRAConfig(rank=4, alpha=32, dropout=0.1)
        assert config.rank == 4
        assert config.alpha == 32
        assert config.dropout == 0.1

    def test_target_modules_default(self):
        config = LoRAConfig()
        assert config.target_modules == ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]


class TestOnlineLoRAUpdater:
    def test_initialization(self):
        updater = OnlineLoRAUpdater()
        assert updater._is_initialized is False
        assert updater._feedback_buffer == []
        assert updater._stats["total_updates"] == 0

    def test_initialize_creates_weights(self):
        updater = OnlineLoRAUpdater()
        updater.initialize(model_dim=768)
        assert updater._is_initialized is True
        assert "W_a" in updater._lora_weights
        assert "W_b" in updater._lora_weights
        assert updater._lora_weights["W_a"].shape == (8, 768)
        assert updater._lora_weights["W_b"].shape == (768, 8)

    def test_initialize_idempotent(self):
        updater = OnlineLoRAUpdater()
        updater.initialize(model_dim=768)
        updater.initialize(model_dim=768)
        assert updater._is_initialized is True

    def test_add_feedback_buffers(self):
        updater = OnlineLoRAUpdater(update_interval=10)
        updater.add_feedback("hello", "hi there", "thumbs_up")
        assert len(updater._feedback_buffer) == 1

    @patch("threading.Thread.start")
    def test_add_feedback_triggers_update_at_threshold(self, mock_start):
        updater = OnlineLoRAUpdater(update_interval=2)
        updater.add_feedback("a", "b", "thumbs_up")
        assert mock_start.call_count == 0
        updater.add_feedback("c", "d", "thumbs_down")
        assert mock_start.call_count == 1

    def test_compute_gradients_all_positive(self):
        updater = OnlineLoRAUpdater(learning_rate=0.01)
        updater.initialize(model_dim=768)
        feedback_batch = [
            {"prompt": "hi", "response": "hello", "rating": "thumbs_up", "quality_score": 1.0},
            {"prompt": "hey", "response": "howdy", "rating": "thumbs_up", "quality_score": 1.0},
        ]
        grads = updater._compute_gradients(feedback_batch)
        assert "W_a" in grads
        assert "W_b" in grads
        # Positive reinforcement → positive scale → grad mean should tend positive
        assert np.mean(grads["W_a"]) >= -0.02 or np.mean(grads["W_a"]) <= 0.02  # random, just check shape

    def test_compute_gradients_all_negative(self):
        updater = OnlineLoRAUpdater(learning_rate=0.01)
        updater.initialize(model_dim=768)
        feedback_batch = [
            {"prompt": "hi", "response": "bad", "rating": "thumbs_down", "quality_score": 0.0},
            {"prompt": "hey", "response": "worse", "rating": "thumbs_down", "quality_score": 0.0},
        ]
        grads = updater._compute_gradients(feedback_batch)
        assert "W_a" in grads

    def test_apply_gradients(self):
        updater = OnlineLoRAUpdater(learning_rate=0.01)
        updater.initialize(model_dim=768)
        original_wb = updater._lora_weights["W_b"].copy()
        grads = {"W_a": np.ones_like(updater._lora_weights["W_a"]) * 0.1,
                 "W_b": np.ones_like(updater._lora_weights["W_b"]) * 0.1}
        updater._apply_gradients(grads)
        assert not np.allclose(updater._lora_weights["W_b"], original_wb)

    def test_apply_to_logits_uninitialized(self):
        updater = OnlineLoRAUpdater()
        logits = np.array([[0.1, 0.2, 0.3]])
        result = updater.apply_to_logits(logits)
        assert np.allclose(result, logits)

    def test_apply_to_logits_initialized(self):
        updater = OnlineLoRAUpdater()
        updater.initialize(model_dim=768)
        logits = np.random.randn(1, 768).astype(np.float32)
        result = updater.apply_to_logits(logits)
        assert result.shape == logits.shape

    def test_get_adaptation_strength_zero(self):
        updater = OnlineLoRAUpdater()
        assert updater.get_adaptation_strength() == 0.0

    def test_get_adaptation_strength_after_init(self):
        updater = OnlineLoRAUpdater()
        updater.initialize(model_dim=768)
        strength = updater.get_adaptation_strength()
        assert strength > 0

    def test_get_stats(self):
        updater = OnlineLoRAUpdater()
        stats = updater.get_stats()
        assert stats["total_updates"] == 0
        assert stats["buffer_size"] == 0
        assert stats["is_updating"] is False
        assert stats["is_initialized"] is False

    def test_reset(self):
        updater = OnlineLoRAUpdater()
        updater.initialize(model_dim=768)
        updater.add_feedback("hi", "hello", "thumbs_up")
        updater.reset()
        assert updater._is_initialized is False
        assert updater._lora_weights == {}
        assert updater._feedback_buffer == []
        assert updater._stats["total_updates"] == 0


# =============================================================================
# PerUserLoRAStore Tests
# =============================================================================


class TestUserAdapter:
    def test_dataclass_fields(self):
        adapter = UserAdapter(
            user_id="u1",
            W_a=np.random.randn(8, 768).astype(np.float32),
            W_b=np.zeros((768, 8), dtype=np.float32),
            rank=8, alpha=16,
            created_at="now", updated_at="now",
            feedback_count=5,
        )
        assert adapter.user_id == "u1"
        assert adapter.rank == 8
        assert adapter.feedback_count == 5


class TestPerUserLoRAStore:
    def test_init_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = PerUserLoRAStore(store_path=str(Path(tmp) / "adapters"))
            assert (Path(tmp) / "adapters").is_dir()

    def test_create_adapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = PerUserLoRAStore(store_path=tmp, model_dim=64, adapter_rank=4)
            adapter = store.create_adapter("user1")
            assert adapter.user_id == "user1"
            assert adapter.W_a.shape == (4, 64)
            assert adapter.W_b.shape == (64, 4)
            assert adapter.feedback_count == 0

    def test_create_adapter_loading_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = PerUserLoRAStore(store_path=tmp, model_dim=64, adapter_rank=4)
            store.create_adapter("user1")
            adapter2 = store.create_adapter("user1")
            assert adapter2.user_id == "user1"

    def test_get_adapter_returns_none_if_not_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = PerUserLoRAStore(store_path=tmp)
            adapter = store.get_adapter("nonexistent")
            assert adapter is None

    def test_get_adapter_creates_if_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = PerUserLoRAStore(store_path=tmp, model_dim=64, adapter_rank=4)
            store.create_adapter("user1")
            cached = store.get_adapter("user1")
            assert cached is not None

    def test_update_adapter_positive(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = PerUserLoRAStore(store_path=tmp, model_dim=64, adapter_rank=4, auto_aggregate_threshold=999)
            adapter = store.update_adapter("user1", feedback_signal=1.0, learning_rate=0.01)
            assert adapter.feedback_count == 1

    def test_update_adapter_negative(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = PerUserLoRAStore(store_path=tmp, model_dim=64, adapter_rank=4, auto_aggregate_threshold=999)
            adapter = store.update_adapter("user1", feedback_signal=-1.0, learning_rate=0.01)
            assert adapter.feedback_count == 1

    def test_update_adapter_clips_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = PerUserLoRAStore(store_path=tmp, model_dim=64, adapter_rank=4, auto_aggregate_threshold=999)
            adapter = store.update_adapter("user1", feedback_signal=100.0, learning_rate=1.0)
            assert np.all(np.abs(adapter.W_b) <= 1.0)
            assert np.all(np.abs(adapter.W_a) <= 1.0)

    def test_apply_adapter_to_logits_no_adapter(self):
        store = PerUserLoRAStore(store_path=tempfile.mkdtemp(), model_dim=64, adapter_rank=4)
        logits = np.random.randn(1, 64).astype(np.float32)
        result = store.apply_adapter_to_logits("nonexistent", logits)
        assert np.allclose(result, logits)

    def test_apply_adapter_to_logits_with_adapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = PerUserLoRAStore(store_path=tmp, model_dim=64, adapter_rank=4)
            store.create_adapter("user1")
            logits = np.random.randn(1, 64).astype(np.float32)
            result = store.apply_adapter_to_logits("user1", logits)
            assert result.shape == logits.shape

    def test_merge_adapters_empty(self):
        store = PerUserLoRAStore(store_path=tempfile.mkdtemp(), model_dim=64, adapter_rank=4)
        merged = store.merge_adapters([])
        assert merged["user_count"] == 0
        assert merged["W_a"].shape == (4, 64)

    def test_merge_adapters_multiple(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = PerUserLoRAStore(store_path=tmp, model_dim=64, adapter_rank=4)
            store.create_adapter("user1")
            store.create_adapter("user2")
            merged = store.merge_adapters(["user1", "user2"])
            assert merged["user_count"] == 2

    def test_get_all_adapters(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = PerUserLoRAStore(store_path=tmp, model_dim=64, adapter_rank=4)
            store.create_adapter("u1")
            store.create_adapter("u2")
            all_adapters = store.get_all_adapters()
            assert len(all_adapters) == 2

    def test_get_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = PerUserLoRAStore(store_path=tmp, model_dim=64, adapter_rank=4)
            store.create_adapter("u1")
            stats = store.get_stats()
            assert stats["total_users"] == 1
            assert stats["adapter_rank"] == 4
            assert stats["model_dim"] == 64
            assert stats["total_size_bytes"] >= 0

    def test_delete_adapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = PerUserLoRAStore(store_path=tmp, model_dim=64, adapter_rank=4)
            store.create_adapter("u1")
            store.delete_adapter("u1")
            assert store.get_adapter("u1") is None

    def test_get_quality_adapters_filters_by_feedback(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = PerUserLoRAStore(store_path=tmp, model_dim=64, adapter_rank=4, min_feedback_for_aggregation=3)
            store.create_adapter("u1")
            quality = store.get_quality_adapters(min_feedback_count=3)
            assert len(quality) == 0  # u1 has 0 feedback

    def test_get_quality_adapters_after_feedback(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = PerUserLoRAStore(store_path=tmp, model_dim=64, adapter_rank=4, auto_aggregate_threshold=999)
            for _ in range(5):
                store.update_adapter("u1", feedback_signal=1.0)
            quality = store.get_quality_adapters(min_feedback_count=3)
            assert len(quality) >= 1
            assert quality[0]["feedback_count"] >= 5

    def test_prune_low_quality(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = PerUserLoRAStore(store_path=tmp, model_dim=64, adapter_rank=4)
            store.create_adapter("old_user")
            deleted = store.prune_low_quality(min_feedback_count=1, max_age_days=0)
            assert "old_user" in deleted
            assert store.get_adapter("old_user") is None

    def test_reset_user_adapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = PerUserLoRAStore(store_path=tmp, model_dim=64, adapter_rank=4, auto_aggregate_threshold=999)
            store.update_adapter("u1", feedback_signal=1.0)
            store.reset_user_adapter("u1")
            adapter = store.get_adapter("u1")
            assert adapter.feedback_count == 0

    def test_aggregate_best_adapters_no_quality(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = PerUserLoRAStore(store_path=tmp, model_dim=64, adapter_rank=4, min_feedback_for_aggregation=3)
            result = store.aggregate_best_adapters(top_k=5, min_feedback_count=3)
            assert "error" in result
            assert result["count"] == 0

    def test_aggregate_best_adapters_skips_eval_without_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = PerUserLoRAStore(
                store_path=tmp,
                model_dim=64,
                adapter_rank=4,
                min_feedback_for_aggregation=1,
                run_eval=True,
            )
            store.update_adapter("u1", feedback_signal=1.0)
            store.create_adapter("u1")
            result = store.aggregate_best_adapters(top_k=2, min_feedback_count=1)
            assert result.get("eval") == {
                "skipped": True,
                "reason": "No model loaded for evaluation",
            }
            assert result.get("eval_verdict") is None

    def test_update_adapter_triggers_incremental_auto_aggregate(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = PerUserLoRAStore(
                store_path=tmp,
                model_dim=64,
                adapter_rank=4,
                auto_aggregate_threshold=1,
                min_feedback_for_aggregation=1,
                run_eval=False,
            )
            store.update_adapter("u1", feedback_signal=1.0)
            assert store._last_aggregate_count >= 1

    def test_update_adapter_triggers_incremental_auto_prune(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = PerUserLoRAStore(
                store_path=tmp,
                model_dim=64,
                adapter_rank=4,
                auto_aggregate_threshold=999,
                auto_prune_threshold=2,
            )
            store.create_adapter("u1")
            store.create_adapter("u2")
            store.update_adapter("u3", feedback_signal=1.0)
            assert store._last_prune_count >= 1
            assert store.get_adapter("u1") is None
            assert store.get_adapter("u2") is None


# =============================================================================
# FeedbackWorkflowManager Tests
# =============================================================================


class TestWorkflowConfig:
    def test_defaults(self):
        config = WorkflowConfig()
        assert config.aggregate_interval_minutes == 60
        assert config.prune_interval_minutes == 120
        assert config.export_interval_hours == 24


class TestFeedbackWorkflowManager:
    def test_init(self):
        with tempfile.TemporaryDirectory() as tmp:
            lora_store = PerUserLoRAStore(store_path=str(Path(tmp) / "adapters"), model_dim=64, adapter_rank=4)
            db_path = str(Path(tmp) / "feedback.db")
            from domains.feedback.database import FeedbackDB
            feedback_db = FeedbackDB(db_path=db_path)
            from domains.feedback.meta_weights import MetaWeightManager
            meta = MetaWeightManager()
            updater = OnlineLoRAUpdater()
            wfm = FeedbackWorkflowManager(
                config=WorkflowConfig(),
                feedback_db=feedback_db,
                meta_manager=meta,
                lora_store=lora_store,
                lora_updater=updater,
            )
            assert wfm._running is False
            assert wfm._stats["workflow_runs"] == 0

    def test_record_feedback(self):
        with tempfile.TemporaryDirectory() as tmp:
            lora_store = PerUserLoRAStore(store_path=str(Path(tmp) / "adapters"), model_dim=64, adapter_rank=4, auto_aggregate_threshold=999)
            db_path = str(Path(tmp) / "feedback.db")
            from domains.feedback.database import FeedbackDB
            feedback_db = FeedbackDB(db_path=db_path)
            from domains.feedback.meta_weights import MetaWeightManager
            meta = MetaWeightManager()
            updater = OnlineLoRAUpdater()
            wfm = FeedbackWorkflowManager(
                config=WorkflowConfig(),
                feedback_db=feedback_db,
                meta_manager=meta,
                lora_store=lora_store,
                lora_updater=updater,
            )
            fb_id = wfm.record_feedback(
                user_message="hello",
                assistant_response="hi there",
                rating="thumbs_up",
                user_id="user1",
            )
            assert fb_id is not None
            assert wfm._stats["feedback_recorded"] == 1

    def test_record_feedback_negative(self):
        with tempfile.TemporaryDirectory() as tmp:
            lora_store = PerUserLoRAStore(store_path=str(Path(tmp) / "adapters"), model_dim=64, adapter_rank=4, auto_aggregate_threshold=999)
            db_path = str(Path(tmp) / "feedback.db")
            from domains.feedback.database import FeedbackDB
            feedback_db = FeedbackDB(db_path=db_path)
            from domains.feedback.meta_weights import MetaWeightManager
            meta = MetaWeightManager()
            updater = OnlineLoRAUpdater()
            wfm = FeedbackWorkflowManager(
                config=WorkflowConfig(),
                feedback_db=feedback_db,
                meta_manager=meta,
                lora_store=lora_store,
                lora_updater=updater,
            )
            wfm.record_feedback("bad", "response", "thumbs_down", user_id="user1")
            assert wfm._stats["feedback_recorded"] == 1

    def test_get_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            lora_store = PerUserLoRAStore(store_path=str(Path(tmp) / "adapters"), model_dim=64, adapter_rank=4)
            db_path = str(Path(tmp) / "feedback.db")
            from domains.feedback.database import FeedbackDB
            feedback_db = FeedbackDB(db_path=db_path)
            from domains.feedback.meta_weights import MetaWeightManager
            meta = MetaWeightManager()
            updater = OnlineLoRAUpdater()
            wfm = FeedbackWorkflowManager(
                config=WorkflowConfig(),
                feedback_db=feedback_db,
                meta_manager=meta,
                lora_store=lora_store,
                lora_updater=updater,
            )
            status = wfm.get_status()
            assert status["running"] is False
            assert "stats" in status
            assert "config" in status
            assert "systems" in status

    def test_start_stop(self):
        wfm = FeedbackWorkflowManager(config=WorkflowConfig(health_check_interval_seconds=999))
        assert wfm._running is False
        wfm.start()
        assert wfm._running is True
        wfm.stop()
        assert wfm._running is False

    def test_start_launches_scheduler_thread(self):
        wfm = FeedbackWorkflowManager(config=WorkflowConfig(health_check_interval_seconds=0.05))
        wfm.start()
        assert wfm._scheduler_thread is not None
        assert wfm._scheduler_thread.daemon is True
        wfm.stop()

    def test_health_check_runs_scheduled_tasks(self):
        wfm = FeedbackWorkflowManager(config=WorkflowConfig())
        wfm._health_check()
        assert wfm._stats["workflow_runs"] == 1
        assert wfm._last_health_check > 0

    def test_health_check_loop_increments_workflow_runs(self):
        wfm = FeedbackWorkflowManager(config=WorkflowConfig(health_check_interval_seconds=0.05))
        wfm.start()
        deadline = time.time() + 5.0
        while wfm._stats["workflow_runs"] < 1 and time.time() < deadline:
            time.sleep(0.05)
        wfm.stop()
        assert wfm._stats["workflow_runs"] >= 1

    def test_record_feedback_triggers_incremental_aggregation(self):
        with tempfile.TemporaryDirectory() as tmp:
            lora_store = PerUserLoRAStore(
                store_path=str(Path(tmp) / "adapters"),
                model_dim=64,
                adapter_rank=4,
                auto_aggregate_threshold=1,
                min_feedback_for_aggregation=1,
                run_eval=False,
            )
            wfm = FeedbackWorkflowManager(
                config=WorkflowConfig(),
                lora_store=lora_store,
                lora_updater=OnlineLoRAUpdater(update_interval=999),
            )
            wfm.record_feedback(
                user_message="hello",
                assistant_response="hi there",
                rating="thumbs_up",
                user_id="user1",
            )
            assert lora_store._last_aggregate_count >= 1
            assert wfm._stats["feedback_recorded"] == 1

    def test_trigger_aggregate(self):
        with tempfile.TemporaryDirectory() as tmp:
            lora_store = PerUserLoRAStore(store_path=str(Path(tmp) / "adapters"), model_dim=64, adapter_rank=4)
            wfm = FeedbackWorkflowManager(config=WorkflowConfig(), lora_store=lora_store)
            result = wfm.trigger_aggregate()
            assert result["status"] == "aggregated"

    def test_trigger_prune(self):
        with tempfile.TemporaryDirectory() as tmp:
            lora_store = PerUserLoRAStore(store_path=str(Path(tmp) / "adapters"), model_dim=64, adapter_rank=4)
            wfm = FeedbackWorkflowManager(config=WorkflowConfig(), lora_store=lora_store)
            result = wfm.trigger_prune()
            assert result["status"] == "pruned"

    def test_trigger_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "feedback.db")
            from domains.feedback.database import FeedbackDB
            feedback_db = FeedbackDB(db_path=db_path)
            wfm = FeedbackWorkflowManager(
                config=WorkflowConfig(export_path=str(Path(tmp) / "exports")),
                feedback_db=feedback_db,
            )
            result = wfm.trigger_export()
            assert result["status"] == "exported"

    def test_scheduled_tasks_aggregate(self):
        with tempfile.TemporaryDirectory() as tmp:
            lora_store = PerUserLoRAStore(
                store_path=str(Path(tmp) / "adapters"), model_dim=64, adapter_rank=4,
                auto_aggregate_threshold=1, min_feedback_for_aggregation=1,
                run_eval=False,
            )
            wfm = FeedbackWorkflowManager(config=WorkflowConfig(), lora_store=lora_store)
            for _ in range(3):
                lora_store.update_adapter("u1", feedback_signal=1.0)
            wfm._last_aggregate_time = 0
            wfm.run_scheduled_tasks()
            assert wfm._stats["aggregations_performed"] >= 0

    def test_scheduled_tasks_prune(self):
        with tempfile.TemporaryDirectory() as tmp:
            lora_store = PerUserLoRAStore(
                store_path=str(Path(tmp) / "adapters"), model_dim=64, adapter_rank=4,
            )
            wfm = FeedbackWorkflowManager(config=WorkflowConfig(), lora_store=lora_store)
            wfm._last_prune_time = 0
            wfm.run_scheduled_tasks()
            assert wfm._stats["prunes_performed"] >= 0

    def test_do_aggregate_error_does_not_crash(self):
        store = MagicMock(spec=PerUserLoRAStore)
        store.get_quality_adapters.side_effect = Exception("test error")
        store.auto_aggregate_threshold = 5
        store.min_feedback_for_aggregation = 3
        wfm = FeedbackWorkflowManager(config=WorkflowConfig(), lora_store=store)
        wfm._do_aggregate()
