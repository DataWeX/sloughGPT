"""Tests for clean architecture domains: ChatDomain, BenchmarkDomain, CompanionSystem."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from domains.chat.domain import ChatDomain, ChatRequest, ChatResponse, get_chat_domain
from domains.benchmark.domain import BenchmarkDomain, QualityMetrics, get_benchmark_domain

# companion.py (top-level, single file)
from domains.companion import (
    CompanionSystem,
    CompanionTraits,
    ConversationContext,
    create_companion,
    get_companion,
)


# =============================================================================
# ChatDomain Tests
# =============================================================================


class TestChatDataclasses:
    def test_chat_request_defaults(self):
        req = ChatRequest(messages=[{"role": "user", "content": "hi"}])
        assert req.model == "gpt2"
        assert req.temperature == 0.8
        assert req.max_tokens == 256
        assert req.session_id is None

    def test_chat_response_fields(self):
        resp = ChatResponse(text="hello", session_id="s1", tokens_generated=5, duration_ms=100)
        assert resp.text == "hello"
        assert resp.session_id == "s1"
        assert resp.done is True
        assert resp.tokens_generated == 5
        assert resp.duration_ms == 100


class TestChatDomain:
    def test_constructor_creates_log_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "chat_logs"
            chat = ChatDomain(log_dir=str(log_dir))
            assert log_dir.is_dir()

    @pytest.mark.asyncio
    @patch("domains.chat.domain.ChatDomain._generate", return_value="mock")
    async def test_respond_no_user_message_returns_placeholder(self, mock_gen):
        chat = ChatDomain(log_dir=tempfile.mkdtemp())
        resp = await chat.respond(messages=[{"role": "assistant", "content": "Hey"}])
        assert isinstance(resp, ChatResponse)
        assert resp.text != ""

    @pytest.mark.asyncio
    @patch("domains.chat.domain.ChatDomain._generate", return_value="mock")
    async def test_respond_empty_messages(self, mock_gen):
        chat = ChatDomain(log_dir=tempfile.mkdtemp())
        resp = await chat.respond(messages=[])
        assert isinstance(resp, ChatResponse)

    def test_get_recent_responses_no_file(self):
        chat = ChatDomain(log_dir=tempfile.mkdtemp())
        result = chat.get_recent_responses(limit=5)
        assert result == []

    def test_get_stats_empty(self):
        chat = ChatDomain(log_dir=tempfile.mkdtemp())
        stats = chat.get_stats()
        assert stats == {"total": 0}

    def test_log_writes_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            chat = ChatDomain(log_dir=tmp)
            chat._log(
                user_message="hi",
                assistant_response="hello",
                model="gpt2",
                temperature=0.8,
                max_tokens=256,
                session_id="s1",
                user_id="u1",
                tokens_generated=5,
                duration_ms=10,
            )
            log_files = list(Path(tmp).glob("responses_*.jsonl"))
            assert len(log_files) == 1
            with open(log_files[0]) as f:
                entry = json.loads(f.readline())
            assert entry["user_message"] == "hi"
            assert entry["assistant_response"] == "hello"

    @pytest.mark.asyncio
    @patch("domains.chat.domain.ChatDomain._generate", return_value="mock response")
    async def test_respond_delegates_to_generate(self, mock_gen):
        chat = ChatDomain(log_dir=tempfile.mkdtemp())
        resp = await chat.respond(messages=[{"role": "user", "content": "hello"}])
        mock_gen.assert_called_once()
        assert resp.text == "mock response"

    @pytest.mark.asyncio
    async def test_respond_picks_last_user_message(self):
        chat = ChatDomain(log_dir=tempfile.mkdtemp())
        with patch.object(chat, "_generate", return_value="ok") as mock:
            await chat.respond(messages=[
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "middle"},
                {"role": "user", "content": "last"},
            ])
            assert mock.call_args[1]["user_msg"] == "last"

    def test_get_recent_responses_returns_limited(self):
        with tempfile.TemporaryDirectory() as tmp:
            chat = ChatDomain(log_dir=tmp)
            for i in range(5):
                chat._log(
                    user_message=f"msg{i}", assistant_response=f"resp{i}",
                    model="t", temperature=0.5, max_tokens=10,
                    session_id="s", user_id="u", tokens_generated=1, duration_ms=1,
                )
            result = chat.get_recent_responses(limit=2)
            assert len(result) == 2
            assert result[-1]["user_message"] == "msg4"

    def test_get_stats_with_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            chat = ChatDomain(log_dir=tmp)
            for i in range(3):
                chat._log(
                    user_message=f"msg{i}", assistant_response=f"resp{i}",
                    model="gpt2", temperature=0.5, max_tokens=10,
                    session_id="s", user_id="u", tokens_generated=5 + i, duration_ms=10 + i,
                )
            stats = chat.get_stats()
            assert stats["total"] == 3
            assert stats["avg_tokens"] > 0
            assert stats["avg_duration_ms"] > 0
            assert "gpt2" in stats["unique_models"]

    def test_get_chat_domain_singleton(self):
        a = get_chat_domain()
        b = get_chat_domain()
        assert a is b


# =============================================================================
# BenchmarkDomain Tests
# =============================================================================


class TestQualityMetrics:
    def test_defaults_all_zero(self):
        m = QualityMetrics()
        assert m.coherence_score == 0.0
        assert m.quality_score == 0.0
        assert m.repetition_rate == 0.0
        assert m.avg_length == 0.0
        assert m.empty_rate == 0.0


class TestBenchmarkDomain:
    def test_evaluate_empty_list(self):
        bench = BenchmarkDomain()
        metrics = bench.evaluate_responses([])
        assert metrics.coherence_score == 0.0
        assert metrics.quality_score == 0.0

    def test_evaluate_single_good_response(self):
        bench = BenchmarkDomain()
        metrics = bench.evaluate_responses([
            {"assistant": "The quick brown fox jumps over the lazy dog."}
        ])
        assert metrics.coherence_score > 0.5
        assert metrics.quality_score > 0.5
        assert metrics.repetition_rate < 0.1
        assert metrics.empty_rate == 0.0

    def test_evaluate_high_repetition(self):
        bench = BenchmarkDomain()
        metrics = bench.evaluate_responses([
            {"assistant": "hello hello hello hello hello hello hello hello hello hello hello hello"}
        ])
        assert metrics.repetition_rate > 0.5

    def test_evaluate_all_empty(self):
        bench = BenchmarkDomain()
        metrics = bench.evaluate_responses([
            {"assistant": ""},
            {"assistant": "   "},
        ])
        assert metrics.empty_rate == 1.0
        assert metrics.avg_length == 0.0

    def test_evaluate_mixed_empty_and_good(self):
        bench = BenchmarkDomain()
        metrics = bench.evaluate_responses([
            {"assistant": ""},
            {"assistant": "A normal response here."},
        ])
        assert metrics.empty_rate == 0.5
        assert metrics.avg_length > 0

    def test_evaluate_missing_assistant_key(self):
        bench = BenchmarkDomain()
        metrics = bench.evaluate_responses([
            {"user": "hello"}
        ])
        assert metrics.empty_rate == 1.0

    def test_evaluate_coherence_bounds(self):
        bench = BenchmarkDomain()
        metrics = bench.evaluate_responses([
            {"assistant": "x"} for _ in range(10)
        ])
        assert 0 <= metrics.coherence_score <= 1
        assert 0 <= metrics.quality_score <= 1

    def test_evaluate_latest_no_log_file(self):
        bench = BenchmarkDomain(log_dir=tempfile.mkdtemp())
        result = bench.evaluate_latest()
        assert result == {"status": "no_data", "message": "No responses logged"}

    def test_get_stats_no_data(self):
        bench = BenchmarkDomain(log_dir=tempfile.mkdtemp())
        stats = bench.get_stats()
        assert stats == {"total": 0}

    def test_get_benchmark_domain_singleton(self):
        a = get_benchmark_domain()
        b = get_benchmark_domain()
        assert a is b


# =============================================================================
# CompanionSystem Tests (domains/companion.py)
# =============================================================================


class TestCompanionTraits:
    def test_default_values(self):
        t = CompanionTraits()
        assert t.name == "Friend"
        assert t.warmth == 0.7
        assert t.curiosity == 0.6
        assert t.creativity == 0.5
        assert t.confidence == 0.5
        assert t.humor == 0.4

    def test_custom_values(self):
        t = CompanionTraits(name="Alex", warmth=0.9, humor=0.8)
        assert t.name == "Alex"
        assert t.warmth == 0.9
        assert t.humor == 0.8


class TestConversationContext:
    def test_defaults(self):
        ctx = ConversationContext()
        assert ctx.user_name is None
        assert ctx.topics == []
        assert ctx.turn_count == 0

    def test_custom(self):
        ctx = ConversationContext(user_name="Bob", topics=["sports"], turn_count=5)
        assert ctx.user_name == "Bob"
        assert ctx.turn_count == 5


class TestCompanionSystem:
    def test_default_initialization(self):
        comp = CompanionSystem()
        assert comp.traits.name == "Friend"
        assert comp.traits.warmth == 0.7
        assert comp._system_prompt != ""

    def test_set_personality_updates_traits(self):
        comp = CompanionSystem()
        comp.set_personality(name="Alex", warmth=0.9, humor=0.8)
        assert comp.traits.name == "Alex"
        assert comp.traits.warmth == 0.9
        assert comp.traits.humor == 0.8
        assert comp.traits.curiosity == 0.6  # unchanged default

    def test_set_personality_rebuilds_system_prompt(self):
        comp = CompanionSystem()
        old_prompt = comp._system_prompt
        comp.set_personality(warmth=0.3)
        assert comp._system_prompt != old_prompt

    def test_build_system_prompt_contains_name(self):
        comp = CompanionSystem()
        comp.set_personality(name="TestBot")
        prompt = comp.get_system_prompt()
        assert "TestBot" in prompt

    def test_build_system_prompt_includes_warmth(self):
        comp = CompanionSystem()
        comp.set_personality(warmth=0.9)
        prompt = comp.get_system_prompt()
        assert "caring" in prompt.lower()

    def test_build_system_prompt_high_humor_adds_style(self):
        comp = CompanionSystem()
        comp.set_personality(humor=0.8)
        prompt = comp.get_system_prompt()
        assert "humor" in prompt.lower()

    def test_build_system_prompt_low_humor_no_style(self):
        comp = CompanionSystem()
        comp.set_personality(humor=0.3)
        prompt = comp.get_system_prompt()
        assert "humor" not in prompt.lower()

    def test_clean_response_removes_robot_phrases(self):
        comp = CompanionSystem()
        cleaned = comp.clean_response("As an AI, I think the answer is 42.")
        assert "As an AI" not in cleaned

    def test_clean_response_removes_multiple_robot_phrases(self):
        comp = CompanionSystem()
        cleaned = comp.clean_response("As an AI, I am an AI language model. I don't have feelings.")
        assert "As an AI" not in cleaned
        assert "AI language model" not in cleaned

    def test_clean_response_ensures_punctuation(self):
        comp = CompanionSystem()
        cleaned = comp.clean_response("Hello world")
        assert cleaned.endswith(".")

    def test_clean_response_preserves_existing_punctuation(self):
        comp = CompanionSystem()
        cleaned = comp.clean_response("Hello world!")
        assert cleaned.endswith("!")

    def test_clean_response_handles_empty(self):
        comp = CompanionSystem()
        cleaned = comp.clean_response("")
        assert cleaned == ""

    def test_respond_increments_turn_count(self):
        comp = CompanionSystem()
        assert comp.context.turn_count == 0
        comp.respond("hello")
        assert comp.context.turn_count == 1
        comp.respond("how are you?")
        assert comp.context.turn_count == 2

    def test_respond_returns_prompt_string(self):
        comp = CompanionSystem()
        result = comp.respond("hello")
        assert isinstance(result, str)
        assert "hello" in result
        assert comp.traits.name in result

    def test_adjust_for_mood_sad_increases_warmth(self):
        comp = CompanionSystem()
        comp.set_personality(warmth=0.5)
        comp.adjust_for_mood("sad")
        assert comp.traits.warmth > 0.5

    def test_adjust_for_mood_sad_decreases_humor(self):
        comp = CompanionSystem()
        comp.set_personality(humor=0.5)
        comp.adjust_for_mood("sad")
        assert comp.traits.humor < 0.5

    def test_adjust_for_mood_happy_slight_warmth_boost(self):
        comp = CompanionSystem()
        comp.set_personality(warmth=0.5)
        comp.adjust_for_mood("happy")
        assert comp.traits.warmth > 0.5

    def test_adjust_for_mood_unknown_no_change(self):
        comp = CompanionSystem()
        comp.set_personality(warmth=0.5, humor=0.5)
        comp.adjust_for_mood("neutral")
        assert comp.traits.warmth == 0.5
        assert comp.traits.humor == 0.5

    def test_adjust_for_mood_warmth_caps_at_one(self):
        comp = CompanionSystem()
        comp.set_personality(warmth=0.9)
        comp.adjust_for_mood("sad")
        assert comp.traits.warmth <= 1.0

    def test_adjust_for_mood_humor_floor_at_zero(self):
        comp = CompanionSystem()
        comp.set_personality(humor=0.1)
        comp.adjust_for_mood("sad")
        assert comp.traits.humor >= 0.0

    def test_to_dict_returns_all_fields(self):
        comp = CompanionSystem()
        comp.set_personality(name="Bob", warmth=0.8, humor=0.6)
        d = comp.to_dict()
        assert d["traits"]["name"] == "Bob"
        assert d["traits"]["warmth"] == 0.8
        assert "system_prompt" in d

    def test_get_system_prompt_returns_non_empty(self):
        comp = CompanionSystem()
        assert len(comp.get_system_prompt()) > 20

    def test_create_companion_warm_preset(self):
        comp = create_companion(name="Alice", personality="warm")
        assert comp.traits.warmth == 0.9
        assert comp.traits.name == "Alice"

    def test_create_companion_playful_preset(self):
        comp = create_companion(personality="playful")
        assert comp.traits.humor == 0.8

    def test_create_companion_invalid_preset_falls_back_to_balanced(self):
        comp = create_companion(personality="nonexistent")
        assert comp.traits.warmth == 0.7

    def test_respond_with_context_updates_topics(self):
        comp = CompanionSystem()
        ctx = ConversationContext(topics=["ai", "python"])
        comp.respond("hello", context=ctx)
        assert "ai" in comp.context.topics

    def test_get_companion_singleton(self):
        a = get_companion()
        b = get_companion()
        assert a is b
