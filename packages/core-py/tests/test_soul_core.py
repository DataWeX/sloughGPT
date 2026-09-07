"""Tests for core.soul — SloEngine and GenerationContext."""

from __future__ import annotations

import pytest
import numpy as np
from unittest.mock import MagicMock, patch, PropertyMock

from domains.core.soul import GenerationContext, SloEngine


# ── GenerationContext ───────────────────────────────────────────────────────


class TestGenerationContext:

    def test_defaults(self):
        ctx = GenerationContext(prompt="hello", prompt_tokens=np.array([[1]]))
        assert ctx.prompt == "hello"
        assert ctx.system_prompt == ""
        assert ctx.temperature == 0.8
        assert ctx.top_k == 40
        assert ctx.top_p == 0.9
        assert ctx.max_tokens == 2048
        assert ctx.stop_tokens == []
        assert ctx.reasoning_depth == "balanced"
        assert ctx.cognitive_boost is True
        assert ctx.emotional_context == {}
        assert ctx.soul_overrides == {}
        assert ctx.reasoning_chain == []
        assert ctx.repetition_penalty == 1.2
        assert ctx.frequency_penalty == 0.0
        assert ctx.presence_penalty == 0.0

    def test_custom_values(self):
        ctx = GenerationContext(
            prompt="test",
            prompt_tokens=np.array([[0]]),
            temperature=0.5,
            top_k=10,
            max_tokens=512,
        )
        assert ctx.temperature == 0.5
        assert ctx.top_k == 10
        assert ctx.max_tokens == 512


# ── SloEngine basics ───────────────────────────────────────────────────────


class TestSloEngine:

    def test_init(self):
        engine = SloEngine()
        assert engine._model is None
        assert engine._soul.name == "default"
        assert engine._device == "cpu"
        assert engine.is_loaded is False

    def test_init_custom_soul(self):
        from domains.inference import SloProfile
        soul = SloProfile(name="custom")
        engine = SloEngine(soul=soul)
        assert engine.soul.name == "custom"

    def test_init_custom_device(self):
        engine = SloEngine(device="cuda")
        assert engine._device == "cuda"

    def test_init_max_history_min(self):
        engine = SloEngine(max_history_messages=1)
        assert engine._max_history_messages == 4

    def test_soul_property(self):
        engine = SloEngine()
        assert engine.soul is engine._soul

    def test_model_property(self):
        engine = SloEngine()
        assert engine.model is None

    def test_model_property_loaded(self):
        engine = SloEngine()
        mock_model = MagicMock()
        engine._model = mock_model
        assert engine.model is mock_model

    def test_is_loaded_false(self):
        engine = SloEngine()
        assert engine.is_loaded is False

    def test_is_loaded_true(self):
        engine = SloEngine()
        engine._model = MagicMock()
        assert engine.is_loaded is True

    def test_set_soul(self):
        engine = SloEngine()
        from domains.inference import SloProfile
        soul = SloProfile(name="new")
        result = engine.set_soul(soul)
        assert engine.soul.name == "new"
        assert result is engine

    def test_set_system_prompt(self):
        engine = SloEngine()
        engine.set_system_prompt("Be helpful")
        assert engine._soul.system_prompt == "Be helpful"

    def test_set_vocab(self):
        engine = SloEngine()
        stoi = {0: "a", 1: "b"}
        itos = {"a": 0, "b": 1}
        result = engine.set_vocab(stoi, itos)
        assert engine._stoi == stoi
        assert engine._itos == itos
        assert result is engine

    def test_set_tokenizer(self):
        engine = SloEngine()
        tok = MagicMock()
        result = engine.set_tokenizer(tok)
        assert engine._tokenizer is tok
        assert result is engine


# ── REASONING_TYPE_MAP ──────────────────────────────────────────────────────


class TestReasoningTypeMap:

    def test_balanced(self):
        assert SloEngine.REASONING_TYPE_MAP["balanced"] == "deductive"

    def test_deductive(self):
        assert SloEngine.REASONING_TYPE_MAP["deductive"] == "deductive"

    def test_inductive(self):
        assert SloEngine.REASONING_TYPE_MAP["inductive"] == "inductive"

    def test_creative(self):
        assert SloEngine.REASONING_TYPE_MAP["creative"] == "creative"

    def test_abductive(self):
        assert SloEngine.REASONING_TYPE_MAP["abductive"] == "abductive"

    def test_analogical(self):
        assert SloEngine.REASONING_TYPE_MAP["analogical"] == "analogical"


# ── _build_system_prompt ───────────────────────────────────────────────────


class TestBuildSystemPrompt:

    def test_basic(self):
        engine = SloEngine()
        prompt = engine._build_system_prompt()
        assert "You are default" in prompt

    def test_with_warmth(self):
        engine = SloEngine()
        engine._soul.personality.warmth = 0.8
        prompt = engine._build_system_prompt()
        assert "warm and empathetic" in prompt

    def test_with_coldness(self):
        engine = SloEngine()
        engine._soul.personality.warmth = 0.2
        prompt = engine._build_system_prompt()
        assert "precise and analytical" in prompt

    def test_with_curiosity(self):
        engine = SloEngine()
        engine._soul.personality.curiosity = 0.8
        prompt = engine._build_system_prompt()
        assert "curious and exploratory" in prompt

    def test_with_confidence(self):
        engine = SloEngine()
        engine._soul.personality.confidence = 0.8
        prompt = engine._build_system_prompt()
        assert "confident and direct" in prompt

    def test_with_low_confidence(self):
        engine = SloEngine()
        engine._soul.personality.confidence = 0.2
        prompt = engine._build_system_prompt()
        assert "thoughtful and measured" in prompt

    def test_with_creativity(self):
        engine = SloEngine()
        engine._soul.personality.creativity = 0.8
        prompt = engine._build_system_prompt()
        assert "creative and innovative" in prompt

    def test_with_humor(self):
        engine = SloEngine()
        engine._soul.personality.humor = 0.8
        prompt = engine._build_system_prompt()
        assert "witty and playful" in prompt


# ── _build_reasoning_chain_text ────────────────────────────────────────────


class TestBuildReasoningChainText:

    def test_basic(self):
        engine = SloEngine()
        text = engine._build_reasoning_chain_text("hello")
        assert "[SOUL_REASONING]" in text
        assert "[/SOUL_REASONING]" in text
        assert "reasoning_type:" in text

    def test_with_sentiment(self):
        engine = SloEngine()
        engine._cognitive_state["last_sentiment"] = 0.5
        engine._cognitive_state["last_emotion"] = "happy"
        text = engine._build_reasoning_chain_text("hello")
        assert "happy" in text
        assert "0.50" in text


# ── _get_generation_params ─────────────────────────────────────────────────


class TestGetGenerationParams:

    def test_defaults(self):
        engine = SloEngine()
        ctx = GenerationContext(prompt="t", prompt_tokens=np.array([[0]]))
        params = engine._get_generation_params(ctx)
        assert params["temperature"] == 0.8
        assert params["top_k"] == 40
        assert params["top_p"] == 0.9

    def test_deep_reasoning(self):
        engine = SloEngine()
        ctx = GenerationContext(prompt="t", prompt_tokens=np.array([[0]]), reasoning_depth="deep")
        params = engine._get_generation_params(ctx)
        assert params["temperature"] < 0.8

    def test_creative_reasoning(self):
        engine = SloEngine()
        ctx = GenerationContext(prompt="t", prompt_tokens=np.array([[0]]), reasoning_depth="creative")
        params = engine._get_generation_params(ctx)
        assert params["temperature"] > 0.8

    def test_high_warmth(self):
        engine = SloEngine()
        engine._soul.personality.warmth = 0.8
        ctx = GenerationContext(prompt="t", prompt_tokens=np.array([[0]]))
        params = engine._get_generation_params(ctx)
        assert params["temperature"] >= 0.8

    def test_soul_overrides(self):
        engine = SloEngine()
        ctx = GenerationContext(
            prompt="t", prompt_tokens=np.array([[0]]),
            soul_overrides={"temperature": 1.5}
        )
        params = engine._get_generation_params(ctx)
        assert params["temperature"] == 1.5


# ── _apply_hebbian_learning ────────────────────────────────────────────────


class TestHebbianLearning:

    def test_basic(self):
        engine = SloEngine()
        engine._apply_hebbian_learning(["hello", "world"], ["response"])
        assert "hello" in engine._hebbian_connections
        assert "world" in engine._hebbian_connections["hello"]

    def test_accumulation(self):
        engine = SloEngine()
        engine._apply_hebbian_learning(["a", "b"], [])
        engine._apply_hebbian_learning(["a", "b"], [])
        assert engine._hebbian_connections["a"]["b"] == 0.02

    def test_empty(self):
        engine = SloEngine()
        engine._apply_hebbian_learning([], [])
        assert engine._hebbian_connections == {}


# ── generate (no model) ────────────────────────────────────────────────────


class TestGenerateNoModel:

    def test_generate_no_model(self):
        engine = SloEngine()
        result = engine.generate("hello")
        assert "[Slo: default]" in result
        assert "no model loaded" in result


# ── _build_full_prompt ─────────────────────────────────────────────────────


class TestBuildFullPrompt:

    def test_basic(self):
        engine = SloEngine()
        prompt = engine._build_full_prompt("hello", include_reasoning=False)
        assert "User: hello" in prompt
        assert "Assistant:" in prompt

    def test_with_session_history(self):
        engine = SloEngine()
        engine._session_history = [
            {"role": "user", "content": "prev question"},
            {"role": "assistant", "content": "prev answer"},
        ]
        prompt = engine._build_full_prompt("hello", include_reasoning=False)
        assert "prev question" in prompt
        assert "CONVERSATION_HISTORY" in prompt

    def test_with_system_prompt(self):
        engine = SloEngine()
        engine._soul.system_prompt = "Be helpful"
        prompt = engine._build_full_prompt("hello", include_reasoning=False)
        assert "Be helpful" in prompt
