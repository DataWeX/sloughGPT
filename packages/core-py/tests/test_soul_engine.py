"""Tests for domains/core/soul.py (SloEngine, GenerationContext)."""

import asyncio
from unittest.mock import patch

import numpy as np
import pytest

from domains.core.soul import GenerationContext, SloEngine
from domains.inference import SloProfile


CHARSET = "abcdefghijklmnopqrstuvwxyz "


class FakeModel:
    """Minimal ModelInterface stand-in."""

    def __init__(self, charset: str = CHARSET):
        self.itos = {i: c for i, c in enumerate(charset)}
        self.stoi = {c: i for i, c in enumerate(charset)}
        self.device = "cpu"
        self.calls = []

    def generate(self, idx, max_new_tokens=10, temperature=0.8, top_k=40, top_p=0.9, **kwargs):
        self.calls.append((idx, max_new_tokens, temperature, top_k, top_p))
        hello = np.array([[ord(c) for c in "hello"]], dtype=np.int64)
        return np.concatenate([idx, hello], axis=1)

    def num_parameters(self):
        return 42

    def config(self):
        return {"vocab_size": len(self.itos)}

    def to(self, device):
        self.device = device
        return self


class RoundTripTokenizer:
    """Identity tokenizer whose encode/decode round-trips text exactly."""

    def encode(self, text):
        return [ord(c) for c in text]

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)


@pytest.fixture
def engine():
    """SloEngine with slow components (sentiment, HD memory) disabled for speed."""
    e = SloEngine()
    e._sentiment_analyzer = None
    e._hd_memory = None
    return e


@pytest.fixture
def model_engine():
    """SloEngine with a FakeModel and round-trip tokenizer."""
    model = FakeModel()
    e = SloEngine(model=model)
    e._sentiment_analyzer = None
    e._hd_memory = None
    e.set_tokenizer(RoundTripTokenizer())
    return e


# =============================================================================
# GenerationContext
# =============================================================================

class TestGenerationContext:
    def test_defaults(self):
        ctx = GenerationContext(prompt="hi", prompt_tokens=np.array([[1]]))
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

    def test_custom_fields(self):
        ctx = GenerationContext(
            prompt="p",
            prompt_tokens=np.array([[1]]),
            temperature=1.0,
            soul_overrides={"temperature": 0.3},
            stop_tokens=["\n"],
        )
        assert ctx.temperature == 1.0
        assert ctx.soul_overrides == {"temperature": 0.3}
        assert ctx.stop_tokens == ["\n"]


# =============================================================================
# Initialization and properties
# =============================================================================

class TestSloEngineInit:
    def test_default_soul(self):
        e = SloEngine()
        assert e.soul.name == "default"
        assert e.model is None
        assert e.is_loaded is False
        assert e._device == "cpu"

    def test_custom_soul(self):
        soul = SloProfile(name="sage", system_prompt="Be wise.")
        e = SloEngine(soul=soul)
        assert e.soul is soul

    def test_custom_model(self):
        model = FakeModel()
        e = SloEngine(model=model)
        assert e.model is model
        assert e.is_loaded is True

    def test_custom_device(self):
        e = SloEngine(device="gpu0")
        assert e._device == "gpu0"

    def test_max_history_clamped_low(self):
        e = SloEngine(max_history_messages=1)
        assert e._max_history_messages == 4

    def test_max_history_kept(self):
        e = SloEngine(max_history_messages=48)
        assert e._max_history_messages == 48

    def test_cognitive_components_loaded(self):
        e = SloEngine()
        assert e._reasoning_engine is not None
        assert e._deep_reasoning is not None
        assert e._logic_engine is not None
        assert e._working_memory is not None

    def test_sentiment_analyzer_loaded(self):
        e = SloEngine()
        assert e._sentiment_analyzer is not None

    def test_hd_memory_loaded(self):
        e = SloEngine()
        assert e._hd_memory is not None

    def test_initial_state(self):
        e = SloEngine()
        assert e._session_history == []
        assert e._cognitive_state["session_turns"] == 0
        assert e._hebbian_connections == {}
        assert e._cache_enabled is False
        assert e._generation_stats == {
            "total_generations": 0,
            "total_tokens": 0,
            "avg_latency_ms": 0.0,
        }

    def test_repr(self):
        assert "SloEngine" in repr(SloEngine())
        assert "no model" in repr(SloEngine())

    def test_repr_loaded(self):
        assert "loaded" in repr(SloEngine(model=FakeModel()))


# =============================================================================
# Setters
# =============================================================================

class TestSetters:
    def test_set_soul_returns_self(self, engine):
        soul = SloProfile(name="new_soul")
        assert engine.set_soul(soul) is engine
        assert engine.soul is soul

    def test_set_system_prompt(self, engine):
        engine.set_system_prompt("New instructions.")
        assert engine.soul.system_prompt == "New instructions."

    def test_set_vocab(self, engine):
        stoi = {"a": 0, "b": 1}
        itos = {0: "a", 1: "b"}
        assert engine.set_vocab(stoi, itos) is engine
        assert engine._stoi == stoi
        assert engine._itos == itos

    def test_set_tokenizer(self, engine):
        tok = object()
        assert engine.set_tokenizer(tok) is engine
        assert engine._tokenizer is tok


# =============================================================================
# Prompt building
# =============================================================================

class TestBuildSystemPrompt:
    def test_default_soul(self, engine):
        text = engine._build_system_prompt()
        assert text.startswith("You are default.")

    def test_includes_system_prompt(self, engine):
        engine.set_system_prompt("Always be concise.")
        text = engine._build_system_prompt()
        assert "Always be concise." in text

    def test_warm_trait(self, engine):
        engine._soul.personality.warmth = 0.9
        text = engine._build_system_prompt()
        assert "warm and empathetic" in text

    def test_analytical_trait(self, engine):
        engine._soul.personality.warmth = 0.1
        text = engine._build_system_prompt()
        assert "precise and analytical" in text

    def test_curious_trait(self, engine):
        engine._soul.personality.curiosity = 0.9
        text = engine._build_system_prompt()
        assert "curious and exploratory" in text

    def test_confidence_trait(self, engine):
        engine._soul.personality.confidence = 0.9
        text = engine._build_system_prompt()
        assert "confident and direct" in text

    def test_low_confidence_trait(self, engine):
        engine._soul.personality.confidence = 0.1
        text = engine._build_system_prompt()
        assert "thoughtful and measured" in text

    def test_creative_trait(self, engine):
        engine._soul.personality.creativity = 0.9
        text = engine._build_system_prompt()
        assert "creative and innovative" in text

    def test_humor_trait(self, engine):
        engine._soul.personality.humor = 0.9
        text = engine._build_system_prompt()
        assert "witty and playful" in text

    def test_neutral_personality_no_extra(self, engine):
        text = engine._build_system_prompt()
        assert "You are default." in text


class TestBuildReasoningChainText:
    def test_contains_blocks(self, engine):
        text = engine._build_reasoning_chain_text("hello")
        assert "[SOUL_REASONING]" in text
        assert "[/SOUL_REASONING]" in text

    def test_reasoning_type_from_approach(self, engine):
        engine._soul.behavior.reasoning_approach = "creative"
        text = engine._build_reasoning_chain_text("hello")
        assert "reasoning_type: creative" in text

    def test_reasoning_approach_line(self, engine):
        text = engine._build_reasoning_chain_text("hello")
        assert "reasoning_approach: balanced" in text

    def test_emotional_context_from_state(self, engine):
        engine._cognitive_state["last_emotion"] = "joy"
        engine._cognitive_state["last_sentiment"] = 0.8
        text = engine._build_reasoning_chain_text("hello")
        assert "joy (sentiment=0.80)" in text

    def test_session_turns(self, engine):
        engine._cognitive_state["session_turns"] = 7
        text = engine._build_reasoning_chain_text("hello")
        assert "session_turns: 7" in text

    def test_cognitive_scores(self, engine):
        engine._soul.cognition.pattern_recognition = 0.9
        text = engine._build_reasoning_chain_text("hello")
        assert "pattern_recognition=0.90" in text

    def test_personality_scores(self, engine):
        engine._soul.personality.warmth = 0.8
        text = engine._build_reasoning_chain_text("hello")
        assert "warmth=0.80" in text

    def test_reasoning_engine_active_line(self, engine):
        assert engine._reasoning_engine is not None
        text = engine._build_reasoning_chain_text("hello")
        assert "reasoning_engine: active" in text

    def test_hd_memory_line(self):
        e = SloEngine()
        text = e._build_reasoning_chain_text("hello")
        assert "hd_memory:" in text


class TestBuildFullPrompt:
    def test_includes_user_prompt(self, engine):
        text = engine._build_full_prompt("what is ai?")
        assert text.endswith("User: what is ai?\nAssistant:")

    def test_includes_system(self, engine):
        text = engine._build_full_prompt("hi")
        assert "You are default." in text

    def test_no_reasoning_without_context(self, engine):
        engine._reasoning_engine = None
        engine._cognitive_state["session_turns"] = 0
        text = engine._build_full_prompt("hi")
        assert "[SOUL_REASONING]" not in text

    def test_reasoning_with_turns(self, engine):
        engine._cognitive_state["session_turns"] = 1
        text = engine._build_full_prompt("hi")
        assert "[SOUL_REASONING]" in text

    def test_reasoning_with_engine(self, engine):
        text = engine._build_full_prompt("hi")
        assert "[SOUL_REASONING]" in text

    def test_conversation_history_included(self, engine):
        engine._session_history = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
        ]
        text = engine._build_full_prompt("third")
        assert "[CONVERSATION_HISTORY]" in text
        assert "User: first" in text
        assert "Assistant: second" in text

    def test_conversation_history_capped(self, engine):
        engine._max_history_messages = 4
        engine._session_history = [
            {"role": "user", "content": f"msg{i}"} for i in range(10)
        ]
        text = engine._build_full_prompt("now")
        assert "msg0" not in text
        assert "msg9" in text

    def test_include_reasoning_false(self, engine):
        engine._cognitive_state["session_turns"] = 1
        text = engine._build_full_prompt("hi", include_reasoning=False)
        assert "[SOUL_REASONING]" not in text


# =============================================================================
# Generation params
# =============================================================================

class TestGetGenerationParams:
    def _ctx(self, **overrides):
        defaults = dict(
            prompt="p",
            prompt_tokens=np.array([[1]]),
            temperature=0.8,
            top_k=40,
            top_p=0.9,
            max_tokens=2048,
            soul_overrides={},
            reasoning_depth="balanced",
        )
        defaults.update(overrides)
        return GenerationContext(**defaults)

    def test_defaults(self, engine):
        params = engine._get_generation_params(self._ctx())
        assert params["temperature"] == 0.8
        assert params["top_k"] == 40
        assert params["top_p"] == 0.9
        assert params["max_tokens"] == 2048
        assert params["repetition_penalty"] == 1.2

    def test_soul_overrides_temperature(self, engine):
        params = engine._get_generation_params(
            self._ctx(soul_overrides={"temperature": 0.1})
        )
        assert params["temperature"] == 0.1

    def test_soul_overrides_max_tokens(self, engine):
        params = engine._get_generation_params(
            self._ctx(soul_overrides={"max_tokens": 64})
        )
        assert params["max_tokens"] == 64

    def test_deep_reasoning_lowers_temp(self, engine):
        params = engine._get_generation_params(
            self._ctx(reasoning_depth="deep")
        )
        assert params["temperature"] == pytest.approx(0.5)

    def test_creative_reasoning_raises_temp(self, engine):
        params = engine._get_generation_params(
            self._ctx(reasoning_depth="creative")
        )
        assert params["temperature"] == pytest.approx(1.1)

    def test_warmth_raises_temp(self, engine):
        engine._soul.personality.warmth = 0.9
        params = engine._get_generation_params(self._ctx())
        assert params["temperature"] == pytest.approx(0.9)

    def test_warmth_temp_capped(self, engine):
        engine._soul.personality.warmth = 0.9
        params = engine._get_generation_params(
            self._ctx(reasoning_depth="creative", temperature=1.4)
        )
        assert params["temperature"] == pytest.approx(1.2)


# =============================================================================
# Hebbian learning
# =============================================================================

class TestHebbianLearning:
    def test_creates_connections(self, engine):
        engine._apply_hebbian_learning(["a", "b", "c"], [])
        assert "a" in engine._hebbian_connections
        assert "b" in engine._hebbian_connections
        assert engine._hebbian_connections["a"]["b"] == pytest.approx(0.01)
        assert engine._hebbian_connections["b"]["c"] == pytest.approx(0.01)

    def test_accumulates_strength(self, engine):
        engine._apply_hebbian_learning(["a", "b"], [])
        engine._apply_hebbian_learning(["a", "b"], [])
        assert engine._hebbian_connections["a"]["b"] == pytest.approx(0.02)

    def test_no_connections_for_single_token(self, engine):
        engine._apply_hebbian_learning(["a"], [])
        assert engine._hebbian_connections == {}


# =============================================================================
# Tokenization
# =============================================================================

class TestTokenization:
    def test_tokenize_fallback_ord(self, engine):
        tokens = engine._tokenize("ab")
        assert tokens.shape == (1, 2)
        assert tokens[0, 0] == ord("a") % 256

    def test_tokenize_with_stoi(self, engine):
        engine.set_vocab({"a": 7, "b": 8}, {7: "a", 8: "b"})
        tokens = engine._tokenize("ab")
        assert tokens.tolist() == [[7, 8]]

    def test_tokenize_unknown_char_defaults_zero(self, engine):
        engine.set_vocab({"a": 7}, {7: "a"})
        tokens = engine._tokenize("az")
        assert tokens.tolist() == [[7, 0]]

    def test_tokenize_with_tokenizer(self, engine):
        class Tok:
            def encode(self, text):
                return [1, 2, 3]

        engine.set_tokenizer(Tok())
        tokens = engine._tokenize("anything")
        assert tokens.tolist() == [[1, 2, 3]]

    def test_detokenize_with_itos(self, engine):
        engine.set_vocab({"a": 7, "b": 8}, {7: "a", 8: "b"})
        assert engine._detokenize(np.array([7, 8, 7])) == "aba"

    def test_detokenize_unknown_is_question(self, engine):
        engine.set_vocab({"a": 7}, {7: "a"})
        assert engine._detokenize(np.array([7, 99])) == "a?"

    def test_detokenize_fallback_chr(self, engine):
        assert engine._detokenize(np.array([65, 66])) == "AB"

    def test_detokenize_with_tokenizer(self, engine):
        class Tok:
            def decode(self, tokens):
                return "".join(chr(t) for t in tokens)

        engine.set_tokenizer(Tok())
        assert engine._detokenize(np.array([65, 66])) == "AB"


# =============================================================================
# generate()
# =============================================================================

class TestGenerate:
    def test_no_model_placeholder(self, engine):
        text = engine.generate("hello world")
        assert text == "[Slo: default] hello world... (no model loaded)"

    def test_appends_session_history(self, engine):
        engine.generate("hello")
        assert len(engine._session_history) == 2
        assert engine._session_history[0] == {"role": "user", "content": "hello"}
        assert engine._session_history[1]["role"] == "assistant"

    def test_increments_session_turns(self, engine):
        engine.generate("one")
        engine.generate("two")
        assert engine._cognitive_state["session_turns"] == 2

    def test_generation_stats_updated(self, engine):
        engine.generate("hello")
        assert engine._generation_stats["total_generations"] == 1

    def test_hebbian_learning_applied(self, engine):
        engine.generate("alpha beta gamma")
        assert engine._hebbian_connections != {}

    def test_history_capped_at_100(self, engine):
        for i in range(60):
            engine.generate(f"message {i}")
        assert len(engine._session_history) == 100

    def test_with_model(self, model_engine):
        text = model_engine.generate("say something")
        assert text == "hello"

    def test_with_model_uses_soul_generation_defaults(self, model_engine):
        model_engine.generate("hi", max_new_tokens=32, temperature=0.5)
        idx, max_new_tokens, temperature, top_k, top_p = model_engine._model.calls[0]
        assert max_new_tokens == 32
        assert temperature == 0.5

    def test_with_model_tokens_generated(self, model_engine):
        model_engine.generate("hi")
        assert model_engine._generation_stats["total_tokens"] == 5

    def test_model_error_captured(self):
        class BadModel:
            def generate(self, *args, **kwargs):
                raise RuntimeError("boom")

        e = SloEngine(model=BadModel())
        e._sentiment_analyzer = None
        e._hd_memory = None
        text = e.generate("hi")
        assert text.startswith("[Error:")
        assert "boom" in text

    def test_return_reasoning(self, engine):
        text, extra = engine.generate("hello", return_reasoning=True)
        assert "reasoning_chain" in extra
        assert "soul_context" in extra
        assert "full_prompt" in extra
        assert "user_message" in extra
        assert "latency_ms" in extra
        assert "tokens_generated" in extra
        assert "generation_params" in extra
        assert extra["user_message"] == "hello"

    def test_generate_calls_set_active_user(self):
        class UserModel(FakeModel):
            def set_active_user(self, user_id):
                self.active_user = user_id

        model = UserModel()
        e = SloEngine(model=model)
        e._sentiment_analyzer = None
        e._hd_memory = None
        e.set_vocab(model.stoi, model.itos)
        e.generate("hi", user_id="u42")
        assert model.active_user == "u42"


# =============================================================================
# generate_async()
# =============================================================================

class TestGenerateAsync:
    async def test_async_matches_sync(self, engine):
        sync_result = engine.generate("hello")
        async_result = await engine.generate_async("hello")
        assert async_result == sync_result

    async def test_async_appends_history(self, engine):
        await engine.generate_async("hello")
        assert len(engine._session_history) == 2


# =============================================================================
# chat()
# =============================================================================

class TestChat:
    def test_empty_messages_returns_empty(self, engine):
        assert engine.chat([]) == ""

    def test_last_message_must_be_user(self, engine):
        with pytest.raises(ValueError):
            engine.chat([{"role": "assistant", "content": "hi"}])

    def test_sets_history_from_prior(self, engine):
        engine.chat(
            [
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "two"},
                {"role": "user", "content": "three"},
            ]
        )
        assert len(engine._session_history) == 4
        assert engine._session_history[0] == {"role": "user", "content": "one"}
        assert engine._session_history[1] == {"role": "assistant", "content": "two"}
        assert engine._session_history[2] == {"role": "user", "content": "three"}
        assert engine._session_history[3]["role"] == "assistant"

    def test_skips_unknown_roles(self, engine):
        engine.chat(
            [
                {"role": "tool", "content": "ignored"},
                {"role": "user", "content": "final"},
            ]
        )
        assert len(engine._session_history) == 2
        assert engine._session_history[0]["role"] == "user"
        assert engine._session_history[0]["content"] == "final"
        assert engine._session_history[1]["role"] == "assistant"

    def test_chat_with_soul_alias(self, engine):
        result = engine.chat_with_soul([{"role": "user", "content": "hello"}])
        assert "no model loaded" in result


# =============================================================================
# Conversation state
# =============================================================================

class TestConversation:
    def test_clear_conversation(self, engine):
        engine.generate("hello")
        assert engine._cognitive_state["session_turns"] == 1
        engine.clear_conversation()
        assert engine._session_history == []
        assert engine._cognitive_state["session_turns"] == 0


# =============================================================================
# Personality
# =============================================================================

class TestApplyPersonality:
    def test_updates_traits(self, engine):
        engine.apply_personality(warmth=0.9, creativity=0.2)
        assert engine._soul.personality.warmth == 0.9
        assert engine._soul.personality.creativity == 0.2

    def test_partial_update(self, engine):
        engine.apply_personality(humor=0.1)
        assert engine._soul.personality.warmth == 0.5
        assert engine._soul.personality.humor == 0.1

    def test_returns_self(self, engine):
        assert engine.apply_personality(warmth=0.6) is engine

    def test_integrity_hash_recomputed(self, engine):
        engine.apply_personality(warmth=0.9)
        h1 = engine._soul.integrity_hash
        assert len(h1) == 16
        engine.apply_personality(warmth=0.1)
        h2 = engine._soul.integrity_hash
        assert h1 != h2


# =============================================================================
# Stats
# =============================================================================

class TestStats:
    def test_get_stats_structure(self, engine):
        stats = engine.get_stats()
        assert "soul" in stats
        assert "model" in stats
        assert "cognitive" in stats
        assert "generation" in stats
        assert stats["soul"]["name"] == "default"
        assert stats["model"]["loaded"] is False
        assert stats["model"]["params"] == 0
        assert stats["model"]["config"] == {}
        assert stats["cognitive"]["session_turns"] == 0
        assert stats["cognitive"]["hebbian_connections"] == 0

    def test_get_stats_with_model(self):
        e = SloEngine(model=FakeModel())
        e._sentiment_analyzer = None
        e._hd_memory = None
        stats = e.get_stats()
        assert stats["model"]["loaded"] is True
        assert stats["model"]["params"] == 42
        assert stats["model"]["config"]["vocab_size"] == len(CHARSET)

    def test_get_stats_after_generation(self, engine):
        engine.generate("hello")
        stats = engine.get_stats()
        assert stats["generation"]["total_generations"] == 1
        assert stats["cognitive"]["session_turns"] == 1

    def test_get_reasoning_stats(self, engine):
        stats = engine.get_reasoning_stats()
        assert stats["deep_reasoning"] is True
        assert stats["logic_engine"] is True
        assert stats["working_memory_items"] == 0
        assert stats["session_turns"] == 0
        assert stats["hebbian_connections"] == 0

    def test_status_structure(self, engine):
        status = engine.status()
        assert status["soul"]["name"] == "default"
        assert status["model"]["loaded"] is False
        assert status["tokenizer"]["trained"] is False
        assert "training" in status

    def test_status_with_soul_personality(self, engine):
        engine._soul.personality.warmth = 0.8
        status = engine.status()
        assert status["soul"]["personality"]["warmth"] == 0.8


# =============================================================================
# Working memory
# =============================================================================

class TestWorkingMemory:
    def test_add_and_get(self, engine):
        engine.add_to_working_memory("item1")
        engine.add_to_working_memory("item2")
        items = engine.get_working_memory(5)
        assert "item1" in items
        assert "item2" in items

    def test_clear(self, engine):
        engine.add_to_working_memory("item1")
        engine.clear_working_memory()
        assert engine.get_working_memory() == []

    def test_capacity_eviction(self, engine):
        for i in range(12):
            engine.add_to_working_memory(f"item{i}")
        assert len(engine.get_working_memory(20)) <= 7


# =============================================================================
# HD memory
# =============================================================================

class TestHDMemory:
    def test_stats(self):
        e = SloEngine()
        stats = e.get_hd_memory_stats()
        assert stats["enabled"] is True
        assert "total_items" in stats

    def test_add_and_search(self):
        e = SloEngine()
        mem_id = e.add_to_hd_memory("the quick brown fox", role="user")
        assert mem_id
        results = e.search_hd_memory("brown fox", top_k=3)
        assert len(results) >= 1

    def test_clear(self):
        e = SloEngine()
        e.add_to_hd_memory("something")
        cleared = e.clear_hd_memory()
        assert cleared >= 1
        stats = e.get_hd_memory_stats()
        assert stats["total_items"] == 0

    def test_get_context(self):
        e = SloEngine()
        e.add_to_hd_memory("Paris is the capital of France")
        ctx = e.get_hd_context("France")
        assert isinstance(ctx, str)

    def test_disabled_returns_defaults(self, engine):
        assert engine.get_hd_memory_stats() == {
            "enabled": False,
            "error": "HD memory not initialized",
        }
        assert engine.search_hd_memory("q") == []
        assert engine.add_to_hd_memory("c") == ""
        assert engine.clear_hd_memory() == 0
        assert engine.get_hd_context("q") == ""


# =============================================================================
# Semantic cache
# =============================================================================

class TestSemanticCache:
    def test_enable_cache(self, engine):
        result = engine.enable_cache()
        assert result["enabled"] is True
        assert result["max_entries"] == 500
        assert engine._cache_enabled is True

    def test_enable_cache_custom_params(self, engine):
        result = engine.enable_cache(max_entries=100, similarity_threshold=0.5)
        assert result["max_entries"] == 100
        assert result["similarity_threshold"] == 0.5

    def test_disable_cache(self, engine):
        engine.enable_cache()
        assert engine.disable_cache() is True
        assert engine._cache_enabled is False

    def test_get_cache_stats(self, engine):
        engine.enable_cache()
        stats = engine.get_cache_stats()
        assert stats["enabled"] is True
        assert stats["cache_enabled"] is True
        assert stats["entries"] == 0

    def test_cache_hit_short_circuits(self, model_engine):
        model_engine.enable_cache()
        model_engine.generate("alpha beta gamma delta epsilon")
        text = model_engine.generate("alpha beta gamma delta epsilon")
        assert text == "hello"

    def test_cache_hit_skips_model_call(self, model_engine):
        model_engine.enable_cache()
        model_engine.generate("alpha beta gamma delta epsilon")
        calls_before = len(model_engine._model.calls)
        model_engine.generate("alpha beta gamma delta epsilon")
        assert len(model_engine._model.calls) == calls_before

    def test_clear_cache(self, model_engine):
        model_engine.enable_cache()
        model_engine.generate("cache me now")
        cleared = model_engine.clear_cache()
        assert cleared >= 1

    def test_invalidate_cache_entry(self, model_engine):
        model_engine.enable_cache()
        model_engine.generate("unique query words here")
        assert model_engine.invalidate_cache_entry("unique query words here") is True

    def test_invalidate_nonexistent(self, engine):
        engine.enable_cache()
        assert engine.invalidate_cache_entry("no such entry exists") is False

    def test_get_cache_stats_without_cache(self, engine):
        assert engine.get_cache_stats() == {
            "enabled": False,
            "error": "Cache not initialized",
        }

    def test_clear_cache_without_cache(self, engine):
        assert engine.clear_cache() == 0

    def test_invalidate_without_cache(self, engine):
        assert engine.invalidate_cache_entry("q") is False


# =============================================================================
# Optimization / benchmarking (removed stubs)
# =============================================================================

class TestOptimization:
    def test_optimize_inference_removed(self, engine):
        result = engine.optimize_inference()
        assert result["success"] is False
        assert "removed" in result["error"]

    def test_benchmark_inference_removed(self, engine):
        result = engine.benchmark_inference()
        assert result["error"]
        assert "removed" in result["error"]


# =============================================================================
# Device
# =============================================================================

class TestDevice:
    def test_to_updates_device(self, engine):
        assert engine.to("gpu1") is engine
        assert engine._device == "gpu1"

    def test_to_moves_model(self):
        model = FakeModel()
        e = SloEngine(model=model)
        e.to("gpu1")
        assert model.device == "gpu1"


# =============================================================================
# Reasoning wrappers
# =============================================================================

class TestReasoningWrappers:
    async def test_deep_reason(self, engine):
        result = await engine.deep_reason("test problem", max_depth=1)
        assert "conclusion" in result
        assert "confidence" in result
        assert "steps" in result
        assert isinstance(result["steps"], list)

    def test_prove_syllogism_valid(self, engine):
        result = engine.prove_syllogism(
            ("All", "humans", "mortal"),
            ("All", "socrates", "human"),
            ("All", "socrates", "mortal"),
        )
        assert isinstance(result, dict)
        assert result["valid"] is True
        assert result["mood"] == "AAA"

    def test_prove_syllogism_invalid(self, engine):
        result = engine.prove_syllogism(
            ("No", "As", "Bs"),
            ("All", "Cs", "Ds"),
            ("All", "Es", "Fs"),
        )
        assert result["valid"] is False

    def test_assert_and_query_knowledge(self, engine):
        engine.assert_knowledge("human", "socrates")
        assert engine.query_knowledge("human", "socrates") is True
        assert engine.query_knowledge("human", "plato") is False

    def test_query_knowledge_missing_predicate(self, engine):
        assert engine.query_knowledge("mortal", "socrates") is False

    def test_deep_reason_unavailable(self):
        e = SloEngine()
        e._deep_reasoning = None
        result = asyncio.run(e.deep_reason("p"))
        assert result == {"error": "Deep reasoning not available"}

    def test_prove_syllogism_unavailable(self):
        e = SloEngine()
        e._logic_engine = None
        assert e.prove_syllogism(("A", "x", "y"), ("A", "y", "z"), ("A", "x", "z")) == {
            "error": "Logic engine not available"
        }

    def test_query_knowledge_unavailable(self):
        e = SloEngine()
        e._logic_engine = None
        assert e.query_knowledge("human", "socrates") is False


# =============================================================================
# Grounding
# =============================================================================

class TestGrounding:
    def test_enable_grounding(self, engine):
        result = engine.enable_grounding()
        assert result["enabled"] is True
        assert engine._grounding is not None

    def test_add_knowledge(self, engine):
        engine.enable_grounding()
        result = engine.add_knowledge("Paris is located in France")
        assert result["success"] is True

    def test_add_knowledge_auto_enables(self, engine):
        result = engine.add_knowledge("Water causes rust")
        assert result["success"] is True
        assert engine._grounding is not None

    def test_ground_output(self, engine):
        engine.enable_grounding()
        result = engine.ground_output("Paris is a city", "Where is Paris?")
        assert "confidence" in result
        assert "verified" in result

    def test_ground_output_before_enable(self, engine):
        result = engine.ground_output("x", "y")
        assert "error" in result

    def test_get_knowledge_context(self, engine):
        engine.enable_grounding()
        ctx = engine.get_knowledge_context("Paris")
        assert isinstance(ctx, str)

    def test_get_knowledge_context_before_enable(self, engine):
        assert engine.get_knowledge_context("q") == ""


# =============================================================================
# save_soul
# =============================================================================

class TestSaveSoul:
    def test_requires_model(self, engine):
        with pytest.raises(ValueError):
            engine.save_soul("/tmp/out.sou")

    def test_save_soul_calls_save_soul(self):
        e = SloEngine(model=FakeModel())
        e._sentiment_analyzer = None
        e._hd_memory = None
        with patch("domains.core.soul.save_soul") as mock_save:
            with patch("builtins.open") as mock_open:
                path = e.save_soul("/tmp/out.sou")
        assert path == "/tmp/out.sou"
        assert mock_save.call_count == 1
        assert mock_open.call_count == 1
