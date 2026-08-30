"""Tests for domains.core.soul — GenerationContext."""

import numpy as np
import pytest
from domains.core.soul import GenerationContext


class TestGenerationContextDefaults:
    def test_init_minimal(self):
        gc = GenerationContext(
            prompt="hello",
            prompt_tokens=np.array([1, 2, 3]),
        )
        assert gc.prompt == "hello"
        assert np.array_equal(gc.prompt_tokens, np.array([1, 2, 3]))
        assert gc.temperature == 0.8
        assert gc.top_k == 40
        assert gc.top_p == 0.9
        assert gc.max_tokens == 2048

    def test_defaults_system_prompt(self):
        gc = GenerationContext(prompt="t", prompt_tokens=np.array([1]))
        assert gc.system_prompt == ""

    def test_defaults_stop_tokens(self):
        gc = GenerationContext(prompt="t", prompt_tokens=np.array([1]))
        assert gc.stop_tokens == []

    def test_defaults_reasoning_depth(self):
        gc = GenerationContext(prompt="t", prompt_tokens=np.array([1]))
        assert gc.reasoning_depth == "balanced"

    def test_defaults_repetition_penalty(self):
        gc = GenerationContext(prompt="t", prompt_tokens=np.array([1]))
        assert gc.repetition_penalty == 1.2

    def test_defaults_frequency_penalty(self):
        gc = GenerationContext(prompt="t", prompt_tokens=np.array([1]))
        assert gc.frequency_penalty == 0.0

    def test_defaults_presence_penalty(self):
        gc = GenerationContext(prompt="t", prompt_tokens=np.array([1]))
        assert gc.presence_penalty == 0.0

    def test_defaults_cognitive_boost(self):
        gc = GenerationContext(prompt="t", prompt_tokens=np.array([1]))
        assert gc.cognitive_boost is True

    def test_defaults_emotional_context(self):
        gc = GenerationContext(prompt="t", prompt_tokens=np.array([1]))
        assert gc.emotional_context == {}

    def test_defaults_soul_overrides(self):
        gc = GenerationContext(prompt="t", prompt_tokens=np.array([1]))
        assert gc.soul_overrides == {}

    def test_defaults_reasoning_chain(self):
        gc = GenerationContext(prompt="t", prompt_tokens=np.array([1]))
        assert gc.reasoning_chain == []


class TestGenerationContextCustomTemperature:
    def test_custom_temperature_low(self):
        gc = GenerationContext(prompt="t", prompt_tokens=np.array([1]), temperature=0.1)
        assert gc.temperature == 0.1

    def test_custom_temperature_high(self):
        gc = GenerationContext(prompt="t", prompt_tokens=np.array([1]), temperature=2.0)
        assert gc.temperature == 2.0

    def test_custom_temperature_zero(self):
        gc = GenerationContext(prompt="t", prompt_tokens=np.array([1]), temperature=0.0)
        assert gc.temperature == 0.0

    def test_custom_temperature_negative(self):
        gc = GenerationContext(prompt="t", prompt_tokens=np.array([1]), temperature=-0.5)
        assert gc.temperature == -0.5

    def test_custom_temperature_default_unchanged(self):
        gc = GenerationContext(prompt="t", prompt_tokens=np.array([1]))
        assert gc.temperature == 0.8


class TestGenerationContextCustomTopK:
    def test_custom_top_k_small(self):
        gc = GenerationContext(prompt="t", prompt_tokens=np.array([1]), top_k=1)
        assert gc.top_k == 1

    def test_custom_top_k_large(self):
        gc = GenerationContext(prompt="t", prompt_tokens=np.array([1]), top_k=1000)
        assert gc.top_k == 1000

    def test_custom_top_k_zero(self):
        gc = GenerationContext(prompt="t", prompt_tokens=np.array([1]), top_k=0)
        assert gc.top_k == 0

    def test_custom_top_k_default_unchanged(self):
        gc = GenerationContext(prompt="t", prompt_tokens=np.array([1]))
        assert gc.top_k == 40


class TestGenerationContextCustomTopP:
    def test_custom_top_p_low(self):
        gc = GenerationContext(prompt="t", prompt_tokens=np.array([1]), top_p=0.1)
        assert gc.top_p == 0.1

    def test_custom_top_p_one(self):
        gc = GenerationContext(prompt="t", prompt_tokens=np.array([1]), top_p=1.0)
        assert gc.top_p == 1.0

    def test_custom_top_p_zero(self):
        gc = GenerationContext(prompt="t", prompt_tokens=np.array([1]), top_p=0.0)
        assert gc.top_p == 0.0

    def test_custom_top_p_default_unchanged(self):
        gc = GenerationContext(prompt="t", prompt_tokens=np.array([1]))
        assert gc.top_p == 0.9


class TestGenerationContextCustomMaxTokens:
    def test_custom_max_tokens_small(self):
        gc = GenerationContext(prompt="t", prompt_tokens=np.array([1]), max_tokens=1)
        assert gc.max_tokens == 1

    def test_custom_max_tokens_large(self):
        gc = GenerationContext(prompt="t", prompt_tokens=np.array([1]), max_tokens=100000)
        assert gc.max_tokens == 100000

    def test_custom_max_tokens_default_unchanged(self):
        gc = GenerationContext(prompt="t", prompt_tokens=np.array([1]))
        assert gc.max_tokens == 2048


class TestGenerationContextSystemPrompt:
    def test_custom_system_prompt(self):
        gc = GenerationContext(
            prompt="t", prompt_tokens=np.array([1]),
            system_prompt="Be helpful.",
        )
        assert gc.system_prompt == "Be helpful."

    def test_custom_system_prompt_empty(self):
        gc = GenerationContext(
            prompt="t", prompt_tokens=np.array([1]),
            system_prompt="",
        )
        assert gc.system_prompt == ""

    def test_custom_system_prompt_long(self):
        long_prompt = "x " * 500
        gc = GenerationContext(
            prompt="t", prompt_tokens=np.array([1]),
            system_prompt=long_prompt,
        )
        assert len(gc.system_prompt) > 500


class TestGenerationContextStopTokens:
    def test_custom_stop_tokens(self):
        gc = GenerationContext(
            prompt="t", prompt_tokens=np.array([1]),
            stop_tokens=["EOF", "STOP"],
        )
        assert gc.stop_tokens == ["EOF", "STOP"]

    def test_stop_tokens_single(self):
        gc = GenerationContext(
            prompt="t", prompt_tokens=np.array([1]),
            stop_tokens=["END"],
        )
        assert len(gc.stop_tokens) == 1

    def test_stop_tokens_multiple(self):
        gc = GenerationContext(
            prompt="t", prompt_tokens=np.array([1]),
            stop_tokens=["A", "B", "C"],
        )
        assert len(gc.stop_tokens) == 3

    def test_stop_tokens_empty_string(self):
        gc = GenerationContext(
            prompt="t", prompt_tokens=np.array([1]),
            stop_tokens=[""],
        )
        assert gc.stop_tokens == [""]


class TestGenerationContextReasoningDepth:
    def test_custom_reasoning_depth_deep(self):
        gc = GenerationContext(
            prompt="t", prompt_tokens=np.array([1]),
            reasoning_depth="deep",
        )
        assert gc.reasoning_depth == "deep"

    def test_custom_reasoning_depth_creative(self):
        gc = GenerationContext(
            prompt="t", prompt_tokens=np.array([1]),
            reasoning_depth="creative",
        )
        assert gc.reasoning_depth == "creative"

    def test_custom_reasoning_depth_balanced(self):
        gc = GenerationContext(
            prompt="t", prompt_tokens=np.array([1]),
            reasoning_depth="balanced",
        )
        assert gc.reasoning_depth == "balanced"

    def test_custom_reasoning_depth_analytical(self):
        gc = GenerationContext(
            prompt="t", prompt_tokens=np.array([1]),
            reasoning_depth="analytical",
        )
        assert gc.reasoning_depth == "analytical"


class TestGenerationContextCognitiveBoost:
    def test_custom_cognitive_boost_false(self):
        gc = GenerationContext(
            prompt="t", prompt_tokens=np.array([1]),
            cognitive_boost=False,
        )
        assert gc.cognitive_boost is False

    def test_custom_cognitive_boost_true(self):
        gc = GenerationContext(
            prompt="t", prompt_tokens=np.array([1]),
            cognitive_boost=True,
        )
        assert gc.cognitive_boost is True


class TestGenerationContextEmotionalContext:
    def test_custom_emotional_context(self):
        ec = {"sentiment": 0.5, "emotion": "happy"}
        gc = GenerationContext(
            prompt="t", prompt_tokens=np.array([1]),
            emotional_context=ec,
        )
        assert gc.emotional_context == ec

    def test_emotional_context_empty(self):
        gc = GenerationContext(
            prompt="t", prompt_tokens=np.array([1]),
            emotional_context={},
        )
        assert gc.emotional_context == {}

    def test_emotional_context_nested(self):
        ec = {"primary": {"emotion": "joy", "intensity": 0.9}, "secondary": "calm"}
        gc = GenerationContext(
            prompt="t", prompt_tokens=np.array([1]),
            emotional_context=ec,
        )
        assert gc.emotional_context["primary"]["intensity"] == 0.9

    def test_emotional_context_many_keys(self):
        ec = {f"key_{i}": i * 0.1 for i in range(10)}
        gc = GenerationContext(
            prompt="t", prompt_tokens=np.array([1]),
            emotional_context=ec,
        )
        assert len(gc.emotional_context) == 10


class TestGenerationContextSoulOverrides:
    def test_custom_soul_overrides(self):
        so = {"temperature": 0.3}
        gc = GenerationContext(
            prompt="t", prompt_tokens=np.array([1]),
            soul_overrides=so,
        )
        assert gc.soul_overrides == so

    def test_soul_overrides_empty(self):
        gc = GenerationContext(
            prompt="t", prompt_tokens=np.array([1]),
            soul_overrides={},
        )
        assert gc.soul_overrides == {}

    def test_soul_overrides_many_keys(self):
        so = {"temperature": 0.1, "top_k": 1, "top_p": 0.1, "max_tokens": 10}
        gc = GenerationContext(
            prompt="t", prompt_tokens=np.array([1]),
            soul_overrides=so,
        )
        assert len(gc.soul_overrides) == 4

    def test_soul_overrides_nested(self):
        so = {"nested": {"key": "value"}}
        gc = GenerationContext(
            prompt="t", prompt_tokens=np.array([1]),
            soul_overrides=so,
        )
        assert gc.soul_overrides["nested"]["key"] == "value"


class TestGenerationContextReasoningChain:
    def test_custom_reasoning_chain(self):
        chain = ["step 1", "step 2"]
        gc = GenerationContext(
            prompt="t", prompt_tokens=np.array([1]),
            reasoning_chain=chain,
        )
        assert gc.reasoning_chain == chain

    def test_reasoning_chain_empty(self):
        gc = GenerationContext(
            prompt="t", prompt_tokens=np.array([1]),
            reasoning_chain=[],
        )
        assert gc.reasoning_chain == []

    def test_reasoning_chain_single(self):
        gc = GenerationContext(
            prompt="t", prompt_tokens=np.array([1]),
            reasoning_chain=["only step"],
        )
        assert len(gc.reasoning_chain) == 1

    def test_reasoning_chain_multiple(self):
        chain = [f"step_{i}" for i in range(20)]
        gc = GenerationContext(
            prompt="t", prompt_tokens=np.array([1]),
            reasoning_chain=chain,
        )
        assert len(gc.reasoning_chain) == 20


class TestGenerationContextPenalties:
    def test_custom_repetition_penalty(self):
        gc = GenerationContext(
            prompt="t", prompt_tokens=np.array([1]),
            repetition_penalty=2.0,
        )
        assert gc.repetition_penalty == 2.0

    def test_custom_repetition_penalty_one(self):
        gc = GenerationContext(
            prompt="t", prompt_tokens=np.array([1]),
            repetition_penalty=1.0,
        )
        assert gc.repetition_penalty == 1.0

    def test_custom_frequency_penalty(self):
        gc = GenerationContext(
            prompt="t", prompt_tokens=np.array([1]),
            frequency_penalty=0.5,
        )
        assert gc.frequency_penalty == 0.5

    def test_custom_frequency_penalty_zero(self):
        gc = GenerationContext(
            prompt="t", prompt_tokens=np.array([1]),
            frequency_penalty=0.0,
        )
        assert gc.frequency_penalty == 0.0

    def test_custom_presence_penalty(self):
        gc = GenerationContext(
            prompt="t", prompt_tokens=np.array([1]),
            presence_penalty=0.3,
        )
        assert gc.presence_penalty == 0.3

    def test_custom_presence_penalty_zero(self):
        gc = GenerationContext(
            prompt="t", prompt_tokens=np.array([1]),
            presence_penalty=0.0,
        )
        assert gc.presence_penalty == 0.0


class TestGenerationContextAllFieldsCustom:
    def test_all_fields_custom(self):
        gc = GenerationContext(
            prompt="full test",
            prompt_tokens=np.array([10, 20]),
            system_prompt="sys",
            temperature=0.1,
            top_k=5,
            top_p=0.3,
            max_tokens=50,
            stop_tokens=["EOS"],
            reasoning_depth="deep",
            cognitive_boost=False,
            emotional_context={"anger": 0.9},
            soul_overrides={"temperature": 0.9},
            reasoning_chain=["r1"],
            repetition_penalty=3.0,
            frequency_penalty=0.7,
            presence_penalty=0.8,
        )
        assert gc.prompt == "full test"
        assert gc.temperature == 0.1
        assert gc.top_k == 5
        assert gc.top_p == 0.3
        assert gc.max_tokens == 50
        assert gc.stop_tokens == ["EOS"]
        assert gc.reasoning_depth == "deep"
        assert gc.cognitive_boost is False
        assert gc.emotional_context == {"anger": 0.9}
        assert gc.soul_overrides == {"temperature": 0.9}
        assert gc.reasoning_chain == ["r1"]
        assert gc.repetition_penalty == 3.0
        assert gc.frequency_penalty == 0.7
        assert gc.presence_penalty == 0.8


class TestGenerationContextTokenArray:
    def test_prompt_tokens_is_numpy(self):
        gc = GenerationContext(prompt="t", prompt_tokens=np.array([1]))
        assert isinstance(gc.prompt_tokens, np.ndarray)

    def test_prompt_tokens_single_element(self):
        gc = GenerationContext(prompt="t", prompt_tokens=np.array([42]))
        assert gc.prompt_tokens[0] == 42

    def test_prompt_tokens_multiple_elements(self):
        gc = GenerationContext(prompt="t", prompt_tokens=np.array([1, 2, 3, 4, 5]))
        assert len(gc.prompt_tokens) == 5

    def test_prompt_tokens_empty_array(self):
        gc = GenerationContext(prompt="t", prompt_tokens=np.array([]))
        assert len(gc.prompt_tokens) == 0

    def test_prompt_tokens_2d_array(self):
        gc = GenerationContext(prompt="t", prompt_tokens=np.array([[1, 2], [3, 4]]))
        assert gc.prompt_tokens.shape == (2, 2)

    def test_prompt_tokens_large_array(self):
        big_arr = np.arange(1000)
        gc = GenerationContext(prompt="t", prompt_tokens=big_arr)
        assert len(gc.prompt_tokens) == 1000


class TestGenerationContextPromptVariations:
    def test_empty_prompt(self):
        gc = GenerationContext(prompt="", prompt_tokens=np.array([]))
        assert gc.prompt == ""

    def test_large_prompt(self):
        big = "x" * 10000
        gc = GenerationContext(prompt=big, prompt_tokens=np.array([1]))
        assert len(gc.prompt) == 10000

    def test_prompt_with_newlines(self):
        gc = GenerationContext(prompt="line1\nline2\nline3", prompt_tokens=np.array([1]))
        assert "\n" in gc.prompt

    def test_prompt_with_special_chars(self):
        gc = GenerationContext(prompt="!@#$%^&*()", prompt_tokens=np.array([1]))
        assert gc.prompt == "!@#$%^&*()"

    def test_prompt_unicode(self):
        gc = GenerationContext(prompt="hello world", prompt_tokens=np.array([1]))
        assert gc.prompt == "hello world"


class TestGenerationContextEdgeCases:
    def test_temperature_very_high(self):
        gc = GenerationContext(prompt="t", prompt_tokens=np.array([1]), temperature=100.0)
        assert gc.temperature == 100.0

    def test_top_k_very_large(self):
        gc = GenerationContext(prompt="t", prompt_tokens=np.array([1]), top_k=999999)
        assert gc.top_k == 999999

    def test_max_tokens_one(self):
        gc = GenerationContext(prompt="t", prompt_tokens=np.array([1]), max_tokens=1)
        assert gc.max_tokens == 1

    def test_repetition_penalty_zero(self):
        gc = GenerationContext(prompt="t", prompt_tokens=np.array([1]), repetition_penalty=0.0)
        assert gc.repetition_penalty == 0.0

    def test_frequency_penalty_negative(self):
        gc = GenerationContext(prompt="t", prompt_tokens=np.array([1]), frequency_penalty=-1.0)
        assert gc.frequency_penalty == -1.0

    def test_presence_penalty_large(self):
        gc = GenerationContext(prompt="t", prompt_tokens=np.array([1]), presence_penalty=5.0)
        assert gc.presence_penalty == 5.0

    def test_stop_tokens_many(self):
        stops = [f"STOP_{i}" for i in range(50)]
        gc = GenerationContext(prompt="t", prompt_tokens=np.array([1]), stop_tokens=stops)
        assert len(gc.stop_tokens) == 50

    def test_reasoning_chain_many_steps(self):
        chain = [f"reasoning step {i}" for i in range(100)]
        gc = GenerationContext(prompt="t", prompt_tokens=np.array([1]), reasoning_chain=chain)
        assert len(gc.reasoning_chain) == 100

    def test_soul_overrides_complex(self):
        so = {
            "temperature": 0.5,
            "top_k": 10,
            "top_p": 0.8,
            "max_tokens": 512,
            "repetition_penalty": 1.5,
            "frequency_penalty": 0.2,
            "presence_penalty": 0.3,
        }
        gc = GenerationContext(prompt="t", prompt_tokens=np.array([1]), soul_overrides=so)
        assert len(gc.soul_overrides) == 7

    def test_emotional_context_complex(self):
        ec = {
            "primary": {"emotion": "joy", "intensity": 0.9},
            "secondary": {"emotion": "calm", "intensity": 0.6},
            "sentiment": 0.75,
            "arousal": 0.3,
            "valence": 0.8,
        }
        gc = GenerationContext(prompt="t", prompt_tokens=np.array([1]), emotional_context=ec)
        assert len(gc.emotional_context) == 5
