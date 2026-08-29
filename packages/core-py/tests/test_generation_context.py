"""Tests for domains.core.soul — GenerationContext."""

import numpy as np
from domains.core.soul import GenerationContext


class TestGenerationContext:
    def test_init(self):
        gc = GenerationContext(
            prompt="hello",
            prompt_tokens=np.array([1, 2, 3]),
        )
        assert gc.prompt == "hello"
        assert gc.temperature == 0.8
        assert gc.top_k == 40
        assert gc.max_tokens == 2048

    def test_defaults(self):
        gc = GenerationContext(
            prompt="test",
            prompt_tokens=np.array([1]),
        )
        assert gc.system_prompt == ""
        assert gc.stop_tokens == []
        assert gc.reasoning_depth == "balanced"
        assert gc.repetition_penalty == 1.2

    def test_custom(self):
        gc = GenerationContext(
            prompt="test",
            prompt_tokens=np.array([1]),
            temperature=0.5,
            top_k=10,
            max_tokens=100,
        )
        assert gc.temperature == 0.5
        assert gc.top_k == 10
        assert gc.max_tokens == 100
