"""Tests for the shared inference generation adapter."""

import asyncio

from infrastructure.inference_generation import (
    count_generated_tokens,
    generate_text,
    stream_text,
)


class _Tokenizer:
    def encode(self, text):
        return [text, text]


class _Provider:
    def __init__(self):
        self.call = None

    async def chat(self, messages, **kwargs):
        self.call = (messages, kwargs)
        return "A generated answer."

    async def chat_stream(self, messages, **kwargs):
        self.stream_call = (messages, kwargs)
        yield "A generated "
        yield "answer."


def test_generate_text_delegates_with_shared_options_and_result_metrics():
    provider = _Provider()
    messages = [{"role": "user", "content": "Explain this."}]

    result = asyncio.run(
        generate_text(
            provider,
            messages,
            max_tokens=24,
            generation_params={"temperature": 0.2, "top_p": 0.9},
            tokenizer=_Tokenizer(),
        )
    )

    assert provider.call == (
        messages,
        {"max_tokens": 24, "temperature": 0.2, "top_p": 0.9},
    )
    assert result.text == "A generated answer."
    assert result.tokens_generated == 2
    assert result.elapsed_ms >= 0


def test_count_generated_tokens_falls_back_to_word_count():
    assert count_generated_tokens("one two three") == 3


def test_stream_text_uses_provider_in_process_and_forwards_controls():
    provider = _Provider()
    messages = [{"role": "user", "content": "Say hello"}]

    async def collect():
        return [
            token
            async for token in stream_text(
                provider,
                messages,
                max_tokens=12,
                generation_params={"temperature": 0.1},
                session_id="session-1",
            )
        ]

    assert asyncio.run(collect()) == ["A generated ", "answer."]
    assert provider.stream_call == (
        messages,
        {"max_tokens": 12, "cancel_event": None, "session_id": "session-1", "temperature": 0.1},
    )
