"""
Tests for SloNetChatProvider.chat_stream() robustness paths.

FEATURE: slonet-provider-wave-i — Producer-error propagation, total-generation
timeout, per-token wait timeout, and error-queue draining. DO NOT DELETE.
"""
import asyncio
import queue
import threading
import time

import pytest

from domains.inference import slonet_provider as slonet_provider_module
from domains.inference.slonet_provider import SloNetChatProvider


class MockTokenizer:
    """Minimal tokenizer — token IDs map to ASCII-ish text."""

    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.eos_token_id = 2

    def encode(self, text):
        return [abs(hash(c)) % self.vocab_size for c in text[:20]]

    def decode(self, token_ids):
        if isinstance(token_ids, (list, tuple)):
            return "".join(chr(65 + (tid % 26)) for tid in token_ids)
        return chr(65 + (token_ids % 26))

    def apply_chat_template(self, messages):
        if messages and isinstance(messages[-1], dict):
            return messages[-1].get("content", "")
        return ""


def _make_provider(model):
    provider = SloNetChatProvider.__new__(SloNetChatProvider)
    provider._hf_model_id = "wave-i"
    provider._model_id = "wave-i"
    provider._device = "cpu"
    provider._model = model
    provider._tokenizer = MockTokenizer()
    provider._quant_engine = None
    provider._parser = None
    return provider


def _collect(generator):
    """Fully drain an async generator into a list."""
    async def _run():
        return [item async for item in generator]
    return asyncio.run(_run())


class _StreamingModel:
    """Model stub whose generate_numpy_stream behavior is injectable."""

    def __init__(self, streamer):
        self._streamer = streamer

    def generate_numpy_stream(self, input_ids, **kwargs):
        return iter(self._streamer())


class TestProducerError:
    def test_error_after_tokens_surfaces_via_error_queue(self):
        def streamer():
            yield 1
            yield 2
            raise RuntimeError("kernel exploded")

        provider = _make_provider(_StreamingModel(streamer))
        out = _collect(provider.chat_stream([{"role": "user", "content": "hi"}], max_tokens=10))
        joined = "".join(out)
        assert "Generation error" in joined
        assert "kernel exploded" in joined

    def test_immediate_error_surfaces(self):
        def streamer():
            raise ValueError("no tokens at all")

        provider = _make_provider(_StreamingModel(streamer))
        out = _collect(provider.chat_stream([{"role": "user", "content": "hi"}], max_tokens=10))
        joined = "".join(out)
        assert "Generation error" in joined
        assert "no tokens at all" in joined


class TestTotalTimeout:
    def test_generation_timeout_yields_message_and_sets_cancel(self, monkeypatch):
        monkeypatch.setattr(slonet_provider_module, "_STREAM_TOTAL_TIMEOUT_S", 0.5)

        async def flaky_wait_for(fut, timeout=None):
            fut.close()
            await asyncio.sleep(0.1)
            raise asyncio.TimeoutError()

        monkeypatch.setattr(asyncio, "wait_for", flaky_wait_for)

        def streamer():
            while True:
                time.sleep(0.05)

        provider = _make_provider(_StreamingModel(streamer))
        cancel_event = threading.Event()
        out = _collect(provider.chat_stream(
            [{"role": "user", "content": "hi"}], max_tokens=10, cancel_event=cancel_event
        ))
        assert out[-1] == "\n\n[Generation timed out after 1s]"
        assert cancel_event.is_set()


class TestWaitTimeout:
    def test_timeout_with_dead_thread_drains_remaining_tokens(self, monkeypatch):
        async def flaky_wait_for(fut, timeout=None):
            fut.close()
            await asyncio.sleep(0.2)
            raise asyncio.TimeoutError()

        monkeypatch.setattr(asyncio, "wait_for", flaky_wait_for)

        def streamer():
            yield 1

        provider = _make_provider(_StreamingModel(streamer))
        out = _collect(provider.chat_stream([{"role": "user", "content": "hi"}], max_tokens=10))
        assert out == ["B"]

    def test_timeout_with_alive_thread_and_error_yields_error(self, monkeypatch):
        real_wait_for = asyncio.wait_for
        calls = []

        async def flaky_wait_for(fut, timeout=None):
            calls.append(1)
            if len(calls) >= 2:
                fut.close()
                raise asyncio.TimeoutError()
            return await real_wait_for(fut, timeout)

        monkeypatch.setattr(asyncio, "wait_for", flaky_wait_for)

        captured = []
        real_queue = queue.Queue

        class RecordingQueue(real_queue):
            def __init__(self):
                super().__init__()
                captured.append(self)

        monkeypatch.setattr(queue, "Queue", RecordingQueue)

        started = threading.Event()

        def streamer():
            started.set()
            yield 1
            while True:
                time.sleep(0.05)

        provider = _make_provider(_StreamingModel(streamer))

        async def run():
            it = provider.chat_stream([{"role": "user", "content": "hi"}], max_tokens=10)
            first = await anext(it)
            assert first == "B"
            started.wait(timeout=5)
            captured[1].put(RuntimeError("injected failure"))
            second = await anext(it)
            assert "Generation error" in second
            assert "injected failure" in second
            with pytest.raises(StopAsyncIteration):
                await anext(it)

        asyncio.run(run())

    def test_error_between_tokens_surfaces_via_err_q(self, monkeypatch):
        captured = []
        real_queue = queue.Queue

        class RecordingQueue(real_queue):
            def __init__(self):
                super().__init__()
                captured.append(self)

        monkeypatch.setattr(queue, "Queue", RecordingQueue)

        def streamer():
            yield 1
            yield 2

        provider = _make_provider(_StreamingModel(streamer))

        async def run():
            it = provider.chat_stream([{"role": "user", "content": "hi"}], max_tokens=10)
            first = await anext(it)
            assert first == "B"
            captured[1].put(RuntimeError("mid-stream failure"))
            second = await anext(it)
            assert second == "\n\n[Generation error: mid-stream failure]"
            with pytest.raises(StopAsyncIteration):
                await anext(it)

        asyncio.run(run())


class TestNoServerDelegation:
    def test_server_shortcut_skipped_when_none(self):
        def streamer():
            yield 1
            yield 2

        provider = _make_provider(_StreamingModel(streamer))
        out = _collect(provider.chat_stream([{"role": "user", "content": "hi"}], max_tokens=10))
        assert "".join(out) == "BC"
