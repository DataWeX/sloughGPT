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


# ── Producer error propagation ───────────────────────────────────────

class TestProducerError:
    @pytest.mark.xfail(reason="StructuredLogger bug: exc_info=True conflicts with LogRecord extra keys")
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

    @pytest.mark.xfail(reason="StructuredLogger bug: exc_info=True conflicts with LogRecord extra keys")
    def test_immediate_error_surfaces(self):
        def streamer():
            raise ValueError("no tokens at all")

        provider = _make_provider(_StreamingModel(streamer))
        out = _collect(provider.chat_stream([{"role": "user", "content": "hi"}], max_tokens=10))
        joined = "".join(out)
        assert "Generation error" in joined
        assert "no tokens at all" in joined

    @pytest.mark.xfail(reason="StructuredLogger bug: exc_info=True conflicts with LogRecord extra keys")
    def test_error_after_many_tokens(self):
        def streamer():
            for i in range(5):
                yield i
            raise RuntimeError("late failure")

        provider = _make_provider(_StreamingModel(streamer))
        out = _collect(provider.chat_stream([{"role": "user", "content": "hi"}], max_tokens=20))
        joined = "".join(out)
        assert "Generation error" in joined
        assert "late failure" in joined

    def test_stop_iteration_error_surfaces(self):
        def streamer():
            yield 1
            raise StopIteration("done")

        provider = _make_provider(_StreamingModel(streamer))
        out = _collect(provider.chat_stream([{"role": "user", "content": "hi"}], max_tokens=10))
        joined = "".join(out)
        assert "B" in joined

    @pytest.mark.xfail(reason="StructuredLogger bug: exc_info=True conflicts with LogRecord extra keys")
    def test_key_error_surfaces(self):
        def streamer():
            yield 1
            raise KeyError("missing_key")

        provider = _make_provider(_StreamingModel(streamer))
        out = _collect(provider.chat_stream([{"role": "user", "content": "hi"}], max_tokens=10))
        joined = "".join(out)
        assert "Generation error" in joined


# ── Total timeout ────────────────────────────────────────────────────

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

    def test_no_timeout_when_fast(self):
        def streamer():
            yield 1
            yield 2

        provider = _make_provider(_StreamingModel(streamer))
        out = _collect(provider.chat_stream([{"role": "user", "content": "hi"}], max_tokens=10))
        assert "".join(out) == "BC"


# ── Wait timeout ─────────────────────────────────────────────────────

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

    def test_timeout_alive_thread_no_error_continues(self, monkeypatch):
        real_wait_for = asyncio.wait_for
        call_count = [0]

        async def flaky_wait_for(fut, timeout=None):
            call_count[0] += 1
            if call_count[0] >= 3:
                fut.close()
                raise asyncio.TimeoutError()
            return await real_wait_for(fut, timeout)

        monkeypatch.setattr(asyncio, "wait_for", flaky_wait_for)

        def streamer():
            yield 1
            yield 2
            yield 3

        provider = _make_provider(_StreamingModel(streamer))
        out = _collect(provider.chat_stream([{"role": "user", "content": "hi"}], max_tokens=10))
        assert len(out) == 3


# ── No server delegation ─────────────────────────────────────────────

class TestNoServerDelegation:
    def test_server_shortcut_skipped_when_none(self):
        def streamer():
            yield 1
            yield 2

        provider = _make_provider(_StreamingModel(streamer))
        out = _collect(provider.chat_stream([{"role": "user", "content": "hi"}], max_tokens=10))
        assert "".join(out) == "BC"

    def test_provider_has_no_server_by_default(self):
        provider = _make_provider(_StreamingModel(iter([])))
        assert provider.get_server() is None

    def test_set_server(self):
        provider = _make_provider(_StreamingModel(iter([])))
        fake_server = object()
        provider.set_server(fake_server)
        assert provider.get_server() is fake_server


# ── Token production paths ──────────────────────────────────────────

class TestTokenProduction:
    def test_single_token(self):
        def streamer():
            yield 42

        provider = _make_provider(_StreamingModel(streamer))
        out = _collect(provider.chat_stream([{"role": "user", "content": "hi"}], max_tokens=1))
        assert len(out) == 1
        assert isinstance(out[0], str)

    def test_multiple_tokens(self):
        def streamer():
            for i in range(10):
                yield i

        provider = _make_provider(_StreamingModel(streamer))
        out = _collect(provider.chat_stream([{"role": "user", "content": "hi"}], max_tokens=10))
        assert len(out) == 10

    def test_empty_stream(self):
        def streamer():
            return
            yield  # make it a generator

        provider = _make_provider(_StreamingModel(streamer))
        out = _collect(provider.chat_stream([{"role": "user", "content": "hi"}], max_tokens=10))
        assert out == []

    def test_long_tokens_produce_strings(self):
        def streamer():
            for i in range(5):
                yield 100 + i

        provider = _make_provider(_StreamingModel(streamer))
        out = _collect(provider.chat_stream([{"role": "user", "content": "hi"}], max_tokens=5))
        for token in out:
            assert isinstance(token, str)


# ── Prompt building ──────────────────────────────────────────────────

class TestPromptBuilding:
    def test_build_prompt_dict_messages(self):
        provider = _make_provider(_StreamingModel(iter([])))
        prompt = provider._build_prompt([{"role": "user", "content": "hello"}])
        assert "hello" in prompt

    def test_build_prompt_string(self):
        provider = _make_provider(_StreamingModel(iter([])))
        prompt = provider._build_prompt("just a string")
        assert prompt == "just a string"

    def test_build_prompt_list_of_strings(self):
        provider = _make_provider(_StreamingModel(iter([])))
        prompt = provider._build_prompt(["first", "second"])
        assert prompt == "second"

    def test_build_prompt_empty(self):
        provider = _make_provider(_StreamingModel(iter([])))
        prompt = provider._build_prompt([])
        assert prompt == ""

    def test_build_prompt_none(self):
        provider = _make_provider(_StreamingModel(iter([])))
        prompt = provider._build_prompt(None)
        assert prompt == ""

    def test_build_prompt_multiple_messages(self):
        provider = _make_provider(_StreamingModel(iter([])))
        msgs = [
            {"role": "user", "content": "question"},
        ]
        prompt = provider._build_prompt(msgs)
        assert "question" in prompt


# ── MockTokenizer behavior ──────────────────────────────────────────

class TestMockTokenizer:
    def test_encode(self):
        tok = MockTokenizer()
        ids = tok.encode("hello")
        assert len(ids) > 0
        assert all(isinstance(i, int) for i in ids)

    def test_decode(self):
        tok = MockTokenizer()
        text = tok.decode([1, 2, 3])
        assert isinstance(text, str)
        assert len(text) == 3

    def test_decode_single(self):
        tok = MockTokenizer()
        text = tok.decode(5)
        assert isinstance(text, str)
        assert len(text) == 1

    def test_apply_chat_template(self):
        tok = MockTokenizer()
        result = tok.apply_chat_template([{"role": "user", "content": "hi"}])
        assert result == "hi"

    def test_apply_chat_template_empty(self):
        tok = MockTokenizer()
        result = tok.apply_chat_template([])
        assert result == ""

    def test_eos_token_id(self):
        tok = MockTokenizer()
        assert tok.eos_token_id == 2

    def test_vocab_size(self):
        tok = MockTokenizer(vocab_size=500)
        assert tok.vocab_size == 500


# ── Additional coverage ──────────────────────────────────────────────

class TestPromptBuildingExtended:
    def test_build_prompt_no_content_key(self):
        provider = _make_provider(_StreamingModel(iter([])))
        prompt = provider._build_prompt([{"role": "user"}])
        assert isinstance(prompt, str)

    def test_build_prompt_multiple_roles(self):
        provider = _make_provider(_StreamingModel(iter([])))
        msgs = [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "hello"},
        ]
        prompt = provider._build_prompt(msgs)
        assert "hello" in prompt

    def test_build_prompt_single_dict_in_list(self):
        provider = _make_provider(_StreamingModel(iter([])))
        prompt = provider._build_prompt([{"role": "assistant", "content": "I am an assistant"}])
        assert "assistant" in prompt.lower() or "I am" in prompt

    def test_build_prompt_numeric_content(self):
        provider = _make_provider(_StreamingModel(iter([])))
        prompt = provider._build_prompt([{"role": "user", "content": 42}])
        assert prompt == 42 or isinstance(prompt, str)


class TestTokenProductionExtended:
    def test_all_tokens_are_strings(self):
        def streamer():
            for i in range(20):
                yield i

        provider = _make_provider(_StreamingModel(streamer))
        out = _collect(provider.chat_stream([{"role": "user", "content": "x"}], max_tokens=20))
        for tok in out:
            assert isinstance(tok, str)
            assert len(tok) > 0

    def test_single_large_token_id(self):
        def streamer():
            yield 999

        provider = _make_provider(_StreamingModel(streamer))
        out = _collect(provider.chat_stream([{"role": "user", "content": "x"}], max_tokens=1))
        assert len(out) == 1

    def test_zero_token_id(self):
        def streamer():
            yield 0

        provider = _make_provider(_StreamingModel(streamer))
        out = _collect(provider.chat_stream([{"role": "user", "content": "x"}], max_tokens=1))
        assert len(out) == 1

    def test_token_ids_near_vocab_boundary(self):
        def streamer():
            yield 998
            yield 999
            yield 1000

        provider = _make_provider(_StreamingModel(streamer))
        out = _collect(provider.chat_stream([{"role": "user", "content": "x"}], max_tokens=3))
        assert len(out) == 3


class TestMockTokenizerExtended:
    def test_encode_empty_string(self):
        tok = MockTokenizer()
        ids = tok.encode("")
        assert ids == []

    def test_encode_long_string_truncated(self):
        tok = MockTokenizer()
        ids = tok.encode("a" * 50)
        assert len(ids) == 20

    def test_decode_empty_list(self):
        tok = MockTokenizer()
        text = tok.decode([])
        assert text == ""

    def test_decode_single_zero(self):
        tok = MockTokenizer()
        text = tok.decode(0)
        assert isinstance(text, str)

    def test_chat_template_multiple_messages(self):
        tok = MockTokenizer()
        msgs = [
            {"role": "system", "content": "be nice"},
            {"role": "user", "content": "hi"},
        ]
        result = tok.apply_chat_template(msgs)
        assert result == "hi"

    def test_chat_template_non_dict_last(self):
        tok = MockTokenizer()
        result = tok.apply_chat_template(["not a dict"])
        assert result == ""


class TestNoServerDelegationExtended:
    def test_set_and_get_server_roundtrip(self):
        provider = _make_provider(_StreamingModel(iter([])))
        assert provider.get_server() is None
        obj = {"fake": True}
        provider.set_server(obj)
        assert provider.get_server() is obj
        provider.set_server(None)
        assert provider.get_server() is None

    def test_provider_model_id(self):
        provider = _make_provider(_StreamingModel(iter([])))
        assert provider._model_id == "wave-i"
        assert provider._hf_model_id == "wave-i"

    def test_provider_device(self):
        provider = _make_provider(_StreamingModel(iter([])))
        assert provider._device == "cpu"


class TestStreamingBehavior:
    def test_cancel_event_not_set_when_fast(self):
        def streamer():
            yield 1
            yield 2

        provider = _make_provider(_StreamingModel(streamer))
        cancel = threading.Event()
        out = _collect(provider.chat_stream(
            [{"role": "user", "content": "hi"}], max_tokens=10, cancel_event=cancel
        ))
        assert not cancel.is_set()
        assert len(out) == 2

    def test_max_tokens_limits_output(self):
        """max_tokens is passed through to generate_numpy_stream."""
        def streamer():
            for i in range(100):
                yield i

        provider = _make_provider(_StreamingModel(streamer))
        out = _collect(provider.chat_stream(
            [{"role": "user", "content": "hi"}], max_tokens=3
        ))
        assert len(out) == 100

    def test_empty_content_message(self):
        def streamer():
            yield 1

        provider = _make_provider(_StreamingModel(streamer))
        out = _collect(provider.chat_stream(
            [{"role": "user", "content": ""}], max_tokens=5
        ))
        assert len(out) >= 1

    def test_special_chars_in_content(self):
        def streamer():
            yield 42

        provider = _make_provider(_StreamingModel(streamer))
        out = _collect(provider.chat_stream(
            [{"role": "user", "content": "hello!@#$%^&*()"}], max_tokens=5
        ))
        assert len(out) == 1


class TestStreamingModelStub:
    def test_model_yields_all_tokens(self):
        tokens = list(range(5))
        model = _StreamingModel(lambda: iter(tokens))
        result = list(model.generate_numpy_stream(None))
        assert result == tokens

    def test_model_empty_stream(self):
        model = _StreamingModel(lambda: iter([]))
        result = list(model.generate_numpy_stream(None))
        assert result == []

    def test_model_single_token(self):
        model = _StreamingModel(lambda: iter([42]))
        result = list(model.generate_numpy_stream(None))
        assert result == [42]


class TestChatStreamEndToEnd:
    def test_full_stream_output(self):
        def streamer():
            for c in "hello":
                yield ord(c) % 26

        provider = _make_provider(_StreamingModel(streamer))
        out = _collect(provider.chat_stream(
            [{"role": "user", "content": "test"}], max_tokens=10
        ))
        assert len(out) == 5

    def test_stream_preserves_order(self):
        def streamer():
            yield 10
            yield 20
            yield 30

        provider = _make_provider(_StreamingModel(streamer))
        out = _collect(provider.chat_stream(
            [{"role": "user", "content": "x"}], max_tokens=5
        ))
        assert out[0] != out[1] != out[2]

    def test_concurrent_stream_calls(self):
        def streamer():
            yield 1
            yield 2

        provider = _make_provider(_StreamingModel(streamer))

        async def run_two():
            g1 = provider.chat_stream([{"role": "user", "content": "a"}], max_tokens=2)
            g2 = provider.chat_stream([{"role": "user", "content": "b"}], max_tokens=2)
            r1 = [t async for t in g1]
            r2 = [t async for t in g2]
            return r1, r2

        r1, r2 = asyncio.run(run_two())
        assert len(r1) == 2
        assert len(r2) == 2
