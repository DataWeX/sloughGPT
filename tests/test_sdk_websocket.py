"""Coverage for sloughgpt_sdk.websocket."""
import asyncio
import json
import sys
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "packages" / "sdk-py"))

from sloughgpt_sdk.websocket import (  # noqa: E402
    StreamGenerator,
    StreamIterator,
    WebSocketClient,
    WebSocketMessage,
)


class FakeModule:
    def __init__(self, create_connection):
        self.create_connection = create_connection


class FakeWS:
    def __init__(self, responses, park=False):
        self.responses = list(responses)
        self.park = park
        self.sent = []
        self.closed = False

    def send(self, payload):
        self.sent.append(payload)

    def recv(self):
        if self.responses:
            return self.responses.pop(0)
        if self.park:
            import time

            while True:
                time.sleep(0.001)
        raise RuntimeError("exhausted")

    def close(self):
        self.closed = True


class TestWebSocketMessage:
    def test_from_dict_defaults(self):
        m = WebSocketMessage.from_dict({})
        assert m.type == "unknown"
        assert m.data == ""
        assert m.raw == {}

    def test_from_dict_full(self):
        m = WebSocketMessage.from_dict({"type": "token", "data": "hi", "other": 1})
        assert m.type == "token"
        assert m.data == "hi"
        assert m.raw == {"type": "token", "data": "hi", "other": 1}

    def test_direct_construction(self):
        m = WebSocketMessage("token", "hello")
        assert m.type == "token"
        assert m.data == "hello"
        assert m.raw is None


class TestWebSocketClientInit:
    def test_base_url_scheme_conversion(self):
        assert WebSocketClient("http://host:8000").base_url == "ws://host:8000"
        assert WebSocketClient("https://host:8000").base_url == "wss://host:8000"

    def test_base_url_strips_trailing_slash(self):
        assert WebSocketClient("http://host:8000/").base_url == "ws://host:8000"

    def test_jwt_passed_to_connect_kwargs(self):
        client = WebSocketClient(jwt_token="tok")
        assert client.jwt_token == "tok"


class TestWebSocketConnect:
    def test_missing_credentials_raises_value_error(self):
        fake_mod = FakeModule(lambda url, timeout=None: FakeWS([]))
        client = WebSocketClient()
        with patch.dict(sys.modules, {"websocket": fake_mod}):
            with pytest.raises(ValueError):
                client.connect()

    def test_missing_websocket_package_raises_import_error(self):
        client = WebSocketClient(api_key="k")
        with patch.dict(sys.modules, {"websocket": None}):
            with pytest.raises(ImportError):
                client.connect()

    def test_auth_failure_raises(self):
        ws = FakeWS([json.dumps({"status": "denied", "error": "bad key"})])
        fake_mod = FakeModule(lambda url, timeout=None: ws)
        client = WebSocketClient(api_key="k")
        with patch.dict(sys.modules, {"websocket": fake_mod}):
            with pytest.raises(ConnectionError, match="bad key"):
                client.connect()

    def test_auth_failure_closes_socket(self):
        ws = FakeWS([json.dumps({"status": "denied"})])
        fake_mod = FakeModule(lambda url, timeout: ws)
        client = WebSocketClient(api_key="k")
        with patch.dict(sys.modules, {"websocket": fake_mod}):
            with pytest.raises(ConnectionError):
                client.connect()
        assert ws.closed is True

    def test_auth_failure_close_error_swallowed(self):
        class CloseCrashWS(FakeWS):
            def close(self):
                raise RuntimeError("close fail")

        ws = CloseCrashWS([json.dumps({"status": "denied"})])
        fake_mod = FakeModule(lambda url, timeout: ws)
        client = WebSocketClient(api_key="k")
        with patch.dict(sys.modules, {"websocket": fake_mod}):
            with pytest.raises(ConnectionError):
                client.connect()

    def test_connect_success_api_key_auth_message(self):
        ws = FakeWS([json.dumps({"status": "authenticated"})], park=True)
        fake_mod = FakeModule(lambda url, timeout=None: ws)
        client = WebSocketClient(api_key="secret")
        with patch.dict(sys.modules, {"websocket": fake_mod}):
            assert client.connect() is True
        assert client.is_connected is True
        assert json.loads(ws.sent[0]) == {"api_key": "secret"}
        client.close()

    def test_connect_success_token_auth_message(self):
        ws = FakeWS([json.dumps({"status": "authenticated"})], park=True)
        fake_mod = FakeModule(lambda url, timeout=None: ws)
        client = WebSocketClient(jwt_token="jwt123")
        with patch.dict(sys.modules, {"websocket": fake_mod}):
            client.connect()
        assert json.loads(ws.sent[0]) == {"token": "jwt123"}
        client.close()


class TestWebSocketDispatch:
    def _make_ws(self, recv_values):
        fake_mod = FakeModule(lambda url, timeout=None: FakeWS(recv_values))
        return fake_mod

    def test_dispatch_error_message(self):
        client = WebSocketClient(api_key="k")
        seen = []
        client.on("error", lambda m: seen.append(m))
        client._dispatch({"status": "error", "error": "boom"})
        assert seen[0].type == "error"
        assert seen[0].data == "boom"

    def test_dispatch_error_falls_through_to_wildcard(self):
        client = WebSocketClient(api_key="k")
        seen = []
        client.on("*", lambda m: seen.append(m))
        client._dispatch({"status": "error", "error": "boom"})
        assert len(seen) == 1
        assert seen[0].type == "error"
        assert seen[0].data == "boom"

    def test_dispatch_token(self):
        client = WebSocketClient(api_key="k")
        seen = []
        client.on("token", lambda m: seen.append(m))
        client._dispatch({"token": "hi"})
        assert seen[0].data == "hi"

    def test_dispatch_complete(self):
        client = WebSocketClient(api_key="k")
        seen = []
        client.on("complete", lambda m: seen.append(m))
        client._dispatch({"done": True, "status": "done", "text": "done txt"})
        assert seen[0].data == "done txt"

    def test_dispatch_token_then_complete(self):
        client = WebSocketClient(api_key="k")
        seq = []
        client.on("token", lambda m: seq.append(("t", m.data)))
        client.on("complete", lambda m: seq.append(("c", m.data)))
        client._dispatch({"token": "a"})
        client._dispatch({"done": True, "status": "done", "text": "z"})
        assert seq == [("t", "a"), ("c", "z")]

    def test_recv_loop_dispatches_and_breaks_on_exception(self):
        ws = FakeWS(
            [
                json.dumps({"status": "authenticated"}),
                json.dumps({"token": "x"}),
                json.dumps({"done": True, "status": "done"}),
            ]
        )
        fake_mod = FakeModule(lambda url, timeout=None: ws)
        client = WebSocketClient(api_key="k")
        with patch.dict(sys.modules, {"websocket": fake_mod}):
            client.connect()
        client._recv_thread.join(timeout=2)
        assert client._connected is False
        assert client._recv_thread is not None

    def test_recv_loop_skips_empty_raw(self):
        ws = FakeWS(
            [
                json.dumps({"status": "authenticated"}),
                "",
                json.dumps({"token": "skip-empty"}),
                json.dumps({"done": True, "status": "done"}),
            ]
        )
        fake_mod = FakeModule(lambda url, timeout=None: ws)
        client = WebSocketClient(api_key="k")
        with patch.dict(sys.modules, {"websocket": fake_mod}):
            client.connect()
        client._recv_thread.join(timeout=2)
        assert client._connected is False


class TestWebSocketSend:
    def test_send_generate_requires_connection(self):
        client = WebSocketClient(api_key="k")
        with pytest.raises(ConnectionError, match="Not connected"):
            client.send_generate("hi")

    def test_send_generate_payload(self):
        ws = FakeWS([json.dumps({"status": "authenticated"})], park=True)
        fake_mod = FakeModule(lambda url, timeout=None: ws)
        client = WebSocketClient(api_key="k")
        with patch.dict(sys.modules, {"websocket": fake_mod}):
            client.connect()
        client.send_generate("CN")
        payload = json.loads(ws.sent[-1])
        assert payload["prompt"] == "CN"
        assert payload["max_tokens"] == 100
        assert payload["temperature"] == 0.8
        client.close()

    def test_send_generate_custom_params_and_model(self):
        ws = FakeWS([json.dumps({"status": "authenticated"})], park=True)
        fake_mod = FakeModule(lambda url, timeout=None: ws)
        client = WebSocketClient(api_key="k")
        with patch.dict(sys.modules, {"websocket": fake_mod}):
            client.connect()
        client.send_generate("hi", max_tokens=42, temperature=0.2, model="m1", extra="e")
        payload = json.loads(ws.sent[-1])
        assert payload["max_tokens"] == 42
        assert payload["temperature"] == 0.2
        assert payload["model"] == "m1"
        assert payload["extra"] == "e"
        client.close()

    def test_send_chat_formats_messages(self):
        ws = FakeWS([json.dumps({"status": "authenticated"})], park=True)
        fake_mod = FakeModule(lambda url, timeout=None: ws)
        client = WebSocketClient(api_key="k")
        with patch.dict(sys.modules, {"websocket": fake_mod}):
            client.connect()
        client.send_chat([{"role": "user", "content": "hi"}])
        payload = json.loads(ws.sent[-1])
        assert payload["prompt"].startswith("user: hi")
        assert payload["prompt"].endswith("assistant:")
        client.close()

    def test_send_ping(self):
        ws = FakeWS([json.dumps({"status": "authenticated"})], park=True)
        fake_mod = FakeModule(lambda url, timeout=None: ws)
        client = WebSocketClient(api_key="k")
        with patch.dict(sys.modules, {"websocket": fake_mod}):
            client.connect()
        client.send_ping()
        assert json.loads(ws.sent[-1]) == {"type": "ping"}
        client.close()

    def test_send_ping_swallows_send_error(self):
        class BoomWS(FakeWS):
            def send(self, payload):
                if json.loads(payload).get("type") == "ping":
                    raise RuntimeError("send fail")
                super().send(payload)

        ws = BoomWS([json.dumps({"status": "authenticated"})], park=True)
        fake_mod = FakeModule(lambda url, timeout=None: ws)
        client = WebSocketClient(api_key="k")
        with patch.dict(sys.modules, {"websocket": fake_mod}):
            client.connect()
        client.send_ping()
        assert json.loads(ws.sent[-1]) == {"api_key": "k"}
        client.close()

    def test_close_swallows_close_error(self):
        class BoomCloseWS(FakeWS):
            def close(self):
                raise RuntimeError("close fail")

        ws = BoomCloseWS([json.dumps({"status": "authenticated"})])
        fake_mod = FakeModule(lambda url, timeout=None: ws)
        client = WebSocketClient(api_key="k")
        with patch.dict(sys.modules, {"websocket": fake_mod}):
            client.connect()
        client.close()
        assert client.is_connected is False

    def test_close_when_never_connected(self):
        client = WebSocketClient(api_key="k")
        client.close()
        assert client.is_connected is False


class TestStreamIterator:
    def test_sync_iteration_over_buffer(self):
        gen = StreamGenerator()
        gen._buffer = ["a", "b"]
        gen._complete = True
        assert list(StreamIterator(gen)) == ["a", "b"]

    def test_sync_iteration_waits_for_tokens(self):
        gen = StreamGenerator()
        gen._buffer = []
        gen._complete = False
        iterator = StreamIterator(gen)

        def fill():
            import time

            time.sleep(0.05)
            gen._buffer.append("late")
            gen._complete = True

        t = threading.Thread(target=fill)
        t.start()
        try:
            tokens = list(iterator)
        finally:
            t.join()
        assert tokens == ["late"]

    def test_async_iteration_waits_for_tokens(self):
        async def collect():
            gen = StreamGenerator()
            gen._buffer = []
            gen._complete = False

            async def fill():
                await asyncio.sleep(0.02)
                gen._buffer.append("late-a")
                gen._complete = True

            async def consume():
                out = []
                async for token in StreamIterator(gen):
                    out.append(token)
                return out

            out, _ = await asyncio.gather(consume(), fill())
            return out

        assert asyncio.run(collect()) == ["late-a"]

    def test_sync_stop_when_empty_and_complete(self):
        gen = StreamGenerator()
        gen._complete = True
        assert list(StreamIterator(gen)) == []

    def test_async_iteration_over_buffer(self):
        async def collect(iterator):
            out = []
            async for token in iterator:
                out.append(token)
            return out

        gen = StreamGenerator()
        gen._buffer = ["x", "y"]
        gen._complete = True
        assert asyncio.run(collect(StreamIterator(gen))) == ["x", "y"]

    def test_async_stop_when_empty_and_complete(self):
        async def collect(iterator):
            out = []
            async for token in iterator:
                out.append(token)
            return out

        gen = StreamGenerator()
        gen._complete = True
        assert asyncio.run(collect(StreamIterator(gen))) == []


class TestStreamGenerator:
    def test_context_manager_lifecycle(self):
        fake = StreamWSFake()
        gen = StreamGenerator(api_key="k")
        with patch.object(gen, "client", fake):
            assert gen.__enter__() is gen
            assert fake.connected is True
            gen.__exit__(None, None, None)
            assert fake.connected is False

    def test_generate_resets_buffer_and_sends(self):
        gen = StreamGenerator(api_key="k")
        fake = StreamWSFake()
        with patch.object(gen, "client", fake):
            gen.__enter__()
            iterator = gen.generate("hello", max_new_tokens=3)
            assert fake.sent == ("hello", {"max_new_tokens": 3})
            assert isinstance(iterator, StreamIterator)
            token_handler = [h for n, h in fake.handlers if n == "token"][0]
            complete_handler = [h for n, h in fake.handlers if n == "complete"][0]
            token_handler(WebSocketMessage("token", "he"))
            token_handler(WebSocketMessage("token", "llo"))
            complete_handler(WebSocketMessage("complete", ""))
            assert gen.get_full_text() == "hello"
            gen.__exit__(None, None, None)

    def test_generate_resets_buffer(self):
        gen = StreamGenerator(api_key="k")
        fake = StreamWSFake()
        with patch.object(gen, "client", fake):
            gen.__enter__()
            gen.generate("one")
            token_handler = [h for n, h in fake.handlers if n == "token"][0]
            token_handler(WebSocketMessage("token", "ab"))
            complete_handler = [h for n, h in fake.handlers if n == "complete"][0]
            complete_handler(WebSocketMessage("complete", ""))
            assert gen.get_full_text() == "ab"
            gen.generate("two")
            assert gen.get_full_text() == ""
            gen.__exit__(None, None, None)

    def test_default_construction_uses_full_client(self):
        gen = StreamGenerator("http://localhost:9000", api_key="k")
        assert gen.client.base_url == "ws://localhost:9000"
        assert gen.client.api_key == "k"


class StreamWSFake:
    def __init__(self):
        self.handlers = []
        self.connected = False
        self.sent = None

    def connect(self):
        self.connected = True

    def on(self, event, handler):
        self.handlers.append((event, handler))

    def send_generate(self, prompt, **kwargs):
        self.sent = (prompt, kwargs)

    def close(self):
        self.connected = False