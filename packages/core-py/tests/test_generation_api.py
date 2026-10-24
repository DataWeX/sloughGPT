"""Generation API — one contract, real wire, tolerant replies.

No mocks: a stdlib ThreadingHTTPServer speaks the hosted
chat-completions shape and records what actually arrived.
"""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from domain.inference.generation import extract_json, hosted

SEEN: dict = {}


class _ChatHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # noqa: ARG002 — silence access log
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        SEEN.clear()
        SEEN.update(
            {
                "path": self.path,
                "auth": self.headers.get("Authorization"),
                "body": body,
            }
        )
        reply = json.dumps({"choices": [{"message": {"content": "hello from the wire"}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)


class TestHostedClient:
    def test_chat_roundtrip_bearer_and_payload(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), _ChatHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            base = f"http://127.0.0.1:{server.server_address[1]}/v1"
            client = hosted(base, api_key="sk-t", model="m1")
            out = asyncio.run(
                client.chat(
                    [{"role": "user", "content": "hi"}],
                    max_tokens=32,
                    temperature=0.1,
                )
            )
            assert out == "hello from the wire"
            assert client.model_id == f"chat-api:{base}"
            assert SEEN["path"] == "/v1/chat/completions"
            assert SEEN["auth"] == "Bearer sk-t"
            assert SEEN["body"]["model"] == "m1"
            assert SEEN["body"]["max_tokens"] == 32
            # the key never rides anything but the header
            assert "sk-t" not in json.dumps(SEEN["body"])
        finally:
            server.shutdown()
            server.server_close()

    def test_no_key_means_no_auth_header(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), _ChatHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            base = f"http://127.0.0.1:{server.server_address[1]}/v1"
            asyncio.run(hosted(base).chat([{"role": "user", "content": "hi"}]))
            assert SEEN["auth"] is None
        finally:
            server.shutdown()
            server.server_close()


class TestExtractJson:
    def test_prose_wrapped_array(self):
        assert extract_json('Sure! [{"question": "q", "answer": "A"}] Done.') == [
            {"question": "q", "answer": "A"}
        ]

    def test_plain_object(self):
        assert extract_json('{"action_type": "wait", "params": {"ms": 500}}') == {
            "action_type": "wait",
            "params": {"ms": 500},
        }

    def test_garbage_returns_none_not_error(self):
        assert extract_json("no structured output at all") is None
        assert extract_json("broken: {not json") is None
