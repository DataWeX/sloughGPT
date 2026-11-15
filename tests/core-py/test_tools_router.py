"""Tests for the tools API router (routers/tools.py).

Covers: list_tools, generate_tool (success, unknown tool, no provider, render error).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

_server_dir = str(Path(__file__).resolve().parents[2] / "apps" / "api" / "server")
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _mock_tools_engine():
    engine = MagicMock()
    engine.list_tools.return_value = [
        {
            "id": "write",
            "name": "Write",
            "description": "Help me write",
            "category": "creative",
            "fields": [],
            "max_tokens": 500,
        },
        {
            "id": "translate",
            "name": "Translate",
            "description": "Translate text",
            "category": "utility",
            "fields": [],
            "max_tokens": 300,
        },
    ]
    profile = MagicMock()
    profile.max_tokens = 500
    engine.get_profile.return_value = profile
    engine.render_prompt.return_value = "Write a poem about AI"
    return engine


def _app():
    from routers.tools import ToolsRouter

    app = FastAPI()
    app.include_router(ToolsRouter().router)
    from infrastructure.exception_handlers import register_all_handlers

    register_all_handlers(app)
    return app


class TestListTools:
    @patch("routers.tools._tools_engine")
    def test_list_tools(self, mock_engine):
        mock_engine.list_tools.return_value = _mock_tools_engine().list_tools()
        client = TestClient(_app())
        resp = client.get("/tools")
        assert resp.status_code == 200
        tools = resp.json()["data"]["tools"]
        assert len(tools) == 2
        assert tools[0]["id"] == "write"


class TestGenerateTool:
    @patch("routers.tools._tools_engine")
    def test_generate_unknown_tool(self, mock_engine):
        mock_engine.get_profile.return_value = None
        client = TestClient(_app())
        resp = client.post(
            "/tools/nonexistent/generate",
            json={"payload": {}},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "text/event-stream; charset=utf-8"
        lines = resp.text.strip().split("\n")
        data_lines = [l for l in lines if l.startswith("data:")]
        assert len(data_lines) >= 1
        last = json.loads(data_lines[-1].removeprefix("data: ").strip())
        assert last["status"] == "error"

    @patch("routers.tools.get_provider")
    @patch("routers.tools._tools_engine")
    def test_generate_no_provider(self, mock_engine, mock_get_provider):
        engine = _mock_tools_engine()
        mock_engine.get_profile.return_value = engine.get_profile()
        mock_engine.render_prompt.return_value = "prompt"
        mock_get_provider.return_value = None
        client = TestClient(_app())
        resp = client.post(
            "/tools/write/generate",
            json={"payload": {}},
        )
        assert resp.status_code == 200
        lines = resp.text.strip().split("\n")
        data_lines = [l for l in lines if l.startswith("data:")]
        last = json.loads(data_lines[-1].removeprefix("data: ").strip())
        assert last["status"] == "error"

    @patch("routers.tools._tools_engine")
    def test_generate_render_error(self, mock_engine):
        engine = _mock_tools_engine()
        mock_engine.get_profile.return_value = engine.get_profile()
        mock_engine.render_prompt.side_effect = ValueError("bad payload")
        client = TestClient(_app())
        resp = client.post(
            "/tools/write/generate",
            json={"payload": {"bad": True}},
        )
        assert resp.status_code == 200
        lines = resp.text.strip().split("\n")
        data_lines = [l for l in lines if l.startswith("data:")]
        last = json.loads(data_lines[-1].removeprefix("data: ").strip())
        assert last["status"] == "error"

    @patch("routers.tools.get_cancel_manager")
    @patch("routers.tools.get_provider")
    @patch("routers.tools._tools_engine")
    def test_generate_success(self, mock_engine, mock_get_provider, mock_cancel):
        engine = _mock_tools_engine()
        mock_engine.get_profile.return_value = engine.get_profile()
        mock_engine.render_prompt.return_value = "prompt"

        provider = MagicMock()

        async def fake_stream(messages, max_tokens, temperature, cancel_event):
            yield "Hello"
            yield " World"

        provider.chat_stream = fake_stream
        mock_get_provider.return_value = provider

        mgr = MagicMock()
        mgr.register.return_value = "op-1"
        mock_cancel.return_value = mgr

        client = TestClient(_app())
        resp = client.post(
            "/tools/write/generate",
            json={"payload": {"topic": "AI"}},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "text/event-stream; charset=utf-8"
        lines = resp.text.strip().split("\n")
        data_lines = [l for l in lines if l.startswith("data:")]
        tokens = [json.loads(l.removeprefix("data: ").strip()) for l in data_lines if "token" in l]
        assert len(tokens) >= 2
