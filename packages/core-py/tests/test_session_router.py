"""Tests for the session API router (routers/session.py).

Covers: set_session_context, get_session_messages, get_session_inspector, regenerate_session.
Domain deps are mocked; only HTTP-level behavior is tested.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_server_dir = str(Path(__file__).resolve().parents[3] / "apps" / "api" / "server")
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, _server_dir)
from routers.session import SessionRouter  # noqa: E402


def _app(sr: SessionRouter) -> FastAPI:
    app = FastAPI()
    app.include_router(sr.router)
    return app


class TestSetSessionContext:
    def test_store_context(self):
        sr = SessionRouter()
        mock_sc = MagicMock()
        mock_sc.store_context.return_value = {"session_id": "s1", "message_count": 2}
        with patch("domains.infrastructure.session_core.SessionCore", mock_sc):
            client = TestClient(_app(sr))
            resp = client.post("/session/s1/context", json={
                "messages": [{"role": "user", "content": "hi"}]
            })
        assert resp.status_code == 200
        assert resp.json()["data"]["message_count"] == 2

    def test_empty_messages(self):
        sr = SessionRouter()
        client = TestClient(_app(sr))
        resp = client.post("/session/s1/context", json={"messages": []})
        assert resp.status_code == 200
        assert resp.json()["data"]["message_count"] == 0


class TestGetSessionMessages:
    def test_get_messages(self):
        sr = SessionRouter()
        mock_sc = MagicMock()
        mock_sc.get_messages.return_value = [{"role": "user", "content": "hi"}]
        with patch("domains.infrastructure.session_core.SessionCore", mock_sc):
            client = TestClient(_app(sr))
            resp = client.get("/session/s1/messages")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert len(data["messages"]) == 1

    def test_get_messages_empty(self):
        sr = SessionRouter()
        mock_sc = MagicMock()
        mock_sc.get_messages.return_value = []
        with patch("domains.infrastructure.session_core.SessionCore", mock_sc):
            client = TestClient(_app(sr))
            resp = client.get("/session/s1/messages")
        assert resp.status_code == 200
        assert resp.json()["data"]["messages"] == []


class TestGetSessionInspector:
    def test_inspector(self):
        sr = SessionRouter()
        mock_sc = MagicMock()
        mock_sc.get_messages.return_value = [{"role": "user", "content": "hi"}]
        mock_fb = MagicMock()
        mock_fb.get_stats.return_value = {"feedback_total": 5, "thumbs_up": 3, "thumbs_down": 2}
        mock_km = MagicMock()
        mock_km.stats.return_value = {"total_facts": 10}
        mock_km.all_topics.return_value = [("ai", 5)]
        mock_tc = MagicMock()
        mock_tc.all.return_value = {"warmth": 0.8}
        mock_cc = MagicMock()
        mock_cc.get_context_inspector.return_value = {"working_memory": [], "semantic_keys": [], "episodic_count": 0}
        with patch("domains.infrastructure.session_core.SessionCore", mock_sc), \
             patch("domains.feedback.message_feedback.get_message_feedback", return_value=mock_fb), \
             patch("domains.learner.knowledge.get_knowledge_memory", return_value=mock_km), \
             patch("domains.context.managers.get_trait_config", return_value=mock_tc), \
             patch("domains.context.managers.PersonalityManager"), \
             patch("domains.context.managers.MemoryManager"), \
             patch("domains.context.managers.StyleManager"), \
             patch("domains.context.managers.TaskManager"), \
             patch("domains.infrastructure.context_core.get_context_core", return_value=mock_cc):
            client = TestClient(_app(sr))
            resp = client.get("/session/s1/inspector")
        assert resp.status_code == 200
        data = resp.json()
        assert "session" in data
        assert "knowledge" in data
        assert "feedback" in data
