"""Tests for the /session router — context, messages, inspector."""

import pytest
from unittest.mock import patch, MagicMock
from test_support import _data, get_test_client


class TestSetSessionContext:
    def setup_method(self):
        self.client = get_test_client()

    def test_set_context_with_messages(self):
        resp = self.client.post(
            "/session/ctx_001/context",
            json={
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi there"},
                ]
            },
        )
        assert resp.status_code == 200
        body = _data(resp)
        assert body["message_count"] == 2

    def test_set_empty_context(self):
        resp = self.client.post("/session/ctx_002/context", json={"messages": []})
        assert resp.status_code == 200
        body = _data(resp)
        assert body["message_count"] == 0

    def test_set_context_with_system_prompt(self):
        resp = self.client.post(
            "/session/ctx_003/context",
            json={
                "system_prompt": "You are helpful.",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 200

    def test_set_context_with_knowledge(self):
        resp = self.client.post(
            "/session/ctx_004/context",
            json={
                "knowledge": ["fact1", "fact2"],
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 200


class TestGetSessionMessages:
    def setup_method(self):
        self.client = get_test_client()

    def test_get_messages_empty_session(self):
        resp = self.client.get("/session/nonexistent_id/messages")
        assert resp.status_code == 200
        body = _data(resp)
        assert body["messages"] == []

    def test_get_messages_after_set(self):
        sid = "msg_001"
        self.client.post(
            f"/session/{sid}/context",
            json={"messages": [{"role": "user", "content": "test"}]},
        )
        resp = self.client.get(f"/session/{sid}/messages")
        assert resp.status_code == 200
        body = _data(resp)
        assert body["session_id"] == sid
        assert len(body["messages"]) >= 1

    def test_get_messages_preserves_order(self):
        sid = "msg_order"
        self.client.post(
            f"/session/{sid}/context",
            json={
                "messages": [
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "second"},
                    {"role": "user", "content": "third"},
                ]
            },
        )
        resp = self.client.get(f"/session/{sid}/messages")
        msgs = _data(resp)["messages"]
        assert msgs[0]["content"] == "first"
        assert msgs[-1]["content"] == "third"


class TestSessionInspector:
    def setup_method(self):
        self.client = get_test_client()

    def test_inspector_has_all_sections(self):
        sid = "insp_001"
        self.client.post(
            f"/session/{sid}/context",
            json={"messages": [{"role": "user", "content": "inspect me"}]},
        )
        resp = self.client.get(f"/session/{sid}/inspector")
        assert resp.status_code == 200
        body = _data(resp)
        assert "session" in body
        assert "messages" in body["session"]
        assert "knowledge" in body
        assert "traits" in body
        assert "modes" in body
        assert "workspace" in body
        assert "elapsed_ms" in body

    def test_inspector_empty_session(self):
        resp = self.client.get("/session/empty_inspector/inspector")
        assert resp.status_code == 200
        body = _data(resp)
        assert body["session"]["message_count"] == 0

    def test_inspector_session_has_correct_id(self):
        sid = "insp_id_check"
        resp = self.client.get(f"/session/{sid}/inspector")
        assert resp.status_code == 200
        assert _data(resp)["session"]["id"] == sid

    def test_inspector_knowledge_structure(self):
        resp = self.client.get("/session/x/inspector")
        knowledge = _data(resp)["knowledge"]
        assert "total_facts" in knowledge
        assert "topics" in knowledge
        assert isinstance(knowledge["topics"], list)

    def test_inspector_workspace_structure(self):
        resp = self.client.get("/session/x/inspector")
        workspace = _data(resp)["workspace"]
        assert "working_memory" in workspace
        assert "semantic_keys" in workspace
        assert "episodic_count" in workspace
