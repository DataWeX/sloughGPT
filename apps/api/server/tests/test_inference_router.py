"""Tests for inference router endpoints (non-streaming, sessions, suggestions, tools)."""
import uuid
import pytest
from unittest.mock import patch, MagicMock

from tests.test_support import get_test_client

client = get_test_client()


class TestInferenceRoot:
    def test_root_returns_status(self):
        resp = client.get("/")
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "SloughGPT API"
        assert body["version"] == "1.0.0"
        assert body["status"] == "running"

    def test_root_has_endpoints_dict(self):
        resp = client.get("/")
        body = resp.json()
        assert "endpoints" in body
        assert isinstance(body["endpoints"], dict)


class TestInfoEndpoints:
    def test_info_returns_model_info(self):
        resp = client.get("/info")
        assert resp.status_code == 200
        body = resp.json()
        assert "model" in body
        assert "loaded" in body["model"]

    def test_info_soul_returns_dict(self):
        resp = client.get("/info/soul")
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body, dict)


class TestSuggestions:
    def test_suggestions_returns_list(self):
        resp = client.get("/suggestions")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert isinstance(body["data"], list)
        assert len(body["data"]) >= 4

    def test_chat_suggestions_returns_same(self):
        resp1 = client.get("/suggestions")
        resp2 = client.get("/chat/suggestions")
        assert resp1.json()["data"] == resp2.json()["data"]

    def test_suggestions_have_text_and_icon(self):
        resp = client.get("/suggestions")
        for item in resp.json()["data"]:
            assert "text" in item
            assert "icon" in item


class TestChatTools:
    def test_list_tools_returns_dict(self):
        resp = client.get("/chat/tools")
        assert resp.status_code == 200
        body = resp.json()
        assert "tools" in body
        assert isinstance(body["tools"], list)


class TestProviders:
    def test_list_providers_returns_dict(self):
        resp = client.get("/providers")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert isinstance(body["data"], dict)


class TestSessionCRUD:
    def test_create_session(self):
        resp = client.post("/chat/sessions", json={"name": "test session"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert "session_id" in body["data"]

    def test_create_session_with_id(self):
        sid = f"test_{uuid.uuid4().hex[:8]}"
        resp = client.post("/chat/sessions", json={"session_id": sid, "name": "custom id"})
        assert resp.status_code == 200
        assert resp.json()["data"]["session_id"] == sid

    def test_list_sessions(self):
        resp = client.get("/chat/sessions")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert isinstance(body["data"], list)

    def test_list_sessions_filter_archived(self):
        resp = client.get("/chat/sessions?archived=false")
        assert resp.status_code == 200
        assert isinstance(resp.json()["data"], list)

    def test_get_current_session(self):
        resp = client.get("/chat/sessions/current")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"

    def test_upsert_session(self):
        sid = f"upsert_{uuid.uuid4().hex[:8]}"
        client.post("/chat/sessions", json={"session_id": sid, "name": "orig"})
        resp = client.put(f"/chat/sessions/{sid}", json={"name": "updated"})
        assert resp.status_code == 200
        assert resp.json()["message"] == "saved"

    def test_delete_session_not_found(self):
        resp = client.delete(f"/chat/sessions/nonexistent_{uuid.uuid4().hex[:8]}")
        assert resp.status_code == 404

    def test_delete_session_clears_slonet_kv(self):
        """Deleting a session drops its cross-turn KV state on the provider."""
        from domains.models.provider import register_provider, _providers

        class _FakeProvider:
            def __init__(self):
                self.cleared = []

            def clear_session(self, session_id):
                self.cleared.append(session_id)

        fake = _FakeProvider()
        register_provider("slonet-native", fake)
        try:
            sid = f"kvdel_{uuid.uuid4().hex[:8]}"
            client.post("/chat/sessions", json={"session_id": sid, "name": "kv"})
            resp = client.delete(f"/chat/sessions/{sid}")
            assert resp.status_code == 200
            assert sid in fake.cleared
        finally:
            _providers.pop("slonet-native", None)

    def test_delete_session_without_provider_still_succeeds(self):
        """KV clear is best-effort — delete works even with no slonet provider."""
        from domains.models.provider import _providers
        _providers.pop("slonet-native", None)
        sid = f"kvmiss_{uuid.uuid4().hex[:8]}"
        client.post("/chat/sessions", json={"session_id": sid, "name": "kv"})
        resp = client.delete(f"/chat/sessions/{sid}")
        assert resp.status_code == 200

    def test_search_sessions_empty_query(self):
        resp = client.get("/chat/sessions/search?q=")
        assert resp.status_code == 200
        body = resp.json()
        assert body["data"] == []

    def test_search_sessions_with_query(self):
        resp = client.get("/chat/sessions/search?q=test")
        assert resp.status_code == 200
        assert isinstance(resp.json()["data"], list)


class TestContextEndpoints:
    def test_inspect_context(self):
        resp = client.get("/context/inspect")
        assert resp.status_code == 200

    def test_get_facts(self):
        resp = client.get("/context/facts")
        assert resp.status_code == 200

    def test_reset_context(self):
        resp = client.post("/context/reset")
        assert resp.status_code == 200
        body = resp.json()
        assert body.get("reset") == "session"

    def test_reset_context_all(self):
        resp = client.post("/context/reset?all=true")
        assert resp.status_code == 200
        assert resp.json().get("reset") == "all"


class TestGenerateRequiresModel:
    def test_generate_returns_503_without_model(self):
        resp = client.post("/inference/generate", json={"prompt": "hello"})
        assert resp.status_code == 503

    def test_generate_stream_returns_error_without_model(self):
        resp = client.post("/inference/generate/stream", json={"prompt": "hello"})
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")


class TestChatRequiresModel:
    def test_chat_returns_503_without_model(self):
        resp = client.post("/chat", json={"messages": [{"role": "user", "content": "hi"}]})
        assert resp.status_code == 503

    def test_chat_stream_returns_error_without_model(self):
        resp = client.post("/chat/stream", json={"messages": [{"role": "user", "content": "hi"}]})
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")


class TestVoiceEndpoints:
    def test_voice_upload_requires_audio(self):
        from io import BytesIO
        files = {"file": ("test.txt", BytesIO(b"not audio"), "text/plain")}
        resp = client.post("/chat/voice/test_session", files=files)
        assert resp.status_code == 400

    def test_get_voice_audio_not_found(self):
        resp = client.get("/chat/audio/nonexistent/nonexistent.m4a")
        assert resp.status_code in (403, 404)


class TestSchemaValidation:
    def test_generate_missing_prompt(self):
        resp = client.post("/inference/generate", json={})
        assert resp.status_code == 422

    def test_chat_missing_messages(self):
        resp = client.post("/chat", json={})
        assert resp.status_code == 422

    def test_chat_empty_messages_list(self):
        resp = client.post("/chat", json={"messages": []})
        assert resp.status_code in (400, 422, 503)

    def test_generate_temperature_out_of_range(self):
        resp = client.post("/inference/generate", json={"prompt": "hi", "temperature": 5.0})
        assert resp.status_code == 422
