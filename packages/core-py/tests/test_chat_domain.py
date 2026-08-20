"""Tests for domains.chat.domain — ChatRequest, ChatResponse, _build_prompt."""

from domains.chat.domain import ChatRequest, ChatResponse, ChatDomain


class TestChatRequest:
    def test_defaults(self):
        req = ChatRequest(messages=[{"role": "user", "content": "hi"}])
        assert req.model == "gpt2"
        assert req.temperature == 0.8
        assert req.max_tokens == 256
        assert req.session_id is None

    def test_custom(self):
        req = ChatRequest(
            messages=[],
            model="qwen",
            temperature=0.5,
            max_tokens=128,
            session_id="s1",
        )
        assert req.model == "qwen"
        assert req.session_id == "s1"


class TestChatResponse:
    def test_defaults(self):
        resp = ChatResponse(text="hello", session_id="s1")
        assert resp.done is True
        assert resp.tokens_generated == 0

    def test_custom(self):
        resp = ChatResponse(text="hi", session_id="s1", tokens_generated=10, duration_ms=500)
        assert resp.tokens_generated == 10
        assert resp.duration_ms == 500


class TestBuildPrompt:
    def test_system_only(self):
        prompt = ChatDomain._build_prompt("You are helpful.", [], "Hello")
        assert "System: You are helpful." in prompt
        assert "User: Hello" in prompt
        assert prompt.endswith("Assistant:")

    def test_with_history(self):
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]
        prompt = ChatDomain._build_prompt("", messages, "Thanks!")
        assert "User: What is 2+2?" in prompt
        assert "User: Thanks!" in prompt

    def test_empty_system(self):
        prompt = ChatDomain._build_prompt("", [], "Hi")
        assert prompt.startswith("User: Hi")
