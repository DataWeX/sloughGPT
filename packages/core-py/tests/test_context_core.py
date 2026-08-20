"""Tests for domains.infrastructure.context_core — ContextCore."""

from domains.infrastructure.context_core import ContextCore


class TestContextCore:
    def test_init(self):
        cc = ContextCore()
        assert cc.max_tokens == 2048
        assert cc.memory_enabled is True
        assert cc.rag_enabled is True
        assert cc.working_capacity == 7
        assert len(cc.session_messages) == 0
        assert len(cc.working_memory) == 0

    def test_custom(self):
        cc = ContextCore(max_tokens=4096, memory_enabled=False)
        assert cc.max_tokens == 4096
        assert cc.memory_enabled is False

    def test_system_prompt(self):
        cc = ContextCore()
        assert "SloughGPT" in cc.system_prompt
