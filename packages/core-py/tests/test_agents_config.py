"""Tests for domains.agents — SecurityConfig, SecurityBoundary, ToolDefinition, ToolExecutionContext, AgentConfig."""

from domains.agents import (
    SecurityConfig, SecurityBoundary, ToolCapability, ToolDefinition,
    ToolExecutionContext, AgentConfig,
)


class TestSecurityConfig:
    def test_defaults(self):
        sc = SecurityConfig()
        assert sc.max_execution_time == 30
        assert sc.max_memory_mb == 512
        assert sc.allow_network is False
        assert sc.rate_limit_per_minute == 60


class TestSecurityBoundary:
    def test_init(self):
        sb = SecurityBoundary()
        assert len(sb._blocked_re) > 0

    def test_is_allowed_safe(self):
        sb = SecurityBoundary()
        allowed, msg = sb.is_allowed("print('hello')")
        assert allowed is True

    def test_is_blocked_eval(self):
        sb = SecurityBoundary()
        allowed, msg = sb.is_allowed("eval('bad')")
        assert allowed is False
        assert "Blocked" in msg


class TestToolDefinition:
    def test_fields(self):
        td = ToolDefinition(
            name="calc", description="calc", parameters={},
            capability=ToolCapability.CODE_EXECUTION,
        )
        assert td.name == "calc"
        assert td.requires_approval is False


class TestToolExecutionContext:
    def test_fields(self):
        tec = ToolExecutionContext(session_id="s1", user_id="u1", timestamp=1.0)
        assert tec.session_id == "s1"
        assert tec.user_id == "u1"
        assert tec.metadata == {}


class TestAgentConfig:
    def test_defaults(self):
        ac = AgentConfig()
        assert ToolCapability.CODE_EXECUTION in ac.tools
        assert ac.max_iterations == 10
        assert ac.timeout == 120
