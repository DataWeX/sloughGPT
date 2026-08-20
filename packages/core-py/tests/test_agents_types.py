"""Tests for domains.agents — SecurityBoundary, SecurityConfig, ToolCapability, ToolDefinition, AgentConfig."""

import time
from domains.agents import (
    SecurityConfig, SecurityBoundary, ToolCapability,
    ToolDefinition, ToolExecutionContext, AgentConfig,
)


class TestSecurityConfig:
    def test_defaults(self):
        cfg = SecurityConfig()
        assert cfg.max_execution_time == 30
        assert cfg.max_memory_mb == 512
        assert cfg.allow_network is False
        assert cfg.rate_limit_per_minute == 60

    def test_custom(self):
        cfg = SecurityConfig(max_execution_time=10, allow_network=True)
        assert cfg.max_execution_time == 10
        assert cfg.allow_network is True


class TestSecurityBoundary:
    def test_allowed_code(self):
        sb = SecurityBoundary()
        ok, msg = sb.is_allowed("x = 1 + 2")
        assert ok is True
        assert msg == ""

    def test_blocked_eval(self):
        sb = SecurityBoundary()
        ok, msg = sb.is_allowed("eval('os.system(\"ls\")')")
        assert ok is False
        assert "eval" in msg

    def test_blocked_exec(self):
        sb = SecurityBoundary()
        ok, _ = sb.is_allowed("exec(code)")
        assert ok is False

    def test_blocked_pickle(self):
        sb = SecurityBoundary()
        ok, _ = sb.is_allowed("pickle.dumps(data)")
        assert ok is False

    def test_blocked_import_os_remove(self):
        sb = SecurityBoundary()
        ok, _ = sb.is_allowed("import os\nos.remove('file')")
        assert ok is False

    def test_blocked_subprocess_shell(self):
        sb = SecurityBoundary()
        ok, _ = sb.is_allowed("subprocess.call(cmd, shell=True)")
        assert ok is False

    def test_custom_config(self):
        cfg = SecurityConfig(rate_limit_per_minute=5)
        sb = SecurityBoundary(cfg)
        assert sb.config.rate_limit_per_minute == 5

    def test_resource_limit_context(self):
        sb = SecurityBoundary()
        with sb.resource_limit("test"):
            pass


class TestToolCapability:
    def test_all_members(self):
        assert len(ToolCapability) == 8

    def test_values(self):
        assert ToolCapability.CODE_EXECUTION.value == "code_execution"
        assert ToolCapability.FILE_READ.value == "file_read"


class TestToolDefinition:
    def test_fields(self):
        td = ToolDefinition(
            name="test_tool",
            description="A test tool",
            parameters={"arg": "str"},
            capability=ToolCapability.CODE_EXECUTION,
        )
        assert td.name == "test_tool"
        assert td.requires_approval is False

    def test_requires_approval(self):
        td = ToolDefinition(
            name="dangerous",
            description="Dangerous",
            parameters={},
            capability=ToolCapability.CODE_EXECUTION,
            requires_approval=True,
        )
        assert td.requires_approval is True


class TestToolExecutionContext:
    def test_fields(self):
        ctx = ToolExecutionContext(
            session_id="s1",
            user_id="u1",
            timestamp=time.time(),
        )
        assert ctx.session_id == "s1"
        assert ctx.metadata == {}


class TestAgentConfig:
    def test_defaults(self):
        cfg = AgentConfig()
        assert ToolCapability.CODE_EXECUTION in cfg.tools
        assert cfg.max_iterations == 10
        assert cfg.timeout == 120

    def test_custom(self):
        cfg = AgentConfig(max_iterations=5, timeout=60)
        assert cfg.max_iterations == 5
        assert cfg.timeout == 60
