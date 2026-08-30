"""Tests for domains.agents — SecurityConfig, SecurityBoundary, ToolDefinition, ToolExecutionContext, AgentConfig."""

import time
import pytest
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

    def test_custom_values(self):
        sc = SecurityConfig(
            max_execution_time=60,
            max_memory_mb=1024,
            allow_network=True,
            rate_limit_per_minute=120
        )
        assert sc.max_execution_time == 60
        assert sc.max_memory_mb == 1024
        assert sc.allow_network is True
        assert sc.rate_limit_per_minute == 120

    def test_file_size_limit(self):
        sc = SecurityConfig(max_file_size_mb=50)
        assert sc.max_file_size_mb == 50

    def test_allowed_directories(self):
        sc = SecurityConfig(allowed_directories=["/tmp", "/data"])
        assert len(sc.allowed_directories) == 2
        assert "/tmp" in sc.allowed_directories

    def test_blocked_patterns(self):
        sc = SecurityConfig(blocked_patterns=["import os", "eval("])
        assert len(sc.blocked_patterns) == 2

    def test_empty_config(self):
        sc = SecurityConfig()
        assert sc.allowed_directories == []
        assert sc.blocked_patterns == []

    def test_default_file_size(self):
        sc = SecurityConfig()
        assert sc.max_file_size_mb == 100

    def test_network_disabled_by_default(self):
        sc = SecurityConfig()
        assert sc.allow_network is False

    def test_rate_limit_default(self):
        sc = SecurityConfig()
        assert sc.rate_limit_per_minute == 60

    def test_max_memory_default(self):
        sc = SecurityConfig()
        assert sc.max_memory_mb == 512

    def test_max_execution_time_default(self):
        sc = SecurityConfig()
        assert sc.max_execution_time == 30

    def test_single_allowed_directory(self):
        sc = SecurityConfig(allowed_directories=["/only"])
        assert sc.allowed_directories == ["/only"]

    def test_single_blocked_pattern(self):
        sc = SecurityConfig(blocked_patterns=["eval"])
        assert sc.blocked_patterns == ["eval"]

    def test_all_fields_custom(self):
        sc = SecurityConfig(
            max_execution_time=1,
            max_memory_mb=2,
            max_file_size_mb=3,
            allow_network=True,
            allowed_directories=["/a"],
            blocked_patterns=["b"],
            rate_limit_per_minute=4,
        )
        assert sc.max_execution_time == 1
        assert sc.max_memory_mb == 2
        assert sc.max_file_size_mb == 3
        assert sc.allow_network is True
        assert sc.allowed_directories == ["/a"]
        assert sc.blocked_patterns == ["b"]
        assert sc.rate_limit_per_minute == 4


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

    def test_is_blocked_exec(self):
        sb = SecurityBoundary()
        allowed, msg = sb.is_allowed("exec('malicious')")
        assert allowed is False
        assert "Blocked" in msg

    def test_is_blocked_pickle(self):
        sb = SecurityBoundary()
        allowed, msg = sb.is_allowed("import pickle")
        assert allowed is False

    def test_is_blocked_marshal(self):
        sb = SecurityBoundary()
        allowed, msg = sb.is_allowed("import marshal")
        assert allowed is False

    def test_is_blocked_compile(self):
        sb = SecurityBoundary()
        allowed, msg = sb.is_allowed("compile('code', 'exec')")
        assert allowed is False

    def test_is_blocked_os_remove(self):
        sb = SecurityBoundary()
        allowed, msg = sb.is_allowed("import os\nos.remove('file')")
        assert allowed is False

    def test_is_blocked_shutil_rmtree(self):
        sb = SecurityBoundary()
        allowed, msg = sb.is_allowed("import shutil\nshutil.rmtree('dir')")
        assert allowed is False

    def test_is_blocked_subprocess_shell(self):
        sb = SecurityBoundary()
        allowed, msg = sb.is_allowed("subprocess.call(shell=True)")
        assert allowed is False

    def test_is_blocked_dunder_import(self):
        sb = SecurityBoundary()
        allowed, msg = sb.is_allowed("__import__('os')")
        assert allowed is False

    def test_with_custom_config(self):
        config = SecurityConfig(max_execution_time=10)
        sb = SecurityBoundary(config)
        assert sb.config.max_execution_time == 10

    def test_resource_limit_context_manager(self):
        sb = SecurityBoundary()
        with sb.resource_limit("test_tool"):
            pass  # Should not raise

    def test_allowed_dirs(self):
        sb = SecurityBoundary()
        assert "data" in sb.ALLOWED_DIRS
        assert "models" in sb.ALLOWED_DIRS

    def test_safe_arithmetic(self):
        sb = SecurityBoundary()
        allowed, _ = sb.is_allowed("result = 2 + 3")
        assert allowed is True

    def test_safe_string_operations(self):
        sb = SecurityBoundary()
        allowed, _ = sb.is_allowed("text = 'hello'.upper()")
        assert allowed is True

    def test_blocked_patterns_count(self):
        sb = SecurityBoundary()
        assert len(sb._blocked_re) == len(SecurityBoundary.BLOCKED_PATTERNS)

    def test_allowed_dirs_count(self):
        sb = SecurityBoundary()
        assert len(sb.ALLOWED_DIRS) == 4

    def test_is_allowed_returns_tuple(self):
        sb = SecurityBoundary()
        result = sb.is_allowed("x = 1")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_is_blocked_multiple_patterns(self):
        sb = SecurityBoundary()
        allowed, _ = sb.is_allowed("eval(exec('x'))")
        assert allowed is False

    def test_resource_limit_yields(self):
        sb = SecurityBoundary()
        with sb.resource_limit("tool") as ctx:
            assert ctx is None

    def test_is_allowed_multiline_safe(self):
        sb = SecurityBoundary()
        code = "x = 1\ny = 2\nz = x + y"
        allowed, _ = sb.is_allowed(code)
        assert allowed is True

    def test_is_blocked_subprocess_with_args(self):
        sb = SecurityBoundary()
        allowed, _ = sb.is_allowed("subprocess.call('ls', shell=True)")
        assert allowed is False

    def test_is_blocked_os_remove_inline(self):
        sb = SecurityBoundary()
        allowed, _ = sb.is_allowed("import os; os.remove('file.txt')")
        assert allowed is False

    def test_is_blocked_shutil_rmtree_inline(self):
        sb = SecurityBoundary()
        allowed, _ = sb.is_allowed("import shutil; shutil.rmtree('/tmp/dir')")
        assert allowed is False

    def test_custom_config_passthrough(self):
        sc = SecurityConfig(rate_limit_per_minute=5)
        sb = SecurityBoundary(sc)
        assert sb.config.rate_limit_per_minute == 5

    def test_blocked_patterns_list(self):
        sb = SecurityBoundary()
        patterns = sb.BLOCKED_PATTERNS
        assert isinstance(patterns, list)
        assert len(patterns) > 0

    def test_is_allowed_empty_string(self):
        sb = SecurityBoundary()
        allowed, _ = sb.is_allowed("")
        assert allowed is True

    def test_is_allowed_whitespace_only(self):
        sb = SecurityBoundary()
        allowed, _ = sb.is_allowed("   \n  \t  ")
        assert allowed is True

    def test_resource_limit_multiple_uses(self):
        sb = SecurityBoundary()
        with sb.resource_limit("a"):
            pass
        with sb.resource_limit("b"):
            pass

    def test_is_blocked_open_read_write_mode(self):
        sb = SecurityBoundary()
        allowed, _ = sb.is_allowed("open('file', r/w)")
        assert allowed is False

    def test_is_allowed_open_quoted_modes(self):
        sb = SecurityBoundary()
        allowed, _ = sb.is_allowed("open('file', 'r')")
        assert allowed is True


class TestToolCapability:
    def test_all_members(self):
        assert len(ToolCapability) == 8

    def test_values(self):
        assert ToolCapability.CODE_EXECUTION.value == "code_execution"
        assert ToolCapability.FILE_READ.value == "file_read"
        assert ToolCapability.FILE_SEARCH.value == "file_search"
        assert ToolCapability.WEB_SEARCH.value == "web_search"
        assert ToolCapability.KNOWLEDGE_RETRIEVAL.value == "knowledge_retrieval"
        assert ToolCapability.IMAGE_ANALYSIS.value == "image_analysis"
        assert ToolCapability.DATA_ANALYSIS.value == "data_analysis"
        assert ToolCapability.CITATION.value == "citation"

    def test_member_names(self):
        names = [m.name for m in ToolCapability]
        assert "CODE_EXECUTION" in names
        assert "FILE_READ" in names

    def test_is_enum(self):
        from enum import Enum
        assert issubclass(ToolCapability, Enum)

    def test_unique_values(self):
        values = [m.value for m in ToolCapability]
        assert len(values) == len(set(values))

    def test_iteration(self):
        count = 0
        for cap in ToolCapability:
            count += 1
        assert count == 8

    def test_membership(self):
        assert ToolCapability.CODE_EXECUTION in ToolCapability

    def test_from_value(self):
        assert ToolCapability("code_execution") == ToolCapability.CODE_EXECUTION


class TestToolDefinition:
    def test_fields(self):
        td = ToolDefinition(
            name="calc", description="calc", parameters={},
            capability=ToolCapability.CODE_EXECUTION,
        )
        assert td.name == "calc"
        assert td.requires_approval is False

    def test_with_parameters(self):
        td = ToolDefinition(
            name="search",
            description="Search files",
            parameters={"query": str, "limit": int},
            capability=ToolCapability.FILE_SEARCH,
        )
        assert "query" in td.parameters

    def test_requires_approval(self):
        td = ToolDefinition(
            name="exec",
            description="Execute code",
            parameters={},
            capability=ToolCapability.CODE_EXECUTION,
            requires_approval=True,
        )
        assert td.requires_approval is True

    def test_different_capabilities(self):
        for cap in ToolCapability:
            td = ToolDefinition(
                name=cap.value,
                description="test",
                parameters={},
                capability=cap,
            )
            assert td.capability == cap

    def test_empty_parameters(self):
        td = ToolDefinition(
            name="noop",
            description="No operation",
            parameters={},
            capability=ToolCapability.FILE_READ,
        )
        assert td.parameters == {}

    def test_description_preserved(self):
        td = ToolDefinition(
            name="t", description="A long description",
            parameters={}, capability=ToolCapability.FILE_READ,
        )
        assert td.description == "A long description"

    def test_default_requires_approval(self):
        td = ToolDefinition(
            name="t", description="d", parameters={},
            capability=ToolCapability.FILE_READ,
        )
        assert td.requires_approval is False

    def test_complex_parameters(self):
        params = {
            "query": str,
            "limit": int,
            "filters": dict,
            "nested": {"a": [1, 2]},
        }
        td = ToolDefinition(
            name="search", description="d", parameters=params,
            capability=ToolCapability.FILE_SEARCH,
        )
        assert td.parameters["nested"]["a"] == [1, 2]

    def test_name_preserved(self):
        td = ToolDefinition(
            name="my_tool", description="d", parameters={},
            capability=ToolCapability.CODE_EXECUTION,
        )
        assert td.name == "my_tool"

    def test_dataclass_fields(self):
        from dataclasses import fields
        field_names = [f.name for f in fields(ToolDefinition)]
        assert "name" in field_names
        assert "description" in field_names
        assert "parameters" in field_names
        assert "capability" in field_names
        assert "requires_approval" in field_names


class TestToolExecutionContext:
    def test_fields(self):
        tec = ToolExecutionContext(session_id="s1", user_id="u1", timestamp=1.0)
        assert tec.session_id == "s1"
        assert tec.user_id == "u1"
        assert tec.metadata == {}

    def test_with_metadata(self):
        tec = ToolExecutionContext(
            session_id="s1",
            user_id="u1",
            timestamp=time.time(),
            metadata={"source": "test", "priority": "high"},
        )
        assert tec.metadata["source"] == "test"

    def test_different_users(self):
        tec1 = ToolExecutionContext(session_id="s1", user_id="user_a", timestamp=1.0)
        tec2 = ToolExecutionContext(session_id="s1", user_id="user_b", timestamp=1.0)
        assert tec1.user_id != tec2.user_id

    def test_timestamp_is_float(self):
        tec = ToolExecutionContext(session_id="s1", user_id="u1", timestamp=1234567890.0)
        assert isinstance(tec.timestamp, float)

    def test_metadata_mutable(self):
        tec = ToolExecutionContext(session_id="s1", user_id="u1", timestamp=1.0)
        tec.metadata["key"] = "value"
        assert tec.metadata["key"] == "value"

    def test_default_metadata_empty(self):
        tec = ToolExecutionContext(session_id="s", user_id="u", timestamp=0.0)
        assert tec.metadata == {}

    def test_session_id_preserved(self):
        tec = ToolExecutionContext(session_id="unique-session", user_id="u", timestamp=0.0)
        assert tec.session_id == "unique-session"

    def test_different_sessions(self):
        tec1 = ToolExecutionContext(session_id="s1", user_id="u", timestamp=0.0)
        tec2 = ToolExecutionContext(session_id="s2", user_id="u", timestamp=0.0)
        assert tec1.session_id != tec2.session_id

    def test_metadata_nested(self):
        tec = ToolExecutionContext(
            session_id="s", user_id="u", timestamp=0.0,
            metadata={"outer": {"inner": [1, 2]}},
        )
        assert tec.metadata["outer"]["inner"] == [1, 2]

    def test_timestamp_int_stored(self):
        tec = ToolExecutionContext(session_id="s", user_id="u", timestamp=42)
        assert tec.timestamp == 42

    def test_metadata_keys(self):
        tec = ToolExecutionContext(
            session_id="s", user_id="u", timestamp=0.0,
            metadata={"a": 1, "b": 2},
        )
        assert set(tec.metadata.keys()) == {"a", "b"}

    def test_metadata_update(self):
        tec = ToolExecutionContext(
            session_id="s", user_id="u", timestamp=0.0,
            metadata={"a": 1},
        )
        tec.metadata.update({"b": 2, "c": 3})
        assert tec.metadata["b"] == 2
        assert tec.metadata["c"] == 3

    def test_metadata_delete(self):
        tec = ToolExecutionContext(
            session_id="s", user_id="u", timestamp=0.0,
            metadata={"key": "val"},
        )
        del tec.metadata["key"]
        assert "key" not in tec.metadata

    def test_dataclass_fields(self):
        from dataclasses import fields
        field_names = [f.name for f in fields(ToolExecutionContext)]
        assert "session_id" in field_names
        assert "user_id" in field_names
        assert "timestamp" in field_names
        assert "metadata" in field_names


class TestAgentConfig:
    def test_defaults(self):
        ac = AgentConfig()
        assert ToolCapability.CODE_EXECUTION in ac.tools
        assert ac.max_iterations == 10
        assert ac.timeout == 120

    def test_custom_iterations(self):
        ac = AgentConfig(max_iterations=5)
        assert ac.max_iterations == 5

    def test_custom_timeout(self):
        ac = AgentConfig(timeout=300)
        assert ac.timeout == 300

    def test_instructions(self):
        ac = AgentConfig(instructions="Be helpful")
        assert ac.instructions == "Be helpful"

    def test_security_config(self):
        config = SecurityConfig(max_execution_time=10)
        ac = AgentConfig(security=config)
        assert ac.security.max_execution_time == 10

    def test_empty_tools(self):
        ac = AgentConfig(tools=[])
        assert len(ac.tools) == 0

    def test_all_capabilities_in_default(self):
        ac = AgentConfig()
        assert len(ac.tools) == 4

    def test_custom_tools_list(self):
        tools = [ToolCapability.FILE_READ, ToolCapability.FILE_SEARCH]
        ac = AgentConfig(tools=tools)
        assert len(ac.tools) == 2
        assert ToolCapability.FILE_READ in ac.tools

    def test_default_security_none(self):
        ac = AgentConfig()
        assert ac.security is None

    def test_default_instructions_empty(self):
        ac = AgentConfig()
        assert ac.instructions == ""

    def test_default_max_iterations(self):
        ac = AgentConfig()
        assert ac.max_iterations == 10

    def test_default_timeout(self):
        ac = AgentConfig()
        assert ac.timeout == 120

    def test_all_tool_capabilities(self):
        ac = AgentConfig(tools=list(ToolCapability))
        assert len(ac.tools) == 8

    def test_tools_is_list(self):
        ac = AgentConfig()
        assert isinstance(ac.tools, list)

    def test_tools_contains_only_tool_capability(self):
        ac = AgentConfig()
        for tool in ac.tools:
            assert isinstance(tool, ToolCapability)

    def test_dataclass_fields(self):
        from dataclasses import fields
        field_names = [f.name for f in fields(AgentConfig)]
        assert "tools" in field_names
        assert "security" in field_names
        assert "max_iterations" in field_names
        assert "timeout" in field_names
        assert "instructions" in field_names

    def test_iterations_zero(self):
        ac = AgentConfig(max_iterations=0)
        assert ac.max_iterations == 0

    def test_timeout_zero(self):
        ac = AgentConfig(timeout=0)
        assert ac.timeout == 0

    def test_negative_iterations(self):
        ac = AgentConfig(max_iterations=-1)
        assert ac.max_iterations == -1

    def test_large_timeout(self):
        ac = AgentConfig(timeout=999999)
        assert ac.timeout == 999999

    def test_long_instructions(self):
        long_text = "x" * 10000
        ac = AgentConfig(instructions=long_text)
        assert len(ac.instructions) == 10000
