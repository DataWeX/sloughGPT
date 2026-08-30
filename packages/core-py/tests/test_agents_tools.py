"""Tests for domains.agents.tools — ToolParam, ToolResult, ToolSpec, ToolRegistry; domains.agents.multi — TaskStatus, AgentTask."""

import pytest
import re
import asyncio
import time
from dataclasses import FrozenInstanceError

from domains.agents.tools import ToolParam, ToolResult, ToolSpec, ToolRegistry, get_tool_registry
from domains.agents.multi import TaskStatus, AgentTask


class TestToolParam:
    def test_fields(self):
        tp = ToolParam(name="expr", type="string", description="math expr", required=True)
        assert tp.name == "expr"
        assert tp.type == "string"
        assert tp.description == "math expr"
        assert tp.required is True

    def test_defaults(self):
        tp = ToolParam(name="x", type="int", description="val")
        assert tp.required is False

    def test_equality(self):
        a = ToolParam(name="a", type="int", description="x")
        b = ToolParam(name="a", type="int", description="x")
        assert a == b

    def test_inequality(self):
        a = ToolParam(name="a", type="int", description="x")
        b = ToolParam(name="b", type="int", description="x")
        assert a != b

    def test_repr(self):
        tp = ToolParam(name="q", type="str", description="query")
        r = repr(tp)
        assert "q" in r
        assert "query" in r

    def test_string_type(self):
        tp = ToolParam(name="mode", type="enum", description="mode", required=False)
        assert tp.type == "enum"

    def test_long_description(self):
        desc = "x" * 500
        tp = ToolParam(name="p", type="str", description=desc)
        assert len(tp.description) == 500


class TestToolResult:
    def test_fields(self):
        tr = ToolResult(success=True, output="42", duration_ms=1.5)
        assert tr.success is True
        assert tr.output == "42"
        assert tr.error is None
        assert tr.metadata == {}

    def test_error(self):
        tr = ToolResult(success=False, output="", error="bad input")
        assert tr.success is False
        assert tr.error == "bad input"

    def test_defaults(self):
        tr = ToolResult(success=True, output="ok")
        assert tr.duration_ms == 0.0
        assert tr.metadata == {}
        assert tr.error is None

    def test_metadata(self):
        tr = ToolResult(success=True, output="x", metadata={"key": "val"})
        assert tr.metadata["key"] == "val"

    def test_metadata_mutable(self):
        tr = ToolResult(success=True, output="x")
        tr.metadata["a"] = 1
        assert tr.metadata["a"] == 1

    def test_equality(self):
        a = ToolResult(success=True, output="ok")
        b = ToolResult(success=True, output="ok")
        assert a == b

    def test_repr(self):
        tr = ToolResult(success=False, output="", error="err")
        r = repr(tr)
        assert "err" in r

    def test_failed_result(self):
        tr = ToolResult(success=False, output="partial", error="timeout", duration_ms=5000.0)
        assert tr.success is False
        assert tr.duration_ms == 5000.0
        assert tr.error == "timeout"


class TestToolSpec:
    def test_creation(self):
        async def dummy(**kwargs):
            return {"output": "test"}
        spec = ToolSpec(name="t", description="desc", parameters=[], execute=dummy)
        assert spec.name == "t"
        assert spec.requires_approval is False
        assert spec.pattern is None

    def test_with_pattern(self):
        async def dummy(**kwargs):
            return {"output": ""}
        pat = re.compile(r"^test$")
        spec = ToolSpec(name="t", description="d", parameters=[], execute=dummy, pattern=pat, requires_approval=True)
        assert spec.pattern == pat
        assert spec.requires_approval is True

    def test_parameters_list(self):
        async def dummy(**kwargs):
            return {}
        params = [
            ToolParam(name="a", type="int", description="first"),
            ToolParam(name="b", type="str", description="second"),
        ]
        spec = ToolSpec(name="t", description="d", parameters=params, execute=dummy)
        assert len(spec.parameters) == 2
        assert spec.parameters[0].name == "a"


class TestTaskStatus:
    def test_values(self):
        assert TaskStatus.PENDING == "pending"
        assert TaskStatus.IN_PROGRESS == "in_progress"
        assert TaskStatus.COMPLETED == "completed"
        assert TaskStatus.FAILED == "failed"

    def test_string_type(self):
        assert isinstance(TaskStatus.PENDING, str)
        assert isinstance(TaskStatus.COMPLETED, str)

    def test_distinct_values(self):
        vals = {TaskStatus.PENDING, TaskStatus.IN_PROGRESS, TaskStatus.COMPLETED, TaskStatus.FAILED}
        assert len(vals) == 4

    def test_comparison(self):
        assert TaskStatus.PENDING != TaskStatus.IN_PROGRESS

    def test_identity(self):
        assert TaskStatus.PENDING == "pending"
        assert TaskStatus.FAILED == "failed"


class TestAgentTask:
    def test_fields(self):
        at = AgentTask(id="t1", description="do something", assigned_agent="code")
        assert at.id == "t1"
        assert at.status == TaskStatus.PENDING
        assert at.depends_on == []

    def test_to_dict(self):
        at = AgentTask(id="t1", description="test", assigned_agent="a", result="done")
        d = at.to_dict()
        assert d["id"] == "t1"
        assert d["agent"] == "a"
        assert d["result_preview"] == "done"

    def test_defaults(self):
        at = AgentTask(id="x", description="d", assigned_agent="r")
        assert at.context == ""
        assert at.result == ""
        assert at.error == ""
        assert at.depends_on == []

    def test_depends_on(self):
        at = AgentTask(id="t2", description="d", assigned_agent="r", depends_on=["t1", "t3"])
        assert len(at.depends_on) == 2
        assert "t1" in at.depends_on

    def test_to_dict_empty_result(self):
        at = AgentTask(id="t1", description="d", assigned_agent="r")
        d = at.to_dict()
        assert d["result_preview"] == ""

    def test_to_dict_long_result(self):
        long_result = "x" * 200
        at = AgentTask(id="t1", description="d", assigned_agent="r", result=long_result)
        d = at.to_dict()
        assert len(d["result_preview"]) == 100

    def test_status_set(self):
        at = AgentTask(id="t1", description="d", assigned_agent="r")
        at.status = TaskStatus.IN_PROGRESS
        assert at.status == "in_progress"

    def test_error_field(self):
        at = AgentTask(id="t1", description="d", assigned_agent="r", error="fail")
        assert at.error == "fail"

    def test_context_field(self):
        at = AgentTask(id="t1", description="d", assigned_agent="r", context="ctx")
        assert at.context == "ctx"

    def test_to_dict_keys(self):
        at = AgentTask(id="t1", description="d", assigned_agent="r")
        d = at.to_dict()
        expected_keys = {"id", "description", "agent", "status", "result_preview", "depends_on"}
        assert set(d.keys()) == expected_keys


class TestToolRegistry:
    def test_init_creates_defaults(self):
        reg = ToolRegistry()
        tools = reg.list_tools()
        assert len(tools) >= 8

    def test_get_existing(self):
        reg = ToolRegistry()
        spec = reg.get("calculator")
        assert spec is not None
        assert spec.name == "calculator"

    def test_get_missing(self):
        reg = ToolRegistry()
        assert reg.get("nonexistent") is None

    def test_register_custom(self):
        reg = ToolRegistry()
        async def custom_exec(**kwargs):
            return {"output": "custom"}
        spec = ToolSpec(name="custom", description="Custom tool", parameters=[], execute=custom_exec)
        reg.register(spec)
        assert reg.get("custom") is not None
        tools = reg.list_tools()
        names = [t["name"] for t in tools]
        assert "custom" in names

    def test_list_tools_format(self):
        reg = ToolRegistry()
        tools = reg.list_tools()
        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "parameters" in tool
            assert "requires_approval" in tool

    def test_detect_calculator(self):
        reg = ToolRegistry()
        result = reg.detect_tool_intent("calc 2+2")
        assert result is not None
        name, args = result
        assert name == "calculator"
        assert "expression" in args

    def test_detect_current_time(self):
        reg = ToolRegistry()
        result = reg.detect_tool_intent("time")
        assert result is not None
        assert result[0] == "current_time"

    def test_detect_web_search(self):
        reg = ToolRegistry()
        result = reg.detect_tool_intent("search python tutorials")
        assert result is not None
        assert result[0] == "web_search"
        assert result[1]["query"] == "python tutorials"

    def test_detect_file_read(self):
        reg = ToolRegistry()
        result = reg.detect_tool_intent("read /path/to/file.py")
        assert result is not None
        assert result[0] == "file_read"

    def test_detect_knowledge_retrieval(self):
        reg = ToolRegistry()
        result = reg.detect_tool_intent("knowledge how to deploy")
        assert result is not None
        assert result[0] == "knowledge_retrieval"

    def test_detect_image_analysis(self):
        reg = ToolRegistry()
        result = reg.detect_tool_intent("analyze image photo.png")
        assert result is not None
        assert result[0] == "image_analysis"

    def test_detect_data_analysis(self):
        reg = ToolRegistry()
        result = reg.detect_tool_intent("analyze data dataset.csv")
        assert result is not None
        assert result[0] == "data_analysis"

    def test_detect_no_match(self):
        reg = ToolRegistry()
        result = reg.detect_tool_intent("hello world")
        assert result is None

    def test_calculator_execute(self):
        reg = ToolRegistry()
        result = asyncio.get_event_loop().run_until_complete(
            reg.execute("calculator", {"expression": "2+3"})
        )
        assert result.success is True
        assert result.output == "5"

    def test_calculator_math_fn(self):
        reg = ToolRegistry()
        result = asyncio.get_event_loop().run_until_complete(
            reg.execute("calculator", {"expression": "sqrt(144)"})
        )
        assert result.success is True
        assert result.output == "12.0"

    def test_calculator_empty(self):
        reg = ToolRegistry()
        result = asyncio.get_event_loop().run_until_complete(
            reg.execute("calculator", {"expression": ""})
        )
        assert result.success is False

    def test_calculator_disallowed(self):
        reg = ToolRegistry()
        result = asyncio.get_event_loop().run_until_complete(
            reg.execute("calculator", {"expression": "__import__('os').system('ls')"})
        )
        assert result.success is False

    def test_current_time_execute(self):
        reg = ToolRegistry()
        result = asyncio.get_event_loop().run_until_complete(
            reg.execute("current_time", {})
        )
        assert result.success is True
        assert len(result.output) > 10

    def test_execute_unknown_tool(self):
        reg = ToolRegistry()
        result = asyncio.get_event_loop().run_until_complete(
            reg.execute("nonexistent_tool", {})
        )
        assert result.success is False
        assert "Unknown tool" in result.error

    def test_execute_with_timing(self):
        reg = ToolRegistry()
        result = asyncio.get_event_loop().run_until_complete(
            reg.execute("calculator", {"expression": "1+1"})
        )
        assert result.duration_ms >= 0

    def test_detect_calc_what_is(self):
        reg = ToolRegistry()
        result = reg.detect_tool_intent("what is 5*5")
        assert result is not None
        assert result[0] == "calculator"

    def test_detect_calc_math(self):
        reg = ToolRegistry()
        result = reg.detect_tool_intent("math 10/2")
        assert result is not None

    def test_detect_time_what_time(self):
        reg = ToolRegistry()
        result = reg.detect_tool_intent("what time")
        assert result is not None
        assert result[0] == "current_time"

    def test_detect_search_look_up(self):
        reg = ToolRegistry()
        result = reg.detect_tool_intent("look up neural networks")
        assert result is not None
        assert result[0] == "web_search"

    def test_detect_file_cat(self):
        reg = ToolRegistry()
        result = reg.detect_tool_intent("cat /etc/hosts")
        assert result is not None
        assert result[0] == "file_read"

    def test_detect_knowledge_recall(self):
        reg = ToolRegistry()
        result = reg.detect_tool_intent("recall deployment guide")
        assert result is not None
        assert result[0] == "knowledge_retrieval"

    def test_detect_image_describe(self):
        reg = ToolRegistry()
        result = reg.detect_tool_intent("describe image cat.jpg")
        assert result is not None
        assert result[0] == "image_analysis"

    def test_detect_data_summarize(self):
        reg = ToolRegistry()
        result = reg.detect_tool_intent("summarize data results.json")
        assert result is not None
        assert result[0] == "data_analysis"

    def test_detect_code_block(self):
        reg = ToolRegistry()
        result = reg.detect_tool_intent("```python\nprint('hello')\n```")
        assert result is not None
        assert result[0] == "run_code"
        assert result[1]["language"] == "python"

    def test_detect_code_block_bash(self):
        reg = ToolRegistry()
        result = reg.detect_tool_intent("```bash\necho hello\n```")
        assert result is not None
        assert result[1]["language"] == "bash"

    def test_registry_overwrite(self):
        reg = ToolRegistry()
        async def new_calc(**kwargs):
            return {"output": "overwritten"}
        spec = ToolSpec(name="calculator", description="Override", parameters=[], execute=new_calc)
        reg.register(spec)
        assert reg.get("calculator").description == "Override"


class TestGetToolRegistry:
    def test_singleton(self):
        r1 = get_tool_registry()
        r2 = get_tool_registry()
        assert r1 is r2

    def test_singleton_has_defaults(self):
        reg = get_tool_registry()
        assert len(reg.list_tools()) >= 8
