"""Tests for domains.agents.tools — ToolParam, ToolResult; domains.agents.multi — TaskStatus, AgentTask."""

from domains.agents.tools import ToolParam, ToolResult
from domains.agents.multi import TaskStatus, AgentTask


class TestToolParam:
    def test_fields(self):
        tp = ToolParam(name="expr", type="string", description="math expr", required=True)
        assert tp.name == "expr"
        assert tp.type == "string"
        assert tp.required is True

    def test_defaults(self):
        tp = ToolParam(name="x", type="int", description="val")
        assert tp.required is False


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


class TestTaskStatus:
    def test_values(self):
        assert TaskStatus.PENDING == "pending"
        assert TaskStatus.IN_PROGRESS == "in_progress"
        assert TaskStatus.COMPLETED == "completed"
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
