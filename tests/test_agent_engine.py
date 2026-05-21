"""
Tests for AgentEngine (domains/agent/engine.py).
"""

from unittest.mock import MagicMock
import pytest

from domains.agent.engine import AgentEngine, Tool, AgentRun


def _make_engine(tools=None):
    """Helper to create an AgentEngine with a mock underlying model."""
    mock_model = MagicMock()
    mock_model.generate.return_value = "This is a test response"
    return AgentEngine(mock_model, tools=tools or [])


class TestTool:
    """Tests for the Tool dataclass."""

    def test_tool_creation(self):
        t = Tool(name="echo", description="Echo back text", fn=lambda x: x)
        assert t.name == "echo"
        assert t.description == "Echo back text"
        assert t.fn("hello") == "hello"

    def test_tool_default_parameters(self):
        t = Tool(name="test", description="test", fn=lambda: "ok")
        assert t.parameters["type"] == "object"


class TestAgentRun:
    """Tests for the AgentRun dataclass."""

    def test_defaults(self):
        r = AgentRun(prompt="hello")
        assert r.prompt == "hello"
        assert r.response == ""
        assert r.tool_calls == []
        assert r.steps == []
        assert r.elapsed_ms == 0.0
        assert r.error is None


class TestAgentEngine:
    """Tests for the AgentEngine reasoning loop."""

    def test_init_no_tools(self):
        engine = _make_engine()
        assert engine.list_tools() == []
        assert engine._max_steps == 6

    def test_init_with_tools(self):
        t = Tool(name="echo", description="Echo", fn=lambda x: x)
        engine = _make_engine(tools=[t])
        tools = engine.list_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "echo"

    def test_register_tool(self):
        engine = _make_engine()
        t = Tool(name="get_time", description="Get time", fn=lambda: "12:00")
        engine.register_tool(t)
        assert len(engine.list_tools()) == 1

    def test_unregister_tool(self):
        t = Tool(name="echo", description="Echo", fn=lambda x: x)
        engine = _make_engine(tools=[t])
        engine.unregister_tool("echo")
        assert engine.list_tools() == []

    def test_list_tools_returns_correct_format(self):
        t = Tool(name="echo", description="Echo back text", fn=lambda x: x)
        engine = _make_engine(tools=[t])
        result = engine.list_tools()
        assert result[0] == {"name": "echo", "description": "Echo back text"}

    def test_build_system_prompt(self):
        t = Tool(name="echo", description="Echo back text", fn=lambda x: x)
        engine = _make_engine(tools=[t])
        prompt = engine._build_system_prompt()
        assert "TOOL_CALL:" in prompt
        assert "echo" in prompt
        assert "Echo back text" in prompt

    def test_run_no_tools_returns_generated_text(self):
        engine = _make_engine()
        result = engine.run("hello")
        assert result.response == "This is a test response"
        assert result.tool_calls == []
        assert result.prompt == "hello"

    def test_run_with_tool_call(self):
        """When model outputs a tool call, the engine should execute it."""
        mock_model = MagicMock()
        mock_model.generate.side_effect = [
            "TOOL_CALL: echo\nargs: {\"text\": \"hello\"}",
            "Final response after tool call",
        ]
        echo_fn = MagicMock(return_value="echoed: hello")
        t = Tool(name="echo", description="Echo", fn=echo_fn)
        engine = AgentEngine(mock_model, tools=[t])
        result = engine.run("say hello")
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["tool"] == "echo"
        assert echo_fn.called
        assert result.response == "Final response after tool call"

    def test_run_with_unknown_tool(self):
        """When model calls an unknown tool, an error should be recorded."""
        mock_model = MagicMock()
        mock_model.generate.side_effect = [
            "TOOL_CALL: unknown_tool\nargs: {}",
            "Fallback response",
        ]
        engine = AgentEngine(mock_model)
        result = engine.run("do something")
        assert len(result.steps) == 1
        assert "unknown_tool" in result.steps[0]["result"]

    def test_max_steps_prevents_infinite_loop(self):
        """Agent should stop after max steps with no final response."""
        mock_model = MagicMock()
        mock_model.generate.return_value = "TOOL_CALL: echo\nargs: {}"
        t = Tool(name="echo", description="Echo", fn=lambda: "result")
        engine = AgentEngine(mock_model, tools=[t])
        engine._max_steps = 3
        result = engine.run("loop")
        assert result.response == "[Agent reached max steps without final response]"
        assert len(result.steps) == 3

    def test_session_memory_accumulates(self):
        engine = _make_engine()
        engine.run("first message")
        assert len(engine._session_memory) == 2
        assert engine._session_memory[0]["role"] == "user"
        assert engine._session_memory[1]["role"] == "assistant"
        engine.run("second message")
        assert len(engine._session_memory) == 4

    def test_reset_memory(self):
        engine = _make_engine()
        engine.run("test")
        assert len(engine._session_memory) > 0
        engine.reset_memory()
        assert engine._session_memory == []

    def test_parse_tool_call(self):
        engine = _make_engine()
        name, args = engine._parse_tool_call("TOOL_CALL: echo\nargs: {\"text\": \"hi\"}")
        assert name == "echo"
        assert args == '{"text": "hi"}'

    def test_parse_tool_call_no_args(self):
        engine = _make_engine()
        name, args = engine._parse_tool_call("TOOL_CALL: get_time\n")
        assert name == "get_time"
        assert args == "{}"

    def test_status(self):
        t = Tool(name="echo", description="Echo", fn=lambda x: x)
        engine = _make_engine(tools=[t])
        status = engine.status()
        assert status["tools"] == [{"name": "echo", "description": "Echo"}]
        assert status["memory_size"] == 0
        assert status["max_steps"] == 6

    def test_tool_error_handling(self):
        """When a tool raises an exception, the error should be captured."""
        mock_model = MagicMock()
        mock_model.generate.side_effect = [
            "TOOL_CALL: fail_tool\nargs: {}",
            "Fallback response",
        ]
        def failing_fn():
            raise ValueError("something went wrong")
        t = Tool(name="fail_tool", description="Fails", fn=failing_fn)
        engine = AgentEngine(mock_model, tools=[t])
        result = engine.run("test")
        assert "Error" in result.steps[0]["result"]
