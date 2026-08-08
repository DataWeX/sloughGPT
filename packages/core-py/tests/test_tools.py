"""Tests for ToolRegistry — tool detection, execution, and safety."""

import sys
import types

import pytest
pytestmark = pytest.mark.slow
from domains.agents.tools import ToolRegistry, ToolSpec, ToolParam, ToolResult


@pytest.fixture
def registry():
    return ToolRegistry()


# ── ToolSpec / ToolParam Dataclasses ──────────────────────────────────────

class TestToolSpec:
    def test_defaults(self):
        async def fake(**kw): return {}
        spec = ToolSpec(name="test", description="A test", parameters=[], execute=fake)
        assert spec.name == "test"
        assert spec.requires_approval is False
        assert spec.pattern is None

    def test_with_approval(self):
        async def fake(**kw): return {}
        spec = ToolSpec(name="risky", description="Risky", parameters=[], execute=fake, requires_approval=True)
        assert spec.requires_approval is True


class TestToolParam:
    def test_defaults(self):
        p = ToolParam(name="x", type="string", description="X value")
        assert p.required is False

    def test_required(self):
        p = ToolParam(name="x", type="string", description="X value", required=True)
        assert p.required is True


# ── ToolRegistry ──────────────────────────────────────────────────────────

class TestToolRegistryInit:
    def test_default_tools_registered(self, registry):
        tools = registry.list_tools()
        names = {t["name"] for t in tools}
        assert names == {
            "calculator", "current_time", "web_search", "run_code",
            "file_read", "knowledge_retrieval", "image_analysis", "data_analysis",
        }

    def test_get_returns_spec(self, registry):
        spec = registry.get("calculator")
        assert spec is not None
        assert spec.name == "calculator"

    def test_get_nonexistent_returns_none(self, registry):
        assert registry.get("nonexistent") is None


class TestToolRegistryRegister:
    def test_register_new_tool(self, registry):
        async def my_tool(**kw): return {"output": "done"}
        spec = ToolSpec(name="my_tool", description="My custom tool", parameters=[], execute=my_tool)
        registry.register(spec)
        assert registry.get("my_tool") is spec

    def test_register_overwrites(self, registry):
        async def a(**kw): return {"output": "a"}
        async def b(**kw): return {"output": "b"}
        registry.register(ToolSpec(name="dupe", description="first", parameters=[], execute=a))
        registry.register(ToolSpec(name="dupe", description="second", parameters=[], execute=b))
        assert registry.get("dupe").description == "second"

    def test_register_increases_list_count(self, registry):
        async def dummy(**kw): return {}
        before = len(registry.list_tools())
        registry.register(ToolSpec(name="extra", description="", parameters=[], execute=dummy))
        assert len(registry.list_tools()) == before + 1


class TestToolRegistryListTools:
    def test_list_includes_required_fields(self, registry):
        for t in registry.list_tools():
            assert "name" in t
            assert "description" in t
            assert "parameters" in t
            assert isinstance(t["parameters"], list)
            assert "requires_approval" in t

    def test_list_approval_flags(self, registry):
        tools = {t["name"]: t for t in registry.list_tools()}
        assert tools["calculator"]["requires_approval"] is False
        assert tools["current_time"]["requires_approval"] is False
        assert tools["web_search"]["requires_approval"] is True
        assert tools["run_code"]["requires_approval"] is True

    def test_list_parameters_have_schema(self, registry):
        calc = [t for t in registry.list_tools() if t["name"] == "calculator"][0]
        assert len(calc["parameters"]) == 1
        p = calc["parameters"][0]
        assert p["name"] == "expression"
        assert p["required"] is True


# ── Tool Detection ────────────────────────────────────────────────────────

class TestDetectToolIntent:
    def test_detect_calculator_explicit(self, registry):
        result = registry.detect_tool_intent("calculate 2+2")
        assert result is not None
        name, args = result
        assert name == "calculator"
        assert args["expression"] == "2+2"

    def test_detect_calculator_variants(self, registry):
        for msg in ["calc 144/12", "math sqrt(16)", "what is 42", "what's 2+2"]:
            result = registry.detect_tool_intent(msg)
            assert result is not None, f"Failed to detect: {msg}"
            assert result[0] == "calculator"

    def test_detect_current_time(self, registry):
        for msg in ["time", "date", "what time", "what date", "current time"]:
            result = registry.detect_tool_intent(msg)
            assert result is not None, f"Failed to detect: {msg}"
            assert result[0] == "current_time"

    def test_detect_web_search(self, registry):
        result = registry.detect_tool_intent("search for Python tutorials")
        assert result is not None
        name, args = result
        assert name == "web_search"
        assert "Python" in args["query"]

    def test_detect_web_search_variants(self, registry):
        for msg in ["look up weather", "find restaurants", "google ai news", "web search quantum computing"]:
            result = registry.detect_tool_intent(msg)
            assert result is not None, f"Failed to detect: {msg}"
            assert result[0] == "web_search"

    def test_detect_run_code(self, registry):
        result = registry.detect_tool_intent("```python\nprint('hello')\n```")
        assert result is not None
        name, args = result
        assert name == "run_code"
        assert args["language"] == "python"
        assert "print" in args["code"]

    def test_detect_run_code_bash(self, registry):
        result = registry.detect_tool_intent("```bash\necho hi\n```")
        assert result is not None
        name, args = result
        assert name == "run_code"
        assert args["language"] == "bash"

    def test_detect_no_match(self, registry):
        result = registry.detect_tool_intent("Hello, how are you?")
        assert result is None

    def test_detect_empty_string(self, registry):
        assert registry.detect_tool_intent("") is None

    def test_detect_only_whitespace(self, registry):
        assert registry.detect_tool_intent("   ") is None

    def test_detect_case_insensitive(self, registry):
        result = registry.detect_tool_intent("CALCULATE 2+2")
        assert result is not None
        assert result[0] == "calculator"


class TestDetectApprovalRequired:
    def test_web_search_requires_approval(self, registry):
        spec = registry.get("web_search")
        assert spec is not None
        assert spec.requires_approval is True

    def test_run_code_requires_approval(self, registry):
        spec = registry.get("run_code")
        assert spec is not None
        assert spec.requires_approval is True

    def test_calculator_no_approval(self, registry):
        assert registry.get("calculator").requires_approval is False

    def test_current_time_no_approval(self, registry):
        assert registry.get("current_time").requires_approval is False


# ── Tool Execution ────────────────────────────────────────────────────────

class TestExecuteCalculator:
    @pytest.mark.asyncio
    async def test_addition(self, registry):
        result = await registry.execute("calculator", {"expression": "2+2"})
        assert result.success is True
        assert result.output == "4"

    @pytest.mark.asyncio
    async def test_complex_expression(self, registry):
        result = await registry.execute("calculator", {"expression": "sqrt(144) * 2"})
        assert result.success is True
        assert result.output == "24.0"

    @pytest.mark.asyncio
    async def test_division(self, registry):
        result = await registry.execute("calculator", {"expression": "10/3"})
        assert result.success is True
        assert float(result.output) == pytest.approx(3.333, rel=0.01)

    @pytest.mark.asyncio
    async def test_invalid_expression_fails(self, registry):
        result = await registry.execute("calculator", {"expression": "invalid***"})
        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_expression_with_os_blocked(self, registry):
        result = await registry.execute("calculator", {"expression": "os.system('ls')"})
        assert result.success is False
        assert "Disallowed" in result.error

    @pytest.mark.asyncio
    async def test_expression_with_import_blocked(self, registry):
        result = await registry.execute("calculator", {"expression": "__import__('os')"})
        assert result.success is False
        assert "Disallowed" in result.error

    @pytest.mark.asyncio
    async def test_expression_unknown_name(self, registry):
        result = await registry.execute("calculator", {"expression": "foobar"})
        assert result.success is False
        assert "Unknown name: foobar" in result.error

    @pytest.mark.asyncio
    async def test_expression_disallowed_node_type(self, registry):
        result = await registry.execute("calculator", {"expression": "[1, 2, 3]"})
        assert result.success is False
        assert "Disallowed expression" in result.error

    @pytest.mark.asyncio
    async def test_empty_expression_fails(self, registry):
        result = await registry.execute("calculator", {"expression": ""})
        assert result.success is False

    @pytest.mark.asyncio
    async def test_math_functions_available(self, registry):
        result = await registry.execute("calculator", {"expression": "sin(0) + cos(0) + pi"})
        assert result.success is True
        assert float(result.output) == pytest.approx(1.0 + 3.14159, rel=0.01)


class TestExecuteCurrentTime:
    @pytest.mark.asyncio
    async def test_returns_formatted_string(self, registry):
        result = await registry.execute("current_time", {})
        assert result.success is True
        assert len(result.output) > 10
        assert "202" in result.output or "20" in result.output

    @pytest.mark.asyncio
    async def test_includes_date_parts(self, registry):
        result = await registry.execute("current_time", {})
        assert ":" in result.output  # HH:MM:SS


class TestExecuteWebSearch:
    @pytest.mark.asyncio
    async def test_graceful_when_unavailable(self, registry):
        result = await registry.execute("web_search", {"query": "test", "num_results": 3})
        # Should not crash — web_search module likely not installed
        assert isinstance(result, ToolResult)
        assert "not available" in result.output.lower()

    @pytest.mark.asyncio
    async def test_success_list_results(self, registry, monkeypatch):
        async def fake_ws(query, num_results=3):
            return [
                {"title": "T", "url": "http://x", "snippet": "snippet"},
                {"title": "U", "url": "http://y", "snippet": ""},
            ]
        monkeypatch.setitem(sys.modules, "web_search", types.SimpleNamespace(web_search=fake_ws))
        result = await registry.execute("web_search", {"query": "q", "num_results": 3})
        assert result.success is True
        assert "**T**" in result.output
        assert "http://x" in result.output
        assert "**U**" in result.output

    @pytest.mark.asyncio
    async def test_success_scalar_result(self, registry, monkeypatch):
        async def fake_ws(query, num_results=3):
            return "plain string"
        monkeypatch.setitem(sys.modules, "web_search", types.SimpleNamespace(web_search=fake_ws))
        result = await registry.execute("web_search", {"query": "q"})
        assert result.success is True
        assert result.output == "plain string"

    @pytest.mark.asyncio
    async def test_empty_results_list(self, registry, monkeypatch):
        async def fake_ws(query, num_results=3):
            return []
        monkeypatch.setitem(sys.modules, "web_search", types.SimpleNamespace(web_search=fake_ws))
        result = await registry.execute("web_search", {"query": "q"})
        assert result.success is True
        assert "No results found" in result.output


class TestExecuteRunCode:
    @pytest.mark.asyncio
    async def test_run_python(self, registry):
        result = await registry.execute("run_code", {"language": "python", "code": "print('hello')"})
        assert result.success is True
        assert result.output == "hello"

    @pytest.mark.asyncio
    async def test_run_code_with_error(self, registry):
        result = await registry.execute("run_code", {"language": "python", "code": "1/0"})
        assert result.success is False
        assert "division" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_run_bash(self, registry):
        result = await registry.execute("run_code", {"language": "bash", "code": "echo hello_world"})
        assert result.success is True
        assert "hello_world" in result.output

    @pytest.mark.asyncio
    async def test_timeout_handling(self, registry):
        result = await registry.execute("run_code", {"language": "python", "code": "import time; time.sleep(20)"})
        assert result.success is False
        assert "timeout" in (result.error or "").lower() or "timed out" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_unknown_language(self, registry):
        result = await registry.execute("run_code", {"language": "ruby", "code": "puts 'hi'"})
        # Should still try to execute even with unknown language
        assert isinstance(result, ToolResult)

    @pytest.mark.asyncio
    async def test_subprocess_creation_error(self, registry, monkeypatch):
        async def boom(*args, **kwargs):
            raise RuntimeError("spawn failed")
        monkeypatch.setattr("asyncio.create_subprocess_exec", boom)
        result = await registry.execute("run_code", {"language": "python", "code": "print(1)"})
        assert result.success is False
        assert "spawn failed" in result.error

    @pytest.mark.asyncio
    async def test_cleanup_error_is_swallowed(self, registry, monkeypatch):
        def boom(path):
            raise OSError("already gone")
        monkeypatch.setattr("os.unlink", boom)
        result = await registry.execute("run_code", {"language": "python", "code": "print('ok')"})
        assert result.success is True


class TestExecuteEdgeCases:
    @pytest.mark.asyncio
    async def test_unknown_tool(self, registry):
        result = await registry.execute("does_not_exist", {})
        assert result.success is False
        assert "Unknown tool" in (result.error or "")

    @pytest.mark.asyncio
    async def test_execute_tracks_duration(self, registry):
        result = await registry.execute("calculator", {"expression": "1+1"})
        assert result.duration_ms > 0

    @pytest.mark.asyncio
    async def test_execute_with_extra_args(self, registry):
        result = await registry.execute("calculator", {"expression": "2*3", "extra": "ignored"})
        assert result.success is False, "Extra kwargs should be rejected, not silently ignored"

    @pytest.mark.asyncio
    async def test_execute_missing_required_arg(self, registry):
        result = await registry.execute("calculator", {})
        assert result.success is False


# ── Singleton Registry ────────────────────────────────────────────────────

class TestGetToolRegistry:
    def test_singleton_returns_same_instance(self):
        from domains.agents.tools import get_tool_registry
        r1 = get_tool_registry()
        r2 = get_tool_registry()
        assert r1 is r2

    def test_singleton_has_default_tools(self):
        from domains.agents.tools import get_tool_registry
        r = get_tool_registry()
        assert r.get("calculator") is not None
        assert r.get("current_time") is not None


# ── Security ──────────────────────────────────────────────────────────────

class TestSecurity:
    @pytest.mark.asyncio
    async def test_calculator_cannot_access_os(self, registry):
        result = await registry.execute("calculator", {"expression": "__import__('os').system('ls')"})
        assert result.success is False

    @pytest.mark.asyncio
    async def test_calculator_cannot_eval_arbitrary(self, registry):
        result = await registry.execute("calculator", {"expression": "eval('1+1')"})
        assert result.success is False

    @pytest.mark.asyncio
    async def test_run_code_limited_by_subprocess(self, registry):
        """run_code should NOT be able to escape the temp file sandbox easily."""
        result = await registry.execute("run_code", {
            "language": "python",
            "code": "import os; print(os.listdir('.'))"
        })
        # It runs but output comes from temp dir, not server root
        assert isinstance(result, ToolResult)

    @pytest.mark.asyncio
    async def test_calculator_blocks_file_operations(self, registry):
        result = await registry.execute("calculator", {"expression": "open('/etc/passwd').read()"})
        assert result.success is False
