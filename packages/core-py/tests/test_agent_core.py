"""Tests for the Agent core class, ToolRunner, and SecurityBoundary."""

import asyncio
import sys

import pytest

import domains.agents as agents_pkg
from domains.agents import (
    Agent,
    AgentConfig,
    SecurityConfig,
    SecurityBoundary,
    ToolCapability,
    ToolExecutionContext,
    ToolRunner,
    get_agent,
    get_runner,
)


def make_context() -> ToolExecutionContext:
    return ToolExecutionContext(session_id="sess", user_id="user", timestamp=0.0)


# ── SecurityBoundary ───────────────────────────────────────────────────────


class TestSecurityBoundary:
    def test_allows_plain_code(self):
        ok, msg = SecurityBoundary().is_allowed("print(1)")
        assert ok is True
        assert msg == ""

    def test_blocks_os_remove(self):
        ok, msg = SecurityBoundary().is_allowed("import os\nos.remove('x')")
        assert ok is False
        assert "Blocked pattern" in msg

    def test_blocks_eval(self):
        ok, _ = SecurityBoundary().is_allowed("eval('1+1')")
        assert ok is False

    def test_blocks_subprocess_shell_true(self):
        ok, _ = SecurityBoundary().is_allowed("subprocess.run('ls', shell=True)")
        assert ok is False

    def test_default_config_values(self):
        config = SecurityBoundary().config
        assert config.max_execution_time == 30
        assert config.max_memory_mb == 512
        assert config.rate_limit_per_minute == 60

    def test_resource_limit_context(self):
        with SecurityBoundary().resource_limit("tool"):
            assert True


# ── ToolRunner: routing & rate limiting ────────────────────────────────────


class TestToolRunnerRouting:
    async def test_unknown_tool(self):
        runner = ToolRunner()
        res = await runner.execute("nope", {}, make_context())
        assert res["success"] is False
        assert "Unknown tool" in res["error"]

    async def test_code_execution_stdout(self):
        runner = ToolRunner()
        res = await runner.execute(
            ToolCapability.CODE_EXECUTION.value,
            {"code": "print(6*7)", "language": "python"},
            make_context(),
        )
        assert res["success"] is True
        assert res["stdout"].strip() == "42"

    async def test_code_blocked_by_security(self):
        runner = ToolRunner()
        res = await runner.execute(
            ToolCapability.CODE_EXECUTION.value,
            {"code": "eval('1')", "language": "python"},
            make_context(),
        )
        assert res["success"] is False
        assert "Blocked pattern" in res["error"]

    async def test_code_timeout(self, monkeypatch):
        runner = ToolRunner()

        async def boom(code, language):
            raise asyncio.TimeoutError()

        monkeypatch.setattr(runner, "_execute_subprocess", boom)
        res = await runner.execute(
            ToolCapability.CODE_EXECUTION.value,
            {"code": "x", "language": "python"},
            make_context(),
        )
        assert res["success"] is False
        assert res["error"] == "Execution timed out"

    async def test_code_exception(self, monkeypatch):
        runner = ToolRunner()

        async def boom(code, language):
            raise RuntimeError("kaboom")

        monkeypatch.setattr(runner, "_execute_subprocess", boom)
        res = await runner.execute(
            ToolCapability.CODE_EXECUTION.value,
            {"code": "x", "language": "python"},
            make_context(),
        )
        assert res["success"] is False
        assert "kaboom" in res["error"]


class TestToolRunnerRateLimit:
    async def test_exceeded_returns_error(self):
        runner = ToolRunner(SecurityBoundary(SecurityConfig(rate_limit_per_minute=1)))
        ctx = make_context()
        first = await runner.execute(ToolCapability.CITATION.value, {"text": "t", "sources": []}, ctx)
        assert first["success"] is True
        second = await runner.execute(ToolCapability.CITATION.value, {"text": "t", "sources": []}, ctx)
        assert second["success"] is False
        assert "Rate limit exceeded" in second["error"]

    async def test_window_reset(self):
        runner = ToolRunner(SecurityBoundary(SecurityConfig(rate_limit_per_minute=1)))
        runner._executed_count = 5
        runner._last_reset = asyncio.get_event_loop().time() - 61
        res = await runner.execute(ToolCapability.CITATION.value, {"text": "t", "sources": []}, make_context())
        assert res["success"] is True


class TestToolRunnerFileSearch:
    async def test_missing_query(self):
        runner = ToolRunner()
        res = await runner.execute(ToolCapability.FILE_SEARCH.value, {}, make_context())
        assert res["success"] is False
        assert "query required" in res["error"]

    async def test_success(self, monkeypatch):
        runner = ToolRunner()

        async def fake(query, path, limit):
            return ["a.py"]

        monkeypatch.setattr(runner, "_search_files", fake)
        res = await runner.execute(ToolCapability.FILE_SEARCH.value, {"query": "x", "path": ".", "limit": 5}, make_context())
        assert res["success"] is True
        assert res["files"] == ["a.py"]
        assert res["count"] == 1

    async def test_error(self, monkeypatch):
        runner = ToolRunner()

        async def fake(query, path, limit):
            raise OSError("bad path")

        monkeypatch.setattr(runner, "_search_files", fake)
        res = await runner.execute(ToolCapability.FILE_SEARCH.value, {"query": "x"}, make_context())
        assert res["success"] is False

    async def test_real_grep(self, tmp_path):
        (tmp_path / "f.txt").write_text("needle")
        runner = ToolRunner()
        files = await runner._search_files("needle", str(tmp_path), 10)
        assert isinstance(files, list)


class TestToolRunnerWebSearch:
    async def test_missing_query(self):
        runner = ToolRunner()
        res = await runner.execute(ToolCapability.WEB_SEARCH.value, {}, make_context())
        assert res["success"] is False
        assert "query required" in res["error"]

    async def test_success(self, monkeypatch):
        runner = ToolRunner()

        async def fake(query, limit=10):
            return [{"title": "T", "snippet": "s"}]

        monkeypatch.setattr(runner, "_search_web", fake)
        res = await runner.execute(ToolCapability.WEB_SEARCH.value, {"query": "q"}, make_context())
        assert res["success"] is True
        assert res["count"] == 1

    async def test_error(self, monkeypatch):
        runner = ToolRunner()

        async def fake(query, limit=10):
            raise RuntimeError("net down")

        monkeypatch.setattr(runner, "_search_web", fake)
        res = await runner.execute(ToolCapability.WEB_SEARCH.value, {"query": "q"}, make_context())
        assert res["success"] is False


class _FakeText:
    def __init__(self, text):
        self._text = text

    def get_text(self, strip=False):
        text = self._text or ""
        return text.strip() if strip else text


class _FakeResult:
    def __init__(self, title, snippet):
        self._title = _FakeText(title)
        self._snippet = _FakeText(snippet)

    def select_one(self, sel):
        if sel == ".result__title":
            return self._title
        if sel == ".result__snippet":
            return self._snippet
        return None


class _FakeSoup:
    def __init__(self, results):
        self._results = results

    def select(self, sel):
        return self._results


class _FakeBS4:
    def __init__(self, results):
        self._results = results

    def BeautifulSoup(self, text, parser):
        return _FakeSoup(self._results)


class _FakeResponse:
    def __init__(self, text):
        self.text = text


class _FakeClient:
    def __init__(self, text):
        self._text = text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, data=None, timeout=None):
        return _FakeResponse(self._text)


class _FakeHttpx:
    def __init__(self, text):
        self._text = text

    def AsyncClient(self):
        return _FakeClient(self._text)


class TestToolRunnerSearchWebImpl:
    async def test_search_web_parses_results(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "httpx", _FakeHttpx("<html></html>"))
        monkeypatch.setitem(sys.modules, "bs4", _FakeBS4([_FakeResult("Hello", "World")]))
        runner = ToolRunner()
        results = await runner._search_web("q", limit=2)
        assert results == [{"title": "Hello", "snippet": "World"}]

    async def test_search_web_missing_fields(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "httpx", _FakeHttpx("<html></html>"))
        monkeypatch.setitem(sys.modules, "bs4", _FakeBS4([_FakeResult(None, None)]))
        runner = ToolRunner()
        results = await runner._search_web("q", limit=2)
        assert results == [{"title": "", "snippet": ""}]


class TestToolRunnerCitation:
    async def test_missing_text(self):
        runner = ToolRunner()
        res = await runner.execute(ToolCapability.CITATION.value, {}, make_context())
        assert res["success"] is False
        assert "text required" in res["error"]

    async def test_success(self):
        runner = ToolRunner()
        sources = [{"text": "python is a language", "url": "http://py"}]
        res = await runner.execute(
            ToolCapability.CITATION.value, {"text": "I love python", "sources": sources}, make_context()
        )
        assert res["success"] is True
        assert res["count"] == 1
        assert res["citations"][0]["url"] == "http://py"

    async def test_sorted_by_relevance(self):
        runner = ToolRunner()
        sources = [
            {"text": "hello world", "url": "a"},
            {"text": "hello hello hello", "url": "b"},
        ]
        res = await runner.execute(
            ToolCapability.CITATION.value, {"text": "hello", "sources": sources}, make_context()
        )
        assert res["citations"][0]["url"] == "b"

    async def test_no_overlap(self):
        runner = ToolRunner()
        assert runner._generate_citations("xyz", [{"text": "abc"}]) == []


# ── Agent: planning ────────────────────────────────────────────────────────


class TestAgentPlanning:
    async def test_keywords_code(self):
        agent = Agent()
        plan = agent._plan_with_keywords("run ```python\nprint(1)\n```")
        assert plan == [
            (ToolCapability.CODE_EXECUTION.value, {"code": "print(1)\n", "language": "python"})
        ]

    async def test_keywords_search(self):
        agent = Agent()
        plan = agent._plan_with_keywords("search for 'cats'")
        assert plan == [(ToolCapability.FILE_SEARCH.value, {"query": "cats"})]

    async def test_keywords_cite(self):
        agent = Agent()
        plan = agent._plan_with_keywords("cite your sources")
        assert plan[0][0] == ToolCapability.CITATION.value

    async def test_keywords_none(self):
        agent = Agent()
        assert agent._plan_with_keywords("hello there") == []

    async def test_plan_execution_keyword_fallback(self):
        agent = Agent()
        plan = await agent._plan_execution("run ```python\nprint(1)\n```", make_context())
        assert plan[0][0] == ToolCapability.CODE_EXECUTION.value

    async def test_plan_with_llm_success(self, monkeypatch):
        agent = Agent()
        monkeypatch.setattr(
            agent, "_inference_fn",
            lambda prompt: '[{"tool": "code_execution", "args": {"code": "print(1)"}}]',
        )
        plan = await agent._plan_with_llm("run code")
        assert plan == [(ToolCapability.CODE_EXECUTION.value, {"code": "print(1)"})]

    async def test_plan_with_llm_exception_falls_back(self, monkeypatch):
        agent = Agent()

        def bad(prompt):
            raise RuntimeError("nope")

        monkeypatch.setattr(agent, "_inference_fn", bad)
        plan = await agent._plan_with_llm("search for 'cats'")
        assert plan[0][0] == ToolCapability.FILE_SEARCH.value


# ── Agent: response generation ─────────────────────────────────────────────


class TestAgentGenerateResponse:
    async def test_dict_with_text(self, monkeypatch):
        agent = Agent()
        monkeypatch.setattr(agent, "_inference_fn", lambda prompt: {"text": "hi there"})
        assert agent._generate_response("q", make_context()) == "hi there"

    async def test_str(self, monkeypatch):
        agent = Agent()
        monkeypatch.setattr(agent, "_inference_fn", lambda prompt: "plain")
        assert agent._generate_response("q", make_context()) == "plain"

    async def test_other_type(self, monkeypatch):
        agent = Agent()
        monkeypatch.setattr(agent, "_inference_fn", lambda prompt: 123)
        assert agent._generate_response("q", make_context()) == "123"

    async def test_exception(self, monkeypatch):
        agent = Agent()

        def bad(prompt):
            raise RuntimeError("x")

        monkeypatch.setattr(agent, "_inference_fn", bad)
        assert agent._generate_response("q", make_context()) == "No results"

    async def test_instructions_in_system_prompt(self, monkeypatch):
        agent = Agent(AgentConfig(instructions="Be brief."))
        captured = {}

        def fake(prompt):
            captured["prompt"] = prompt
            return {"text": "ok"}

        monkeypatch.setattr(agent, "_inference_fn", fake)
        assert agent._generate_response("q", make_context()) == "ok"
        assert "Be brief." in captured["prompt"]


class TestAgentComposeResponse:
    async def test_stdout(self):
        agent = Agent()
        out = agent._compose_response("q", [{"tool": "code", "result": {"success": True, "stdout": "hello"}}])
        assert out == "hello"

    async def test_files(self):
        agent = Agent()
        out = agent._compose_response("q", [{"tool": "f", "result": {"success": True, "files": ["a"], "count": 1}}])
        assert out == "Found 1 files"

    async def test_results(self):
        agent = Agent()
        out = agent._compose_response("q", [{"tool": "w", "result": {"success": True, "results": [{}], "count": 2}}])
        assert out == "Found 2 results"

    async def test_citations(self):
        agent = Agent()
        out = agent._compose_response("q", [{"tool": "c", "result": {"success": True, "citations": [{}], "count": 3}}])
        assert out == "Generated 3 citations"

    async def test_error(self):
        agent = Agent()
        out = agent._compose_response("q", [{"tool": "t", "result": {"success": False, "error": "boom"}}])
        assert out == "Error: boom"

    async def test_empty(self):
        agent = Agent()
        assert agent._compose_response("q", []) == "No results"


class TestAgentExecute:
    async def test_keyword_code_tool_full_execute(self):
        agent = Agent()
        result = await agent.execute("run ```python\nprint(2*21)\n```", "sess")
        assert result["session_id"] == "sess"
        assert result["tools_used"]
        assert result["tools_used"][0]["result"]["stdout"].strip() == "42"

    async def test_set_inference_fn_replaces(self, monkeypatch):
        agent = Agent()
        fake = lambda prompt: "replaced"
        agent.set_inference_fn(fake)
        assert agent._inference_fn is fake


# ── Singletons & lazy imports ──────────────────────────────────────────────


class TestSingletons:
    async def test_get_agent_singleton(self, monkeypatch):
        monkeypatch.setattr(agents_pkg, "_agent", None)
        a = get_agent()
        b = get_agent()
        assert a is b

    async def test_get_runner_singleton(self, monkeypatch):
        monkeypatch.setattr(agents_pkg, "_runner", None)
        a = get_runner()
        b = get_runner()
        assert a is b


class TestLazyImports:
    def test_lazy_multi_import(self):
        from domains.agents import MultiAgentOrchestrator

        assert MultiAgentOrchestrator is not None

    def test_lazy_import_bad_name(self):
        with pytest.raises(AttributeError):
            getattr(agents_pkg, "does_not_exist")
