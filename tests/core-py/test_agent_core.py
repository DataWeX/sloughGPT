"""Tests for the Agent core class, ToolRunner, and SecurityBoundary."""

import asyncio
import sys
import time
import types

import pytest

import domain.agents as agents_pkg
from domain.agents import (
    Agent,
    AgentConfig,
    SecurityBoundary,
    SecurityConfig,
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
            raise TimeoutError()

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
        first = await runner.execute(
            ToolCapability.CITATION.value, {"text": "t", "sources": []}, ctx
        )
        assert first["success"] is True
        second = await runner.execute(
            ToolCapability.CITATION.value, {"text": "t", "sources": []}, ctx
        )
        assert second["success"] is False
        assert "Rate limit exceeded" in second["error"]

    async def test_window_reset(self):
        runner = ToolRunner(SecurityBoundary(SecurityConfig(rate_limit_per_minute=1)))
        runner._executed_count = 5
        runner._last_reset = asyncio.get_event_loop().time() - 61
        res = await runner.execute(
            ToolCapability.CITATION.value, {"text": "t", "sources": []}, make_context()
        )
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
        res = await runner.execute(
            ToolCapability.FILE_SEARCH.value,
            {"query": "x", "path": ".", "limit": 5},
            make_context(),
        )
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
        # Real assertion: grep must actually find the file (the old
        # --max-count=0 bug made this return [] silently).
        assert any("f.txt" in f for f in files)


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


class _FakeAnchor:
    def __init__(self, href):
        self._href = href

    def get(self, attr, default=None):
        return self._href if attr == "href" else default


class _FakeResult:
    def __init__(self, title, snippet, href=None):
        self._title = _FakeText(title)
        self._snippet = _FakeText(snippet)
        self._href = href

    def select_one(self, sel):
        if sel == ".result__title":
            return self._title
        if sel == ".result__snippet":
            return self._snippet
        if sel == ".result__a":
            return _FakeAnchor(self._href) if self._href else None
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
    def __init__(self, text, status=200):
        self.text = text
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeClient:
    def __init__(self, text, status=200, **kwargs):
        self._text = text
        self._status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, data=None, timeout=None):
        return _FakeResponse(self._text, self._status)


class _FakeHttpx:
    def __init__(self, text, status=200):
        self._text = text
        self._status = status

    def AsyncClient(self, **kwargs):
        return _FakeClient(self._text, self._status, **kwargs)


class TestToolRunnerSearchWebImpl:
    async def test_search_web_parses_results(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "httpx", _FakeHttpx("<html></html>"))
        monkeypatch.setitem(sys.modules, "bs4", _FakeBS4([_FakeResult("Hello", "World")]))
        runner = ToolRunner()
        results = await runner._search_web("q", limit=2)
        assert results == [{"title": "Hello", "snippet": "World", "url": ""}]

    async def test_search_web_missing_fields(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "httpx", _FakeHttpx("<html></html>"))
        monkeypatch.setitem(sys.modules, "bs4", _FakeBS4([_FakeResult(None, None)]))
        runner = ToolRunner()
        results = await runner._search_web("q", limit=2)
        assert results == [{"title": "", "snippet": "", "url": ""}]

    async def test_search_web_extracts_and_unwraps_url(self, monkeypatch):
        href = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage&rut=x"
        monkeypatch.setitem(sys.modules, "httpx", _FakeHttpx("<html></html>"))
        monkeypatch.setitem(sys.modules, "bs4", _FakeBS4([_FakeResult("T", "s", href=href)]))
        runner = ToolRunner()
        results = await runner._search_web("q", limit=1)
        assert results[0]["url"] == "https://example.com/page"

    async def test_search_web_http_error_is_error_not_empty_success(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "httpx", _FakeHttpx("<html></html>", status=403))
        monkeypatch.setitem(sys.modules, "bs4", _FakeBS4([]))
        runner = ToolRunner()
        res = await runner.execute(ToolCapability.WEB_SEARCH.value, {"query": "q"}, make_context())
        assert res["success"] is False
        assert "403" in res["error"]

    def test_unwrap_ddg_plain_link_passthrough(self):
        assert ToolRunner._unwrap_ddg("https://example.com/a") == "https://example.com/a"

    def test_unwrap_ddg_protocol_relative(self):
        assert ToolRunner._unwrap_ddg("//example.com/a") == "https://example.com/a"


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
            ToolCapability.CITATION.value,
            {"text": "I love python", "sources": sources},
            make_context(),
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


class TestToolRunnerKnowledgeRetrieval:
    async def test_missing_query(self):
        runner = ToolRunner()
        res = await runner.execute(ToolCapability.KNOWLEDGE_RETRIEVAL.value, {}, make_context())
        assert res["success"] is False
        assert "query required" in res["error"]

    async def test_rag_hits(self, monkeypatch):
        runner = ToolRunner()
        fact = {
            "id": "f1",
            "content": "Deploy uses Argo CD on a staging cluster.",
            "topic": "deploy",
            "source": "docs/ops.md",
            "url": "",
            "score": 0.87,
        }

        def fake_engine():
            return _FakeEngine([fact])

        monkeypatch.setattr("domain.knowledge.get_knowledge_engine", fake_engine)
        res = await runner.execute(
            ToolCapability.KNOWLEDGE_RETRIEVAL.value,
            {"query": "deploy argocd"},
            make_context(),
        )
        assert res["success"] is True
        assert res["engine"] == "rag"
        assert res["count"] == 1
        match = res["matches"][0]
        assert match["content"] == fact["content"]
        assert match["topic"] == "deploy"
        assert match["source"] == "docs/ops.md"
        assert match["score"] == 0.87

    async def test_rag_prioritized_over_files(self, monkeypatch):
        runner = ToolRunner()
        fact = {
            "id": "f1",
            "content": "How to configure the api provider",
            "topic": "config",
            "source": "docs/config.md",
            "url": "",
            "score": 0.9,
        }

        def fake_engine():
            return _FakeEngine([fact])

        monkeypatch.setattr("domain.knowledge.get_knowledge_engine", fake_engine)
        monkeypatch.setenv("WORKSPACE_ROOT", "/nonexistent")
        res = await runner.execute(
            ToolCapability.KNOWLEDGE_RETRIEVAL.value,
            {"query": "config provider"},
            make_context(),
        )
        assert res["engine"] == "rag"
        assert res["matches"][0]["source"] == "docs/config.md"

    async def test_rag_empty_falls_back_to_files(self, monkeypatch, tmp_path):
        runner = ToolRunner()

        def fake_engine():
            return _FakeEngine([])

        monkeypatch.setattr("domain.knowledge.get_knowledge_engine", fake_engine)
        doc = tmp_path / "docs" / "guide.md"
        doc.parent.mkdir(parents=True)
        doc.write_text("redis caching: the tuning harness stores index chunks here.")
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        res = await runner.execute(
            ToolCapability.KNOWLEDGE_RETRIEVAL.value,
            {"query": "redis caching"},
            make_context(),
        )
        assert res["success"] is True
        assert res["engine"] == "files"
        assert res["count"] == 1
        assert "guide.md" in res["matches"][0]["file"]

    async def test_rag_failure_falls_back_to_files(self, monkeypatch, tmp_path):
        runner = ToolRunner()

        def fake_engine():
            raise RuntimeError("boom")

        monkeypatch.setattr("domain.knowledge.get_knowledge_engine", fake_engine)
        doc = tmp_path / "docs" / "guide.md"
        doc.parent.mkdir(parents=True)
        doc.write_text("redis caching: the tuning harness stores index chunks here.")
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        res = await runner.execute(
            ToolCapability.KNOWLEDGE_RETRIEVAL.value,
            {"query": "redis caching"},
            make_context(),
        )
        assert res["success"] is True
        assert res["engine"] == "files"
        assert res["count"] == 1


class _FakeEngine:
    """Minimal engine double — returns canned facts via .query()."""

    def __init__(self, facts):
        self._facts = facts

    def query(self, search, top_k=10):
        from domain.knowledge.engine import KnowledgeResult

        return KnowledgeResult(success=True, data=self._facts, metadata={"count": len(self._facts)})


# ── Tool correctness: path guard, code exec, data analysis ─────────────────


class TestToolRunnerPathGuard:
    async def test_file_read_denies_sensitive_dir(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        secret = tmp_path / ".ssh" / "known_hosts"
        secret.parent.mkdir()
        secret.write_text("key material")
        runner = ToolRunner()
        res = await runner.execute(
            ToolCapability.FILE_READ.value, {"path": str(secret)}, make_context()
        )
        assert res["success"] is False
        assert "sensitive" in res["error"]

    async def test_file_read_denies_dotenv(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("API_KEY=abc")
        runner = ToolRunner()
        res = await runner.execute(
            ToolCapability.FILE_READ.value, {"path": str(env_file)}, make_context()
        )
        assert res["success"] is False
        assert "sensitive" in res["error"]

    async def test_file_search_denies_sensitive_path(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        (tmp_path / ".aws").mkdir()
        runner = ToolRunner()
        res = await runner.execute(
            ToolCapability.FILE_SEARCH.value,
            {"query": "x", "path": str(tmp_path / ".aws")},
            make_context(),
        )
        assert res["success"] is False
        assert "sensitive" in res["error"]

    async def test_allowed_directories_allowlist_enforced(self, tmp_path):
        inside = tmp_path / "ok"
        inside.mkdir()
        (inside / "a.txt").write_text("hello")
        outside = tmp_path / "outside.txt"
        outside.write_text("secret")
        runner = ToolRunner(SecurityBoundary(SecurityConfig(allowed_directories=[str(inside)])))
        res_in = await runner.execute(
            ToolCapability.FILE_READ.value, {"path": str(inside / "a.txt")}, make_context()
        )
        assert res_in["success"] is True
        res_out = await runner.execute(
            ToolCapability.FILE_READ.value, {"path": str(outside)}, make_context()
        )
        assert res_out["success"] is False
        assert "allowed_directories" in res_out["error"]

    async def test_data_analysis_denies_sensitive_path(self, tmp_path):
        runner = ToolRunner()
        res = await runner.execute(
            ToolCapability.DATA_ANALYSIS.value,
            {"data_path": str(tmp_path / ".env")},
            make_context(),
        )
        assert res["success"] is False
        assert "sensitive" in res["error"]


class TestToolRunnerCodeExec:
    async def test_python_runs_in_project_interpreter(self, monkeypatch):
        runner = ToolRunner()
        res = await runner.execute(
            ToolCapability.CODE_EXECUTION.value,
            {"code": "import sys; print(sys.executable)", "language": "python"},
            make_context(),
        )
        assert res["success"] is True
        assert res["stdout"].strip() == sys.executable

    async def test_timeout_kills_child_process(self, monkeypatch, tmp_path):
        boundary = SecurityBoundary(SecurityConfig(max_execution_time=1))
        runner = ToolRunner(boundary)
        sentinel = tmp_path / "tick.txt"
        code = (
            "import time, pathlib\n"
            f"p = pathlib.Path({str(sentinel)!r})\n"
            "while True:\n"
            "    p.write_text(str(time.time()))\n"
            "    time.sleep(0.05)\n"
        )
        res = await runner.execute(
            ToolCapability.CODE_EXECUTION.value,
            {"code": code, "language": "python"},
            make_context(),
        )
        assert res["success"] is False
        assert "timed out" in res["error"].lower()
        # if the child outlived the timeout, the sentinel keeps ticking
        time.sleep(0.3)
        first = sentinel.read_text()
        time.sleep(0.4)
        assert sentinel.read_text() == first


class TestToolRunnerDataAnalysis:
    async def test_jsonl_skips_malformed_tail_line(self, tmp_path):
        data = tmp_path / "rows.jsonl"
        data.write_text('{"a": 1}\n{"a": 2}\n{"a": 3')  # third line cut off
        runner = ToolRunner()
        res = await runner.execute(
            ToolCapability.DATA_ANALYSIS.value, {"data_path": str(data)}, make_context()
        )
        assert res["success"] is True
        assert res["items"] == 2
        assert res["bad_lines"] == 1
        assert res["truncated"] is False

    async def test_large_file_flags_truncation(self, tmp_path):
        data = tmp_path / "big.jsonl"
        line = '{"a": 1, "b": "xxxxxxxxxxxxxxxxxxxxxxxx"}\n'
        data.write_text(line * 20_000)  # ~900KB > 500KB cap
        runner = ToolRunner()
        res = await runner.execute(
            ToolCapability.DATA_ANALYSIS.value, {"data_path": str(data)}, make_context()
        )
        assert res["success"] is True
        assert res["truncated"] is True
        assert res["bad_lines"] >= 1  # cut mid-line tail


class TestKnowledgeFallbackTokenMatch:
    async def test_words_split_across_lines_still_match(self, monkeypatch, tmp_path):
        runner = ToolRunner()

        def fake_engine():
            return _FakeEngine([])

        monkeypatch.setattr("domain.knowledge.get_knowledge_engine", fake_engine)
        doc = tmp_path / "notes.md"
        doc.write_text("redis\ncaching: the harness stores chunks here.")
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        res = await runner.execute(
            ToolCapability.KNOWLEDGE_RETRIEVAL.value,
            {"query": "redis caching"},
            make_context(),
        )
        # old whole-phrase substring check missed "redis\ncaching"
        assert res["engine"] == "files"
        assert res["count"] == 1


class TestImageAnalysisHonesty:
    async def test_vision_unavailable_reports_failure(self, monkeypatch, tmp_path):
        img = tmp_path / "pic.png"
        img.write_bytes(b"\x89PNG fake")
        stub = types.ModuleType("domain.multimodal._internal.vision")
        monkeypatch.setitem(sys.modules, "domain.multimodal._internal.vision", stub)
        runner = ToolRunner()
        res = await runner.execute(
            ToolCapability.IMAGE_ANALYSIS.value,
            {"image_path": str(img)},
            make_context(),
        )
        assert res["success"] is False
        assert "Vision model not available" in res["error"]


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
            agent,
            "_inference_fn",
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
        out = agent._compose_response(
            "q", [{"tool": "code", "result": {"success": True, "stdout": "hello"}}]
        )
        assert out == "hello"

    async def test_files(self):
        agent = Agent()
        out = agent._compose_response(
            "q", [{"tool": "f", "result": {"success": True, "files": ["a"], "count": 1}}]
        )
        assert out == "Found 1 files"

    async def test_results(self):
        agent = Agent()
        out = agent._compose_response(
            "q", [{"tool": "w", "result": {"success": True, "results": [{}], "count": 2}}]
        )
        assert out == "Found 2 results"

    async def test_citations(self):
        agent = Agent()
        out = agent._compose_response(
            "q", [{"tool": "c", "result": {"success": True, "citations": [{}], "count": 3}}]
        )
        assert out == "Generated 3 citations"

    async def test_error(self):
        agent = Agent()
        out = agent._compose_response(
            "q", [{"tool": "t", "result": {"success": False, "error": "boom"}}]
        )
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

        def fake(prompt):
            return "replaced"

        agent.set_inference_fn(fake)
        assert agent._inference_fn is fake


# ── Singletons & lazy imports ──────────────────────────────────────────────


class TestSingletons:
    async def test_get_agent_singleton(self, monkeypatch):
        import domain.agents._internal.agents as _mod

        monkeypatch.setattr(_mod, "_agent", None)
        a = get_agent()
        b = get_agent()
        assert a is b

    async def test_get_runner_singleton(self, monkeypatch):
        import domain.agents._internal.agents as _mod

        monkeypatch.setattr(_mod, "_runner", None)
        a = get_runner()
        b = get_runner()
        assert a is b


class TestLazyImports:
    def test_lazy_multi_import(self):
        from domain.agents._internal.multi import MultiAgentOrchestrator

        assert MultiAgentOrchestrator is not None

    def test_lazy_import_bad_name(self):
        with pytest.raises(AttributeError):
            agents_pkg.does_not_exist
