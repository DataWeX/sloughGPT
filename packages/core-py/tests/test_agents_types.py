"""Tests for domains.agents — SecurityBoundary, SecurityConfig, ToolCapability, ToolDefinition, AgentConfig."""

import asyncio
import os
import tempfile
import time
from domains.agents import (
    SecurityConfig, SecurityBoundary, ToolCapability,
    ToolDefinition, ToolExecutionContext, AgentConfig,
    Agent, get_agent, get_runner,
    ToolRunner,
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

    def test_all_fields(self):
        cfg = SecurityConfig(
            max_execution_time=60,
            max_memory_mb=1024,
            max_file_size_mb=200,
            allow_network=True,
            allowed_directories=["/tmp", "/data"],
            blocked_patterns=["pattern1"],
            rate_limit_per_minute=120,
        )
        assert cfg.max_execution_time == 60
        assert cfg.max_memory_mb == 1024
        assert cfg.max_file_size_mb == 200
        assert cfg.allow_network is True
        assert cfg.allowed_directories == ["/tmp", "/data"]
        assert cfg.blocked_patterns == ["pattern1"]
        assert cfg.rate_limit_per_minute == 120

    def test_default_allowed_directories_empty(self):
        cfg = SecurityConfig()
        assert cfg.allowed_directories == []

    def test_default_blocked_patterns_empty(self):
        cfg = SecurityConfig()
        assert cfg.blocked_patterns == []


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

    def test_blocked_marshal(self):
        sb = SecurityBoundary()
        ok, _ = sb.is_allowed("marshal.loads(data)")
        assert ok is False

    def test_blocked_compile(self):
        sb = SecurityBoundary()
        ok, _ = sb.is_allowed("compile(code, '<string>', 'exec')")
        assert ok is False

    def test_blocked_shutil_rmtree(self):
        sb = SecurityBoundary()
        ok, _ = sb.is_allowed("import shutil\nshutil.rmtree('/tmp/dir')")
        assert ok is False

    def test_blocked_dunder_import(self):
        sb = SecurityBoundary()
        ok, _ = sb.is_allowed("__import__('os')")
        assert ok is False

    def test_allowed_simple_math(self):
        sb = SecurityBoundary()
        ok, _ = sb.is_allowed("result = sum(range(10))")
        assert ok is True

    def test_allowed_list_comprehension(self):
        sb = SecurityBoundary()
        ok, _ = sb.is_allowed("[x**2 for x in range(10)]")
        assert ok is True

    def test_allowed_string_operations(self):
        sb = SecurityBoundary()
        ok, _ = sb.is_allowed("s = 'hello'.upper()")
        assert ok is True

    def test_allowed_open_r_mode(self):
        sb = SecurityBoundary()
        ok, _ = sb.is_allowed("open('file.txt', 'r')")
        # The regex only blocks 'r/w' together, not 'r' alone
        assert ok is True

    def test_allowed_open_w_mode(self):
        sb = SecurityBoundary()
        ok, _ = sb.is_allowed("open('file.txt', 'w')")
        # The regex only blocks 'r/w' together, not 'w' alone
        assert ok is True

    def test_blocked_open_rw_mode(self):
        sb = SecurityBoundary()
        ok, _ = sb.is_allowed("open('file.txt', 'r/w')")
        assert ok is False

    def test_empty_code_allowed(self):
        sb = SecurityBoundary()
        ok, _ = sb.is_allowed("")
        assert ok is True

    def test_block_message_contains_pattern(self):
        sb = SecurityBoundary()
        ok, msg = sb.is_allowed("eval('bad')")
        assert ok is False
        assert "Blocked" in msg

    def test_default_config_when_none(self):
        sb = SecurityBoundary(None)
        assert sb.config is not None
        assert sb.config.max_execution_time == 30

    def test_blocked_patterns_compiled(self):
        sb = SecurityBoundary()
        assert len(sb._blocked_re) > 0
        assert all(hasattr(p, "search") for p in sb._blocked_re)

    def test_resource_limit_exception_handled(self):
        sb = SecurityBoundary()
        try:
            with sb.resource_limit("test"):
                raise ValueError("boom")
        except ValueError:
            pass
        # Context manager should not swallow exceptions

    def test_resource_limit_nested(self):
        sb = SecurityBoundary()
        with sb.resource_limit("outer"):
            with sb.resource_limit("inner"):
                pass


class TestToolCapability:
    def test_all_members(self):
        assert len(ToolCapability) == 8

    def test_values(self):
        assert ToolCapability.CODE_EXECUTION.value == "code_execution"
        assert ToolCapability.FILE_READ.value == "file_read"

    def test_member_names(self):
        names = [c.name for c in ToolCapability]
        assert "CODE_EXECUTION" in names
        assert "FILE_READ" in names
        assert "FILE_SEARCH" in names
        assert "WEB_SEARCH" in names
        assert "KNOWLEDGE_RETRIEVAL" in names
        assert "IMAGE_ANALYSIS" in names
        assert "DATA_ANALYSIS" in names
        assert "CITATION" in names

    def test_member_values_are_strings(self):
        for cap in ToolCapability:
            assert isinstance(cap.value, str)

    def test_member_values_unique(self):
        values = [c.value for c in ToolCapability]
        assert len(values) == len(set(values))

    def test_file_search_value(self):
        assert ToolCapability.FILE_SEARCH.value == "file_search"

    def test_web_search_value(self):
        assert ToolCapability.WEB_SEARCH.value == "web_search"

    def test_knowledge_retrieval_value(self):
        assert ToolCapability.KNOWLEDGE_RETRIEVAL.value == "knowledge_retrieval"

    def test_image_analysis_value(self):
        assert ToolCapability.IMAGE_ANALYSIS.value == "image_analysis"

    def test_data_analysis_value(self):
        assert ToolCapability.DATA_ANALYSIS.value == "data_analysis"

    def test_citation_value(self):
        assert ToolCapability.CITATION.value == "citation"


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

    def test_empty_parameters(self):
        td = ToolDefinition(
            name="noop",
            description="Does nothing",
            parameters={},
            capability=ToolCapability.FILE_READ,
        )
        assert td.parameters == {}

    def test_complex_parameters(self):
        params = {
            "code": {"type": "string", "required": True},
            "language": {"type": "string", "default": "python"},
        }
        td = ToolDefinition(
            name="run",
            description="Run code",
            parameters=params,
            capability=ToolCapability.CODE_EXECUTION,
        )
        assert td.parameters == params

    def test_description_preserved(self):
        td = ToolDefinition(
            name="tool",
            description="My tool description",
            parameters={},
            capability=ToolCapability.WEB_SEARCH,
        )
        assert td.description == "My tool description"

    def test_capability_is_enum(self):
        td = ToolDefinition(
            name="tool",
            description="desc",
            parameters={},
            capability=ToolCapability.FILE_SEARCH,
        )
        assert isinstance(td.capability, ToolCapability)

    def test_all_capabilities_work(self):
        for cap in ToolCapability:
            td = ToolDefinition(
                name=cap.value,
                description="test",
                parameters={},
                capability=cap,
            )
            assert td.capability == cap


class TestToolExecutionContext:
    def test_fields(self):
        ctx = ToolExecutionContext(
            session_id="s1",
            user_id="u1",
            timestamp=time.time(),
        )
        assert ctx.session_id == "s1"
        assert ctx.metadata == {}

    def test_with_metadata(self):
        ctx = ToolExecutionContext(
            session_id="s1",
            user_id="u1",
            timestamp=time.time(),
            metadata={"key": "value", "count": 42},
        )
        assert ctx.metadata["key"] == "value"
        assert ctx.metadata["count"] == 42

    def test_timestamp_is_float(self):
        ctx = ToolExecutionContext(
            session_id="s1",
            user_id="u1",
            timestamp=1234567890.0,
        )
        assert ctx.timestamp == 1234567890.0

    def test_session_id_preserved(self):
        ctx = ToolExecutionContext(
            session_id="unique-session-123",
            user_id="user-456",
            timestamp=time.time(),
        )
        assert ctx.session_id == "unique-session-123"
        assert ctx.user_id == "user-456"

    def test_default_metadata_is_dict(self):
        ctx = ToolExecutionContext(
            session_id="s1",
            user_id="u1",
            timestamp=time.time(),
        )
        assert isinstance(ctx.metadata, dict)
        assert len(ctx.metadata) == 0


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

    def test_default_tools(self):
        cfg = AgentConfig()
        assert len(cfg.tools) == 4
        assert ToolCapability.CODE_EXECUTION in cfg.tools
        assert ToolCapability.FILE_READ in cfg.tools
        assert ToolCapability.WEB_SEARCH in cfg.tools
        assert ToolCapability.KNOWLEDGE_RETRIEVAL in cfg.tools

    def test_custom_tools(self):
        cfg = AgentConfig(tools=[ToolCapability.FILE_SEARCH, ToolCapability.CITATION])
        assert len(cfg.tools) == 2
        assert ToolCapability.FILE_SEARCH in cfg.tools

    def test_instructions_default(self):
        cfg = AgentConfig()
        assert cfg.instructions == ""

    def test_custom_instructions(self):
        cfg = AgentConfig(instructions="You are a coding assistant.")
        assert cfg.instructions == "You are a coding assistant."

    def test_security_default_none(self):
        cfg = AgentConfig()
        assert cfg.security is None

    def test_custom_security(self):
        sc = SecurityConfig(max_execution_time=60)
        cfg = AgentConfig(security=sc)
        assert cfg.security.max_execution_time == 60


class TestAgent:
    def test_default_config(self):
        agent = Agent()
        assert agent.config is not None
        assert agent.config.max_iterations == 10

    def test_custom_config(self):
        cfg = AgentConfig(max_iterations=3, timeout=30)
        agent = Agent(config=cfg)
        assert agent.config.max_iterations == 3
        assert agent.config.timeout == 30

    def test_set_inference_fn(self):
        agent = Agent()
        fn = lambda x: {"text": "response"}
        agent.set_inference_fn(fn)
        assert agent._inference_fn is fn

    def test_compose_response_empty(self):
        agent = Agent()
        response = agent._compose_response("hello", [])
        assert response == "No results"

    def test_compose_response_with_results(self):
        agent = Agent()
        results = [
            {"tool": "code_execution", "result": {"success": True, "stdout": "output"}},
            {"tool": "file_search", "result": {"success": True, "files": ["a.py"], "count": 1}},
        ]
        response = agent._compose_response("hello", results)
        assert "output" in response
        assert "Found 1 files" in response

    def test_compose_response_with_error(self):
        agent = Agent()
        results = [
            {"tool": "code_execution", "result": {"success": False, "error": "bad code"}},
        ]
        response = agent._compose_response("hello", results)
        assert "Error: bad code" in response

    def test_plan_with_keywords_code(self):
        agent = Agent()
        plan = agent._plan_with_keywords("run this code ```python\nprint(1)\n```")
        assert any(t[0] == "code_execution" for t in plan)

    def test_plan_with_keywords_search(self):
        agent = Agent()
        plan = agent._plan_with_keywords('search for "hello world"')
        assert any(t[0] == "file_search" for t in plan)

    def test_plan_with_keywords_web(self):
        agent = Agent()
        plan = agent._plan_with_keywords('web search "python docs"')
        assert any(t[0] == "web_search" for t in plan)

    def test_plan_with_keywords_read(self):
        agent = Agent()
        plan = agent._plan_with_keywords('read "myfile.txt"')
        assert any(t[0] == "file_read" for t in plan)

    def test_plan_with_keywords_citation(self):
        agent = Agent()
        plan = agent._plan_with_keywords("cite your sources research paper")
        assert any(t[0] == "citation" for t in plan)

    def test_plan_with_keywords_knowledge(self):
        agent = Agent()
        plan = agent._plan_with_keywords("knowledge deep learning basics")
        assert any(t[0] == "knowledge_retrieval" for t in plan)

    def test_plan_with_keywords_data(self):
        agent = Agent()
        plan = agent._plan_with_keywords("analyze data /tmp/data.csv")
        assert any(t[0] == "data_analysis" for t in plan)

    def test_plan_with_keywords_image(self):
        agent = Agent()
        plan = agent._plan_with_keywords("image /tmp/photo.jpg")
        assert any(t[0] == "image_analysis" for t in plan)

    def test_plan_with_keywords_no_match(self):
        agent = Agent()
        plan = agent._plan_with_keywords("hello world")
        assert plan == []

    def test_plan_with_llm_invalid_json(self):
        agent = Agent()
        agent.set_inference_fn(lambda x: {"text": "not json"})
        import asyncio
        plan = asyncio.get_event_loop().run_until_complete(
            agent._plan_with_llm("do something")
        )
        # Should fall back to keywords
        assert isinstance(plan, list)

    def test_plan_with_llm_valid_json(self):
        agent = Agent()
        agent.set_inference_fn(lambda x: {"text": '[{"tool": "code_execution", "args": {"code": "print(1)"}}]'})
        import asyncio
        plan = asyncio.get_event_loop().run_until_complete(
            agent._plan_with_llm("run code")
        )
        assert len(plan) == 1
        assert plan[0][0] == "code_execution"

    def test_plan_with_llm_empty_array(self):
        agent = Agent()
        agent.set_inference_fn(lambda x: {"text": "[]"})
        import asyncio
        plan = asyncio.get_event_loop().run_until_complete(
            agent._plan_with_llm("hello")
        )
        assert plan == []


class TestToolRunner:
    def test_check_rate_limit(self):
        runner = ToolRunner()
        assert runner._check_rate_limit() is True

    def test_generate_citations(self):
        runner = ToolRunner()
        text = "The quick brown fox jumps over the lazy dog"
        sources = [
            {"text": "The quick brown fox jumps", "url": "http://example.com"},
            {"text": "unrelated text about cooking", "url": "http://other.com"},
        ]
        citations = runner._generate_citations(text, sources)
        assert len(citations) >= 1
        assert citations[0]["id"] == "source_0"

    def test_generate_citations_empty_sources(self):
        runner = ToolRunner()
        citations = runner._generate_citations("hello world", [])
        assert citations == []

    def test_generate_citations_no_overlap(self):
        runner = ToolRunner()
        citations = runner._generate_citations("hello world", [{"text": "completely different", "url": ""}])
        assert citations == []


class TestSingletons:
    def test_get_agent_returns_agent(self):
        import domains.agents as mod
        mod._agent = None
        agent = get_agent()
        assert isinstance(agent, Agent)

    def test_get_agent_returns_same_instance(self):
        import domains.agents as mod
        mod._agent = None
        a1 = get_agent()
        a2 = get_agent()
        assert a1 is a2

    def test_get_runner_returns_runner(self):
        import domains.agents as mod
        mod._runner = None
        runner = get_runner()
        assert isinstance(runner, ToolRunner)

    def test_get_runner_returns_same_instance(self):
        import domains.agents as mod
        mod._runner = None
        r1 = get_runner()
        r2 = get_runner()
        assert r1 is r2


# ── ToolRunner Execution (pure logic, no external APIs) ──────────────────────

class TestToolRunnerRateLimit:
    def test_rate_limit_allows_first_call(self):
        runner = ToolRunner()
        assert runner._check_rate_limit() is True
        assert runner._executed_count == 1

    def test_rate_limit_counts_calls(self):
        runner = ToolRunner()
        for _ in range(5):
            runner._check_rate_limit()
        assert runner._executed_count == 5

    def test_rate_limit_blocks_at_max(self):
        runner = ToolRunner(SecurityBoundary(SecurityConfig(rate_limit_per_minute=3)))
        for _ in range(3):
            assert runner._check_rate_limit() is True
        assert runner._check_rate_limit() is False

    def test_rate_limit_resets_after_60s(self):
        runner = ToolRunner(SecurityBoundary(SecurityConfig(rate_limit_per_minute=1)))
        runner._check_rate_limit()
        runner._last_reset = asyncio.get_event_loop().time() - 61
        assert runner._check_rate_limit() is True

    def test_rate_limit_tracks_count(self):
        runner = ToolRunner(SecurityBoundary(SecurityConfig(rate_limit_per_minute=10)))
        for _ in range(7):
            runner._check_rate_limit()
        assert runner._executed_count == 7


class TestToolRunnerExecute:
    def test_execute_unknown_tool(self):
        runner = ToolRunner()
        ctx = ToolExecutionContext(session_id="s1", user_id="u1", timestamp=time.time())
        result = asyncio.get_event_loop().run_until_complete(
            runner.execute("unknown_tool", {}, ctx)
        )
        assert result["success"] is False
        assert "Unknown tool" in result["error"]

    def test_execute_rate_limit_exceeded(self):
        runner = ToolRunner(SecurityBoundary(SecurityConfig(rate_limit_per_minute=1)))
        ctx = ToolExecutionContext(session_id="s1", user_id="u1", timestamp=time.time())
        runner._check_rate_limit()
        result = asyncio.get_event_loop().run_until_complete(
            runner.execute("code_execution", {"code": "print(1)"}, ctx)
        )
        assert result["success"] is False
        assert "Rate limit" in result["error"]

    def test_execute_code_blocked_pattern(self):
        runner = ToolRunner()
        ctx = ToolExecutionContext(session_id="s1", user_id="u1", timestamp=time.time())
        result = asyncio.get_event_loop().run_until_complete(
            runner.execute("code_execution", {"code": "eval('bad')"}, ctx)
        )
        assert result["success"] is False
        assert "Blocked" in result["error"]

    def test_execute_code_success(self):
        runner = ToolRunner()
        ctx = ToolExecutionContext(session_id="s1", user_id="u1", timestamp=time.time())
        result = asyncio.get_event_loop().run_until_complete(
            runner.execute("code_execution", {"code": "print('hello')", "language": "python"}, ctx)
        )
        assert result["success"] is True
        assert "hello" in result.get("stdout", "")

    def test_execute_code_timeout(self):
        runner = ToolRunner(SecurityBoundary(SecurityConfig(max_execution_time=1)))
        ctx = ToolExecutionContext(session_id="s1", user_id="u1", timestamp=time.time())
        result = asyncio.get_event_loop().run_until_complete(
            runner.execute("code_execution", {"code": "import time; time.sleep(5)"}, ctx)
        )
        assert result["success"] is False

    def test_execute_code_js(self):
        runner = ToolRunner()
        ctx = ToolExecutionContext(session_id="s1", user_id="u1", timestamp=time.time())
        result = asyncio.get_event_loop().run_until_complete(
            runner.execute("code_execution", {"code": "console.log('hi')", "language": "javascript"}, ctx)
        )
        assert result["success"] is True

    def test_execute_code_empty(self):
        runner = ToolRunner()
        ctx = ToolExecutionContext(session_id="s1", user_id="u1", timestamp=time.time())
        result = asyncio.get_event_loop().run_until_complete(
            runner.execute("code_execution", {"code": ""}, ctx)
        )
        assert result["success"] is True

    def test_execute_code_no_code_key(self):
        runner = ToolRunner()
        ctx = ToolExecutionContext(session_id="s1", user_id="u1", timestamp=time.time())
        result = asyncio.get_event_loop().run_until_complete(
            runner.execute("code_execution", {}, ctx)
        )
        assert result["success"] is True


class TestToolRunnerFileRead:
    def test_file_read_no_path(self):
        runner = ToolRunner()
        ctx = ToolExecutionContext(session_id="s1", user_id="u1", timestamp=time.time())
        result = asyncio.get_event_loop().run_until_complete(
            runner.execute("file_read", {}, ctx)
        )
        assert result["success"] is False
        assert "path required" in result["error"]

    def test_file_read_not_found(self):
        runner = ToolRunner()
        ctx = ToolExecutionContext(session_id="s1", user_id="u1", timestamp=time.time())
        result = asyncio.get_event_loop().run_until_complete(
            runner.execute("file_read", {"path": "/nonexistent/file.txt"}, ctx)
        )
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    def test_file_read_success(self):
        runner = ToolRunner()
        ctx = ToolExecutionContext(session_id="s1", user_id="u1", timestamp=time.time())
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test content")
            path = f.name
        try:
            result = asyncio.get_event_loop().run_until_complete(
                runner.execute("file_read", {"path": path}, ctx)
            )
            assert result["success"] is True
            assert "test content" in result["content"]
        finally:
            os.unlink(path)

    def test_file_read_too_large(self):
        runner = ToolRunner()
        ctx = ToolExecutionContext(session_id="s1", user_id="u1", timestamp=time.time())
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("x" * 1_000_001)
            path = f.name
        try:
            result = asyncio.get_event_loop().run_until_complete(
                runner.execute("file_read", {"path": path}, ctx)
            )
            assert result["success"] is False
            assert "too large" in result["error"].lower()
        finally:
            os.unlink(path)


class TestToolRunnerFileSearch:
    def test_file_search_no_query(self):
        runner = ToolRunner()
        ctx = ToolExecutionContext(session_id="s1", user_id="u1", timestamp=time.time())
        result = asyncio.get_event_loop().run_until_complete(
            runner.execute("file_search", {}, ctx)
        )
        assert result["success"] is False
        assert "query required" in result["error"]

    def test_file_search_with_query(self):
        runner = ToolRunner()
        ctx = ToolExecutionContext(session_id="s1", user_id="u1", timestamp=time.time())
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.py")
            with open(filepath, "w") as f:
                f.write("def hello_world(): pass")
            result = asyncio.get_event_loop().run_until_complete(
                runner.execute("file_search", {"query": "hello_world", "path": tmpdir}, ctx)
            )
            assert result["success"] is True
            assert isinstance(result["count"], int)


class TestToolRunnerCitation:
    def test_citation_no_text(self):
        runner = ToolRunner()
        ctx = ToolExecutionContext(session_id="s1", user_id="u1", timestamp=time.time())
        result = asyncio.get_event_loop().run_until_complete(
            runner.execute("citation", {}, ctx)
        )
        assert result["success"] is False
        assert "text required" in result["error"]

    def test_citation_with_text(self):
        runner = ToolRunner()
        ctx = ToolExecutionContext(session_id="s1", user_id="u1", timestamp=time.time())
        result = asyncio.get_event_loop().run_until_complete(
            runner.execute("citation", {
                "text": "The quick brown fox",
                "sources": [{"text": "The quick brown fox jumps", "url": "http://example.com"}]
            }, ctx)
        )
        assert result["success"] is True
        assert result["count"] >= 1

    def test_citation_sorted_by_relevance(self):
        runner = ToolRunner()
        sources = [
            {"text": "unrelated cooking recipe", "url": "http://a.com"},
            {"text": "The quick brown fox jumps over", "url": "http://b.com"},
        ]
        citations = runner._generate_citations("The quick brown fox", sources)
        assert len(citations) >= 1
        # The better match should come first
        if len(citations) >= 2:
            assert citations[0]["relevance"] >= citations[1]["relevance"]

    def test_citation_empty_source_text(self):
        runner = ToolRunner()
        sources = [{"text": "", "url": ""}]
        citations = runner._generate_citations("hello world", sources)
        assert citations == []

    def test_citation_single_word_match(self):
        runner = ToolRunner()
        sources = [{"text": "hello", "url": "http://example.com"}]
        citations = runner._generate_citations("hello", sources)
        assert len(citations) == 1
        assert citations[0]["relevance"] == 1.0


class TestToolRunnerWebSearch:
    def test_web_search_no_query(self):
        runner = ToolRunner()
        ctx = ToolExecutionContext(session_id="s1", user_id="u1", timestamp=time.time())
        result = asyncio.get_event_loop().run_until_complete(
            runner.execute("web_search", {}, ctx)
        )
        assert result["success"] is False
        assert "query required" in result["error"]

    def test_web_search_with_query(self):
        runner = ToolRunner()
        ctx = ToolExecutionContext(session_id="s1", user_id="u1", timestamp=time.time())
        result = asyncio.get_event_loop().run_until_complete(
            runner.execute("web_search", {"query": "python"}, ctx)
        )
        # Will fail or succeed depending on network, but should not crash
        assert "success" in result


class TestToolRunnerKnowledgeRetrieval:
    def test_knowledge_no_query(self):
        runner = ToolRunner()
        ctx = ToolExecutionContext(session_id="s1", user_id="u1", timestamp=time.time())
        result = asyncio.get_event_loop().run_until_complete(
            runner.execute("knowledge_retrieval", {}, ctx)
        )
        assert result["success"] is False
        assert "query required" in result["error"]

    def test_knowledge_with_query(self):
        runner = ToolRunner()
        ctx = ToolExecutionContext(session_id="s1", user_id="u1", timestamp=time.time())
        result = asyncio.get_event_loop().run_until_complete(
            runner.execute("knowledge_retrieval", {"query": "python"}, ctx)
        )
        assert result["success"] is True
        assert "matches" in result


class TestToolRunnerImageAnalysis:
    def test_image_analysis_no_path(self):
        runner = ToolRunner()
        ctx = ToolExecutionContext(session_id="s1", user_id="u1", timestamp=time.time())
        result = asyncio.get_event_loop().run_until_complete(
            runner.execute("image_analysis", {}, ctx)
        )
        assert result["success"] is False
        assert "image_path required" in result["error"]

    def test_image_analysis_not_found(self):
        runner = ToolRunner()
        ctx = ToolExecutionContext(session_id="s1", user_id="u1", timestamp=time.time())
        result = asyncio.get_event_loop().run_until_complete(
            runner.execute("image_analysis", {"image_path": "/nonexistent.jpg"}, ctx)
        )
        assert result["success"] is False
        assert "not found" in result["error"].lower()


class TestToolRunnerDataAnalysis:
    def test_data_analysis_no_path(self):
        runner = ToolRunner()
        ctx = ToolExecutionContext(session_id="s1", user_id="u1", timestamp=time.time())
        result = asyncio.get_event_loop().run_until_complete(
            runner.execute("data_analysis", {}, ctx)
        )
        assert result["success"] is False
        assert "data_path required" in result["error"]

    def test_data_analysis_not_found(self):
        runner = ToolRunner()
        ctx = ToolExecutionContext(session_id="s1", user_id="u1", timestamp=time.time())
        result = asyncio.get_event_loop().run_until_complete(
            runner.execute("data_analysis", {"data_path": "/nonexistent.csv"}, ctx)
        )
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    def test_data_analysis_csv(self):
        runner = ToolRunner()
        ctx = ToolExecutionContext(session_id="s1", user_id="u1", timestamp=time.time())
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("name,age\nAlice,30\nBob,25\n")
            path = f.name
        try:
            result = asyncio.get_event_loop().run_until_complete(
                runner.execute("data_analysis", {"data_path": path}, ctx)
            )
            assert result["success"] is True
            assert result["rows"] == 2
            assert result["columns"] == 2
            assert "name" in result["headers"]
        finally:
            os.unlink(path)

    def test_data_analysis_json(self):
        runner = ToolRunner()
        ctx = ToolExecutionContext(session_id="s1", user_id="u1", timestamp=time.time())
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('[{"key": "value"}, {"key": "value2"}]')
            path = f.name
        try:
            result = asyncio.get_event_loop().run_until_complete(
                runner.execute("data_analysis", {"data_path": path}, ctx)
            )
            assert result["success"] is True
            assert result["items"] == 2
        finally:
            os.unlink(path)

    def test_data_analysis_jsonl(self):
        runner = ToolRunner()
        ctx = ToolExecutionContext(session_id="s1", user_id="u1", timestamp=time.time())
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"a": 1}\n{"b": 2}\n')
            path = f.name
        try:
            result = asyncio.get_event_loop().run_until_complete(
                runner.execute("data_analysis", {"data_path": path}, ctx)
            )
            assert result["success"] is True
            assert result["items"] == 2
        finally:
            os.unlink(path)

    def test_data_analysis_json_object(self):
        runner = ToolRunner()
        ctx = ToolExecutionContext(session_id="s1", user_id="u1", timestamp=time.time())
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"name": "test", "version": 1}')
            path = f.name
        try:
            result = asyncio.get_event_loop().run_until_complete(
                runner.execute("data_analysis", {"data_path": path}, ctx)
            )
            assert result["success"] is True
            assert "keys" in result
        finally:
            os.unlink(path)

    def test_data_analysis_unknown_extension(self):
        runner = ToolRunner()
        ctx = ToolExecutionContext(session_id="s1", user_id="u1", timestamp=time.time())
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            f.write("line1\nline2\n")
            path = f.name
        try:
            result = asyncio.get_event_loop().run_until_complete(
                runner.execute("data_analysis", {"data_path": path}, ctx)
            )
            assert result["success"] is True
            assert "lines" in result
        finally:
            os.unlink(path)


class TestAgentExecute:
    def test_execute_basic(self):
        agent = Agent()
        result = asyncio.get_event_loop().run_until_complete(
            agent.execute("hello", "session1", "user1")
        )
        assert "response" in result
        assert result["session_id"] == "session1"
        assert isinstance(result["tools_used"], list)

    def test_execute_with_code(self):
        agent = Agent()
        result = asyncio.get_event_loop().run_until_complete(
            agent.execute("run this code ```python\nprint(1)\n```", "s1", "u1")
        )
        assert result["response"] != ""

    def test_execute_creates_session_context(self):
        agent = Agent()
        result = asyncio.get_event_loop().run_until_complete(
            agent.execute("hello", "s1", "u1")
        )
        assert result["session_id"] == "s1"

    def test_execute_inference_fn_fallback(self):
        agent = Agent()
        agent.set_inference_fn(lambda x: {"text": "AI response"})
        result = asyncio.get_event_loop().run_until_complete(
            agent.execute("tell me something", "s1", "u1")
        )
        # No tool match for "tell me something" → uses inference
        assert "AI response" in result["response"]

    def test_execute_inference_fn_string_return(self):
        agent = Agent()
        agent.set_inference_fn(lambda x: "string response")
        result = asyncio.get_event_loop().run_until_complete(
            agent.execute("tell me something", "s1", "u1")
        )
        assert "string response" in result["response"]

    def test_execute_inference_fn_exception(self):
        agent = Agent()
        agent.set_inference_fn(lambda x: 1 / 0)
        result = asyncio.get_event_loop().run_until_complete(
            agent.execute("tell me something", "s1", "u1")
        )
        # Should fall back to compose_response
        assert isinstance(result["response"], str)

    def test_execute_inference_fn_non_dict_non_str(self):
        agent = Agent()
        agent.set_inference_fn(lambda x: 42)
        result = asyncio.get_event_loop().run_until_complete(
            agent.execute("tell me something", "s1", "u1")
        )
        assert "42" in result["response"]


class TestAgentGenerateResponse:
    def test_generate_response_no_instructions(self):
        agent = Agent()
        agent.set_inference_fn(lambda x: {"text": "reply"})
        ctx = ToolExecutionContext(session_id="s1", user_id="u1", timestamp=time.time())
        response = agent._generate_response("test", ctx)
        assert response == "reply"

    def test_generate_response_with_instructions(self):
        agent = Agent(AgentConfig(instructions="Be helpful"))
        agent.set_inference_fn(lambda x: {"text": "helpful reply"})
        ctx = ToolExecutionContext(session_id="s1", user_id="u1", timestamp=time.time())
        response = agent._generate_response("test", ctx)
        assert response == "helpful reply"

    def test_generate_response_inference_exception(self):
        agent = Agent()
        agent.set_inference_fn(lambda x: (_ for _ in ()).throw(RuntimeError("fail")))
        ctx = ToolExecutionContext(session_id="s1", user_id="u1", timestamp=time.time())
        response = agent._generate_response("test", ctx)
        assert response == "No results"


class TestAgentGetSession:
    def test_get_session_creates_context(self):
        agent = Agent()
        ctx = agent._get_session("sess1", "user1")
        assert isinstance(ctx, ToolExecutionContext)
        assert ctx.user_id == "user1"
        assert "sess1" in ctx.session_id
        assert ctx.timestamp > 0

    def test_get_session_timestamp_is_float(self):
        agent = Agent()
        ctx = agent._get_session("s", "u")
        assert isinstance(ctx.timestamp, float)

    def test_get_session_empty_ids(self):
        agent = Agent()
        ctx = agent._get_session("", "")
        assert ctx.session_id == ":"
        assert ctx.user_id == ""


class TestAgentComposeResponseExtended:
    def test_compose_with_web_results(self):
        agent = Agent()
        results = [
            {"tool": "web_search", "result": {"success": True, "results": ["r1", "r2"], "count": 2}},
        ]
        response = agent._compose_response("query", results)
        assert "Found 2 results" in response

    def test_compose_with_citations(self):
        agent = Agent()
        results = [
            {"tool": "citation", "result": {"success": True, "citations": ["c1"], "count": 1}},
        ]
        response = agent._compose_response("query", results)
        assert "Generated 1 citations" in response

    def test_compose_mixed_success_and_error(self):
        agent = Agent()
        results = [
            {"tool": "code_execution", "result": {"success": True, "stdout": "ok"}},
            {"tool": "file_search", "result": {"success": False, "error": "not found"}},
        ]
        response = agent._compose_response("query", results)
        assert "ok" in response
        assert "Error: not found" in response

    def test_compose_empty_results(self):
        agent = Agent()
        response = agent._compose_response("query", [])
        assert response == "No results"

    def test_compose_result_without_success_key(self):
        agent = Agent()
        results = [{"tool": "test", "result": {"error": "something"}}]
        response = agent._compose_response("query", results)
        assert "Error: something" in response


class TestAgentPlanKeywordsExtended:
    def test_plan_code_with_execute_keyword(self):
        agent = Agent()
        plan = agent._plan_with_keywords("execute this code ```python\nprint(1)\n```")
        assert any(t[0] == "code_execution" for t in plan)

    def test_plan_code_with_run_keyword(self):
        agent = Agent()
        plan = agent._plan_with_keywords("run this ```python\nprint(1)\n```")
        assert any(t[0] == "code_execution" for t in plan)

    def test_plan_search_with_find_keyword(self):
        agent = Agent()
        plan = agent._plan_with_keywords('find "test query"')
        assert any(t[0] == "file_search" for t in plan)

    def test_plan_web_with_online_keyword(self):
        agent = Agent()
        plan = agent._plan_with_keywords('online "python docs"')
        assert any(t[0] == "web_search" for t in plan)

    def test_plan_web_with_internet_keyword(self):
        agent = Agent()
        plan = agent._plan_with_keywords('internet "python docs"')
        assert any(t[0] == "web_search" for t in plan)

    def test_plan_read_with_open_keyword(self):
        agent = Agent()
        plan = agent._plan_with_keywords('open "myfile.txt"')
        assert any(t[0] == "file_read" for t in plan)

    def test_plan_read_with_show_keyword(self):
        agent = Agent()
        plan = agent._plan_with_keywords('show "myfile.txt"')
        assert any(t[0] == "file_read" for t in plan)

    def test_plan_knowledge_with_recall(self):
        agent = Agent()
        plan = agent._plan_with_keywords("recall deep learning basics")
        assert any(t[0] == "knowledge_retrieval" for t in plan)

    def test_plan_data_with_analyze(self):
        agent = Agent()
        plan = agent._plan_with_keywords("analyze /tmp/data.csv")
        assert any(t[0] == "data_analysis" for t in plan)

    def test_plan_data_with_stats(self):
        agent = Agent()
        plan = agent._plan_with_keywords("stats /tmp/data.csv")
        assert any(t[0] == "data_analysis" for t in plan)

    def test_plan_image_with_photo(self):
        agent = Agent()
        plan = agent._plan_with_keywords("photo /tmp/image.jpg")
        assert any(t[0] == "image_analysis" for t in plan)

    def test_plan_image_with_picture(self):
        agent = Agent()
        plan = agent._plan_with_keywords("picture /tmp/image.jpg")
        assert any(t[0] == "image_analysis" for t in plan)

    def test_plan_multiple_tools(self):
        agent = Agent()
        plan = agent._plan_with_keywords('search "test" and read "file.txt"')
        tools = [t[0] for t in plan]
        assert "file_search" in tools
        assert "file_read" in tools

    def test_plan_code_without_fences(self):
        agent = Agent()
        plan = agent._plan_with_keywords("execute the code")
        # No code fences → no code_execution tool
        assert not any(t[0] == "code_execution" for t in plan)

    def test_plan_citation_with_source(self):
        agent = Agent()
        plan = agent._plan_with_keywords("cite source research paper")
        assert any(t[0] == "citation" for t in plan)


class TestAgentPlanLLMExtended:
    def test_plan_llm_malformed_json(self):
        agent = Agent()
        agent.set_inference_fn(lambda x: {"text": "not json at all"})
        plan = asyncio.get_event_loop().run_until_complete(
            agent._plan_with_llm("do something")
        )
        assert isinstance(plan, list)

    def test_plan_llm_json_with_extra_text(self):
        agent = Agent()
        agent.set_inference_fn(lambda x: {"text": 'Here is the plan:\n[{"tool": "code_execution", "args": {"code": "print(1)"}}]\nDone.'})
        plan = asyncio.get_event_loop().run_until_complete(
            agent._plan_with_llm("run code")
        )
        assert len(plan) == 1
        assert plan[0][0] == "code_execution"

    def test_plan_llm_items_without_tool_key(self):
        agent = Agent()
        agent.set_inference_fn(lambda x: {"text": '[{"name": "code_execution"}]'})
        plan = asyncio.get_event_loop().run_until_complete(
            agent._plan_with_llm("run code")
        )
        # Items without "tool" key should be filtered out
        assert len(plan) == 0

    def test_plan_llm_non_dict_items(self):
        agent = Agent()
        agent.set_inference_fn(lambda x: {"text": '["code_execution", "file_read"]'})
        plan = asyncio.get_event_loop().run_until_complete(
            agent._plan_with_llm("do things")
        )
        assert len(plan) == 0

    def test_plan_llm_inference_returns_string(self):
        agent = Agent()
        agent.set_inference_fn(lambda x: '{"text": "[]"}')
        plan = asyncio.get_event_loop().run_until_complete(
            agent._plan_with_llm("hello")
        )
        assert plan == []


class TestAgentConfigExtended:
    def test_custom_all_fields(self):
        sc = SecurityConfig(max_execution_time=60)
        cfg = AgentConfig(
            tools=[ToolCapability.FILE_SEARCH],
            security=sc,
            max_iterations=5,
            timeout=30,
            instructions="Custom instructions",
        )
        assert cfg.tools == [ToolCapability.FILE_SEARCH]
        assert cfg.security.max_execution_time == 60
        assert cfg.max_iterations == 5
        assert cfg.timeout == 30
        assert cfg.instructions == "Custom instructions"

    def test_default_timeout(self):
        cfg = AgentConfig()
        assert cfg.timeout == 120

    def test_default_max_iterations(self):
        cfg = AgentConfig()
        assert cfg.max_iterations == 10

    def test_empty_tools_list(self):
        cfg = AgentConfig(tools=[])
        assert len(cfg.tools) == 0


class TestSecurityBoundaryExtended:
    def test_blocked_patterns_count(self):
        sb = SecurityBoundary()
        assert len(sb._blocked_re) == len(sb.BLOCKED_PATTERNS)

    def test_all_blocked_patterns_compiled(self):
        sb = SecurityBoundary()
        for p in sb._blocked_re:
            assert hasattr(p, "search")
            assert hasattr(p, "pattern")

    def test_is_allowed_returns_tuple(self):
        sb = SecurityBoundary()
        result = sb.is_allowed("x = 1")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_resource_limit_yields(self):
        sb = SecurityBoundary()
        with sb.resource_limit("test"):
            x = 1 + 1
        assert x == 2

    def test_custom_config_direct(self):
        cfg = SecurityConfig(allow_network=True, max_file_size_mb=200)
        sb = SecurityBoundary(cfg)
        assert sb.config.allow_network is True
        assert sb.config.max_file_size_mb == 200
