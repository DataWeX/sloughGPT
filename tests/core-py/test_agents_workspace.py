"""Tests for the agent workspace build capability (card 20261128_054926).

The failure pinned here: "Build a web" on the Agents page answered with a
step-by-step tutorial because (a) the planner had no build intent and (b) no
tool could write a file. These tests assert the opposite end state — the run
writes a project, reports its manifest, and never asks the model to narrate.
"""

from __future__ import annotations

import pytest

from domain.agents._internal import workspace
from domain.agents._internal.agents import (
    Agent,
    AgentConfig,
    ToolCapability,
    ToolExecutionContext,
    ToolRunner,
    _build_plan,
    _build_template,
    _compose_artifact_report,
)
from domain.agents._internal.tools import ToolRegistry


@pytest.fixture
def projects(tmp_path, monkeypatch):
    root = tmp_path / "agent-projects"
    monkeypatch.setattr(workspace, "PROJECTS_ROOT", str(root))
    return root


def _ctx() -> ToolExecutionContext:
    return ToolExecutionContext(session_id="s", user_id="u", timestamp=0.0)


# ── Templates ─────────────────────────────────────────────────────────────


class TestScaffold:
    def test_static_writes_a_runnable_site(self, projects):
        result = workspace.scaffold(template="static", name="portfolio", title="My Portfolio")

        assert result["artifact"] is True
        assert result["template"] == "static"
        assert result["files"] == ["index.html", "script.js", "styles.css"]
        assert (projects / "portfolio" / "index.html").exists()
        assert (projects / "portfolio" / "styles.css").exists()
        assert (projects / "portfolio" / "script.js").exists()

        index = (projects / "portfolio" / "index.html").read_text()
        assert "<title>My Portfolio</title>" in index
        assert 'href="styles.css"' in index
        assert 'src="script.js"' in index

    def test_manifest_lists_every_written_file(self, projects):
        result = workspace.scaffold(template="static", name="site")

        manifest_paths = [entry["path"] for entry in result["manifest"]]
        assert manifest_paths == ["index.html", "script.js", "styles.css"]
        assert all(entry["bytes"] > 0 for entry in result["manifest"])

    def test_vite_react_scaffold(self, projects):
        result = workspace.scaffold(template="vite-react", name="dashboard", title="Dashboard")

        assert "package.json" in result["files"]
        assert "src/App.jsx" in result["files"]
        package = (projects / "dashboard" / "package.json").read_text()
        assert '"name": "dashboard"' in package
        assert "vite" in package

    def test_fastapi_scaffold(self, projects):
        result = workspace.scaffold(template="fastapi", name="api", title="Orders API")

        assert result["files"] == ["README.md", "app.py", "requirements.txt"]
        app_py = (projects / "api" / "app.py").read_text()
        assert "from fastapi import FastAPI" in app_py
        assert "Orders API" in app_py

    def test_custom_sections_land_in_the_page(self, projects):
        workspace.scaffold(template="static", name="docs", title="Docs", sections="Pricing, FAQ")

        index = (projects / "docs" / "index.html").read_text()
        assert "<h2>Pricing</h2>" in index
        assert "<h2>FAQ</h2>" in index

    def test_title_is_html_escaped(self, projects):
        workspace.scaffold(template="static", name="xss", title="<script>alert(1)</script>")

        index = (projects / "xss" / "index.html").read_text()
        assert "<script>alert(1)</script>" not in index
        assert "&lt;script&gt;" in index

    def test_unknown_template_is_an_error(self, projects):
        with pytest.raises(workspace.WorkspaceError, match="unknown template"):
            workspace.scaffold(template="rails", name="blog")

    @pytest.mark.parametrize(
        ("name", "slug"),
        [("../escape", "escape"), ("UPPER", "upper"), ("", "site"), ("with space", "with-space")],
    )
    def test_names_are_slugified_and_contained(self, projects, name, slug):
        result = workspace.scaffold(template="static", name=name)

        assert result["name"] == slug
        assert (projects / slug).is_dir()
        assert str(projects.resolve()) in result["root"]

    def test_second_build_does_not_overwrite_the_first(self, projects):
        first = workspace.scaffold(template="static", name="web")
        second = workspace.scaffold(template="static", name="web")

        assert first["name"] == "web"
        assert second["name"] == "web-2"
        assert (projects / "web" / "index.html").exists()
        assert (projects / "web-2" / "index.html").exists()


# ── Writes ────────────────────────────────────────────────────────────────


class TestWrite:
    def test_writes_nested_file(self, projects):
        workspace.scaffold(template="vite-react", name="app")

        result = workspace.write_file(
            project="app", path="src/components/Hero.jsx", content="export default () => null;"
        )

        assert result["artifact"] is True
        assert result["path"] == "src/components/Hero.jsx"
        assert (projects / "app" / "src" / "components" / "Hero.jsx").exists()

    @pytest.mark.parametrize("path", ["../escape.txt", "src/../../escape.txt", "/etc/passwd"])
    def test_escape_attempts_are_refused(self, projects, path):
        workspace.scaffold(template="static", name="app")

        with pytest.raises(workspace.WorkspaceError):
            workspace.write_file(project="app", path=path, content="x")

    def test_missing_project_is_an_error(self, projects):
        with pytest.raises(workspace.WorkspaceError, match="does not exist"):
            workspace.write_file(project="ghost", path="index.html", content="x")

    def test_directory_path_is_an_error(self, projects):
        workspace.scaffold(template="vite-react", name="app")

        with pytest.raises(workspace.WorkspaceError, match="must name a file"):
            workspace.write_file(project="app", path="src", content="x")

    def test_trailing_slash_is_an_error(self, projects):
        workspace.scaffold(template="static", name="app")

        with pytest.raises(workspace.WorkspaceError, match="must name a file"):
            workspace.write_file(project="app", path="nested/", content="x")


# ── Build intent ──────────────────────────────────────────────────────────


class TestBuildIntent:
    @pytest.mark.parametrize(
        ("prompt", "template"),
        [
            ("Build a web", "static"),
            ("Build a web app", "static"),
            ("create a landing page", "static"),
            ("Build a React app", "vite-react"),
            ("scaffold a vite frontend", "vite-react"),
            ("Build an API", "fastapi"),
            ("create a backend service", "fastapi"),
        ],
    )
    def test_build_requests_scaffold(self, prompt, template):
        plan = _build_plan(prompt)

        assert len(plan) == 1
        tool, args = plan[0]
        assert tool == ToolCapability.WORKSPACE_SCAFFOLD.value
        assert args["template"] == template
        assert args["name"]
        assert args["title"]

    @pytest.mark.parametrize(
        "prompt",
        [
            "Summarize this document",
            "Search the web for rust ownership",
            "Make the app faster",
            "Write a blog post about caching",
            "Run the tests",
        ],
    )
    def test_non_build_requests_do_not_scaffold(self, prompt):
        assert _build_plan(prompt) == []

    def test_template_choice_prefers_frontend_then_backend(self):
        assert _build_template("build a react api dashboard") == "vite-react"
        assert _build_template("build an api") == "fastapi"
        assert _build_template("build a site") == "static"

    def test_agent_without_build_capability_is_not_granted_one(self):
        agent = Agent(config=AgentConfig(tools=[]))

        assert agent._plan_with_keywords("Build a web") == []

    def test_code_execution_implies_build(self):
        agent = Agent(config=AgentConfig(tools=[ToolCapability.CODE_EXECUTION]))

        plan = agent._plan_with_keywords("Build a web")
        assert plan and plan[0][0] == ToolCapability.WORKSPACE_SCAFFOLD.value


# ── Deterministic report ──────────────────────────────────────────────────


class TestArtifactReport:
    def test_report_joins_artifact_outputs(self):
        results = [
            {"tool": "workspace_scaffold", "result": {"artifact": True, "output": "Built X"}},
            {"tool": "workspace_write", "result": {"artifact": True, "output": "Wrote y"}},
            {"tool": "web_search", "result": {"output": "not an artifact"}},
        ]

        assert _compose_artifact_report(results) == "Built X\nWrote y"

    def test_no_artifacts_means_no_report(self):
        assert _compose_artifact_report([{"tool": "web_search", "result": {"output": "x"}}]) == ""


# ── Runner + registry projections ─────────────────────────────────────────


class TestToolRunner:
    async def test_runner_scaffolds(self, projects):
        runner = ToolRunner()

        result = await runner.execute(
            "workspace_scaffold", {"name": "portfolio", "template": "static"}, _ctx()
        )

        assert result["success"] is True
        assert (projects / "portfolio" / "index.html").exists()

    async def test_runner_reports_workspace_errors(self, projects):
        runner = ToolRunner()

        result = await runner.execute(
            "workspace_write", {"project": "../bad", "path": "x", "content": "x"}, _ctx()
        )

        assert result["success"] is False
        assert "invalid project name" in result["error"]

    async def test_runner_writes(self, projects):
        runner = ToolRunner()
        await runner.execute("workspace_scaffold", {"name": "app", "template": "static"}, _ctx())

        result = await runner.execute(
            "workspace_write",
            {"project": "app", "path": "extra.html", "content": "<p>x</p>"},
            _ctx(),
        )

        assert result["success"] is True
        assert (projects / "app" / "extra.html").read_text() == "<p>x</p>"


class TestToolRegistry:
    def test_build_tools_are_registered_with_schema(self):
        registry = ToolRegistry()

        scaffold = registry.get("workspace_scaffold")
        assert scaffold is not None
        assert scaffold.params["required"] == ["name", "template"]
        assert registry.get("workspace_write") is not None

    async def test_registry_executes_scaffold(self, projects):
        registry = ToolRegistry()

        result = await registry.execute(
            "workspace_scaffold", {"name": "site", "template": "static"}
        )

        assert result.success is True
        assert (projects / "site" / "index.html").exists()

    async def test_registry_surfaces_workspace_error(self, projects):
        registry = ToolRegistry()

        result = await registry.execute("workspace_scaffold", {"name": "site", "template": "rails"})

        assert result.success is False
        assert "unknown template" in (result.error or "")


# ── End to end: the screenshot case ───────────────────────────────────────


class TestBuildEndToEnd:
    async def test_build_a_web_writes_files_instead_of_tutorial(self, projects, monkeypatch):
        monkeypatch.setattr(
            "domain.agents._internal.run_history.get_agent_run_store",
            lambda: (_ for _ in ()).throw(RuntimeError("no run store in tests")),
        )
        agent = Agent(
            config=AgentConfig(tools=[ToolCapability.CODE_EXECUTION]),
            # The 0.5B planner returns nothing usable — exactly the live case.
            inference_fn=lambda prompt, max_tokens=200: {"text": ""},
        )

        result = await agent.execute("Build a web", "session-1")

        assert "Built" in result["response"]
        assert "index.html" in result["response"]
        assert result["tools_used"][0]["tool"] == "workspace_scaffold"
        assert (projects / "web" / "index.html").exists()

    async def test_build_intent_outranks_the_model_plan(self, projects, monkeypatch):
        monkeypatch.setattr(
            "domain.agents._internal.run_history.get_agent_run_store",
            lambda: (_ for _ in ()).throw(RuntimeError("no run store in tests")),
        )
        calls: list[str] = []

        def inference(prompt: str, max_tokens: int = 200) -> dict:
            calls.append(prompt)
            return {"text": '[{"tool": "web_search", "args": {"query": "web"}}]'}

        agent = Agent(
            config=AgentConfig(tools=[ToolCapability.CODE_EXECUTION]),
            inference_fn=inference,
        )

        result = await agent.execute("Build a web", "session-3")

        assert result["tools_used"][0]["tool"] == "workspace_scaffold"
        assert calls == []  # neither the planner nor synthesis touched the model

    async def test_plain_question_still_uses_the_model(self, projects, monkeypatch):
        monkeypatch.setattr(
            "domain.agents._internal.run_history.get_agent_run_store",
            lambda: (_ for _ in ()).throw(RuntimeError("no run store in tests")),
        )
        agent = Agent(
            config=AgentConfig(tools=[ToolCapability.CODE_EXECUTION]),
            inference_fn=lambda prompt, max_tokens=200: {"text": "Four."},
        )

        result = await agent.execute("What is 2+2?", "session-2")

        assert result["response"] == "Four."
        assert result["tools_used"] == []
        assert not projects.exists() or not any(projects.iterdir())
