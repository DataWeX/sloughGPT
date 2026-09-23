"""Tools tests — registry, adapter, agent TOOL_CALL, import hygiene."""

import ast
import asyncio
import sys

from avion.ai import Agent, AgentConfig, EchoModel
from avion.ai.models import Action, ActionType, validate_action
from avion.ai.slo import build_prompt
from avion.ai.tools import (
    ToolRegistry,
    ToolSpec,
    make_executor,
    normalize_result,
)
from test_autoclicker import FakeBackend


def run(coro):
    return asyncio.run(coro)


class TestSpecs:
    def test_from_duck_typed_definition(self):
        class FakeDefinition:
            name = "file_search"
            description = "search files"
            parameters = {"query": "text to find"}
            requires_approval = False

        spec = ToolSpec.from_definition(FakeDefinition())
        assert spec.name == "file_search"
        assert spec.params == {"query": "text to find"}

    def test_registry(self):
        reg = ToolRegistry([ToolSpec("b", "second"), ToolSpec("a", "first")])
        assert reg.names == ["a", "b"]
        assert "needs approval" not in reg.describe()
        reg.add(ToolSpec("c", "gated", requires_approval=True))
        assert "[needs approval]" in reg.describe()
        assert reg.remove("c") is True
        assert reg.remove("c") is False
        assert len(reg) == 2

    def test_empty_describe(self):
        assert ToolRegistry().describe() == "(no outside tools)"


class TestNormalize:
    def test_dict_success_shapes(self):
        assert normalize_result({"success": True, "result": "abc"}) == ("abc", True)
        assert normalize_result({"success": True, "output": "o"}) == ("o", True)

    def test_dict_failure(self):
        text, ok = normalize_result({"success": False, "error": "bad"})
        assert ok is False and "bad" in text

    def test_plain_text(self):
        assert normalize_result("hello") == ("hello", True)

    def test_truncated(self):
        text, _ = normalize_result("x" * 5000, max_chars=10)
        assert len(text) == 10


class TestAgentToolCall:
    def _agent(self, actions, registry, executor):
        import tempfile

        cfg = AgentConfig(
            action_delay_ms=0,
            screenshot_on_each_step=False,
            output_dir=tempfile.mkdtemp(prefix="avion-tools-"),
        )
        agent = Agent(
            model=EchoModel(actions), config=cfg, tools=registry, executor=executor
        )
        run(agent.start(backend=FakeBackend()))
        return agent

    def _registry(self):
        return ToolRegistry([
            ToolSpec("file_search", "search files", {"query": "text"}),
            ToolSpec("danger", "gated tool", requires_approval=True),
        ])

    def test_tool_success(self):
        async def fake_runner(tool, args):
            assert tool == "file_search" and args == {"query": "x"}
            return {"success": True, "result": "found it"}

        agent = self._agent(
            [
                Action(ActionType.TOOL_CALL,
                       {"tool": "file_search", "args": {"query": "x"}}),
                Action(ActionType.DONE),
            ],
            self._registry(),
            make_executor(fake_runner),
        )
        result = run(agent.run("search"))
        run(agent.stop())
        assert result.success is True
        assert result.trajectory.steps[0].observation == "found it"

    def test_unknown_tool(self):
        agent = self._agent(
            [Action(ActionType.TOOL_CALL, {"tool": "nope"}), Action(ActionType.DONE)],
            self._registry(),
            make_executor(lambda t, a: "x"),
        )
        result = run(agent.run("t"))
        run(agent.stop())
        assert "Unknown tool" in result.trajectory.steps[0].observation
        assert result.success is True  # run continues to DONE

    def test_approval_gate(self):
        agent = self._agent(
            [Action(ActionType.TOOL_CALL, {"tool": "danger"}), Action(ActionType.DONE)],
            self._registry(),
            make_executor(lambda t, a: "x"),
        )
        result = run(agent.run("t"))
        run(agent.stop())
        assert "needs approval" in result.trajectory.steps[0].observation

    def test_no_executor(self):
        agent = self._agent(
            [Action(ActionType.TOOL_CALL, {"tool": "file_search"}), Action(ActionType.DONE)],
            self._registry(),
            None,
        )
        result = run(agent.run("t"))
        run(agent.stop())
        assert "No executor" in result.trajectory.steps[0].observation

    def test_executor_error(self):
        async def broken(tool, args):
            raise RuntimeError("down")

        agent = self._agent(
            [Action(ActionType.TOOL_CALL, {"tool": "file_search"}), Action(ActionType.DONE)],
            self._registry(),
            make_executor(broken),
        )
        result = run(agent.run("t"))
        run(agent.stop())
        assert "error: down" in result.trajectory.steps[0].observation

    def test_tool_call_validates(self):
        assert "tool" in validate_action(Action(ActionType.TOOL_CALL, {}))
        assert validate_action(
            Action(ActionType.TOOL_CALL, {"tool": "x"})) is None

    def test_prompt_lists_tools(self):
        reg = ToolRegistry([ToolSpec("file_search", "search files")])
        prompt = build_prompt("t", tools=reg)
        assert "file_search" in prompt and "tool_call" in prompt
        assert "Outside tools" in build_prompt("t", tools=reg)
        assert "Outside tools" not in build_prompt("t")


class TestProductRunnerShape:
    def test_runner_style_dict_flows(self):
        """A ToolRunner-shaped (tool, args, ctx) callable adapts in one line."""
        calls = []

        async def runner_execute(tool, args, ctx):
            calls.append((tool, args, ctx))
            return {"success": True, "result": "ok"}

        ctx = object()
        executor = make_executor(lambda t, a: runner_execute(t, a, ctx))
        text, ok = normalize_result(run(executor.run("file_search", {"q": "1"})))
        assert (text, ok) == ("ok", True)
        assert calls[0][2] is ctx


class TestHygiene:
    def test_avion_imports_only_stdlib_and_self(self):
        """Module-level imports must be stdlib + avion only.

        Function-level lazy imports (playwright, PIL, cv2, ...) are
        the sanctioned pattern for optional heavy deps — they stay
        out of module scope so `import avion` never pulls them in.
        """
        import pathlib

        allowed = set(sys.stdlib_module_names) | {"avion", "__future__"}
        offenders = []
        root = pathlib.Path("packages/avion/src/avion")
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text())
            for node in tree.body:
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        top = alias.name.split(".")[0]
                        if top not in allowed:
                            offenders.append(f"{path}:{top}")
                elif isinstance(node, ast.ImportFrom):
                    if node.level == 0 and node.module:
                        top = node.module.split(".")[0]
                        if top not in allowed:
                            offenders.append(f"{path}:{top}")
        assert offenders == [], offenders

    def test_import_avion_pulls_no_heavy_libs(self):
        before = set(sys.modules)
        import importlib

        importlib.import_module("avion")
        added = {m.split(".")[0] for m in sys.modules if m not in before}
        heavy = {"playwright", "selenium", "appium", "pyautogui", "PIL",
                 "cv2", "numpy", "torch", "pytesseract", "websockets",
                 "domain", "apps"}
        assert not (added & heavy), added & heavy
