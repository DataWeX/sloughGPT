"""
Approval gate at the ToolRegistry.execute seam — fail-closed.

``requires_approval`` is declared once (on ToolSpec) but was enforced only
where the caller remembered: loop.py checked, multi.py:453/472 called
``registry.execute()`` directly, so researcher's web_search ran unattended
and a custom agent with ``tools: ["run_code"]`` could execute model code
with neither approval nor a human in the loop. The gate now lives at the
SEAM — the spec already carries the flag, so one check covers every current
and future caller (no second declaration to drift).

Contract pinned here:
  * no approver passed -> refusal BEFORE dispatch (no subprocess, no network)
  * approver passed    -> run proceeds (loop.py prompts upstream, then
    passes its approver through; the seam checks PRESENCE, it must not
    re-call the approver — that would double-prompt the human)
  * multi.py's direct callers get the refusal for free — the bypass card's
    money test: with the gate absent, the code below would RUN and multi
    would return its stdout instead of None

Standalone always-run file: test_tools.py is module-marked ``slow`` and a
gate test deselected by a marker is not a gate.
"""

from __future__ import annotations

import pytest

from domain.agents._internal.multi import MultiAgentOrchestrator, SpecializedAgent
from domain.agents._internal.tools import ToolRegistry


@pytest.fixture
def registry():
    return ToolRegistry()


def _approve(_call) -> bool:
    """Test approver — declares upstream approval for requires_approval tools."""
    return True


class TestSeamGate:
    @pytest.mark.asyncio
    async def test_non_approval_tool_runs_without_approver(self, registry):
        # calculator carries no flag — default surface unchanged
        result = await registry.execute("calculator", {"expression": "2+2"})
        assert result.success is True
        assert "4" in result.output

    @pytest.mark.asyncio
    async def test_run_code_refused_without_approver(self, registry):
        result = await registry.execute(
            "run_code", {"language": "python", "code": "print('GATE_MUST_NOT_RUN')"}
        )
        assert result.success is False
        assert "requires approval (no approver supplied)" in (result.error or "")
        # refused BEFORE dispatch: the code never ran, no stdout leaked
        assert "GATE_MUST_NOT_RUN" not in (result.output or "")

    @pytest.mark.asyncio
    async def test_web_search_refused_without_approver(self, registry):
        # web_search also carries requires_approval=True — refusal happens
        # before _run_web_search, so no network call is attempted here
        result = await registry.execute("web_search", {"query": "q"})
        assert result.success is False
        assert "requires approval (no approver supplied)" in (result.error or "")

    @pytest.mark.asyncio
    async def test_refusal_is_exact_loop_message(self, registry):
        # byte-identical to loop.py's own gate message — one semantics,
        # two lines of defense (loop prompts, seam enforces)
        result = await registry.execute("run_code", {"language": "python", "code": "1"})
        assert result.error == "tool 'run_code' requires approval (no approver supplied)"

    @pytest.mark.asyncio
    async def test_run_code_runs_with_approver(self, registry):
        result = await registry.execute(
            "run_code",
            {"language": "python", "code": "print('APPROVED_RAN')"},
            approve=_approve,
        )
        assert result.success is True
        assert "APPROVED_RAN" in result.output

    @pytest.mark.asyncio
    async def test_web_search_with_approver_passes_the_gate(self, registry):
        # presence of approve is what the seam checks — dispatch follows;
        # result may fail on network, but it must NOT be an approval refusal
        result = await registry.execute(
            "web_search", {"query": "q", "num_results": 1}, approve=_approve
        )
        assert "requires approval" not in (result.error or "")

    @pytest.mark.asyncio
    async def test_seam_checks_presence_not_recalling_approver(self, registry):
        # loop.py already called the approver upstream; the seam must not
        # call it again (double-prompt) — presence declares "approved"
        def boom(_call):
            raise AssertionError("seam must not re-call the approver")

        result = await registry.execute(
            "run_code",
            {"language": "python", "code": "print('ok')"},
            approve=boom,
        )
        assert result.success is True

    @pytest.mark.asyncio
    async def test_approval_gate_runs_before_pattern_guard(self, registry):
        # layer order at the seam: approval refusal (generic, spec-level)
        # fires before the tool's own blocked-pattern check (code_guard) —
        # an unapproved call never reaches the code that would run it
        result = await registry.execute("run_code", {"language": "python", "code": "eval('1')"})
        assert result.success is False
        assert "requires approval" in (result.error or "")
        assert "Blocked pattern" not in (result.error or "")


class TestMultiBypassClosed:
    """multi.py's direct registry.execute() callers are now gated too."""

    @staticmethod
    def _orchestrator() -> MultiAgentOrchestrator:
        coder = SpecializedAgent(
            name="coder",
            role="writes code",
            system_prompt="you write code",
            tools=["run_code"],  # custom-agent config can name ANY tool
        )
        return MultiAgentOrchestrator(agents={"coder": coder})

    @pytest.mark.asyncio
    async def test_async_multi_cannot_execute_run_code_without_approver(self, registry):
        orch = self._orchestrator()
        fenced = "```python\nprint('BYPASS_RAN')\n```"
        result = await orch._async_try_execute_tool(fenced, ["run_code"])
        # with the seam gate absent this returns "BYPASS_RAN\n" (code ran);
        # gated, execute() refuses -> success=False -> None
        assert result is None

    def test_sync_multi_cannot_execute_run_code_without_approver(self):
        orch = self._orchestrator()
        fenced = "```python\nprint('BYPASS_RAN')\n```"
        result = orch._try_execute_tool(fenced, ["run_code"])
        assert result is None

    @pytest.mark.asyncio
    async def test_multi_allowlist_still_blocks_unknown_tools(self, registry):
        # the allowlist gate stays (defense in depth): a tool the agent
        # does not carry is skipped before the seam is even reached
        orch = self._orchestrator()
        fenced = "```python\nprint('NOPE')\n```"
        result = await orch._async_try_execute_tool(fenced, ["memory"])
        assert result is None
