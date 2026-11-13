"""
Code guard — the ONE blocked-pattern list, enforced on BOTH tool surfaces.

``code_guard.BLOCKED_PATTERNS`` is the single list: agents.py's
``SecurityBoundary`` aliases it (and delegates ``is_allowed``), tools.py's
``run_code`` consults it before spawning a subprocess. This file proves
(a) the two surfaces share ONE list by identity (no drift), (b) model
code that trips a pattern is denied on both, (c) ordinary code runs.

Standalone file (not test_tools.py / test_agents_config.py) because it
must run in CI and locally by default — a guard test deselected by a
``slow`` marker is not a guard.
"""

from __future__ import annotations

import sys

import pytest

from domain.agents._internal import code_guard
from domain.agents._internal.tools import ToolRegistry


@pytest.fixture
def registry():
    return ToolRegistry()


def _approve(_call) -> bool:
    """Test approver — the seam gate refuses approval tools without one."""
    return True


class TestCodeGuardSingleSource:
    """Both surfaces must read the SAME list — identity, not equality."""

    def test_securityboundary_aliases_the_shared_list(self):
        from domain.agents._internal.agents import SecurityBoundary

        # `is` (not ==): a copy would let the two lists drift apart again.
        assert SecurityBoundary.BLOCKED_PATTERNS is code_guard.BLOCKED_PATTERNS

    def test_securityboundary_is_allowed_delegates(self):
        from domain.agents._internal.agents import SecurityBoundary

        # behavior parity through delegation (signature + message intact)
        ok, msg = SecurityBoundary().is_allowed("print('hello')")
        assert ok is True
        assert msg == ""
        ok, msg = SecurityBoundary().is_allowed("eval('1+1')")
        assert ok is False
        assert msg.startswith("Blocked pattern: ")

    def test_instance_blocked_re_is_compiled_view(self):
        from domain.agents._internal.agents import SecurityBoundary

        sb = SecurityBoundary()
        assert len(sb._blocked_re) == len(SecurityBoundary.BLOCKED_PATTERNS)
        assert all(hasattr(p, "search") for p in sb._blocked_re)


class TestCodeGuardPatterns:
    @pytest.mark.parametrize(
        "code",
        [
            "eval('1+1')",
            "exec('x = 1')",
            "import pickle; pickle.load(f)",
            "import marshal; marshal.loads(d)",
            "compile(code, '<s>', 'exec')",
            "import os\nos.remove('file')",
            "import os; os.remove('file.txt')",
            "import shutil\nshutil.rmtree('/tmp/dir')",
            "subprocess.call('ls', shell=True)",
            "__import__('os')",
            "open('file', r/w)",
        ],
    )
    def test_blocked(self, code):
        assert code_guard.is_allowed(code)[0] is False

    @pytest.mark.parametrize(
        "code",
        [
            "print('hello')",
            "x = 2 + 3",
            "open('file.txt', 'r')",
            "import os; print(os.listdir('.'))",
            "",
            "x = 1\ny = 2\nz = x + y",
        ],
    )
    def test_allowed(self, code):
        assert code_guard.is_allowed(code)[0] is True


class TestToolRegistryRunCodeGuard:
    """tools.py run_code must deny before spawning a subprocess."""

    @pytest.mark.asyncio
    async def test_eval_denied(self, registry):
        result = await registry.execute(
            "run_code", {"language": "python", "code": "eval('1+1')"}, approve=_approve
        )
        assert result.success is False
        assert "Blocked pattern" in result.error

    @pytest.mark.asyncio
    async def test_pickle_denied(self, registry):
        result = await registry.execute(
            "run_code", {"language": "python", "code": "import pickle"}, approve=_approve
        )
        assert result.success is False
        assert "Blocked pattern" in result.error

    @pytest.mark.asyncio
    async def test_os_remove_denied(self, registry):
        result = await registry.execute(
            "run_code",
            {"language": "python", "code": "import os; os.remove('x')"},
            approve=_approve,
        )
        assert result.success is False
        assert "Blocked pattern" in result.error

    @pytest.mark.asyncio
    async def test_denied_before_subprocess_no_output(self, registry):
        # a blocked snippet must not run at all: no stdout, just the refusal
        result = await registry.execute(
            "run_code",
            {"language": "python", "code": "import pickle; print('RAN')"},
            approve=_approve,
        )
        assert result.success is False
        assert "RAN" not in (result.output or "")

    @pytest.mark.asyncio
    async def test_ordinary_code_still_runs(self, registry):
        result = await registry.execute(
            "run_code", {"language": "python", "code": "print('hello')"}, approve=_approve
        )
        assert result.success is True
        assert "hello" in result.output

    @pytest.mark.asyncio
    async def test_bash_still_runs(self, registry):
        result = await registry.execute(
            "run_code", {"language": "bash", "code": "echo hi"}, approve=_approve
        )
        assert result.success is True
        assert "hi" in result.output

    @pytest.mark.asyncio
    async def test_python_uses_project_interpreter(self, registry):
        """sys.executable, not a bare `python3` that may miss the env."""
        result = await registry.execute(
            "run_code",
            {"language": "python", "code": "import sys; print(sys.executable)"},
            approve=_approve,
        )
        assert result.success is True
        assert result.output.strip() == sys.executable
