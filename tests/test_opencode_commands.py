"""Conformance guard for OpenCode command → agent targets.

A command file's ``agent: <name>`` frontmatter is how ``/command`` resolves
to a subagent. When the target does not exist, the command fails at use
time — inside a session, where nobody sees the cause. This repo has already
paid for that class twice: ``/coverage`` pointed at the ``coverage`` agent
after it was merged into ``test-writer`` (2026-09-12), and ``/goal`` has
pointed at a ``build`` agent that has never existed in any root. Neither
was caught, because ``test_opencode_skills.py`` guards skills only — the
"command→agent targets resolve" claim in ``docs/AGENT_SYNC.md`` had no test
behind it.

The rule is mechanical: every command's target must be an agent file on
disk. A command that genuinely needs a built-in (``explore``, ``general``)
must add it to ``BUILTIN_AGENTS`` explicitly, so widening the set is a
reviewed edit rather than a silent loophole.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMANDS_DIR = REPO_ROOT / ".opencode" / "commands"
AGENTS_DIR = REPO_ROOT / ".opencode" / "agents"
GLOBAL_COMMANDS_DIR = Path.home() / ".config" / "opencode" / "commands"

# Built-ins shipped by opencode itself — extend deliberately, never by
# inferring from a failing test.
BUILTIN_AGENTS: set[str] = set()

FRONTMATTER_RE = re.compile(r"\A---\s*\n(?P<body>.*?)\n---\s*$", re.S | re.M)


def command_files() -> list[Path]:
    if not COMMANDS_DIR.is_dir():
        return []
    return sorted(COMMANDS_DIR.glob("*.md"))


def target_of(path: Path) -> str | None:
    """The declared agent target, or None when frontmatter is unreadable."""
    match = FRONTMATTER_RE.match(path.read_text(encoding="utf-8"))
    if not match:
        return None
    data = yaml.safe_load(match.group("body"))
    if not isinstance(data, dict):
        return None
    agent = data.get("agent")
    return agent if isinstance(agent, str) and agent else None


def test_commands_are_actually_discovered():
    """Non-vacuity: an empty scan must fail, not pass every loop below."""
    assert COMMANDS_DIR.is_dir(), f"missing commands dir: {COMMANDS_DIR}"
    assert command_files(), f"no command .md files found under {COMMANDS_DIR}"


@pytest.mark.parametrize("path", command_files(), ids=lambda p: p.stem)
def test_command_has_agent_target(path: Path):
    target = target_of(path)
    assert target, (
        f"{path.relative_to(REPO_ROOT)} must declare a non-empty `agent: <name>` "
        "frontmatter field — a command with no target cannot resolve"
    )


@pytest.mark.parametrize("path", command_files(), ids=lambda p: p.stem)
def test_agent_target_resolves(path: Path):
    target = target_of(path)
    if not target:  # reported by test_command_has_agent_target
        pytest.fail(f"{path.name} has no agent target")
    if target in BUILTIN_AGENTS:
        return
    agent_file = AGENTS_DIR / f"{target}.md"
    assert agent_file.is_file(), (
        f"{path.name} targets agent {target!r} but {agent_file.relative_to(REPO_ROOT)} "
        "does not exist — the command would fail at use time. Retarget it, or "
        "add the agent (a built-in goes in BUILTIN_AGENTS, reviewed)."
    )


@pytest.mark.skipif(
    not GLOBAL_COMMANDS_DIR.is_dir(),
    reason="no global opencode commands on this machine",
)
def test_global_commands_targets_resolve():
    """Same rule for the user-global commands (local guard, CI skips it).

    These load on every session outside the repo, so a dangling target here
    is invisible to repo CI — check it where it lives.
    """
    dangling = [
        f"{p.name} -> {target_of(p)}"
        for p in sorted(GLOBAL_COMMANDS_DIR.glob("*.md"))
        if (t := target_of(p))
        and t not in BUILTIN_AGENTS
        and not (AGENTS_DIR / f"{t}.md").is_file()
    ]
    assert not dangling, "global commands with unresolvable agent targets: " + ", ".join(dangling)
