"""Guard the console-script contract: declared targets must actually resolve.

This exists because they did not. ``pyproject.toml`` declared
``planner = "planner.cli:main"`` and ``kanban = "planner.cli:main"``, but
``planner.cli`` does not exist anywhere in the repo — the importable module is
``app_planner``. Installing produced two console scripts that crashed with
``ModuleNotFoundError`` on invocation, and nothing in the suite noticed,
because nothing ever imported the declared target.

It is the same failure shape as a monkeypatch aimed at a symbol production no
longer calls: the reference went stale, and no test was asserting it. So assert
it.
"""

from __future__ import annotations

import importlib
import re
import sys
import tomllib
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]
SRC = PKG_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

REPO = PKG_ROOT.parents[1]


def _scripts() -> dict[str, str]:
    with (PKG_ROOT / "pyproject.toml").open("rb") as fh:
        return dict(tomllib.load(fh)["project"]["scripts"])


def test_every_declared_console_script_resolves():
    scripts = _scripts()
    assert scripts, "no console scripts declared"

    for command, target in scripts.items():
        module_name, _, attr = target.partition(":")
        assert attr, f"{command}: target {target!r} has no 'module:function' shape"

        module = importlib.import_module(module_name)
        resolved = getattr(module, attr, None)
        assert resolved is not None, (
            f"console script {command!r} points at {target!r}, but "
            f"{module_name!r} has no attribute {attr!r} — installing it would "
            f"produce a binary that crashes on import"
        )
        assert callable(resolved), f"console script {command!r} target {target!r} is not callable"


def test_primary_command_is_planner():
    """One app, one command — `planner` is the name the spec fixes."""
    scripts = _scripts()
    assert "planner" in scripts, "the primary `planner` command is not declared"

    # `app-planner` (README, CLI help) and `kanban` (previously shipped) stay
    # as aliases, but all three must be the SAME program, not three modules.
    assert scripts["planner"] == scripts.get("app-planner")
    assert scripts["planner"] == scripts.get("kanban")


def test_declared_module_is_app_planner_not_planner():
    """The stale target, pinned so it cannot come back.

    `planner.cli` was the target of both scripts and never existed.
    """
    for command, target in _scripts().items():
        assert not target.startswith("planner."), (
            f"console script {command!r} targets {target!r}: module `planner` "
            "is a 2-file stub with no cli.py. Use `app_planner.cli:main`."
        )


def test_distribution_name_does_not_collide_with_the_stub():
    """Both packages once claimed the distribution name `planner`.

    Only one can win an install, so the names must differ.
    """
    with (PKG_ROOT / "pyproject.toml").open("rb") as fh:
        dist_name = tomllib.load(fh)["project"]["name"]

    stub = REPO / "packages" / "planner" / "setup.py"
    if stub.exists():
        match = re.search(r'name\s*=\s*["\']([^"\']+)["\']', stub.read_text())
        assert match, f"could not read the distribution name from {stub}"
        assert dist_name != match.group(1), (
            f"both {PKG_ROOT.name} and packages/planner declare the "
            f"distribution name {dist_name!r} — they collide on install"
        )


def test_spec_command_shape_is_supported():
    """`.opencode/plans/planner-notes-specs.md` fixes the shape as
    `planner notes | kanban | sync | gui`. All four must dispatch."""
    from app_planner.cli import build_parser

    parsed = build_parser().parse_args
    top = build_parser()._subparsers._group_actions[0].choices
    assert {"note", "notes", "board", "kanban", "sync", "gui"} <= set(top)

    # the spec writes the group names as `notes` / `kanban`
    parsed(["notes", "new", "x"])
    parsed(["kanban", "stats"])
    parsed(["sync"])
