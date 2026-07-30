"""
Command registry — auto-discovers commands/ submodules exporting a `run` function.

Protocol:
    def run(argv: list[str], out: Console, api: ShellCommands,
            env: dict[str, str]) -> int: ...

Each module may also export:
    help: str   — one-line description shown in `help` output
    names: list[str]  — command names this module handles (default: [module_name])

Loading-state standard:
    Every command that calls the HTTP API MUST wrap the call with ``out.spinner()``.
    This is a codebase standard — no API call should execute without a visible
    loading indicator. Use the pattern::

        with out.spinner("Fetching models") as s:
            result = api.models()
        s.ok("Models loaded")

    Use ``s.ok()`` for success and ``s.fail()`` for errors. Fast sub-second
    calls still get a spinner — it provides visual confirmation that work
    happened and prevents the "did it hang?" uncertainty.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any, Callable

from ..console import Console
from ..commands import ShellCommands

CommandFunc = Callable[
    [list[str], Console, ShellCommands, dict[str, str]],
    int,
]


class CmdModule:
    """Lazy-loaded command module."""

    def __init__(self, name: str):
        self._name = name
        self._mod: Any = None

    def _load(self) -> None:
        if self._mod is not None:
            return
        self._mod = importlib.import_module(f"..cmds.{self._name}", __package__)

    @property
    def run(self) -> CommandFunc:
        self._load()
        return self._mod.run

    @property
    def help(self) -> str:
        self._load()
        return getattr(self._mod, "help", "")

    @property
    def loaded(self) -> bool:
        return self._mod is not None


# Static mapping: module_name -> list of command names.
# Kept in sync with each module's `names` variable so that discover()
# can build the command map without importing every module eagerly.
# Unix standard commands (cp, mv, grep, echo, etc.) are NOT included —
# users already have those in their terminal.
_MODULE_NAMES: dict[str, list[str]] = {
    "data_cmds": ["datasets", "checkpoints", "finetuned", "knowledge", "remember", "recall", "tokenizer"],
    "health": ["health"],
    "models_cmd": ["models", "unload", "precision", "quantize", "dequantize"],
    "souls_cmd": ["souls", "switch", "whoami"],
}


def discover() -> dict[str, CmdModule]:
    """Return {command_name: CmdModule} for every module in commands/.

    Modules are NOT imported eagerly — they are loaded on first use via
    CmdModule._load(). The name mapping is resolved from the static
    _MODULE_NAMES dict instead.
    """
    cmd_map: dict[str, CmdModule] = {}
    pkg_path = __path__[0]
    for _, mod_name, _ in pkgutil.iter_modules([pkg_path]):
        if mod_name == "__init__":
            continue
        mod = CmdModule(mod_name)
        names = _MODULE_NAMES.get(mod_name, [mod_name])
        for n in names:
            cmd_map[n] = mod
    return cmd_map
