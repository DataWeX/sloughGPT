"""The three RetroBIOS option views — registries, not drawings.

Card ``20261026_092427``.  A view is a list of row specs in exactly the
shape ``options_screen.OptionRow.from_spec`` consumes::

    {label, value, flag, setter}

The screen draws the row, ``argv()`` spells the flag, ``apply()`` pushes
the value through the setter — one descriptor, three projections (menu
row, command line, help) that therefore cannot drift.  For the command
views the flags are additionally pinned against the command's own
docstring *and* its parser by ``test_option_views.py``, which is what
keeps a registry honest as the commands evolve.

Pure data: zero terminal imports.  Rendering happens in
``console.run_options_screen`` — the raw loop on a TTY, plain lines
everywhere else.
"""

from __future__ import annotations

import os

from .config import get_api_base

AUTOLOAD_DEFAULT = "Qwen/Qwen2.5-0.5B-Instruct"

# ── session prefs ────────────────────────────────────────────────────
#
# Small key/value store hanging off the REPL, so a preference is real
# state shared by every view that shows it (settings' `--api` row and
# boot's `--api` row are the same pref — they cannot disagree).


def _prefs(repl) -> dict:
    prefs = getattr(repl, "_option_prefs", None)
    if prefs is None:
        prefs = {}
        repl._option_prefs = prefs
    return prefs


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in ("on", "true", "1", "yes")


def boot_api_pref(repl) -> bool:
    """Auto-start the API on boot?  Backs the `--api` row in two views."""
    return _truthy(_prefs(repl).get("boot_api", "on"))


def _set_bool_pref(prefs: dict, key: str):
    def _set(value: str) -> None:
        prefs[key] = "on" if _truthy(value) else "off"

    return _set


def _set_env(name: str):
    def _set(value: str) -> None:
        os.environ[name] = str(value)

    return _set


def _set_api_base(value: str) -> None:
    os.environ["MAN_API_URL"] = str(value)


def _set_history_size(repl):
    def _set(value: str) -> None:
        try:
            size = int(str(value).strip())
        except (TypeError, ValueError):
            return
        if size < 1:
            return
        store = getattr(repl, "_history_store", None)
        if store is not None:
            store.max_entries = size

    return _set


# ── view 1: settings ─────────────────────────────────────────────────


def settings_rows(repl) -> list[dict]:
    """The shell's real settings — every row bound to live state."""
    prefs = _prefs(repl)
    store = getattr(repl, "_history_store", None)
    return [
        {
            "label": "API base",
            "type": "text",
            "value": get_api_base(),
            "flag": "--api-base",
            "setter": _set_api_base,
        },
        {
            "label": "History size",
            "type": "text",
            "value": str(store.max_entries) if store is not None else "500",
            "flag": "--history",
            "setter": _set_history_size(repl),
        },
        {
            "label": "Auto-start API",
            "type": "bool",
            "value": boot_api_pref(repl),
            "flag": "--api",
            "setter": _set_bool_pref(prefs, "boot_api"),
        },
    ]


# ── view 2: boot ─────────────────────────────────────────────────────


def boot_rows(repl) -> list[dict]:
    """Boot-time flags: what the next boot should start, and with what.

    `--api` reads and writes the same pref as the settings view;
    `--model` is the env var `SLO_AUTOLOAD_MODEL` the API server reads
    when it starts (the API process inherits the environment).
    """
    prefs = _prefs(repl)
    return [
        {
            "label": "Auto-start API",
            "type": "bool",
            "value": boot_api_pref(repl),
            "flag": "--api",
            "setter": _set_bool_pref(prefs, "boot_api"),
        },
        {
            "label": "Autoload model",
            "type": "text",
            "value": os.environ.get("SLO_AUTOLOAD_MODEL", AUTOLOAD_DEFAULT),
            "flag": "--model",
            "setter": _set_env("SLO_AUTOLOAD_MODEL"),
        },
    ]


# ── view 3: `options <command>` — a command's own flags as rows ─────
#
# Row flags here are the drift-prone half: the docstring and the parser
# are written by hand in two places.  `test_option_views.py` pins this
# registry against both, so a flag added to one and not the others fails
# the build instead of lying to the user.

_COMMANDS: dict[str, list[dict]] = {
    "logs": [
        {
            "label": "Level",
            "type": "choice",
            "options": ["(any)", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            "value": "(any)",
            "flag": "--level",
            "omit": ("(any)",),
        },
        {"label": "Source", "type": "text", "value": "", "flag": "--source"},
        {"label": "Lines", "type": "text", "value": "30", "flag": "--lines"},
        {"label": "Follow", "type": "bool", "value": False, "flag": "--follow"},
        {"label": "Clear the buffer", "type": "bool", "value": False, "flag": "--clear"},
        {"label": "Export file", "type": "text", "value": "", "flag": "--export"},
        {"label": "Stats", "type": "bool", "value": False, "flag": "--stats"},
        {"label": "Explain", "type": "bool", "value": False, "flag": "--explain"},
        {"label": "Most recent only", "type": "bool", "value": False, "flag": "--last"},
    ],
    "ai": [
        {"label": "Interactive loop", "type": "bool", "value": False, "flag": "--loop"},
        {"label": "Auto-approve tools", "type": "bool", "value": False, "flag": "--auto"},
    ],
}


def option_commands() -> list[str]:
    """Commands that have an option view, sorted."""
    return sorted(_COMMANDS)


def _remember(repl, name: str, key: str):
    """Setter that persists a row's value so the view reopens as you left it."""

    def _set(value: str) -> None:
        _prefs(repl).setdefault("options", {}).setdefault(name, {})[key] = str(value)

    return _set


def command_rows(name: str, repl=None) -> list[dict]:
    """Row specs for *name*'s flags; [] when the command has no view.

    Values are restored from the session's previous visit when *repl*
    is given, and every row then carries a setter that remembers edits.
    """
    spec = _COMMANDS.get(name)
    if not spec:
        return []
    saved: dict = {}
    if repl is not None:
        saved = _prefs(repl).setdefault("options", {}).setdefault(name, {})
    rows: list[dict] = []
    for base in spec:
        row = dict(base)
        key = str(row.get("flag") or row["label"]).lstrip("-")
        if key in saved:
            value = saved[key]
            if row.get("type") == "bool":
                row["value"] = _truthy(value)
            else:
                row["value"] = value
        if repl is not None:
            row["setter"] = _remember(repl, name, key)
        rows.append(row)
    return rows
