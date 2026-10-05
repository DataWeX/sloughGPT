"""Code-hygiene ratchet — the 2026-10-04 cleanup sweep, encoded as tests.

Origin: kanban card 0d4ad67b. Every pattern below was at ZERO true
violations when this file was written, so the bar is zero forever: new
violations fail, and the fix is either real code or a reviewed allowlist
entry with a reason — never silence.

Enforced here (things ruff does NOT catch):
  1. f-string logger calls — must be lazy: logger.info("x %s", x)
  2. bare ``except:`` — ruff's E722 is disabled in pyproject.toml, so
     this test is the only guard
  3. ``<marker>:`` comment markers — kanban cards are the todo system;
     markers rot in the tree (patterns are self-excluded below)
  4. print() in domain library code — stdout-channel work (TUI renderers,
     CLI tools, protocol handshakes, stderr bootstrap warnings, ``__main__``
     demo entry points) is exempt; new library prints are not

Deliberately NOT enforced (open / owned elsewhere):
  - ``from __future__ import annotations`` — scope decision open on card
    0d4ad67b (860-file debt across two trees); do not "fix" piecemeal
  - F401 unused imports — owned by ruff (pyproject [tool.ruff.lint])
"""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()

# Never descend into these (a 7.5G node_modules lives under apps/mobile).
SKIP_DIRS = frozenset({
    "node_modules", ".venv", "__pycache__", "dist", "build", ".git",
    ".next", "coverage", ".mypy_cache", ".ruff_cache", "test_results",
})

SCAN_ROOTS = ("domain", "packages", "apps", "scripts")
PY = (".py",)
MARKER_EXTS = (".py", ".ts", ".tsx")

LOGGER_FSTRING = re.compile(r"(?:logger|logging)\.(?:info|warning|error|debug)\(f['\"]")
BARE_EXCEPT = re.compile(r"except\s*:")
# Built from parts so this source file does not match its own scan.
MARKER = re.compile(r"\b(?:" + "|".join(("TO" + "DO", "FIX" + "ME", "HA" + "CK", "X" + "XX")) + r"):")

# Custom loggers whose contract rejects lazy %-args — there the f-string
# IS the correct call (StructuredLogger.info(message, **fields): one
# positional message only). Extend only via review.
FSTRING_EXEMPT = {
    "packages/avion/src/avion/core/session.py":
        "avion StructuredLogger.info(message, **fields) — single positional; f-string required",
}

# Existing prints that ARE the product surface (2026-10-04 sweep: 92/92
# legitimate). New files are never pre-exempted; extend only via review.
PRINT_ALLOWLIST = {
    "domain/shell/_internal/world_driver.py":
        "TUI/world renderer — print is the frame writer",
    "domain/multimodal/_internal/phoneme_encoder_cli.py":
        "CLI tool — stdout is its output channel",
    "domain/infrastructure/_internal/inference_engine.py":
        "ENGINE_READY parent-process protocol handshake",
    "domain/logging/_internal/config.py":
        "bootstrap warning — cannot log while configuring logging",
    "domain/shell/_internal/input_device.py":
        "terminal echo in readline()",
    "domain/shell/_internal/realm_live.py":
        "argparse CLI run summary",
    "domain/training/_internal/lm_eval_char.py":
        "--json machine-readable stdout mode",
    "domain/journeys/runner.py":
        "argparse CLI journey runner — [PASS]/[FAIL] report IS stdout",
}


def iter_files(exts, roots=SCAN_ROOTS):
    """Yield files under roots, pruning heavy/irrelevant directories."""
    for root in roots:
        base = ROOT / root
        if not base.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for name in filenames:
                if not name.endswith(exts):
                    continue
                path = Path(dirpath) / name
                if path != SELF:
                    yield path


def line_hits(pattern, exts):
    found = []
    for path in iter_files(exts):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if pattern.search(line):
                rel = path.relative_to(ROOT).as_posix()
                found.append(f"{rel}:{lineno}: {line.strip()[:120]}")
    return found


def fmt(found, limit=40):
    body = "\n".join(found[:limit])
    extra = len(found) - limit
    return body + (f"\n... and {extra} more" if extra > 0 else "")


def test_no_fstring_logger_calls():
    """Logging must use lazy %s formatting, not f-strings.

    Exempt: custom loggers that take a single pre-formatted message
    (FSTRING_EXEMPT carries the per-file reason).
    """
    found = line_hits(LOGGER_FSTRING, PY)
    prefixes = tuple(f"{path}:" for path in FSTRING_EXEMPT)
    found = [h for h in found if not h.startswith(prefixes)]
    assert not found, "f-string logger calls (use lazy %s):\n" + fmt(found)


def test_no_bare_except():
    """Bare except: swallows SystemExit/KeyboardInterrupt — catch types."""
    found = line_hits(BARE_EXCEPT, PY)
    assert not found, "bare except clauses (catch explicit types):\n" + fmt(found)


def test_no_todo_markers():
    """Markers rot — follow-ups belong on the kanban board."""
    found = line_hits(MARKER, MARKER_EXTS)
    assert not found, "in-tree task markers (file a card instead):\n" + fmt(found)


def _main_guard_ranges(tree):
    ranges = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
            continue
        test = node.test
        if not (isinstance(test.left, ast.Name) and test.left.id == "__name__"):
            continue
        if not any(isinstance(c, ast.Constant) and c.value == "__main__"
                   for c in test.comparators):
            continue
        end = max((getattr(s, "end_lineno", node.lineno) for s in ast.walk(node)),
                  default=node.lineno)
        ranges.append((node.lineno, end))
    return ranges


def test_print_only_in_exempted_stdout_channels():
    """print() in domain library code needs a reason (or an exemption).

    Allowed: listed stdout-channel files, calls with file=..., and calls
    inside an ``if __name__ == "__main__"`` demo/CLI entry guard.
    """
    violations = []
    for path in iter_files(PY, roots=("domain",)):
        rel = path.relative_to(ROOT).as_posix()
        if path.name == "__main__.py":
            continue  # `python -m` entry point — stdout is its interface
        if rel in PRINT_ALLOWLIST:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError as exc:
            violations.append(f"{rel}: syntax error: {exc}")
            continue
        guards = _main_guard_ranges(tree)
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "print"):
                continue
            if any(kw.arg == "file" for kw in node.keywords):
                continue  # stderr/stderr-targeted warnings
            if any(lo <= node.lineno <= hi for lo, hi in guards):
                continue  # explicit demo/CLI entry point
            violations.append(
                f"{rel}:{node.lineno}: print() in library path — use the logger, "
                f"or get a channel exemption reviewed into PRINT_ALLOWLIST"
            )
    assert not violations, "library prints without an exemption:\n" + fmt(violations)
