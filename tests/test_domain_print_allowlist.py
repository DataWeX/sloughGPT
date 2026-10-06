"""`domain/` may print only from a known set of output contracts.

`print()` in production code is normally a defect — output should go through
the logger so it is levelled, structured and routable. But a blanket rule
cannot be applied here: of the ~92 `print()` calls under `domain/`, every one
is load-bearing, and converting them would break a real contract:

  * ``ENGINE_READY port=...`` with ``flush=True`` is a **stdout IPC readiness
    protocol** — a parent process parses it. Logger output would never reach it.
  * ``logging/_internal/config.py`` writes to **stderr during logging
    bootstrap**; you cannot report a logging failure *through* logging.
  * ``training/_internal/lm_eval_char.py`` print is the ``--json`` output mode;
    the structured path already uses ``logger.info`` on the line below.
  * ``shell/_internal/input_device.py`` emits a bare newline for terminal echo.
  * the rest are headless-harness / CLI report tables where stdout **is** the
    program's result.

So this is an allowlist ratchet rather than a ban: the files above may print
(and grow their output freely), but a *new* file under `domain/` printing into
a production service path fails the test.

The comparison is a subset check, not equality — the allowlist covers every
contract across the branches that carry this test, and a tree simply lacking
one of those files is not a violation.
"""
from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOMAIN = REPO / "domain"

# path -> why this file's print() is its contract
ALLOWLIST: dict[str, str] = {
    "domain/core/_internal/mole/__main__.py": "module CLI entrypoint; stdout is the CLI result",
    "domain/core/_internal/mole/__main__.py": "module CLI entrypoint; stdout is the CLI result",
    "domain/infrastructure/_internal/inference_engine.py": (
        "ENGINE_READY port=... readiness line consumed over stdout by a parent process"
    ),
    "domain/journeys/runner.py": "journey harness runner; prints the run report",
    "domain/logging/_internal/config.py": (
        "writes to stderr during logging bootstrap — a logging failure cannot be logged"
    ),
    "domain/multimodal/_internal/phoneme_encoder_cli.py": "_cli module: CLI output",
    "domain/shell/_internal/input_device.py": "terminal echo (bare newline on Enter)",
    "domain/shell/_internal/realm_live.py": "realm harness; prints the run summary",
    "domain/shell/_internal/world_driver.py": "headless world harness; aligned report tables",
    "domain/training/_internal/lm_eval_char.py": "--json output mode (logger.info covers the rest)",
}


def _docstring_lines(tree: ast.Module) -> set[int]:
    """Line numbers belonging to any docstring.

    Examples inside docstrings are text, not calls, so AST already skips them;
    this guards against a docstring whose *content* would otherwise be parsed
    if a module ever grew a nested literal call on the same line numbers.
    """
    spans: set[int] = set()
    nodes = [tree] + [
        n
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    for node in nodes:
        if not node.body or not isinstance(node.body[0], ast.Expr):
            continue
        val = node.body[0].value
        if isinstance(val, ast.Constant) and isinstance(val.value, str):
            for lineno in range(val.lineno, val.end_lineno + 1):
                spans.add(lineno)
    return spans


def _production_prints(path: Path) -> int:
    """print() calls reachable from real code, excluding docstring text."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return 0
    docs = _docstring_lines(tree)
    return sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
        and node.lineno not in docs
    )


def _domain_files() -> list[Path]:
    return sorted(
        p for p in DOMAIN.rglob("*.py") if "__pycache__" not in str(p)
    )


def test_no_new_print_in_domain():
    """A print outside the allowlist means someone leaked output into a service."""
    offenders: dict[str, int] = {}
    for path in _domain_files():
        count = _production_prints(path)
        if not count:
            continue
        rel = str(path.relative_to(REPO))
        if rel not in ALLOWLIST:
            offenders[rel] = count
    assert not offenders, (
        "print() in domain/ outside the allowed output contracts. Either route it "
        "through logger, or — if stdout is genuinely this program's output — add "
        "the file to ALLOWLIST with a one-line reason.\n"
        + "\n".join(f"  {f}: {n} call(s)" for f, n in sorted(offenders.items()))
    )


def test_allowlist_entries_are_documented():
    """Every entry needs a reason, or the allowlist becomes an unexamined list."""
    for rel, reason in ALLOWLIST.items():
        assert reason.strip(), f"{rel} has no reason"
        assert len(reason.strip()) > 10, f"{rel}: reason too terse to justify an exception"


def test_domain_is_actually_scanned():
    """Fail loudly if domain/ moved or vanished — otherwise the gate passes vacuously."""
    files = _domain_files()
    assert len(files) > 100, f"only {len(files)} python files found under {DOMAIN}"
