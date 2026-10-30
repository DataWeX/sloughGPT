"""Domain import hygiene: the core reads no wire types and no app layer.

The dependency rule the whole engine-port refactor rests on — ``domain`` must
stay app-import-free:

- **no ``pydantic``** — wire contracts belong to the edge (``schemas/*``); a
  domain model that inherits ``BaseModel`` has silently become a transport type
  and can no longer be imported from a script, CLI or test without dragging in
  the HTTP layer.
- **no ``fastapi``** — ``StreamingResponse``, ``Depends``, ``Request`` are
  transport. Streaming is a *contract* the core honours by yielding tokens; how
  they are framed into SSE is the edge's job.
- **no ``routers`` / ``apps`` / ``schemas``** — the core never reaches upward.
  Consumers depend on the core, never the reverse.

Enforced with ``ast`` rather than text matching so docstrings and comments that
*mention* ``from schemas.common import ...`` do not count as violations, while
imports nested inside ``if TYPE_CHECKING:`` or a function body (deferred, but
still a real coupling) do.

The baseline is **shrink-only and exact**: entries name a specific
``file -> forbidden root`` pair. Adding a violation fails, and so does leaving a
stale entry behind once the violation is fixed — the baseline can only shrink.
"""

from __future__ import annotations

import ast
from pathlib import Path

DOMAIN_DIR = Path(__file__).resolve().parents[2] / "domain"

# Wire types, HTTP framework, and the app layer — the core must not bind these.
FORBIDDEN_ROOTS: frozenset[str] = frozenset({"pydantic", "fastapi", "routers", "apps", "schemas"})

# Existing couplings, measured before this gate (2026-10-20): 1 pydantic, 0
# fastapi, 3 app-layer, out of 623 domain modules. Shrink-only — do not add.
# (``domain/infrastructure/_internal/errors.py`` looks like a 4th hit to grep,
# but its ``from schemas.common import raise_error`` is a usage example inside
# the module docstring — AST sees no real import, which is why this gate is
# AST-based rather than text-based.)
_BASELINE: frozenset[tuple[str, str]] = frozenset(
    {
        # Domain-level config predates the rule: it models ServerConfig-shaped
        # settings as pydantic models rather than plain dataclasses.
        ("domain/infrastructure/_internal/config.py", "pydantic"),
        # Deferred (function/TYPE_CHECKING) but still upward-reaching.
        ("domain/infrastructure/_internal/model_config.py", "apps"),
        ("domain/shell/_internal/repl.py", "apps"),
        ("domain/logging/_internal/config.py", "schemas"),
    }
)

# The chat engine port — the specific seam this gate was written to protect.
ENGINE_PORT = "domain/inference/engine.py"


def _imported_roots(py: Path) -> set[str]:
    """Forbidden package roots this module actually imports (AST-accurate)."""
    try:
        tree = ast.parse(py.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        # Unparseable source is a different defect (ruff/compile gate owns it);
        # reporting it as a hygiene violation would blame the wrong tool.
        return set()

    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level or not node.module:
                continue  # relative import — stays inside domain
            modules = [node.module]
        else:
            continue
        for module in modules:
            root = module.split(".", 1)[0]
            if root in FORBIDDEN_ROOTS:
                roots.add(root)
    return roots


def _violations() -> set[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    for py in sorted(DOMAIN_DIR.rglob("*.py")):
        rel = py.relative_to(DOMAIN_DIR.parent).as_posix()
        for root in _imported_roots(py):
            found.add((rel, root))
    return found


def test_domain_imports_no_wire_or_app_layer() -> None:
    found = _violations()

    added = sorted(found - _BASELINE)
    assert not added, (
        "domain must stay app-import-free — a core module may not import pydantic, "
        "fastapi, or the app layer (routers/apps/schemas). Move the wire shape to "
        "schemas/* and translate at the edge, or push the value down as a plain "
        "dataclass. New violations:\n  " + "\n  ".join(f"{f} -> {r}" for f, r in added)
    )

    stale = sorted(_BASELINE - found)
    assert not stale, (
        "The import-hygiene baseline is shrink-only: these couplings no longer "
        "exist, so remove them from _BASELINE:\n  " + "\n  ".join(f"{f} -> {r}" for f, r in stale)
    )


def test_engine_port_is_app_import_free() -> None:
    """The chat engine port specifically: stream-only, domain-native, no wire.

    ``routers/chat.py`` imports this port and ``routers/inference.py`` registers
    into it; neither router imports the other. That only holds if the port itself
    binds neither pydantic (its inputs are ``TurnRequest``, a dataclass) nor
    fastAPI (its output is tokens and a ``TurnResult``, not a response object).
    """
    # ENGINE_PORT is repo-root-relative (same form _violations() reports).
    py = DOMAIN_DIR.parent / ENGINE_PORT
    assert py.exists(), f"engine port moved — update ENGINE_PORT ({ENGINE_PORT})"

    roots = _imported_roots(py)
    assert not roots, (
        f"{ENGINE_PORT} must stay app-import-free but imports: {sorted(roots)}. "
        "Wire contracts live in schemas/*; SSE framing lives at the edge."
    )
