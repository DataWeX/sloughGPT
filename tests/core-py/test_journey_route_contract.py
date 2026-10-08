"""Static contract: every route a journey test navigates must actually exist.

Why this exists
---------------
A dead route does NOT look dead to the journey assertions:

- The Vite dev server serves the SPA shell for unknown paths (HTTP 200), the
  client router renders ``not-found.tsx``, and ``page.inner_text("body")`` is
  comfortably longer than the ``len(body) > 30/50`` guards.
- ``check_no_console_errors()`` passes — a NotFound render logs nothing.

So ``/training/queue`` (dead since the day it was written, card c712a1b7) and
the six ``/training`` subsystem routes (card 59f835fa) sailed through the
journeys green while testing nothing. Body-length and console-error checks
cannot close that hole; route existence can — and it needs no live server,
no Playwright, and writes no shared result files.

Contract
--------
Every route literal a journey test navigates (``*ROUTES`` list entries and
``goto(...)``/``go(...)`` string literals) must be one of:

1. a real page — ``apps/web/app/(app)/<path>/page.tsx`` (or ``apps/web/app``),
2. a redirect source listed in ``apps/web/lib/redirects.ts`` (legacy paths
   legitimately land elsewhere),
3. the root path ``/``.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
APP = REPO / "apps" / "web" / "app"
JOURNEY_DIR = Path(__file__).resolve().parent
REDIRECTS_FILE = REPO / "apps" / "web" / "lib" / "redirects.ts"

# Journey files navigate; other core-py tests do not carry route lists.
ROUTE_CARRIERS = (
    "test_comprehensive_journeys.py",
    "test_comprehensive_training_journeys.py",
    "test_user_journeys.py",
    "test_ui_journey_chrome_devtools.py",
    "test_ui_journey_generic.py",
    "test_ui_journey_library.py",
)


def _redirect_sources() -> set[str]:
    """Pathnames keys of REDIRECTS in lib/redirects.ts (the single source)."""
    text = REDIRECTS_FILE.read_text(encoding="utf-8")
    block = text.split("export const REDIRECTS", 1)[1]
    block = block.split("\n}", 1)[0]
    return set(re.findall(r"'(/[^']*)'\s*:", block))


_DYNAMIC_SEG = re.compile(r"^\[.+\]$")


def _page_exists(route: str) -> bool:
    if route == "/":
        return (APP / "(app)" / "page.tsx").exists() or (APP / "page.tsx").exists()
    segments = [s for s in route.lstrip("/").split("/") if s]
    for base in (APP / "(app)", APP):
        if _segments_resolve(base, segments):
            return True
    return False


def _segments_resolve(base: Path, segments: list[str]) -> bool:
    """Walk ``segments`` under ``base``, matching dynamic ``[id]`` dirs too."""
    if not base.is_dir():
        return False
    current = base
    for seg in segments:
        exact = current / seg
        if exact.is_dir():
            current = exact
            continue
        dynamic = next(
            (d for d in current.iterdir() if d.is_dir() and _DYNAMIC_SEG.match(d.name)),
            None,
        )
        if dynamic is None:
            return False
        current = dynamic
    return (current / "page.tsx").exists()


def _route_literals(tree: ast.AST) -> list[tuple[int, str]]:
    """(line, route) for *ROUTES list entries and goto/go string literals."""
    found: list[tuple[int, str]] = []

    def collect_string(el: ast.expr) -> None:
        if isinstance(el, ast.Constant) and isinstance(el.value, str) and el.value.startswith("/"):
            found.append((el.lineno, el.value))

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.endswith("ROUTES"):
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        for el in node.value.elts:
                            if isinstance(el, (ast.List, ast.Tuple)) and el.elts:
                                collect_string(el.elts[0])  # ("​/path", "name") pairs
                            else:
                                collect_string(el)
        elif isinstance(node, ast.Call):
            func = node.func
            is_goto = isinstance(func, ast.Attribute) and func.attr == "goto"
            is_go = isinstance(func, ast.Name) and func.id == "go"
            if is_goto or is_go:
                for arg in node.args:
                    collect_string(arg)
    return found


def _discover() -> list[tuple[str, int, str]]:
    cases: list[tuple[str, int, str]] = []
    for name in ROUTE_CARRIERS:
        path = JOURNEY_DIR / name
        if not path.exists():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for line, route in _route_literals(tree):
            cases.append((name, line, route))
    return cases


CASES = _discover()


def test_journey_route_carriers_found() -> None:
    """Guard the extractor itself: the known carriers must yield routes."""
    assert len(CASES) >= 20, f"extraction found only {len(CASES)} route literals"


def test_all_journey_routes_exist() -> None:
    sources = _redirect_sources()
    missing: list[str] = []
    for name, line, route in CASES:
        if route in sources or _page_exists(route):
            continue
        missing.append(f"{name}:{line}  {route}")
    assert not missing, (
        "journey tests navigate routes that do not exist (dead links — they "
        "render not-found.tsx and every body-length/console assertion passes "
        "vacuously):\n  " + "\n  ".join(missing)
    )
