"""API envelope contract: two sanctioned response styles, one error shape.

Response styles (docs/API.md → Response Envelope):

- **Envelope** (default): ``success_response()`` → ``{"status": "success", "data": ...}``.
- **Typed**: a route registered with ``response_model=X`` returns the bare typed
  payload (``GET /knowledge`` → ``list[KnowledgeItemOut]``,
  ``POST /shell/exec`` → ``ShellExecResponse``, ``POST /inference/generate`` →
  ``InferResponse``). Clients pass non-enveloped payloads through unchanged
  (``apps/web/lib/http-client.ts`` unwraps iff ``status`` and ``data`` are present).

Error shape (single source: ``schemas.common.error_response``): flat
``{"error": ..., "code": ..., "details"?: ..., "correlation_id"?: ...}`` —
never wrapped in a ``status``/``data`` envelope. Clients read ``j.error`` /
``j.code`` / ``j.correlation_id`` (http-client.ts:676-679).

The AST scan enforces that every endpoint-level ``return`` in
``apps/api/server/routers`` is either non-literal (envelope call, Response
object, name, await, …) or the route declares ``response_model``. The tree is
currently clean — zero-tolerance, no baseline.
"""

from __future__ import annotations

import ast
from pathlib import Path

from schemas.common import error_response, success_response

ROUTERS_DIR = Path(__file__).resolve().parents[2] / "apps" / "api" / "server" / "routers"
_METHODS = ("get", "post", "put", "delete", "patch", "api_route", "websocket")


# ── Envelope shape pins ──────────────────────────────────────────────────────


def test_success_response_shape() -> None:
    body = success_response(data={"a": 1})
    assert body == {"status": "success", "data": {"a": 1}}


def test_success_response_optional_fields_only_when_set() -> None:
    body = success_response(data=[1], message="ok", meta={"count": 1})
    assert set(body) == {"status", "data", "message", "meta"}
    assert success_response() == {"status": "success", "data": None}


def test_error_response_shape_is_flat() -> None:
    body = error_response("boom", code="E_TEST", details={"k": "v"})
    assert set(body) == {"error", "code", "details"}
    assert body["error"] == "boom"
    assert body["code"] == "E_TEST"


def test_error_response_is_never_enveloped() -> None:
    """The stale docs shape ``{status: "error", error: {...}}`` must not creep in."""
    body = error_response("boom", code="E_TEST")
    assert "status" not in body
    assert "data" not in body
    assert not isinstance(body.get("error"), dict)


# ── Router return scan ───────────────────────────────────────────────────────


def _own_returns(fn: ast.FunctionDef | ast.AsyncFunctionDef):
    """Return nodes in fn's own body only (skip nested defs/lambdas/classes)."""
    stack: list[ast.AST] = list(fn.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        if isinstance(node, ast.Return):
            yield node
        for field in ("body", "orelse", "finalbody"):
            child = getattr(node, field, None)
            if isinstance(child, list):
                stack.extend(child)
            elif isinstance(child, ast.AST):
                stack.append(child)
        for field in ("handlers", "captures"):
            child = getattr(node, field, None)
            if isinstance(child, list):
                stack.extend(h for h in child if isinstance(h, ast.AST))


def _declares_response_model(call: ast.Call) -> bool:
    if any(kw.arg == "response_model" for kw in call.keywords):
        return True
    return "response_model" in ast.unparse(call)


def _scan() -> tuple[int, list[str]]:
    """Return (routes_seen, violations) for every registered route handler."""
    routes_seen = 0
    violations: list[str] = []
    for py in sorted(ROUTERS_DIR.glob("*.py")):
        if py.name == "__init__.py":
            continue
        tree = ast.parse(py.read_text(encoding="utf-8"))
        handlers: dict[str, list[ast.Call | None]] = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for dec in node.decorator_list:
                    if any(f".{m}(" in ast.unparse(dec) for m in _METHODS):
                        handlers.setdefault(node.name, []).append(
                            dec if isinstance(dec, ast.Call) else None
                        )
            if (
                isinstance(node, ast.Call)
                and ast.unparse(node.func).endswith("add_api_route")
                and len(node.args) >= 2
            ):
                handlers.setdefault(ast.unparse(node.args[1]).split(".")[-1], []).append(node)
        functions = {
            n.name: n
            for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for name, registrations in handlers.items():
            routes_seen += len(registrations)
            fn = functions.get(name)
            if fn is None:
                violations.append(f"{py.name}::{name}: handler not found for registration")
                continue
            typed = any(r is not None and _declares_response_model(r) for r in registrations)
            for ret in _own_returns(fn):
                if isinstance(ret.value, (ast.Dict, ast.List, ast.DictComp, ast.ListComp)):
                    if not typed:
                        violations.append(
                            f"{py.name}::{name}:{ret.lineno}: bare {type(ret.value).__name__} "
                            f"return without response_model — wrap in success_response() "
                            f"or declare response_model"
                        )
    return routes_seen, violations


def test_router_returns_are_enveloped_or_typed() -> None:
    routes_seen, violations = _scan()
    assert routes_seen > 400, (
        f"route detection found only {routes_seen} routes — scanner logic has rotted"
    )
    assert not violations, (
        "Endpoint-level bare payload without response_model. Envelope the return "
        "(success_response) or declare a typed response_model.\n" + "\n".join(violations)
    )
