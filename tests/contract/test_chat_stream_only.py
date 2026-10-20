"""Chat is stream-only: one SSE endpoint, no non-streaming re-entry (ratchet).

The invariant the stream-first merge bought, frozen here: **one producer,
derived reads**. The generate engine has exactly one method that produces
tokens, so every "complete reply" is a *fold computed by the consumer* — never
a second producer, never a mode flag, never a sibling URL.

This is the shape of failure the gate exists to stop, and it is not
hypothetical: the cancel-channel bug (``aad552524``) lived in exactly the fork
this forbids — ``respond`` (fold) and ``stream`` (SSE) were two wrappers over
one engine, and they drifted until one of them discarded its ``stop`` event and
the API reported ``cancelled: true`` while generation ran to completion.
Merging them removed the fork; this gate keeps it removed.

Layers, and the specific reintroduction each catches:

1. **No ``/chat/stream`` sibling, anywhere** — a literal scan of every router
   (how the last fork happened: a *new* route on *another* router), plus the
   registered route tables of the production mount set so a path built by
   concatenation cannot slip past the literal.
2. **No ``stream`` flag** on ``ChatRequest`` / ``TurnRequest`` — a mode toggle
   forks caller behaviour while the route table still looks clean.
3. **The chat router never calls ``fold()``** — folding is the edge's job (the
   mobile BFF folds in-process through ``handle_chat``); inside the router it
   is a second producer wearing a consumer's name.
4. **Exactly one registered ``POST /chat``**, bound to a handler typed
   ``StreamingResponse``.
5. **No ``response_model`` on the chat router** — a folded body in OpenAPI
   re-advertises the non-streaming contract even while the handler streams.

Deliberately out of scope (each is a different rule): ``/inference/generate``'s
non-streaming variant is grandfathered by the Endpoint Rule (no retroactive
purge); ``companion`` owns a *separate* ``POST /chat`` under its own prefix
with its own ``ChatResponse``; the port's ``fold()`` is the edge fold the
design mandates and must stay. The router-level ``chat_stream``/
``handle_chat_stream`` pair has been retired; its provider-protocol namesake
(``provider.chat_stream()`` — the token producer every engine streams
through) is a different layer and must stay, so this gate never greps for the
bare name.

AST rather than text: ``routers/chat.py`` uses the words "folds" and "/chat/stream
sibling" in five docstring lines, so a grep gate would fail on our own comments
(the lesson ``test_domain_import_hygiene.py`` documents).
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROUTERS_DIR = ROOT / "apps" / "api" / "server" / "routers"
CHAT_ROUTER = ROUTERS_DIR / "chat.py"
CHAT_WIRE = ROOT / "apps" / "api" / "server" / "schemas" / "chat.py"
ENGINE_PORT = ROOT / "domain" / "inference" / "engine.py"

# The sibling URL, by exact literal. Whole-string equality means a docstring
# that *mentions* it never matches: the AST constant is the whole docstring.
FORBIDDEN_PATH = "/chat/stream"
# Mode toggles on the generation contract. `streaming` is included so renaming
# the flag does not dodge the gate.
TOGGLE_NAMES: frozenset[str] = frozenset({"stream", "streaming"})


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


# The live registry is read in a *cold interpreter*, never in-process.
# ``get_all_routers()`` imports all 53 router modules, and their first import
# binds ``state``/startup in whatever interpreter does it. Running that inside
# pytest would freeze real-state bindings before the behaviour suites patch
# ``sys.modules`` (they rely on ``state`` binding their mock on first import)
# — turning this gate into an order-dependent hazard that fails sibling suites
# it has no business touching. One subprocess, zero pollution.
_PROBE = """
import json
from routers import get_all_routers

rows = []
for router in get_all_routers():
    for route in getattr(router, "routes", []):
        endpoint = getattr(route, "endpoint", None)
        annotations = getattr(endpoint, "__annotations__", None) or {}
        rows.append({
            "path": getattr(route, "path", "") or "",
            "methods": sorted(getattr(route, "methods", None) or []),
            "endpoint": getattr(endpoint, "__qualname__", ""),
            "has_response_model": getattr(route, "response_model", None) is not None,
            "returns": getattr(annotations.get("return"), "__name__", None),
        })
print("ROWS=" + json.dumps(rows))
"""


def _registered_routes() -> list[dict]:
    """Snapshot the production route registry (``get_all_routers()``) once."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        str(p)
        for p in (
            ROOT,
            ROOT / "apps" / "api" / "server",
            ROOT / "packages" / "core-py",
            ROOT / "packages" / "sdk-py",
            ROOT / "packages" / "downcraft",
            ROOT / "apps" / "cli" / "src",
        )
    )
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        env=env,
        cwd=ROOT,
        timeout=300,
    )
    assert proc.returncode == 0, (
        f"route-registry probe exited {proc.returncode}:\n{proc.stderr[-3000:]}"
    )
    line = next((ln for ln in reversed(proc.stdout.splitlines()) if ln.startswith("ROWS=")), None)
    assert line is not None, f"probe produced no ROWS marker:\n{proc.stdout[-2000:]}"
    return json.loads(line[len("ROWS=") :])


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


# ── 1. no sibling route ─────────────────────────────────────────────────────


def test_no_chat_stream_route_literal_in_any_router():
    """No router may register the sibling path — including routers that are not chat."""
    hits = [
        f"{_rel(path)}:{node.lineno}"
        for path in sorted(ROUTERS_DIR.glob("*.py"))
        for node in ast.walk(_parse(path))
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value == FORBIDDEN_PATH
    ]
    assert hits == [], (
        f"{FORBIDDEN_PATH} is registered as a literal here: {hits}. "
        "POST /chat is the endpoint's only state; a caller wanting a complete "
        "reply folds the stream at its own edge."
    )


def test_served_chat_endpoint_is_single_and_sse_typed():
    """The registered surface: one POST /chat, streaming, no folded body, no sibling.

    Scanned across **every** router in the production mount set (the registry
    ``get_all_routers()`` returns — what ``infrastructure/startup.py`` and the
    router manifest read), taken from a cold subprocess so this gate cannot
    reorder sibling suites' module imports (see ``_registered_routes``). No
    request is made: with no model materialized ReadinessGateMiddleware
    answers 503 before the handler, so a live POST would inspect the gate,
    not the endpoint.
    """
    rows = _registered_routes()

    sibling = [f"{r['methods']} {r['path']}" for r in rows if r["path"].startswith(FORBIDDEN_PATH)]
    chat_posts = [r for r in rows if r["path"] == "/chat" and "POST" in r["methods"]]

    assert sibling == [], f"a non-streaming sibling is registered: {sibling}"
    assert len(chat_posts) == 1, (
        f"POST /chat must have exactly one registration owner, found {len(chat_posts)}"
    )

    route = chat_posts[0]
    assert not route["has_response_model"], (
        "POST /chat must not declare a folded response_model — a complete reply "
        "is the consumer's fold of the stream, not a response body"
    )
    assert route["endpoint"].startswith("ChatRouter."), (
        f"POST /chat is served by {route['endpoint']}, not the chat router — "
        "ownership moved without this gate noticing"
    )
    assert route["returns"] == "StreamingResponse", (
        f"POST /chat must be typed StreamingResponse, got {route['returns']!r}"
    )


# ── 2. no mode toggle on the generation contract ────────────────────────────


def test_generation_contract_declares_no_stream_toggle():
    """Neither the wire request nor the port's TurnRequest may carry a mode flag.

    A ``stream: bool`` is the quiet way back: the route table stays clean while
    two behaviours re-emerge behind one URL.
    """
    violations: list[str] = []
    for path, class_name in ((CHAT_WIRE, "ChatRequest"), (ENGINE_PORT, "TurnRequest")):
        tree = _parse(path)
        cls = next(
            (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == class_name),
            None,
        )
        assert cls is not None, f"{class_name} not found in {_rel(path)} — gate needs a rewrite"
        for node in cls.body:
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if node.target.id in TOGGLE_NAMES:
                    violations.append(f"{_rel(path)}:{node.lineno} {class_name}.{node.target.id}")
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in TOGGLE_NAMES:
                        violations.append(f"{_rel(path)}:{node.lineno} {class_name}.{target.id}")
    assert violations == [], (
        f"mode toggles on the generation contract: {violations}. "
        "Streaming is the endpoint's only state — the engine yields raw tokens "
        "and edges fold them; there is no second behaviour to select."
    )


# ── 3. the router never folds ───────────────────────────────────────────────


def test_chat_router_never_folds():
    """``fold()`` belongs to the consumer edge, never to the chat router.

    In the router, a fold is the retired ``respond`` route back under another
    name — the exact two-wrapper fork that swallowed the cancel event.
    """
    calls = [
        f"line {node.lineno}"
        for node in ast.walk(_parse(CHAT_ROUTER))
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == "fold")
            or (isinstance(node.func, ast.Attribute) and node.func.attr == "fold")
        )
    ]
    assert calls == [], (
        f"routers/chat.py calls fold() at {calls}. The router frames the engine's "
        "stream; a caller wanting a complete payload folds at its own edge "
        "(the mobile BFF does it in-process through handle_chat)."
    )


# ── 5. no folded response shape on the chat router ──────────────────────────


def test_chat_router_declares_no_response_model():
    """A ``response_model`` on a streaming router re-advertises the old contract.

    OpenAPI is a projection other clients generate from; a folded schema there
    recreates the non-streaming surface even while the handler streams.
    """
    declared = [
        f"line {node.lineno}"
        for node in ast.walk(_parse(CHAT_ROUTER))
        if isinstance(node, ast.Call)
        for kw in node.keywords
        if kw.arg == "response_model"
    ]
    assert declared == [], (
        f"response_model declared in routers/chat.py at {declared}. "
        "POST /chat answers an SSE stream; its contract is the frame, not a body."
    )
