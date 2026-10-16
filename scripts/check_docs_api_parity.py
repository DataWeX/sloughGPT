#!/usr/bin/env python3
"""Doc-vs-code parity check for the API surface.

Cross-references three sources and reports the gaps:

1. **Code truth**  — route decorators in ``apps/api/server/routers/*.py``
   (plus ``main.py`` app-level routes), joined with each router's
   ``APIRouter(prefix=...)``.
2. **Doc claims**  — the endpoint tables in ``docs/routers.md`` and the
   count claims in ``docs/API.md`` (``N routes across M routers``).
3. **Test coverage** — ``tests/server/`` + ``tests/api-server/`` files
   per router (fuzzy name match).

Findings (docs claim but code lacks / code has but docs lack / stale counts
/ untested routers) are what the gap-audit report turns into cards.

4. **Caller reachability** — an AST import graph over tracked ``domain/``
   modules (resolved the way Python resolves: relative imports, facade
   re-exports, ancestor package inits) classifies modules with zero static
   references against the ``CALLERLESS_BASELINE`` ratchet: a NEW orphan
   fails the check, a baselined module that gains a caller is reported
   stale (exit stays 0), untracked files are pending, never dead.
   DOMAIN_CONSOLIDATION.md §7 method / §8 step 9.

Usage::

    python scripts/check_docs_api_parity.py             # human summary
    python scripts/check_docs_api_parity.py --json      # machine output
    python scripts/check_docs_api_parity.py --markdown  # report fragment

Exit codes: 0 = parity, 1 = gaps found, 2 = parse/setup error.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROUTERS_DIR = REPO / "apps" / "api" / "server" / "routers"
CONTROLLERS_DIR = REPO / "apps" / "api" / "server" / "controllers"
MAIN_PY = REPO / "apps" / "api" / "server" / "main.py"
TRAINING_ROUTER = REPO / "apps" / "api" / "server" / "training" / "router.py"
ROUTERS_MD = REPO / "docs" / "routers.md"
API_MD = REPO / "docs" / "API.md"
# the contract block lives with the evidence pack
CLAIMS_MD = REPO / "docs" / "DOC_VS_CODE_GAP_AUDIT.md"
# Fuzzy-matched against router stems by collect_tests(). The second entry was
# `apps/api/server/tests`, which does not exist — and because collect_tests
# filters on `d.exists()`, half the suite went invisible and 22 tested routers
# were reported untested (23 findings vs the true 1). The module docstring
# always said `tests/api-server`; the constant was never updated to match.
#
# Do NOT add `tests/core-py` here: it looks like it clears the last finding, but
# tests/core-py/test_metrics.py imports domain.core.infrastructure._internal.metrics
# (the Prometheus collector), not the metrics ROUTER — its filename would erase
# the one genuine coverage gap.
TEST_DIRS = (REPO / "tests" / "server", REPO / "tests" / "api-server")

# Routers in a concurrent-refactor zone owned by another session — excluded from
# the handler `_internal` assertions until that work lands.
HANDLER_ALLOWLIST = frozenset({"shell", "vm", "world_render"})

# --- caller-reachability ratchet (DOMAIN_CONSOLIDATION.md §7 method) ---------
# Modules with ZERO static references, i.e. indegree 0 over an AST import graph
# built the way Python resolves — §7 calibrated that grep-dotted-paths yields
# 69% false positives (32 flagged vs 10 real), so the graph resolves:
#   * relative imports against __package__ (from .base import X)
#   * facade re-exports (from domain.x import NAME -> the module that actually
#     binds NAME, read from each package's __init__ import table)
#   * ancestor package inits, which load whenever any submodule is imported
#   * `from pkg import sub`, which loads the submodule when one exists
#   * string-literal dotted refs from non-test production code (importlib etc.)
# tests/ are NOT roots (§7: "a test is a weak root" — math_rag.py has a
# 421-line test and is still collectable), and untracked files are PENDING,
# never dead (§7 rule: classify untracked code as pending).
#
# The ratchet: a baselined module that GAINS a caller leaves the callerless
# set and is reported stale (cleanup nudge, exit 0); a module that LOSES all
# callers is a NEW orphan and exits 1. Baseline sealed 2026-10-15 on
# feat/generate-consolidation @ 0e03324c1; all 52 entries hand-verified against
# the same probe run at 89d4e7cb1^ (before the §8 CCGT-four moves): 51 are
# pre-existing debt carried through the renames, and domain.core.ops.__init__
# is the one new orphan — step 5's deliberate wandb_server extraction
# (ops -> training) left the pure-NumPy fused-ops facade with test-only roots.
# Classes present: shell dynamic TUI commands (loaded by directory scan, so no
# per-module reference exists anywhere), tested-but-unwired leaf modules
# (§7's weak-root class), backward-compat shims, and the slolib.py stub whose
# delete-or-keep decision is still pending with the user.
CALLERLESS_BASELINE = frozenset(
    {
        "domain.agents._internal.devtools_follower",
        "domain.cognition._internal.consciousness.training",
        "domain.cognition._internal.math_rag",
        "domain.cognition.multimodal._internal.phoneme_encoder_cli",
        "domain.cognition.multimodal._internal.tts",
        "domain.core._internal.base",
        "domain.core._internal.core",
        "domain.core._internal.grounding",
        "domain.core._internal.knowledge_graph_v2",
        "domain.core._internal.math_rag",
        "domain.core.infrastructure._internal.cache",
        "domain.core.infrastructure._internal.cache.__init__",
        "domain.core.infrastructure._internal.deployment",
        "domain.core.infrastructure._internal.deployment.__init__",
        "domain.core.infrastructure._internal.gpu.compile_shaders",
        "domain.core.infrastructure._internal.gpu.wgpu_be",
        "domain.core.infrastructure._internal.rope",
        "domain.core.infrastructure._internal.singleton",
        "domain.core.infrastructure.safetensors_loader",
        "domain.core.ops.__init__",
        "domain.core.shared._internal.utils",
        "domain.core.slolib._internal.slolib",
        "domain.generate._internal.session_kv_manager",
        "domain.shell._internal.addons.module_loader",
        "domain.shell._internal.chart",
        "domain.shell._internal.cmds.dashboard",
        "domain.shell._internal.cmds.data_cmds",
        "domain.shell._internal.cmds.health",
        "domain.shell._internal.cmds.models_cmd",
        "domain.shell._internal.cmds.souls_cmd",
        "domain.shell._internal.cmds.status",
        "domain.shell._internal.device_decorators",
        "domain.shell._internal.realm_live",
        "domain.shell._internal.term_screen",
        "domain.shell._internal.widgets",
        "domain.shell._internal.window",
        "domain.shell._internal.world_driver",
        "domain.training._internal.comprehensive_trainer",
        "domain.training._internal.dataset",
        "domain.training._internal.model_card",
        "domain.training._internal.owned_objective",
        "domain.training._internal.presets",
        "domain.training._internal.sequence",
        "domain.training._internal.status",
        "domain.training.collections._internal.baby_building",
        "domain.training.collections._internal.baby_community",
        "domain.training.collections._internal.baby_economy",
        "domain.training.collections._internal.baby_evolution",
        "domain.training.collections._internal.baby_perception",
        "domain.training.collections._internal.baby_social",
        "domain.training.feedback._internal.model_health_repo",
        "domain.training.learner._internal.knowledge_storage",
    }
)

# docs/routers.md section title -> router file stem. Sections that map to no
# single file (OpenAPI) or to main.py-registered routers are noted inline.
SECTION_ALIASES = {
    "Health": "health",
    "Status": "status",
    "System": "system",
    "Inference": "inference",
    "Infer": "infer",
    "Models": "models",
    "Souls": "souls",
    "Config": "config",
    "Auth": "auth",
    "Session": "session",
    "Feedback": "feedback",
    "Knowledge Base": "kb",
    "Memory": "memory",
    "Datasets": "datasets",
    "Training": "training/router",
    "Self-Train": "self_train",
    "Learner": "learner",
    "Tokenizer": "tokenizer",
    "Token Tree": "token_tree",
    "Agent": "agents",
    "Multimodal": "multimodal",
    "Benchmark": "benchmark",
    "Companion": "companion",
    "Collections": "collections",
    "Docstore": "docstore",
    "Errors": "errors",
    "Experiments": "experiments",
    "Vector": "vector",
    "User Adapters": "user_adapters",
    "LoRA Eval": "lora_eval",
    "Registry": "registry",
    "Meta-Weights": "meta_weights",
    "VM": "vm",
    "Workflow": "workflow",
    "Shell": "shell",
    "Images": "images",
    "Voice": "voice",
    "Files": "files",
    "Security": "security",
    "Rate Limit": "ratelimit",
    "World Render": "world_render",
}

# Router registration comes in three styles across the codebase:
#   1. class-based:  self.router = APIRouter(prefix="/x")
#                    self.router.add_api_route("/y", handler, methods=["GET"])
#   2. module-style: router = APIRouter(prefix="/x"); @router.get("/y")
#   3. app-level:    @app.get("/y") in main.py
ROUTE_RE = re.compile(
    r"@(?P<chain>\w+(?:\.\w+)*)\.(?P<method>get|post|put|delete|patch|head|options)"
    r'\(\s*(?P<q>["\'])(?P<path>[^"\']*)(?P=q)',
    re.MULTILINE,
)
ADD_ROUTE_RE = re.compile(r"(?P<recv>\w+)\.add_api_route\(")
# path may be positional or path="<x>" (keyword form) — anchored to the call's
# own "(" so a non-literal first arg can't match some later quoted string.
ADD_ROUTE_PATH_RE = re.compile(r"^.*?\(\s*(?:path\s*=\s*)?(?P<q>[\"'])(?P<path>[^\"']*)(?P=q)")
ADD_ROUTE_METHODS_RE = re.compile(r"methods\s*=\s*\[(?P<ms>[^\]]*)\]")
# endpoint sits right after the path: positional (add_api_route("/x", self.h, ...))
# or keyword (add_api_route(path="/x", endpoint=self.h, ...))
ADD_ROUTE_ENDPOINT_RE = re.compile(r"\s*,\s*(?:endpoint\s*\=\s*)?(?P<e>[A-Za-z_]\w*(?:\.\w+)*)")
FULL_ROW_RE = re.compile(
    r"^\|\s*`(?P<m>GET|POST|PUT|DELETE|PATCH)`\s*\|\s*`(?P<p>[^`]+)`"
    r"\s*\|\s*(?P<d>.*?)\s*\|\s*$"
)


def _norm(s: str) -> str:
    """Normalize a section title or router stem for fuzzy matching."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _endpoint_doc(text: str, endpoint: str) -> str:
    """First line of the endpoint handler's docstring (best effort)."""
    name = endpoint.rsplit(".", 1)[-1]
    m = re.search(
        rf"def\s+{re.escape(name)}\s*\([^)]*\)[^:\n]*:\s*\n\s*\"\"\"(?P<d>[^\"\n]+)",
        text,
    )
    if not m:
        return ""
    return re.sub(r"\s+", " ", m.group("d")).strip()


APICONFIG_RE = re.compile(
    r"(?P<var>\w+)\s*=\s*APIRouter\((?P<args>.*?)\)\s*(?:\n|$)",
    re.DOTALL,
)
PREFIX_RE = re.compile(r'prefix\s*=\s*["\']([^"\']*)["\']')
DOC_ROW_RE = re.compile(
    r"^\|\s*`(?P<method>GET|POST|PUT|DELETE|PATCH)`\s*\|\s*`(?P<path>[^`]+)`",
    re.MULTILINE,
)
DOC_SECTION_RE = re.compile(r"^## (?P<title>.+?)(?:\s+Router.*)?$", re.MULTILINE)
API_COUNT_RE = re.compile(r"\*\*(?P<routes>\d+)\s+routes\s+across\s+(?P<routers>\d+)\s+routers\*\*")


@dataclass
class Route:
    router: str  # file stem, e.g. "kb" or "main" / "training/router"
    method: str
    path: str  # full path incl. prefix
    dynamic: bool = False  # decorator path was not a literal — skipped in diffs
    desc: str = ""  # handler docstring first line (used by --render-routers)


@dataclass
class Findings:
    routes: list[Route] = field(default_factory=list)
    doc_claims: list[tuple[str, str]] = field(default_factory=list)  # (METHOD, path)
    doc_sections: list[str] = field(default_factory=list)
    api_md_counts: tuple[int, int] | None = None
    routers_on_disk: list[str] = field(default_factory=list)
    untested: list[str] = field(default_factory=list)
    code_not_documented: list[str] = field(default_factory=list)
    doc_not_in_code: list[tuple[str, str]] = field(default_factory=list)
    undocumented: dict[str, int] = field(default_factory=dict)  # router -> #routes w/o doc row
    dynamic_routes: list[str] = field(default_factory=list)
    collisions: list[str] = field(default_factory=list)  # "METHOD path -> [routers]"
    # doc-vs-code architectural contract (the `parity-claims` block)
    arch_claims: dict[str, int] = field(default_factory=dict)
    arch_measured: dict[str, int] = field(default_factory=dict)
    arch_mismatch: list[str] = field(default_factory=list)  # "key: doc=11 code=8"
    arch_missing: list[str] = field(default_factory=list)  # claim keys absent from docs


def _join(prefix: str, path: str) -> str:
    """Join an APIRouter prefix with a route path (FastAPI semantics)."""
    if not path:
        return prefix or "/"
    return (prefix.rstrip("/") + "/" + path.lstrip("/")) if prefix else "/" + path.lstrip("/")


def _spec_fields(node: ast.AST) -> tuple[bool | None, str]:
    """``(idempotent, description)`` literals read off a ``ToolSpec(...)`` expr.

    Mirrors ``infrastructure.contract.resolve_method``: the verb is derived from
    ``idempotent``, so the gate has to derive it the same way or it will
    disagree with what FastAPI actually serves. Returns ``idempotent=None`` when
    it cannot be resolved statically — the caller marks the route dynamic
    instead of guessing.
    """
    idem: bool | None = None
    desc = ""
    if not isinstance(node, ast.Call):
        return None, ""
    func = node.func
    tail = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
    if tail != "ToolSpec":
        return None, ""
    seen = False
    for kw in node.keywords:
        if kw.arg == "idempotent":
            seen = True
            if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, bool):
                idem = kw.value.value
        elif kw.arg == "description" and isinstance(kw.value, ast.Constant):
            desc = kw.value.value if isinstance(kw.value.value, str) else ""
    if not seen:
        idem = False  # ToolSpec dataclass default
    return idem, desc


def _literal_str(node: ast.AST) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _scan_file(path: Path, router_stem: str, out: list[Route]) -> None:
    """Extract routes from one file in any of the three registration styles."""
    text = path.read_text(encoding="utf-8", errors="replace")

    prefixes: dict[str, str] = {}
    for m in APICONFIG_RE.finditer(text):
        pm = PREFIX_RE.search(m.group("args"))
        prefixes[m.group("var")] = pm.group(1) if pm else ""
    # receiver aliases (e.g. `r = self.router` in kb.py) inherit the prefix so
    # add_api_route calls made through them join paths correctly
    for am in re.finditer(r"(\w+)\s*=\s*([\w.]+)", text):
        alias, rhs = am.group(1), am.group(2).rsplit(".", 1)[-1]
        if rhs in prefixes and alias not in prefixes:
            prefixes[alias] = prefixes[rhs]

    # style 1: class-based add_api_route — bounded per call so a call without
    # methods= can't absorb the next call's methods list.
    positions = list(ADD_ROUTE_RE.finditer(text))
    for i, m in enumerate(positions):
        end = positions[i + 1].start() if i + 1 < len(positions) else len(text)
        chunk = text[m.start() : end]
        pm = ADD_ROUTE_PATH_RE.search(chunk)
        if not pm:
            out.append(Route(router_stem, "?", "<dynamic>", dynamic=True))
            continue
        mm = ADD_ROUTE_METHODS_RE.search(chunk)
        methods = (re.findall(r"[\"']([A-Za-z]+)[\"']", mm.group("ms")) if mm else []) or ["GET"]
        full = _join(prefixes.get(m.group("recv"), ""), pm.group("path"))
        em = ADD_ROUTE_ENDPOINT_RE.match(chunk, pm.end())
        desc = _endpoint_doc(text, em.group("e")) if em else ""
        for method in methods:
            out.append(Route(router_stem, method.upper(), full, desc=desc))

    # styles 2+3: decorators on router/app (incl. @self.router.get chains)
    for m in ROUTE_RE.finditer(text):
        key = m.group("chain").rsplit(".", 1)[-1]
        if key not in prefixes and key not in {"app", "router"} and not key.endswith("router"):
            continue  # decorator on an unrelated object
        dm = re.search(r"def\s+(?P<n>\w+)\s*\(", text[m.end() :])
        desc = _endpoint_doc(text, dm.group("n")) if dm else ""
        out.append(
            Route(
                router_stem,
                m.group("method").upper(),
                _join(prefixes.get(key, ""), m.group("path")),
                desc=desc,
            )
        )

    # style 4: descriptor-projected fragments — create_router(spec, route, h).
    # Both the path and the verb are derived from the descriptor at build time,
    # so resolve them exactly the way resolve_method() does. Anything that
    # cannot be resolved statically becomes dynamic (counted, not diffed) rather
    # than guessed — a wrong verb here would fork the gate from what FastAPI
    # actually serves.
    try:
        tree = ast.parse(text)
    except SyntaxError:
        tree = None
    if tree is not None:
        specs: dict[str, tuple[bool | None, str]] = {}
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                idem, sdesc = _spec_fields(node.value)
                if idem is not None:
                    for t in targets:
                        if isinstance(t, ast.Name):
                            specs[t.id] = (idem, sdesc)
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "create_router"
            ):
                continue
            idem, desc = _spec_fields(node.args[0]) if len(node.args) >= 1 else (None, "")
            if idem is None and len(node.args) >= 1 and isinstance(node.args[0], ast.Name):
                idem, desc = specs.get(node.args[0].id, (None, desc))
            route = _literal_str(node.args[1]) if len(node.args) >= 2 else None
            if route is None or idem is None:
                out.append(Route(router_stem, "?", "<dynamic>", dynamic=True))
                continue
            prefix, override = "", None
            for kw in node.keywords:
                if kw.arg == "prefix":
                    prefix = _literal_str(kw.value) or ""
                elif kw.arg == "method":
                    override = _literal_str(kw.value)
            # resolve_method(): an explicit method= override wins
            method = (override or ("GET" if idem else "POST")).upper()
            out.append(Route(router_stem, method, _join(prefix, route.strip("/")), desc=desc))

    # dynamic decorator paths — counted, not diffed
    for m in re.finditer(
        r"@\w+(?:\.\w+)*\.(?P<method>get|post|put|delete|patch)\(\s*(?![\"'])", text
    ):
        out.append(Route(router_stem, m.group("method").upper(), "<dynamic>", dynamic=True))


def collect_code(findings: Findings) -> None:
    router_files = sorted(p for p in ROUTERS_DIR.glob("*.py") if not p.name.startswith("_"))
    findings.routers_on_disk = [p.stem for p in router_files]
    for p in router_files:
        _scan_file(p, p.stem, findings.routes)
    if MAIN_PY.exists():
        _scan_file(MAIN_PY, "main", findings.routes)
    if TRAINING_ROUTER.exists():
        # the training umbrella + every sub-router (legacy, lora, distill, ...)
        # form one logical router; the doc has a single Training section.
        for p in sorted(TRAINING_ROUTER.parent.glob("*.py")):
            if p.name != "__init__.py":
                _scan_file(p, "training/router", findings.routes)


def collect_docs(findings: Findings) -> None:
    text = ROUTERS_MD.read_text(encoding="utf-8", errors="replace")
    findings.doc_sections = [m.group("title").strip() for m in DOC_SECTION_RE.finditer(text)]
    findings.doc_claims = [
        (m.group("method"), m.group("path").strip()) for m in DOC_ROW_RE.finditer(text)
    ]
    api = API_MD.read_text(encoding="utf-8", errors="replace")
    m = API_COUNT_RE.search(api)
    if m:
        findings.api_md_counts = (int(m.group("routes")), int(m.group("routers")))


def collect_tests(findings: Findings) -> None:
    test_files = [p.name for d in TEST_DIRS if d.exists() for p in d.glob("test_*.py")]
    for stem in findings.routers_on_disk:
        key = stem.replace("_", "")
        if not any(key in f.replace("_", "").replace("-", "") for f in test_files):
            findings.untested.append(stem)


CLAIMS_BLOCK_RE = re.compile(r"<!--\s*parity-claims:v1\s*(?P<body>.*?)-->", re.DOTALL)


def _internal_stats(path: Path) -> tuple[list[int], set[str]]:
    """Return (line numbers, module names) of `domain.*_internal*` imports.

    Parses imports; never greps the token. Docstrings legitimately *mention*
    ``_internal`` to assert they do not import it (``tokenizer.py:6``,
    ``cloud_training.py:4``), which is what made the prose checklist drift.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return [], set()
    lines: list[int] = []
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("domain.") and "_internal" in node.module:
                lines.append(node.lineno)
                mods.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("domain.") and "_internal" in alias.name:
                    lines.append(node.lineno)
                    mods.add(alias.name)
    return lines, mods


def _py_files(directory: Path) -> list[Path]:
    return [p for p in sorted(directory.glob("*.py")) if not p.name.startswith("_")]


def collect_architecture(findings: Findings) -> None:
    """Diff the ``parity-claims`` contract block against measured code."""
    if CLAIMS_MD.exists():
        m = CLAIMS_BLOCK_RE.search(CLAIMS_MD.read_text(encoding="utf-8", errors="replace"))
        if not m:
            findings.arch_missing.append("no parity-claims block in " + CLAIMS_MD.name)
        else:
            for line in m.group("body").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, raw = line.partition("=")
                key = key.strip()
                try:
                    findings.arch_claims[key] = int(raw.strip())
                except ValueError:
                    findings.arch_missing.append(f"{key}: {raw.strip()!r} is not an int")
    else:
        findings.arch_missing.append(f"{CLAIMS_MD.name} missing")

    # --- routers (all, plus the in-scope subset that excludes the allowlist) ---
    r_files = r_stmts = scope_files = 0
    for p in _py_files(ROUTERS_DIR):
        lines, _ = _internal_stats(p)
        if not lines:
            continue
        r_files += 1
        r_stmts += len(lines)
        if p.stem not in HANDLER_ALLOWLIST:
            scope_files += 1

    # --- controllers ---
    c_files = c_lines = c_defs = c_ifiles = c_istmts = 0
    c_mods: set[str] = set()
    for p in _py_files(CONTROLLERS_DIR):
        text = p.read_text(encoding="utf-8", errors="replace")
        lines, mods = _internal_stats(p)
        c_files += 1
        c_lines += len(text.splitlines())
        try:
            tree = ast.parse(text)
            c_defs += sum(
                1 for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            )
        except SyntaxError:
            findings.arch_missing.append(f"{p.name}: unparseable")
        if lines:
            c_ifiles += 1
            c_istmts += len(lines)
            c_mods |= mods

    findings.arch_measured = {
        "routers_internal_files": r_files,
        "routers_internal_stmts": r_stmts,
        "routers_internal_in_scope_files": scope_files,
        "controllers_files": c_files,
        "controllers_lines": c_lines,
        "controllers_defs": c_defs,
        "controllers_internal_files": c_ifiles,
        "controllers_internal_stmts": c_istmts,
        "controllers_internal_modules": len(c_mods),
        "combined_internal_files": r_files + c_ifiles,
        "combined_internal_stmts": r_stmts + c_istmts,
    }

    for key, claimed in findings.arch_claims.items():
        if key not in findings.arch_measured:
            findings.arch_missing.append(f"{key}: claimed but not measurable")
            continue
        got = findings.arch_measured[key]
        if got != claimed:
            findings.arch_mismatch.append(f"{key}: doc={claimed} code={got}")
    for key in findings.arch_measured:
        if key not in findings.arch_claims:
            findings.arch_missing.append(f"{key}: measured but not claimed in docs")


def _mounted_routers() -> set[str] | None:
    """Router stems whose routes are actually served at runtime.

    Reads the generated mount table ``routers/_manifest.py`` (plus the legacy
    inline ``_router_names`` table, for a checkout predating it) and the
    routers registered directly in main.py pre-lifespan. Unmounted-by-design
    files (e.g. api_keys) still scan into findings, but duplicate paths there
    never reach the app, so they must not fail the collision gate.

    ``None`` = mount table unreadable; callers then treat every stem as
    mounted so the gate errs toward flagging.
    """
    for path, pattern in (
        (ROUTERS_DIR / "_manifest.py", r"ROUTER_MODULES[^=]*=\s*\((.*?)\)"),
        (ROUTERS_DIR / "__init__.py", r"_router_names\s*=\s*\[(.*?)\]"),
    ):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        m = re.search(pattern, text, re.DOTALL)
        if not m:
            continue
        names = set(re.findall(r"[\"']([^\"']+)[\"']", m.group(1)))
        names |= {"main", "health", "status"}  # registered directly in main.py
        return names
    return None


def diff(findings: Findings) -> None:
    code_keys = {(r.method, r.path) for r in findings.routes if not r.dynamic}
    findings.doc_not_in_code = [(m, p) for m, p in findings.doc_claims if (m, p) not in code_keys]

    # which routers have any doc section at all (explicit aliases first,
    # then normalized title<->stem match so new sections need no alias edit)
    section_stems = set()
    norm_stems = {_norm(s): s for s in findings.routers_on_disk}
    for title in findings.doc_sections:
        stem = SECTION_ALIASES.get(title) or norm_stems.get(_norm(title))
        if stem:
            section_stems.add(stem)
    findings.code_not_documented = [
        s
        for s in findings.routers_on_disk
        if s not in section_stems
        # main.py-registered routers are documented in Health/Status sections
        and s not in {"health", "status"}
        # unmounted by design (documented in PRODUCT_ENGINEERING Build Order)
        and s != "api_keys"
    ]
    findings.dynamic_routes = sorted(
        {f"{r.router} {r.method}" for r in findings.routes if r.dynamic}
    )

    # one owner per (METHOD, path): first-match wins at runtime while OpenAPI
    # is last-wins, so a duplicate silently forks spec from behaviour (the
    # /chat InferenceRouter-vs-ChatRouter split this gate now blocks). Only
    # mounted routers participate — unmounted duplicates are never served.
    owners: dict[tuple[str, str], set[str]] = {}
    for r in findings.routes:
        if r.dynamic:
            continue
        owners.setdefault((r.method, r.path), set()).add(r.router)
    mounted = _mounted_routers()
    findings.collisions = []
    for (m, p), stems in sorted(owners.items()):
        served = (
            stems
            if mounted is None
            else {s for s in stems if s in mounted or s.split("/")[0] in mounted}
        )
        if len(served) > 1:
            findings.collisions.append(f"{m} {p} -> {sorted(served)}")

    # reverse direction: routes in code with no row anywhere in routers.md
    doc_keys = {(m, p) for m, p in findings.doc_claims}
    seen: set[tuple[str, str, str]] = set()
    for r in findings.routes:
        if r.dynamic or (r.method, r.path) in doc_keys:
            continue
        key = (r.router, r.method, r.path)
        if key in seen:
            continue
        seen.add(key)
        findings.undocumented[r.router] = findings.undocumented.get(r.router, 0) + 1


def _render_routers_md(findings: Findings) -> str:
    """Regenerate docs/routers.md endpoint tables from code truth.

    Section prose and headings are preserved; row descriptions carry over
    from the old table when the (METHOD, path) row survives, else fall back
    to the handler docstring. Sections are kept in file order; routers that
    had no section get a generated one before the OpenAPI tail.
    """
    old = ROUTERS_MD.read_text(encoding="utf-8", errors="replace")

    # carryover: old row descriptions (wrapping continuation lines merged)
    old_rows: dict[tuple[str, str], str] = {}
    cur: tuple[str, str] | None = None
    for line in old.splitlines():
        mrow = FULL_ROW_RE.match(line)
        if mrow:
            cur = (mrow.group("m"), mrow.group("p").strip())
            old_rows[cur] = mrow.group("d").strip().replace("\\|", "|")
        elif line.lstrip().startswith("|"):
            if cur and not re.match(r"^\|[\s\-:|]+$", line):
                parts = [q.strip() for q in line.strip().strip("|").split("|")]
                if len(parts) > 2 and parts[2]:
                    old_rows[cur] = (old_rows[cur] + " " + parts[2]).strip()
        else:
            cur = None

    by_stem: dict[str, list[Route]] = {}
    for r in findings.routes:
        if not r.dynamic:
            by_stem.setdefault(r.router, []).append(r)
    norm_stems = {_norm(s): s for s in findings.routers_on_disk}

    def section_stem(title: str) -> str | None:
        return SECTION_ALIASES.get(title) or norm_stems.get(_norm(title))

    def clean_desc(s: str) -> str:
        s = re.sub(r"\s+", " ", s).replace("|", "\\|").strip()
        return s[:160] + "\u2026" if len(s) > 160 else s

    def table(routes: list[Route]) -> str:
        rows = ["| Method | Path | Description |", "| --- | --- | --- |"]
        for r in sorted(routes, key=lambda x: (x.path, x.method)):
            desc = old_rows.get((r.method, r.path)) or r.desc or ""
            rows.append(f"| `{r.method}` | `{r.path}` | {clean_desc(desc)} |")
        return "\n".join(rows) + "\n\n"

    parts = re.split(r"(?m)^(?=## )", old)
    out: list[str] = [parts[0]]
    covered: set[str] = set()
    openapi_at: int | None = None
    for sec in parts[1:]:
        tm = DOC_SECTION_RE.match(sec)
        title = tm.group("title").strip() if tm else ""
        stem = section_stem(title)
        if stem and stem in by_stem:
            covered.add(stem)
            lines = sec.splitlines(keepends=True)
            if not any(l.startswith("|") for l in lines):
                out.append(sec)
                continue
            # segment into prose/row blocks: prose is preserved verbatim
            # (incl. notes sitting between a section's tables), the first row
            # block is replaced by the regenerated table, later row blocks
            # (stale extra tables) are dropped.
            blocks: list[tuple[bool, str]] = []
            rows_flag: bool | None = None
            buf: list[str] = []
            for line in lines[1:]:
                is_row = line.startswith("|")
                if rows_flag is None or is_row == rows_flag:
                    buf.append(line)
                    rows_flag = is_row
                else:
                    blocks.append((rows_flag, "".join(buf)))
                    buf = [line]
                    rows_flag = is_row
            if buf:
                blocks.append((rows_flag, "".join(buf)))
            body: list[str] = []
            table_done = False
            seen_row = False
            for is_row, text in blocks:
                if is_row:
                    seen_row = True
                    if not table_done:
                        body.append(table(by_stem[stem]))
                        table_done = True
                else:
                    body.append(text.lstrip("\n") if seen_row else text)
            # normalize the section ending (trailing blank-line prose blocks
            # after the table would otherwise grow on every re-render)
            out.append((lines[0] + "".join(body)).rstrip("\n") + "\n\n")
        else:
            if (
                sec.startswith("## OpenAPI")
                and "main" in by_stem
                and not any(DOC_ROW_RE.match(l) for l in sec.splitlines())
            ):
                nl = sec.find("\n\n")
                at = nl + 2 if nl != -1 else len(sec)
                sec = sec[:at] + table(by_stem["main"]) + sec[at:]
            if sec.startswith("## OpenAPI"):
                openapi_at = len(out)
            out.append(sec)

    prefix_by_stem: dict[str, str] = {}
    for pf in sorted(ROUTERS_DIR.glob("*.py")):
        if pf.name.startswith("_"):
            continue
        pm = PREFIX_RE.search(pf.read_text(encoding="utf-8", errors="replace"))
        prefix_by_stem[pf.stem] = pm.group(1) if pm else ""

    new_sections: list[str] = []
    for stem in findings.code_not_documented:
        if stem in covered or stem not in by_stem:
            continue
        title = stem.replace("_", " ").title()
        prefix = prefix_by_stem.get(stem) or "/"
        new_sections.append(
            f"## {title} Router (`{prefix}`)\n\n"
            f"Defined in `apps/api/server/routers/{stem}.py` — "
            f"{len(by_stem[stem])} endpoints.\n\n" + table(by_stem[stem])
        )
    if new_sections:
        at = openapi_at if openapi_at is not None else len(out)
        out[at:at] = new_sections
    return "".join(out)


def caller_reachability() -> tuple[set[str], set[str], set[str]]:
    """Return ``(new_orphans, stale_baseline, pending)`` over tracked ``domain/``.

    See ``CALLERLESS_BASELINE`` for the method and its calibration. Stdlib only
    plus ``git ls-files`` for the tracked/untracked split — CI invokes this
    script bare (no PYTHONPATH), and the split is what keeps in-flight work
    from ever being classified dead.
    """

    def _git_files(*flags: str) -> list[Path]:
        out = subprocess.run(
            ["git", "ls-files", "-z", *flags, "--", "domain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return [REPO / p for p in out.split("\0") if p.endswith(".py")]

    def _modname(path: Path) -> str:
        return ".".join(path.relative_to(REPO).with_suffix("").parts)

    def _package_of(path: Path) -> str:
        parts = list(path.relative_to(REPO).with_suffix("").parts)
        return ".".join(parts[:-1]) if len(parts) > 1 else ""

    def _relative_base(pkg: str, level: int, module: str | None) -> str:
        parts = pkg.split(".") if pkg else []
        up = level - 1
        if up:
            parts = parts[: len(parts) - up] if up <= len(parts) else []
        if module:
            parts += module.split(".")
        return ".".join(parts)

    tracked = _git_files()
    pending_files = _git_files("--others", "--exclude-standard")
    file_nodes = {_modname(f) for f in tracked + pending_files}
    refs: dict[str, set[str]] = {n: set() for n in file_nodes}

    def _resolve(name: str | None) -> str | None:
        """Module file node for a module path: file, or package -> its init."""
        if not name:
            return None
        if name in file_nodes:
            return name
        init = f"{name}.__init__"
        return init if init in file_nodes else None

    def _add(target: str, source: str) -> None:
        if target == source:
            return
        refs[target].add(source)
        # importing X.y.z also loads every ancestor package init
        parts = target.split(".")
        for i in range(1, len(parts)):
            anc = ".".join(parts[:i]) + ".__init__"
            if anc in file_nodes and anc != source:
                refs[anc].add(source)

    def _bindings(init_file: Path) -> dict[str, str]:
        """name -> module it is actually bound from (facade re-export resolution)."""
        try:
            tree = ast.parse(init_file.read_text(encoding="utf-8", errors="ignore"))
        except (SyntaxError, OSError):
            return {}
        pkg = _package_of(init_file)
        table: dict[str, str] = {}
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                base = (
                    _relative_base(pkg, node.level, node.module)
                    if node.level
                    else (node.module or "")
                )
                for a in node.names:
                    if a.name != "*":
                        table[a.asname or a.name] = base
            elif isinstance(node, ast.Import):
                for a in node.names:
                    if a.asname:
                        table[a.asname] = a.name
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        table[t.id] = ""
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                table[node.target.id] = ""
        return table

    def _star_sources(init_file: Path) -> list[str]:
        if init_file.name != "__init__.py":
            return []
        try:
            tree = ast.parse(init_file.read_text(encoding="utf-8", errors="ignore"))
        except (SyntaxError, OSError):
            return []
        return [
            n.module
            for n in tree.body
            if isinstance(n, ast.ImportFrom)
            and not n.level
            and n.module
            and any(a.name == "*" for a in n.names)
        ]

    def _scan_imports(path: Path, source: str) -> None:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except (SyntaxError, OSError):
            return
        pkg = _package_of(path)
        bind_cache: dict[str, dict[str, str]] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.startswith("domain"):
                        target = _resolve(a.name)
                        if target:
                            _add(target, source)
            elif isinstance(node, ast.ImportFrom):
                base = (
                    _relative_base(pkg, node.level, node.module)
                    if node.level
                    else (node.module or "")
                )
                if not base.startswith("domain"):
                    continue
                direct = _resolve(base)
                if direct:
                    _add(direct, source)
                base_init = REPO.joinpath(*base.split(".")) / "__init__.py"
                for a in node.names:
                    if a.name == "*":
                        for star in _star_sources(base_init):
                            target = _resolve(star)
                            if target:
                                _add(target, source)
                        continue
                    tbl = bind_cache.get(base)
                    if tbl is None:
                        tbl = _bindings(base_init) if base_init.is_file() else {}
                        bind_cache[base] = tbl
                    target = _resolve(tbl.get(a.name))  # facade re-export hop
                    if target:
                        _add(target, source)
                    sub = _resolve(f"{base}.{a.name}")  # `from pkg import sub`
                    if sub:
                        _add(sub, source)

    # Importers: every tracked/pending domain module plus production code
    # (apps/ scripts/ packages/ services/ and root-level .py). tests/ is NOT
    # scanned — §7: a test is a weak root, so test-only modules stay flagged.
    prod_files = list(tracked) + list(pending_files)
    for root in ("apps", "scripts", "packages", "services"):
        d = REPO / root
        if d.exists():
            prod_files += [
                p
                for p in d.rglob("*.py")
                if "__pycache__" not in p.parts and "node_modules" not in p.parts
            ]
    prod_files += list(REPO.glob("*.py"))

    string_ref_re = re.compile(r"(?<![\w.])domain(?:\.[A-Za-z_]\w*)+")
    # This script AUTHORS CALLERLESS_BASELINE: its own source names every
    # baselined module as a string literal, so scanning it would reference the
    # baseline into clearing itself (measured: all 52 entries got their only
    # edge from <scripts.check_docs_api_parity>). Exclude it from the string
    # pass only — its AST imports (stdlib, no domain) still count.
    self_source = Path(__file__).resolve()
    for f in prod_files:
        try:
            rel = f.relative_to(REPO)
        except ValueError:
            continue
        source = (
            _modname(f)
            if rel.parts[0] == "domain"
            else "<" + ".".join(rel.with_suffix("").parts) + ">"
        )
        _scan_imports(f, source)
        # string-literal dotted refs from non-test production code
        if rel.parts[0] == "tests" or f == self_source:
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for ref in set(string_ref_re.findall(text)):
            parts = ref.split(".")
            for i in range(len(parts), 0, -1):
                target = _resolve(".".join(parts[:i]))
                if target:
                    if target != source:
                        refs[target].add(source)
                    break

    callerless = {n for n, r in refs.items() if not r}
    pending = {_modname(f) for f in pending_files}
    flagged = callerless - pending  # untracked = pending, never dead
    return flagged - CALLERLESS_BASELINE, CALLERLESS_BASELINE - flagged, pending


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--markdown", action="store_true", help="markdown report fragment")
    ap.add_argument(
        "--render-routers",
        action="store_true",
        help="rewrite docs/routers.md endpoint tables from code truth",
    )
    args = ap.parse_args()

    if not ROUTERS_DIR.exists() or not ROUTERS_MD.exists():
        print(f"setup error: missing {ROUTERS_DIR} or {ROUTERS_MD}", file=sys.stderr)
        return 2

    # A TEST_DIRS entry that has been moved or renamed used to be skipped
    # silently by collect_tests' `d.exists()` filter, which is how 22 tested
    # routers came back as untested. Fail loudly instead of quietly searching
    # fewer files.
    missing_dirs = [str(d) for d in TEST_DIRS if not d.exists()]
    if missing_dirs:
        print(f"setup error: TEST_DIRS entry missing: {missing_dirs}", file=sys.stderr)
        return 2

    findings = Findings()
    collect_code(findings)
    collect_docs(findings)
    collect_tests(findings)
    collect_architecture(findings)
    diff(findings)
    reach_new, reach_stale, reach_pending = caller_reachability()

    if args.render_routers:
        ROUTERS_MD.write_text(_render_routers_md(findings), encoding="utf-8")
        # re-collect so the exit code reflects the rewritten doc
        f2 = Findings()
        collect_code(f2)
        collect_docs(f2)
        collect_tests(f2)
        collect_architecture(f2)
        diff(f2)
        gaps2 = bool(
            f2.doc_not_in_code
            or f2.code_not_documented
            or f2.untested
            or f2.collisions
            or f2.arch_mismatch
            or f2.arch_missing
            or reach_new
        )
        print(
            f"rendered {ROUTERS_MD.name}: {len(f2.doc_claims)} rows, "
            f"{len(f2.doc_sections)} sections, dead={len(f2.doc_not_in_code)}, "
            f"no-section={len(f2.code_not_documented)}, untested={len(f2.untested)}, "
            f"collisions={len(f2.collisions)}, new_orphans={len(reach_new)}"
        )
        return 1 if gaps2 else 0

    literal = [r for r in findings.routes if not r.dynamic]
    actual_routes = len(literal)
    actual_routers = len(findings.routers_on_disk)
    gaps = bool(
        findings.doc_not_in_code
        or findings.code_not_documented
        or findings.untested
        or findings.collisions
        or findings.arch_mismatch
        or findings.arch_missing
        or reach_new
    )

    if args.json:
        print(
            json.dumps(
                {
                    "code": {
                        "routers": actual_routers,
                        "literal_routes": actual_routes,
                        "dynamic_routes": len(findings.dynamic_routes),
                    },
                    "api_md_claims": findings.api_md_counts,
                    "routers_md": {
                        "sections": len(findings.doc_sections),
                        "endpoint_rows": len(findings.doc_claims),
                    },
                    "code_not_documented": findings.code_not_documented,
                    "doc_not_in_code": findings.doc_not_in_code,
                    "undocumented_routes": findings.undocumented,
                    "untested_routers": findings.untested,
                    "dynamic": findings.dynamic_routes,
                    "route_collisions": findings.collisions,
                    "architecture": {
                        "claims": findings.arch_claims,
                        "measured": findings.arch_measured,
                        "mismatch": findings.arch_mismatch,
                        "missing": findings.arch_missing,
                    },
                    "reachability": {
                        "baseline": len(CALLERLESS_BASELINE),
                        "new_orphans": sorted(reach_new),
                        "stale_baseline": sorted(reach_stale),
                        "pending_untracked": sorted(reach_pending),
                    },
                },
                indent=2,
            )
        )
        return 1 if gaps else 0

    if args.markdown:
        print(_render_markdown(findings, actual_routes, actual_routers))
        return 1 if gaps else 0

    print(
        f"code:      {actual_routers} router files, {actual_routes} literal routes "
        f"({len(findings.dynamic_routes)} dynamic)"
    )
    if findings.api_md_counts:
        r, k = findings.api_md_counts
        print(
            f"docs/API.md claims:      {r} routes across {k} routers "
            f"-> {'STALE' if (r, k) != (actual_routes, actual_routers) else 'ok'}"
        )
    print(
        f"docs/routers.md:         {len(findings.doc_sections)} sections, "
        f"{len(findings.doc_claims)} endpoint rows"
    )
    print(
        f"routers w/o doc section: {len(findings.code_not_documented)} "
        f"{findings.code_not_documented}"
    )
    print(f"doc rows not in code:    {len(findings.doc_not_in_code)}")
    for m, p in findings.doc_not_in_code[:10]:
        print(f"   {m:6} {p}")
    if len(findings.doc_not_in_code) > 10:
        print(f"   ... +{len(findings.doc_not_in_code) - 10} more")
    total_undoc = sum(findings.undocumented.values())
    print(f"code routes w/o doc row: {total_undoc} across {len(findings.undocumented)} routers")
    for name, n in sorted(findings.undocumented.items(), key=lambda kv: -kv[1])[:8]:
        print(f"   {name}: {n}")
    print(f"routers w/o tests:       {len(findings.untested)} {findings.untested}")
    print(f"route collisions:        {len(findings.collisions)}")
    for c in findings.collisions[:10]:
        print(f"   {c}")
    if len(findings.collisions) > 10:
        print(f"   ... +{len(findings.collisions) - 10} more")
    if findings.dynamic_routes:
        print(f"dynamic (skipped):       {findings.dynamic_routes}")
    print(
        f"architecture contract:    {len(findings.arch_claims)} claims, "
        f"{len(findings.arch_mismatch)} mismatched, "
        f"{len(findings.arch_missing)} unresolved"
    )
    for row in findings.arch_mismatch:
        print(f"   ALARM  {row}")
    for row in findings.arch_missing:
        print(f"   ALARM  {row}")
    print(
        f"caller reachability:      baseline {len(CALLERLESS_BASELINE)}, "
        f"{len(reach_new)} new orphans, {len(reach_stale)} stale, "
        f"{len(reach_pending)} untracked pending"
    )
    for m in sorted(reach_new):
        print(f"   ORPHAN  {m}  (gained zero callers — wire it, delete it, or baseline it)")
    for m in sorted(reach_stale):
        print(f"   stale   {m}  (gained a caller — remove from CALLERLESS_BASELINE)")
    return 1 if gaps else 0


def _render_markdown(f: Findings, actual_routes: int, actual_routers: int) -> str:
    lines = [
        "### Measured parity (scripts/check_docs_api_parity.py)",
        "",
        f"- Code: **{actual_routers}** router files, **{actual_routes}** literal routes "
        f"({len(f.dynamic_routes)} dynamic — not diffed).",
        f"- `docs/API.md` claims: **{f.api_md_counts}** vs measured "
        f"{(actual_routes, actual_routers)} → stale."
        if f.api_md_counts and f.api_md_counts != (actual_routes, actual_routers)
        else f"- `docs/API.md` claims: {f.api_md_counts} vs measured "
        f"{(actual_routes, actual_routers)} → ok.",
        f"- `docs/routers.md`: {len(f.doc_sections)} sections, "
        f"{len(f.doc_claims)} endpoint rows; **{len(f.doc_not_in_code)}** rows no "
        "longer match a code route (drift),",
        f"- **{len(f.code_not_documented)}** mounted routers have no section in "
        f"`docs/routers.md`: `{', '.join(f.code_not_documented)}`.",
        f"- **{len(f.untested)}** routers have no test file in either suite: "
        f"`{', '.join(f.untested)}`.",
        f"- **{sum(f.undocumented.values())}** code routes have no endpoint row in "
        f"`docs/routers.md` (across {len(f.undocumented)} routers).",
        f"- **{len(f.collisions)}** route collisions (same method+path in 2+ routers)"
        + (f": {'; '.join(f.collisions[:5])}." if f.collisions else " — one owner each."),
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
