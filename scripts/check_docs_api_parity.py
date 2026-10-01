#!/usr/bin/env python3
"""Doc-vs-code parity check for the API surface.

Cross-references three sources and reports the gaps:

1. **Code truth**  — route decorators in ``apps/api/server/routers/*.py``
   (plus ``main.py`` app-level routes), joined with each router's
   ``APIRouter(prefix=...)``.
2. **Doc claims**  — the endpoint tables in ``docs/routers.md`` and the
   count claims in ``docs/API.md`` (``N routes across M routers``).
3. **Test coverage** — ``tests/server/`` + ``apps/api/server/tests/`` files
   per router (fuzzy name match).

Findings (docs claim but code lacks / code has but docs lack / stale counts
/ untested routers) are what the gap-audit report turns into cards.

Usage::

    python scripts/check_docs_api_parity.py             # human summary
    python scripts/check_docs_api_parity.py --json      # machine output
    python scripts/check_docs_api_parity.py --markdown  # report fragment

Exit codes: 0 = parity, 1 = gaps found, 2 = parse/setup error.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROUTERS_DIR = REPO / "apps" / "api" / "server" / "routers"
MAIN_PY = REPO / "apps" / "api" / "server" / "main.py"
TRAINING_ROUTER = REPO / "apps" / "api" / "server" / "training" / "router.py"
ROUTERS_MD = REPO / "docs" / "routers.md"
API_MD = REPO / "docs" / "API.md"
TEST_DIRS = (REPO / "tests" / "server", REPO / "apps" / "api" / "server" / "tests")

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
    r'@(?P<chain>(?:\w+\.)+\w+)\.(?P<method>get|post|put|delete|patch|head|options)'
    r'\(\s*(?P<q>["\'])(?P<path>[^"\']*)(?P=q)',
    re.MULTILINE,
)
ADD_ROUTE_RE = re.compile(r"(?P<recv>\w+)\.add_api_route\(")
# path may be positional or path="<x>" (keyword form) — anchored to the call's
# own "(" so a non-literal first arg can't match some later quoted string.
ADD_ROUTE_PATH_RE = re.compile(
    r"^.*?\(\s*(?:path\s*=\s*)?(?P<q>[\"'])(?P<path>[^\"']*)(?P=q)"
)
ADD_ROUTE_METHODS_RE = re.compile(r"methods\s*=\s*\[(?P<ms>[^\]]*)\]")
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


def _join(prefix: str, path: str) -> str:
    """Join an APIRouter prefix with a route path (FastAPI semantics)."""
    if not path:
        return prefix or "/"
    return (prefix.rstrip("/") + "/" + path.lstrip("/")) if prefix else "/" + path.lstrip("/")


def _scan_file(path: Path, router_stem: str, out: list[Route]) -> None:
    """Extract routes from one file in any of the three registration styles."""
    text = path.read_text(encoding="utf-8", errors="replace")

    prefixes: dict[str, str] = {}
    for m in APICONFIG_RE.finditer(text):
        pm = PREFIX_RE.search(m.group("args"))
        prefixes[m.group("var")] = pm.group(1) if pm else ""

    # style 1: class-based add_api_route — bounded per call so a call without
    # methods= can't absorb the next call's methods list.
    positions = list(ADD_ROUTE_RE.finditer(text))
    for i, m in enumerate(positions):
        end = positions[i + 1].start() if i + 1 < len(positions) else len(text)
        chunk = text[m.start():end]
        pm = ADD_ROUTE_PATH_RE.search(chunk)
        if not pm:
            out.append(Route(router_stem, "?", "<dynamic>", dynamic=True))
            continue
        mm = ADD_ROUTE_METHODS_RE.search(chunk)
        methods = (re.findall(r"[\"']([A-Za-z]+)[\"']", mm.group("ms")) if mm else []) or ["GET"]
        full = _join(prefixes.get(m.group("recv"), ""), pm.group("path"))
        for method in methods:
            out.append(Route(router_stem, method.upper(), full))

    # styles 2+3: decorators on router/app (incl. @self.router.get chains)
    for m in ROUTE_RE.finditer(text):
        key = m.group("chain").rsplit(".", 1)[-1]
        if key not in prefixes and key not in {"app", "router"} and not key.endswith("router"):
            continue  # decorator on an unrelated object
        out.append(
            Route(router_stem, m.group("method").upper(), _join(prefixes.get(key, ""), m.group("path")))
        )

    # dynamic decorator paths — counted, not diffed
    for m in re.finditer(
        r"@(?:\w+\.)+(?P<method>get|post|put|delete|patch)\(\s*(?![\"'])", text
    ):
        out.append(Route(router_stem, m.group("method").upper(), "<dynamic>", dynamic=True))


def collect_code(findings: Findings) -> None:
    router_files = sorted(
        p for p in ROUTERS_DIR.glob("*.py") if p.name != "__init__.py"
    )
    findings.routers_on_disk = [p.stem for p in router_files]
    for p in router_files:
        _scan_file(p, p.stem, findings.routes)
    if MAIN_PY.exists():
        _scan_file(MAIN_PY, "main", findings.routes)
    if TRAINING_ROUTER.exists():
        _scan_file(TRAINING_ROUTER, "training/router", findings.routes)


def collect_docs(findings: Findings) -> None:
    text = ROUTERS_MD.read_text(encoding="utf-8", errors="replace")
    findings.doc_sections = [m.group("title").strip() for m in DOC_SECTION_RE.finditer(text)]
    findings.doc_claims = [
        (m.group("method"), m.group("path").strip())
        for m in DOC_ROW_RE.finditer(text)
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


def diff(findings: Findings) -> None:
    code_keys = {(r.method, r.path) for r in findings.routes if not r.dynamic}
    findings.doc_not_in_code = [
        (m, p) for m, p in findings.doc_claims if (m, p) not in code_keys
    ]

    # which routers have any doc section at all
    section_stems = set()
    for title in findings.doc_sections:
        stem = SECTION_ALIASES.get(title)
        if stem:
            section_stems.add(stem)
    findings.code_not_documented = [
        s for s in findings.routers_on_disk
        if s not in section_stems
        # main.py-registered routers are documented in Health/Status sections
        and s not in {"health", "status"}
        # unmounted by design (documented in PRODUCT_ENGINEERING Build Order)
        and s != "api_keys"
    ]
    findings.dynamic_routes = sorted(
        {f"{r.router} {r.method}" for r in findings.routes if r.dynamic}
    )

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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--markdown", action="store_true", help="markdown report fragment")
    args = ap.parse_args()

    if not ROUTERS_DIR.exists() or not ROUTERS_MD.exists():
        print(f"setup error: missing {ROUTERS_DIR} or {ROUTERS_MD}", file=sys.stderr)
        return 2

    findings = Findings()
    collect_code(findings)
    collect_docs(findings)
    collect_tests(findings)
    diff(findings)

    literal = [r for r in findings.routes if not r.dynamic]
    actual_routes = len(literal)
    actual_routers = len(findings.routers_on_disk)
    gaps = bool(findings.doc_not_in_code or findings.code_not_documented or findings.untested)

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
                },
                indent=2,
            )
        )
        return 1 if gaps else 0

    if args.markdown:
        print(_render_markdown(findings, actual_routes, actual_routers))
        return 1 if gaps else 0

    print(f"code:      {actual_routers} router files, {actual_routes} literal routes "
          f"({len(findings.dynamic_routes)} dynamic)")
    if findings.api_md_counts:
        r, k = findings.api_md_counts
        print(f"docs/API.md claims:      {r} routes across {k} routers "
              f"-> {'STALE' if (r, k) != (actual_routes, actual_routers) else 'ok'}")
    print(f"docs/routers.md:         {len(findings.doc_sections)} sections, "
          f"{len(findings.doc_claims)} endpoint rows")
    print(f"routers w/o doc section: {len(findings.code_not_documented)} "
          f"{findings.code_not_documented}")
    print(f"doc rows not in code:    {len(findings.doc_not_in_code)}")
    for m, p in findings.doc_not_in_code[:10]:
        print(f"   {m:6} {p}")
    if len(findings.doc_not_in_code) > 10:
        print(f"   ... +{len(findings.doc_not_in_code) - 10} more")
    total_undoc = sum(findings.undocumented.values())
    print(f"code routes w/o doc row: {total_undoc} across "
          f"{len(findings.undocumented)} routers")
    for name, n in sorted(findings.undocumented.items(), key=lambda kv: -kv[1])[:8]:
        print(f"   {name}: {n}")
    print(f"routers w/o tests:       {len(findings.untested)} {findings.untested}")
    if findings.dynamic_routes:
        print(f"dynamic (skipped):       {findings.dynamic_routes}")
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
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
