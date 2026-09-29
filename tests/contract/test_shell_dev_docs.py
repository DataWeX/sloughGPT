"""Contract: shell + developer guide docs stay truthful against the code.

Guards docs/SHELL.md and docs/DEVELOPER_GUIDE.md — the drift class found in
the 2026-09-29 seam-7 audit:

- flat module trees (domain/shell/ now lives under domain/shell/_internal/),
- stale method/test counts ("22 API methods", "148 tests", 30-row
  self-reported coverage table),
- dead API rows (`/knowledge/list`, `/training/jobs/{id}/stop` — the real
  param name is {job_id}),
- dead repo paths and examples (iamtowbee clone URL, `make dev-stack`,
  tests/test_knowledge_memory.py, config/development.py, config/production.py,
  `source = ["domains"]` coverage config, flat-tests/flat-docs claims,
  Node 20 vs .nvmrc 22),
- metrics examples calling methods that do not exist (record_training),
- broken/orphaned code fences and fragments.

Any change to the real infrastructure that invalidates these docs must update
them in the same change.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_DOCS = (
    _REPO / "docs" / "SHELL.md",
    _REPO / "docs" / "DEVELOPER_GUIDE.md",
)
_SHELL = _DOCS[0]
_DEVGUIDE = _DOCS[1]
_PATH_PREFIXES = (
    "tests/",
    "scripts/",
    "apps/",
    "packages/",
    "domain/",
    "docs/",
    ".github/",
    "config/",
    "infra/",
    "data/",
)
_SKIP_SUBSTRINGS = ("*", " ", "=", "\t")
_NEGATIVE_CONTEXT = (
    "removed",
    "deleted",
    "superseded",
    "no shims",
    "stub",
    "formerly",
    "moved to",
    "predate",
)

_STALE_CLAIMS = {
    "iamtowbee": "clone URL owner drifted (remote is DataWeX/sloughGPT)",
    "make dev-stack": "Makefile has no dev-stack target (use make stack)",
    "/knowledge/list": "route is GET /knowledge now",
    "/training/jobs/{id}/stop": "openapi param is {job_id}",
    "22 API methods": "ShellCommands has grown (count: @staticmethods in commands.py)",
    "22 API wrappers": "ShellCommands has grown (count: @staticmethods in commands.py)",
    "148 tests": "stale self-reported count",
    "(30 tests": "stale self-reported count",
    "tests/test_knowledge_memory.py": "file does not exist",
    'source = ["domains"]': "coverage source is packages/core-py + domain",
    "config/development.py": "file does not exist",
    "config/production.py": "file does not exist",
    "no subdirectories": "tests/ has server/ and contract/ subdirs",
    "Node.js**: 20": ".nvmrc pins Node 22",
    "Tests are flat": "tests/ has subdirectories",
    "live flat in": "docs/ has subdirectories (plans/, policies/, …)",
    "record_training": "ServerState has no record_training method",
    "db_manager.execute_query": "orphan fragment with no fence/opener",
    "single Python module": "domain/shell is a package (_internal/), not one module",
    "postgresql://": "stack runs MogDB, not postgres",
    "for internal calls and": "dup httpx wording",
}


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_referenced_paths_exist() -> None:
    missing: list[str] = []
    for doc in _DOCS:
        for line in _text(doc).splitlines():
            low = line.lower()
            if any(ctx in low for ctx in _NEGATIVE_CONTEXT):
                continue
            for token in re.findall(r"`([^`\n]+)`", line):
                token = token.strip().rstrip(".,;:")
                if any(s in token for s in _SKIP_SUBSTRINGS):
                    continue
                if token.startswith("./"):
                    token = token[2:]
                if not token.startswith(_PATH_PREFIXES):
                    continue
                if not (_REPO / token).exists():
                    missing.append(f"{doc.name}: {token}")
    assert not missing, f"docs reference dead paths: {missing}"


def test_markdown_links_resolve() -> None:
    broken: list[str] = []
    for doc in _DOCS:
        for t in re.findall(r"\]\(([^)#]+)\)", _text(doc)):
            if t.startswith(("http://", "https://", "mailto:")):
                continue
            if not (_REPO / t).exists():
                broken.append(f"{doc.name}: {t}")
    assert not broken, f"docs contain dead links: {broken}"


def test_no_stale_claims() -> None:
    hits = [
        f"{doc.name}: {claim!r} ({why})"
        for doc in _DOCS
        for claim, why in _STALE_CLAIMS.items()
        if claim in _text(doc)
    ]
    assert not hits, f"stale claims reintroduced: {hits}"


def _live_routes() -> set[tuple[str, str]]:
    spec = importlib.util.spec_from_file_location(
        "_shelldev_routes", _REPO / "tests" / "contract" / "test_routers_doc.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._live_routes()


def test_shell_api_table_serves() -> None:
    text = _text(_SHELL)
    start = text.index("## API Endpoints Used")
    end = text.index("\n---", start)
    section = text[start:end]
    live = _live_routes()
    dead: list[str] = []
    rows = 0
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 3 or not cells[2].strip("`").isupper():
            continue
        method = cells[2].strip("`")
        paths = re.findall(r"`(/[^`]+)`", cells[1])
        if not paths:
            continue
        rows += 1
        for path in paths:
            if (method, path) not in live:
                dead.append(f"{method} {path}")
    assert rows >= 20, f"expected the shell API table, parsed only {rows} rows"
    assert not dead, f"SHELL.md API table rows do not serve: {dead}"


def test_shell_tree_counts_and_commands() -> None:
    text = _text(_SHELL)
    problems: list[str] = []
    for rel in (
        "domain/shell/__init__.py",
        "domain/shell/_internal/kernel.py",
        "domain/shell/_internal/repl.py",
        "domain/shell/_internal/commands.py",
        "domain/shell/_internal/state.py",
        "domain/shell/_internal/permissions.py",
    ):
        if not (_REPO / rel).exists():
            problems.append(f"tree file missing on disk: {rel}")
        if Path(rel).name not in text:
            problems.append(f"tree no longer lists: {Path(rel).name}")
        if rel.startswith("domain/shell/_internal/") and "_internal/" not in text:
            problems.append("tree no longer shows the domain/shell/_internal/ layout")
    n_static = (
        _REPO / "domain" / "shell" / "_internal" / "commands.py"
    ).read_text(encoding="utf-8").count("@staticmethod")
    claims = [int(m) for m in re.findall(r"(\d+) API (?:methods|wrappers)", text)]
    if not claims or any(c != n_static for c in claims):
        problems.append(
            f"doc claims API method counts {claims}, actual @staticmethods={n_static}"
        )
    n_cmds = len(
        re.findall(
            r"^\s+def _cmd_\w+",
            (_REPO / "domain" / "shell" / "_internal" / "repl.py").read_text(
                encoding="utf-8"
            ),
            re.MULTILINE,
        )
    )
    if n_cmds < 40:
        problems.append(f"repl.py has only {n_cmds} _cmd_ handlers; doc claims 40+")
    assert not problems, problems


def test_devguide_node_version_matches_nvmrc() -> None:
    match = re.search(r"Node\.js\*\*:\s*(\d+)", _text(_DEVGUIDE))
    assert match, "DEVELOPER_GUIDE no longer states a Node.js version"
    nvmrc = int((_REPO / ".nvmrc").read_text(encoding="utf-8").strip())
    assert int(match.group(1)) == nvmrc, (
        f"DEVELOPER_GUIDE claims Node {match.group(1)}, .nvmrc pins {nvmrc}"
    )


def test_devguide_metrics_example_calls_real_methods() -> None:
    from domain.infrastructure.server_state import get_server_state

    state = get_server_state()
    called = set(re.findall(r"state\.(record_\w+)\(", _text(_DEVGUIDE)))
    assert called, "no ServerState example found in DEVELOPER_GUIDE"
    dead = sorted(name for name in called if not hasattr(state, name))
    assert not dead, f"DEVELOPER_GUIDE calls missing ServerState methods: {dead}"
