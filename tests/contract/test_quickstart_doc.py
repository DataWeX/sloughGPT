"""Contract: QUICKSTART.md stays truthful against the live CLI and API.

Guards the 5-minute entry doc — the drift class found in the 2026-09-29
seam-6 audit:

- dead top-level CLI forms (`./sloughgpt quick`, `export`, `api-status`, …
  now subcommands under train/model/system/dataset groups),
- dead repo paths and clone URLs (iamtowbee → DataWeX; optimized_trainer,
  torch_runtime, sou_format, ml_infrastructure, helm chart all gone),
- curl examples for routes that no longer serve (/cache, /metrics,
  /inference/batch, /metrics/prometheus),
- Makefile/npm targets that do not exist (`make dev-stack`, `make help`),
- the file-structure tree (bare filenames in the tree were never covered by
  the prefix sweep — pinned here).

Any change to the real infrastructure that invalidates this doc must update
it in the same change.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_DOC = _REPO / "QUICKSTART.md"
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

# Commands that existed when the doc was written but are gone (or only live
# as group subcommands now).
_DEAD_COMMANDS = frozenset(
    {
        "quick",
        "eval",
        "export",
        "datasets",
        "models",
        "personalities",
        "info",
        "api-status",
        "api-test",
        "api-auth",
        "compare",
        "config",
        "data",
        "benchmark",
        "optimize",
        "hf-download",
    }
)

_STALE_CLAIMS = {
    "iamtowbee": "clone/links URL owner drifted (remote is DataWeX/sloughGPT)",
    "make dev-stack": "Makefile has no dev-stack target (use make stack)",
    "make help": "Makefile has no help target",
    "domain.torch_runtime": "module removed",
    "optimized_trainer": "file removed (use train quick --preset)",
    "sou_format.py": "file is domain/inference/_internal/slo_format.py",
    "ml_infrastructure": "package removed",
    "nanogpt.py": "file removed",
    "helm": "infra/k8s has no helm chart",
    "/cache/stats": "route removed",
    "/metrics/prometheus": "route removed",
    "/inference/batch": "route removed",
    "api-status": "top-level command removed",
    "api-test": "top-level command removed",
    "api-auth": "top-level command removed",
    "`tiny`": "dataset no longer ships in data/",
}

# Pinned file-structure tree entries (the tree uses bare filenames, which the
# prefix-based path sweep cannot see).
_TREE_CLAIMS = (
    "cli.py",
    "apps/api/server/main.py",
    "infra/k8s",
    "infra/k8s/k8s",
    "infra/docker/docker-compose.yml",
    "tests",
    "data",
    "sloughgpt_colab.ipynb",
    "domain/training/engine.py",
    "domain/training/tokenizer_engine.py",
    "domain/training/_internal/train_pipeline.py",
    "domain/inference/_internal/slo_format.py",
)

# Swagger UI shell is not an API path (app-level, not in openapi.json).
_CURL_SKIP = frozenset({"/docs"})


def _text() -> str:
    return _DOC.read_text(encoding="utf-8")


def test_referenced_paths_exist() -> None:
    missing: list[str] = []
    for line in _text().splitlines():
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
                missing.append(token)
    assert not missing, f"QUICKSTART.md references dead paths: {missing}"


def test_markdown_links_resolve() -> None:
    broken = [
        t
        for t in re.findall(r"\]\(([^)#]+)\)", _text())
        if not t.startswith(("http://", "https://", "mailto:")) and not (_REPO / t).exists()
    ]
    assert not broken, f"QUICKSTART.md contains dead links: {broken}"


def test_no_stale_claims_or_dead_commands() -> None:
    text = _text()
    hits = [f"{c!r}" for c, why in _STALE_CLAIMS.items() if c in text for _ in (why,)]
    dead_cmds = sorted(
        {
            m.group(1)
            for m in re.finditer(r"\./sloughgpt ([a-z][a-z-]*)", text)
            if m.group(1) in _DEAD_COMMANDS
        }
    )
    assert not dead_cmds, (
        f"QUICKSTART.md calls removed top-level commands: {dead_cmds} — "
        "they live under train/model/system/dataset groups now"
    )
    assert not hits, f"stale claims reintroduced in QUICKSTART.md: {hits}"


def _live_routes() -> set[tuple[str, str]]:
    spec = importlib.util.spec_from_file_location(
        "_quickstart_routes", _REPO / "tests" / "contract" / "test_routers_doc.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._live_routes()


def test_curl_examples_serve() -> None:
    paths = {p for _, p in _live_routes()}
    urls = set(re.findall(r"http://localhost:8000(/[^\s\\\"'`)]*)", _text()))
    assert urls, "no curl examples found in QUICKSTART.md"
    dead: list[str] = []
    for path in sorted(urls):
        if path in _CURL_SKIP:
            continue
        if not any(r == path or r.startswith(path.rstrip("/") + "/") for r in paths):
            dead.append(path)
    assert not dead, f"QUICKSTART.md curls hit dead routes: {dead}"


def test_shell_and_npm_targets_exist() -> None:
    text = _text()
    problems: list[str] = []
    makefile = (_REPO / "Makefile").read_text(encoding="utf-8")
    for name in sorted(set(re.findall(r"\bmake ([a-z0-9][a-z0-9-]*)", text))):
        if not re.search(rf"^{re.escape(name)}:", makefile, re.MULTILINE):
            problems.append(f"Makefile target missing: {name}")
    scripts = json.loads((_REPO / "package.json").read_text(encoding="utf-8")).get("scripts", {})
    for name in sorted(set(re.findall(r"`([a-z]+:[a-z-]+)`", text))):
        if name not in scripts:
            problems.append(f"package.json script missing: {name}")
    assert not problems, problems


def test_tree_claims_exist() -> None:
    problems = [f"missing: {n}" for n in _TREE_CLAIMS if not (_REPO / n).exists()]
    if not (_REPO / "datasets").exists():
        line = next(
            (ln for ln in _text().splitlines() if re.search(r"[├└]── datasets/", ln)),
            "",
        )
        if "runtime" not in line:
            problems.append(
                "tree lists datasets/ but the directory does not exist and the "
                "entry does not mark it runtime"
            )
    assert not problems, problems
