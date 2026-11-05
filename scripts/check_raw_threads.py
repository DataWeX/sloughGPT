#!/usr/bin/env python3
"""Raw-thread lint hint — stage 2 of docs/plans/2026-09-20-execution-consolidation.md.

Scans ``apps/`` + ``domain/`` production files for raw ``threading.Thread``
call sites (any import spelling: ``threading.Thread(``, ``_threading.Thread(``,
``from threading import Thread``) and checks them against the allowlist
(``scripts/raw_threads_allowlist.json``). A site outside the allowlist, a
declared ``count`` that drifted, a schema-less entry, or a stale entry all
fail the check — the grandfathered set moves only by explicit edit.

The same scan backs ``tests/core-py/test_raw_thread_gate.py`` (CI gate);
run this script directly for the developer-facing hint:

    python3 scripts/check_raw_threads.py
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

_KINDS = {"grandfathered", "deferred", "followup"}
_SKIP_DIRS = {"tests", "__pycache__", ".git", "node_modules", "venv", ".venv"}
_SCAN_ROOTS = ("apps", "domain")


def _is_test_path(rel: Path) -> bool:
    if rel.name.startswith("test_") or rel.name == "conftest.py":
        return True
    return any(part in _SKIP_DIRS for part in rel.parts[:-1])


def _thread_bindings(tree: ast.AST) -> tuple[set[str], set[str]]:
    """Names bound to the threading module / the Thread class, anywhere in the file."""
    modules: set[str] = set()
    classes: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "threading":
                    modules.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "threading":
            for alias in node.names:
                if alias.name == "Thread":
                    classes.add(alias.asname or alias.name)
    return modules, classes


def _is_thread_call(node: ast.Call, modules: set[str], classes: set[str]) -> bool:
    fn = node.func
    if isinstance(fn, ast.Attribute) and fn.attr == "Thread":
        return isinstance(fn.value, ast.Name) and fn.value.id in modules
    if isinstance(fn, ast.Name):
        return fn.id in classes
    return False


def _collect_sites(path: Path, root: Path) -> list[dict]:
    rel = path.relative_to(root)
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError) as exc:  # pragma: no cover - corrupt file
        raise SystemExit(f"cannot parse {rel}: {exc}") from exc
    modules, classes = _thread_bindings(tree)
    if not modules and not classes:
        return []

    sites: list[dict] = []
    stack: list[str] = []

    def visit(node: ast.AST) -> None:
        if isinstance(node, ast.Call) and _is_thread_call(node, modules, classes):
            sites.append(
                {
                    "path": rel.as_posix(),
                    "qualname": ".".join(stack) or "<module>",
                    "line": node.lineno,
                }
            )
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            stack.append(node.name)
            for child in ast.iter_child_nodes(node):
                visit(child)
            stack.pop()
            return
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(tree)
    return sites


def _validate_entry(entry: object) -> str | None:
    if not isinstance(entry, dict):
        return "entry must be an object"
    reason = entry.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        return "missing reason"
    if entry.get("kind") not in _KINDS:
        return f"kind must be one of {sorted(_KINDS)}"
    if not isinstance(entry.get("count"), int) or entry["count"] < 0:
        return "count must be a non-negative int"
    return None


def scan(root: Path, allowlist: dict) -> dict:
    """Scan ``root``'s apps/ + domain/ against the allowlist.

    Returns ``{"violations": [...], "site_count": N}``. One violation per
    grouped site (``path::qualname``) or per unmatched allowlist entry.
    """
    sites: list[dict] = []
    for scan_root in _SCAN_ROOTS:
        base = root / scan_root
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            if _is_test_path(path.relative_to(root)):
                continue
            sites.extend(_collect_sites(path, root))

    groups: dict[tuple[str, str], list[dict]] = {}
    for site in sites:
        groups.setdefault((site["path"], site["qualname"]), []).append(site)

    violations: list[dict] = []
    for (rel, qualname), hits in sorted(groups.items()):
        key = f"{rel}::{qualname}"
        entry = allowlist.get(key)
        if entry is None:
            violations.append(
                {
                    "path": rel,
                    "qualname": qualname,
                    "declared": None,
                    "actual": len(hits),
                    "lines": [h["line"] for h in hits],
                    "how": "unlisted",
                }
            )
            continue
        schema_err = _validate_entry(entry)
        if schema_err:
            violations.append(
                {
                    "path": rel,
                    "qualname": qualname,
                    "schema": schema_err,
                    "actual": len(hits),
                    "how": "schema",
                }
            )
            continue
        if entry["count"] != len(hits):
            violations.append(
                {
                    "path": rel,
                    "qualname": qualname,
                    "declared": entry["count"],
                    "actual": len(hits),
                    "lines": [h["line"] for h in hits],
                    "how": "count-drift",
                }
            )

    seen_keys = {f"{p}::{q}" for (p, q) in groups}
    for key, entry in sorted(allowlist.items()):
        if key in seen_keys:
            continue
        rel = key.split("::", 1)[0]
        path_exists = (root / rel).exists()
        schema_err = _validate_entry(entry)
        violations.append(
            {
                "path": rel,
                "qualname": key.split("::", 1)[1] if "::" in key else "",
                "stale": not path_exists,
                "declared": entry.get("count") if isinstance(entry, dict) else None,
                "actual": 0,
                "schema": schema_err,
                "how": "stale" if path_exists is False else "count-drift",
            }
        )

    return {"violations": violations, "site_count": len(sites)}


def load_allowlist(root: Path) -> dict:
    path = root / "scripts" / "raw_threads_allowlist.json"
    if not path.exists():
        return {}
    doc = json.loads(path.read_text(encoding="utf-8"))
    return doc.get("entries", doc) if isinstance(doc, dict) else {}


def _format(violation: dict) -> str:
    loc = f"{violation['path']}::{violation['qualname']}"
    how = violation.get("how", "?")
    if how == "unlisted":
        return (
            f"{loc} (line {violation['lines'][0]}) — raw thread not in allowlist; "
            "migrate per docs/plans/2026-09-20-execution-consolidation.md or "
            "add an entry to scripts/raw_threads_allowlist.json"
        )
    if how == "schema":
        return f"{loc} — allowlist entry invalid: {violation['schema']}"
    if how == "count-drift":
        return (
            f"{loc} — declared count {violation['declared']} but found "
            f"{violation['actual']}; shrink/grow the allowlist entry to match "
            "(migrations shrink, new sites are not allowed)"
        )
    return f"{loc} — stale allowlist entry (no matching sites); delete it"


def main(argv: list[str]) -> int:
    root = Path(__file__).resolve().parents[1]
    if len(argv) > 1:
        root = Path(argv[1])
    report = scan(root, load_allowlist(root))
    for violation in report["violations"]:
        print(f"RAW-THREAD: {_format(violation)}")
    if report["violations"]:
        print(
            f"{len(report['violations'])} violation(s); "
            f"{report['site_count']} grandfathered site(s) scanned"
        )
        return 1
    print(f"ok — {report['site_count']} allowlisted raw-thread site(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
