#!/usr/bin/env python3
"""Detect ``@patch`` targets that can never reach the code under test.

The hazard: a package re-exports a name, production code reads it through the
package, and the test patches the *deep* module instead.

    domain/slolib/__init__.py:          from ._internal.gpu import get_accelerator
    apps/api/server/routers/models.py:  from domain.slolib import get_accelerator
    tests/.../test_models_router.py:    @patch("domain.slolib._internal.gpu.get_accelerator")

``mock.patch`` does ``setattr(domain.slolib._internal.gpu, "get_accelerator",
new)`` — it replaces the binding in the *source* module only.
``domain.slolib.get_accelerator`` was copied into the package namespace when
``__init__`` ran, so it keeps pointing at the original, and because
``models.py`` reads through the package it never sees the mock. The real GTX
1050 then answers ``cuda`` where the test asserts ``cpu``.

Two properties make this expensive to notice:

* It is host-dependent. On a CPU-only machine the real object agrees with the
  assertion, so the test passes and the stale target stays invisible.
* When it does not fail it fails *silently*: the mock never lands, the real
  object happens to satisfy the assertion, and the test passes while testing
  nothing about the code path it names.

This repo has hit it repeatedly — the stale ``os.rename`` patch (f0aabaa2c),
the precision test (8d855acca), and two live failures found by this script.
The rule is mechanical, so check it mechanically.

Advisory by default (exit 0) because the repo carries historical findings;
``--strict`` exits 1, for new code.
"""

from __future__ import annotations

import argparse
import ast
import json
import pathlib
import sys
from collections import defaultdict

REPO = pathlib.Path(__file__).resolve().parents[1]

# Production trees scanned for the immune read. Test trees are skipped.
_PROD_ROOTS = ("domain", "apps", "packages")


def _parse(path: pathlib.Path) -> ast.AST | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None


def _rel(path: pathlib.Path, repo: pathlib.Path) -> str:
    try:
        return str(path.relative_to(repo))
    except ValueError:
        return str(path)


def _patch_targets(test_file: pathlib.Path) -> set[str]:
    """Every string literal passed to ``patch(...)`` in this file."""
    tree = _parse(test_file)
    if tree is None:
        return set()
    found: set[str] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "id", "") == "patch"):
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and "." in arg.value:
                found.add(arg.value)
    return found


def _reexport_source(pkg_dir: pathlib.Path, attr: str) -> str | None:
    """If ``pkg_dir/__init__.py`` imports *attr* from a submodule, name that module.

    That import is the copy: once it runs, ``pkg.attr`` and
    ``submodule.attr`` are two independent bindings to one object.
    """
    init = pkg_dir / "__init__.py"
    if not init.exists():
        return None
    tree = _parse(init)
    if tree is None:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                if alias.name == attr and alias.asname in (None, attr):
                    return node.module
    return None


def _immune_readers(repo: pathlib.Path) -> dict[str, set[str]]:
    """``"pkg.attr" -> {production files doing `from pkg import attr`}``.

    These are the reads a deep-module patch cannot influence.
    """
    readers: dict[str, set[str]] = defaultdict(set)
    for root_name in _PROD_ROOTS:
        base = repo / root_name
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            if "/tests/" in str(path) or path.name.startswith("test_"):
                continue
            tree = _parse(path)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    for alias in node.names:
                        if alias.asname is None:
                            readers[f"{node.module}.{alias.name}"].add(_rel(path, repo))
    return readers


def _test_files(paths: list[str], repo: pathlib.Path) -> list[pathlib.Path]:
    if not paths:
        candidates = [
            repo / "tests",
            *[p for p in (repo / "packages").glob("*/tests") if p.exists()],
        ]
    else:
        candidates = [pathlib.Path(p) for p in paths]
    result: list[pathlib.Path] = []
    for cand in candidates:
        cand = pathlib.Path(cand)
        if not cand.is_absolute():
            cand = (pathlib.Path.cwd() / cand).resolve()
        if cand.is_file():
            result.append(cand)
        elif cand.is_dir():
            result.extend(sorted(cand.rglob("test_*.py")))
    return result


def find_stale_targets(paths: list[str] | None = None, repo: pathlib.Path = REPO) -> list[dict]:
    """Return findings, each naming the patch, the immune read, and its readers."""
    readers = _immune_readers(repo)
    per_target: dict[str, set[str]] = defaultdict(set)
    for test_file in _test_files(paths or [], repo):
        for target in _patch_targets(test_file):
            per_target[target].add(_rel(test_file, repo))

    findings: list[dict] = []
    for target, tests in sorted(per_target.items()):
        module, _, attr = target.rpartition(".")
        parts = module.split(".")
        for depth in range(1, len(parts)):
            pkg = ".".join(parts[:depth])
            if _reexport_source(repo.joinpath(*parts[:depth]), attr) is None:
                continue
            key = f"{pkg}.{attr}"
            if key in readers:
                findings.append(
                    {
                        "patch": target,
                        "immune_read": key,
                        "read_by": sorted(readers[key]),
                        "patched_in": sorted(tests),
                        "suggest": f'patch("{key}")',
                    }
                )
            break
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("paths", nargs="*", help="test files/dirs (default: all test trees)")
    parser.add_argument("--strict", action="store_true", help="exit 1 when anything is found")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()

    findings = find_stale_targets(args.paths)

    if args.json:
        print(json.dumps(findings, indent=2))
        return 1 if (args.strict and findings) else 0

    if not findings:
        print("patch-target lint: ok — no re-export-immune @patch targets")
        return 0

    print(f"patch-target lint: {len(findings)} @patch target(s) cannot reach the code under test\n")
    for item in findings:
        print(f"  {item['patch']}")
        print(f"      production reads {item['immune_read']} (a copy made by the package __init__)")
        for reader in item["read_by"][:2]:
            print(f"        <- {reader}")
        if len(item["read_by"]) > 2:
            print(f"        ... and {len(item['read_by']) - 2} more")
        print(f"      patched in {', '.join(item['patched_in'][:3])}")
        print(f"      fix: {item['suggest']}\n")
    print("The deep-module patch replaces one binding; the package keeps its own copy.")
    return 1 if args.strict else 0


if __name__ == "__main__":
    sys.exit(main())
