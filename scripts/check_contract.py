#!/usr/bin/env python3
"""Contract gate — descriptor validation + route-baseline diff.

Two layers, deliberately different weights:

**Blocking (exit 1)** — the contract must be honest:
    * router manifest freshness (``gen_router_manifest.py --check``);
    * every descriptor declares a literal ``name``/``description``/
      ``auth_scope``/``version`` and parameters with name/type/description in
      JSON Schema terms (validated by AST — no imports, so CI needs no deps);
    * each descriptor's name appears in a server test (the contract is only
      real if something exercises it);
    * runtime: every registry entry is actually served, i.e. the projection
      ``create_router`` claims matches the routes FastAPI registers.

**Report-only (exit 0)** — surfaced, never silently ignored:
    * route inventory drift vs the committed baseline (grandfathered routers
      still churn by hand; the diff is the signal, and the first baseline
      commit freezes today's surface);
    * duplicate ``(method, path)`` across hand-written routers.

Usage:
    python scripts/check_contract.py                  # static + runtime
    python scripts/check_contract.py --static         # CI-safe, stdlib only
    python scripts/check_contract.py --runtime        # needs the app's deps
    python scripts/check_contract.py --runtime --write-baseline
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER_DIR = ROOT / "apps" / "api" / "server"
ROUTERS_DIR = SERVER_DIR / "routers"
TESTS_DIR = SERVER_DIR / "tests"
BASELINE_PATH = SERVER_DIR / "contract-baseline.json"

sys.path.insert(0, str(ROOT / "scripts"))
import gen_router_manifest as manifest_mod  # noqa: E402  (sibling script, not a package)

JSON_TYPES = {"string", "integer", "number", "boolean", "object", "array"}
TOOLSPEC_FIELDS = (
    "name",
    "description",
    "parameters",
    "execute",
    "pattern",
    "requires_approval",
    "result",
    "auth_scope",
    "idempotent",
    "version",
)
TOOLPARAM_FIELDS = ("name", "type", "description", "required")
RESERVED_PARAM_NAMES = {"request"}


def _call_name(func: ast.AST) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _call_fields(call: ast.Call, fields: tuple[str, ...]) -> dict[str, ast.AST]:
    """Positional-or-keyword fields of a dataclass-style call, by field name."""
    out: dict[str, ast.AST] = {}
    for value, field_name in zip(call.args, fields):
        out[field_name] = value
    for kw in call.keywords:
        if kw.arg:
            out[kw.arg] = kw.value
    return out


def _literal(node: ast.AST | None, expected: type) -> object | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, expected):
        return node.value
    return None


def find_descriptors(path: Path) -> list[tuple[ast.Call, dict[str, ast.AST]]]:
    """Every ``ToolSpec(...)`` call in a router module, with its fields."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return []
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_name(node.func) == "ToolSpec":
            found.append((node, _call_fields(node, TOOLSPEC_FIELDS)))
    return found


def check_descriptor(module: str, fields: dict[str, ast.AST]) -> tuple[str | None, list[str]]:
    """Validate one descriptor's literals. Returns (name, errors)."""
    errors: list[str] = []
    name = _literal(fields.get("name"), str)
    if not name:
        errors.append("descriptor has no literal non-empty 'name'")
    for field_name in ("description", "auth_scope", "version"):
        if not _literal(fields.get(field_name), str):
            errors.append(f"descriptor field '{field_name}' must be a literal non-empty string")

    idempotent_node = fields.get("idempotent")
    if idempotent_node is not None and not isinstance(_literal(idempotent_node, bool), bool):
        errors.append("'idempotent' must be a bool literal (it picks the HTTP verb)")

    params_node = fields.get("parameters")
    if params_node is None:
        errors.append("'parameters' is missing (declare an empty list if there are none)")
    elif not isinstance(params_node, (ast.List, ast.Tuple)):
        errors.append("'parameters' must be a literal list of ToolParam(...) calls")
    else:
        for index, element in enumerate(params_node.elts):
            if not (isinstance(element, ast.Call) and _call_name(element.func) == "ToolParam"):
                errors.append(f"parameters[{index}] must be a ToolParam(...) call")
                continue
            pf = _call_fields(element, TOOLPARAM_FIELDS)
            pname = _literal(pf.get("name"), str)
            ptype = _literal(pf.get("type"), str)
            pdesc = _literal(pf.get("description"), str)
            label = f"parameters[{index}] ({pname or '?'})"
            if not pname:
                errors.append(f"{label}: missing literal 'name'")
            elif pname in RESERVED_PARAM_NAMES:
                errors.append(
                    f"{label}: '{pname}' is reserved by the projection (raw Request is injected)"
                )
            if ptype not in JSON_TYPES:
                errors.append(
                    f"{label}: type {ptype!r} is not a JSON Schema type {sorted(JSON_TYPES)}"
                )
            if not pdesc:
                errors.append(f"{label}: missing literal 'description'")

    return name, errors


def contract_tests_exist(name: str, tests: Path) -> bool:
    """The contract is only real if some test names it."""
    if not tests.is_dir():
        return False
    for test_file in sorted(tests.rglob("test_*.py")):
        try:
            if name in test_file.read_text(encoding="utf-8"):
                return True
        except OSError:
            continue
    return False


def run_static() -> int:
    """Stdlib-only checks: manifest freshness + descriptor literals + test presence."""
    print("── static contract checks (AST, no imports)")
    failures: list[str] = []

    if manifest_mod.main(["--check"]) != 0:
        failures.append("router manifest is stale (see gen_router_manifest.py --check)")

    manifest = manifest_mod.read_manifest() or []
    names_seen: dict[str, str] = {}
    descriptor_modules: list[str] = []
    descriptors = 0

    for path in sorted(ROUTERS_DIR.glob("*.py")):
        if path.name.startswith("_"):
            continue
        calls = find_descriptors(path)
        if not calls:
            continue
        module = path.stem
        descriptor_modules.append(module)
        if module not in manifest:
            failures.append(
                f"{module}: declares a descriptor but is not in the router manifest "
                f"(run scripts/gen_router_manifest.py)"
            )
        for _call, fields in calls:
            descriptors += 1
            name, errors = check_descriptor(module, fields)
            for err in errors:
                failures.append(f"{module}: {err}")
            if name:
                owner = names_seen.get(name)
                if owner and owner != module:
                    failures.append(f"{module}: descriptor name '{name}' already used by {owner}")
                names_seen[name] = module
                if not contract_tests_exist(name, TESTS_DIR):
                    failures.append(
                        f"{module}: no test under {TESTS_DIR.relative_to(ROOT)} mentions "
                        f"'{name}' — a contract without a test is a wish"
                    )

    print(
        f"   manifest: {len(manifest)} modules · descriptors found: {descriptors} "
        f"in {len(descriptor_modules)} module(s)"
    )
    for name, module in sorted(names_seen.items()):
        print(f"   ✓ {name} ({module})")

    if failures:
        print(f"\n✗ {len(failures)} blocking contract violation(s):", file=sys.stderr)
        for failure in failures:
            print(f"   {failure}", file=sys.stderr)
        return 1
    print("   ✓ all descriptors satisfy the contract")
    return 0


def _iter_served_routes(routes) -> list[tuple[str, str]]:
    """Flatten an app's route table, recursing into included routers."""
    served: list[tuple[str, str]] = []
    for route in routes:
        inner = getattr(route, "routes", None)
        if inner:
            served.extend(_iter_served_routes(inner))
            continue
        original = getattr(route, "original_router", None)
        if original is not None:
            served.extend(_iter_served_routes(getattr(original, "routes", [])))
            continue
        methods = {m for m in (getattr(route, "methods", None) or []) if m != "HEAD"}
        path = getattr(route, "path", None)
        if path and methods:
            for method in sorted(methods):
                served.append((path, method))
    return served


def _inventory_from_routers(routers_list) -> list[list[str]]:
    """Sorted ``[path, method, ...]`` inventory of the boot router set."""
    inventory: list[list[str]] = []
    for router in routers_list:
        prefix = getattr(router, "prefix", "") or ""
        for route in getattr(router, "routes", []):
            methods = getattr(route, "methods", None)
            if not methods:
                continue  # websockets / mounts carry no HTTP verb
            path = route.path
            if prefix and not path.startswith(prefix):
                path = prefix + path
            for method in sorted(m for m in methods if m != "HEAD"):
                inventory.append([path, method])
    inventory.sort()
    return inventory


def run_runtime(baseline: Path, write_baseline: bool, artifact: Path | None) -> int:
    """Import the boot router set; verify registry ↔ served routes, diff baseline."""
    print("── runtime contract checks (imports the boot router set)")
    try:
        from fastapi import FastAPI
        from infrastructure.contract import REGISTRY
        from routers import get_all_routers
    except ImportError as exc:
        print(f"✗ runtime checks need the app's dependencies: {exc}", file=sys.stderr)
        print("  run --static in a bare environment, or install requirements.txt", file=sys.stderr)
        return 1

    routers_list = get_all_routers()
    inventory = _inventory_from_routers(routers_list)
    entries = REGISTRY.routes()

    # Assemble the app the way main.py does and verify every registry claim is
    # a route FastAPI really serves — projection vs. reality, not a second
    # source of truth.
    app = FastAPI()
    for router in routers_list:
        app.include_router(router)
    served = set(_iter_served_routes(app.routes))

    failures = [
        f"{entry['name']}: registry claims {entry['method']} {entry['path']} "
        f"but FastAPI serves no such route"
        for entry in entries
        if (entry["path"], entry["method"]) not in served
    ]

    print(
        f"   routers: {len(routers_list)} · routes: {len(inventory)} · "
        f"descriptor crossings: {len(entries)}"
    )
    for entry in entries:
        print(f"   ✓ {entry['method']:6} {entry['path']:30} {entry['name']}")

    # A descriptor module that imports but never registers (or fails to import
    # at boot) would otherwise pass vacuously — the contract must reach the wire.
    registered_modules = {entry["module"].rsplit(".", 1)[-1] for entry in entries}
    for path in sorted(ROUTERS_DIR.glob("*.py")):
        if path.name.startswith("_") or not find_descriptors(path):
            continue
        if path.stem not in registered_modules:
            failures.append(
                f"{path.stem}: declares a descriptor but registered nothing at boot — "
                f"the module did not import, or create_router was never called"
            )

    duplicates: list[list[str]] = []
    seen: set[tuple[str, str]] = set()
    for path, method in inventory:
        key = (path, method)
        if key in seen:
            duplicates.append([path, method])
        seen.add(key)

    if artifact:
        artifact.write_text(
            json.dumps(
                {"routers": len(routers_list), "routes": inventory, "contracts": entries},
                indent=1,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"   wrote artifact {artifact.relative_to(ROOT)}")

    if write_baseline:
        baseline.write_text(
            json.dumps(
                {
                    "note": (
                        "Route baseline for scripts/check_contract.py — report-only drift "
                        "signal, regenerated with --write-baseline."
                    ),
                    "routers": len(routers_list),
                    "count": len(inventory),
                    "routes": inventory,
                },
                indent=1,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"   wrote baseline {baseline.relative_to(ROOT)} ({len(inventory)} routes)")
    elif baseline.is_file():
        stored = json.loads(baseline.read_text(encoding="utf-8"))
        if stored.get("routers") not in (None, len(routers_list)):
            print(
                f"   ⚠ environment differs from baseline "
                f"({len(routers_list)} routers vs {stored.get('routers')} recorded — "
                f"some modules skipped an import); route diff suppressed",
            )
        else:
            current = {tuple(row) for row in inventory}
            previous = {tuple(row) for row in stored.get("routes", [])}
            added = sorted(current - previous)
            removed = sorted(previous - current)
            if added or removed:
                print(f"   ⚠ route drift vs {baseline.name} (report-only):")
                for path, method in added:
                    print(f"      + {method:6} {path}")
                for path, method in removed:
                    print(f"      - {method:6} {path}")
            else:
                print(f"   ✓ route inventory matches baseline ({len(previous)} routes)")
    else:
        print(f"   ⚠ no baseline at {baseline.name} — run with --write-baseline")

    if duplicates:
        print(
            f"   ⚠ {len(duplicates)} duplicate route(s) across hand-written routers "
            f"(report-only; migrate via create_router):"
        )
        for path, method in duplicates:
            print(f"      ! {method:6} {path}")

    if failures:
        print(f"\n✗ {len(failures)} blocking runtime violation(s):", file=sys.stderr)
        for failure in failures:
            print(f"   {failure}", file=sys.stderr)
        return 1
    print("   ✓ projection registry matches the served route table")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static", action="store_true", help="AST checks only (no imports)")
    parser.add_argument("--runtime", action="store_true", help="import the app and cross-check")
    parser.add_argument("--baseline", type=Path, default=BASELINE_PATH)
    parser.add_argument("--write-baseline", action="store_true", help="record current routes")
    parser.add_argument("--json", type=Path, help="write a machine-readable artifact")
    args = parser.parse_args(argv)

    run_static_mode = args.static or not args.runtime
    run_runtime_mode = args.runtime or not args.static

    status = 0
    if run_static_mode:
        status |= run_static()
    if run_runtime_mode:
        status |= run_runtime(args.baseline, args.write_baseline, args.json)
    return status


if __name__ == "__main__":
    sys.exit(main())
