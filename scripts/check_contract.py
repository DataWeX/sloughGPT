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
    * the declaration-style ratchet: hand-built route declarations (``@router.get``
      decorators, ``add_api_route`` calls) must equal the grandfathered counts in
      ``handbuilt-routes-baseline.json`` — new sites fail, migrations must shrink
      the baseline in the same reviewed edit;
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
DECLARATION_BASELINE_PATH = SERVER_DIR / "handbuilt-routes-baseline.json"

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


# ── declaration-style ratchet (the endpoint rule's mechanical half) ────
#
# Routes are projections of a descriptor (create_router). The grandfathered
# hand-built backlog — decorators and add_api_route calls — is frozen in
# handbuilt-routes-baseline.json and may only shrink; a new hand-built site
# anywhere in the server tree fails the gate.

HANDBUILT_DECORATOR_METHODS = frozenset(
    {"get", "post", "put", "patch", "delete", "head", "options", "api_route", "websocket", "route"}
)
HANDBUILT_CALL_METHODS = frozenset({"add_api_route", "add_websocket_route", "add_route"})
# create_router's own projection internals — the one legitimate add_api_route.
PROJECTION_INTERNAL_FILES = frozenset({"infrastructure/contract.py"})


def _site_label(func: ast.AST, prefix: str) -> str:
    try:
        return f"{prefix}{ast.unparse(func)}"
    except Exception:  # noqa: BLE001 — a label is never worth failing the gate over
        return prefix or "<call>"


def find_handbuilt_declarations(source: str) -> list[tuple[int, str]]:
    """``(lineno, label)`` for every route declaration that bypasses create_router.

    Decorators (``@router.get(...)``) and direct registration calls
    (``router.add_api_route(...)``) are the hand-built shapes; ``create_router``
    and ``include_router`` are the seam and are never counted.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        for deco in getattr(node, "decorator_list", ()):
            func = deco.func if isinstance(deco, ast.Call) else deco
            if isinstance(func, ast.Attribute):
                attr = func.attr
            elif isinstance(func, ast.Name):
                attr = func.id
            else:
                attr = ""
            if attr in HANDBUILT_DECORATOR_METHODS:
                found.append((getattr(node, "lineno", 0), _site_label(func, "@")))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in HANDBUILT_CALL_METHODS:
                found.append((node.lineno, _site_label(node.func, "")))
    return found


def scan_handbuilt(server_dir: Path) -> dict[str, list[tuple[int, str]]]:
    """Every server module's hand-built declaration sites, keyed by relative path.

    Scope: the whole server tree — test fixtures build throwaway apps by design,
    and ``infrastructure/contract.py`` is the projection engine itself.
    """
    sites: dict[str, list[tuple[int, str]]] = {}
    for path in sorted(server_dir.rglob("*.py")):
        rel = path.relative_to(server_dir).as_posix()
        if path.name.startswith("_"):
            continue
        if "tests" in path.relative_to(server_dir).parts:
            continue
        if rel in PROJECTION_INTERNAL_FILES:
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except OSError:
            continue
        found = find_handbuilt_declarations(source)
        if found:
            sites[rel] = found
    return sites


def check_declaration_style(
    sites: dict[str, list[tuple[int, str]]], baseline_files: dict[str, int]
) -> list[str]:
    """Ratchet: hand-built counts equal the baseline — may only shrink, with review.

    * a file with MORE sites than its baseline entry (or no entry at all) is a
      new hand-built route — exactly what the endpoint rule exists to stop;
    * a file with FEWER sites, or a baseline entry with none left, is a
      migration — the baseline must shrink to record it, as a reviewed edit.
    """
    violations: list[str] = []
    way_out = "declare a ToolSpec + create_router instead (docs/TRANSPORT_PROJECTIONS.md)"
    for rel in sorted(sites):
        actual = len(sites[rel])
        allowed = baseline_files.get(rel, 0)
        if actual > allowed:
            lines = ", ".join(str(lineno) for lineno, _ in sites[rel][:6])
            suffix = ", …" if actual - allowed > 6 else ""
            violations.append(
                f"{rel}: {actual} hand-built route declaration(s), baseline allows "
                f"{allowed} — new hand-built site(s) at line(s) {lines}{suffix}; "
                f"{way_out} (or add a reviewed entry to {DECLARATION_BASELINE_PATH.name})"
            )
        elif actual < allowed:
            violations.append(
                f"{rel}: {actual} hand-built site(s), baseline records {allowed} — "
                f"migration detected; shrink the baseline entry (the baseline may only shrink)"
            )
    for rel in sorted(set(baseline_files) - set(sites)):
        violations.append(
            f"{rel}: baseline entry with no hand-built declarations left — "
            f"remove it (the baseline may only shrink)"
        )
    return violations


def write_declaration_baseline() -> int:
    """Record today's hand-built counts as the grandfathered ratchet baseline."""
    sites = scan_handbuilt(SERVER_DIR)
    files = {rel: len(found) for rel, found in sorted(sites.items())}
    DECLARATION_BASELINE_PATH.write_text(
        json.dumps(
            {
                "note": (
                    "Grandfathered hand-built route declarations for "
                    "scripts/check_contract.py — the endpoint rule's ratchet: "
                    "counts may only shrink, as a reviewed edit. Regenerate with "
                    "--write-declaration-baseline."
                ),
                "total": sum(files.values()),
                "files": files,
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    return len(files)


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

    # Declaration-style ratchet: the grandfathered hand-built backlog is
    # frozen in a baseline; new decorator/add_api_route sites fail CI.
    declared_sites = scan_handbuilt(SERVER_DIR)
    if DECLARATION_BASELINE_PATH.is_file():
        stored = json.loads(DECLARATION_BASELINE_PATH.read_text(encoding="utf-8"))
        style_violations = check_declaration_style(declared_sites, stored.get("files", {}))
    else:
        style_violations = [
            f"missing {DECLARATION_BASELINE_PATH.name} — generate it with "
            f"check_contract.py --write-declaration-baseline"
        ]
    failures.extend(style_violations)
    total_sites = sum(len(found) for found in declared_sites.values())
    print(
        f"   hand-built declarations: {total_sites} in {len(declared_sites)} module(s) "
        f"(grandfathered baseline — may only shrink)"
    )

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


def _openapi_operations(app) -> dict[str, dict[str, dict]]:
    """Per-operation contract facts, read back out of the app's OpenAPI doc.

    Path + method alone cannot see a renamed query param, a newly required
    field or a dropped response code — these facts can. This is the "review
    the OpenAPI diff like code" half of the contract gate; it is report-only,
    because the grandfathered backlog changes shape constantly.
    """
    doc = app.openapi()
    ops: dict[str, dict[str, dict]] = {}
    for path, methods in doc.get("paths", {}).items():
        for method, op in methods.items():
            if not isinstance(op, dict):
                continue  # $ref stubs / non-operation entries
            params = sorted(
                (
                    {
                        "name": p.get("name"),
                        "in": p.get("in"),
                        "required": bool(p.get("required", False)),
                        "type": (p.get("schema") or {}).get("type"),
                    }
                    for p in op.get("parameters", [])
                ),
                key=lambda d: (str(d["in"]), str(d["name"])),
            )
            body = None
            content = (op.get("requestBody") or {}).get("content") or {}
            if content:
                ctype = sorted(content)[0]  # deterministic: one body per op
                schema = content[ctype].get("schema") or {}
                ref = schema.get("$ref")
                body = {
                    "content_type": ctype,
                    # named model -> its name (stable); inline -> the shape
                    "schema": ref.rsplit("/", 1)[-1] if ref else schema,
                }
            ops.setdefault(path, {})[method.upper()] = {
                "operationId": op.get("operationId"),
                "params": params,
                "body": body,
                "responses": sorted((op.get("responses") or {}).keys()),
            }
    return ops


def _diff_operations(current: dict, previous: dict) -> list[str]:
    """Human-readable contract drift between two OpenAPI fact sets.

    Each line is one reviewable change: ``+ added``, ``- removed`` or
    ``~ changed`` with the exact field delta. Sorted for stable output.
    """
    changes: list[str] = []
    cur_paths, prev_paths = set(current), set(previous)
    for path in sorted(prev_paths - cur_paths):
        changes.append(f"- {path} removed")
    for path in sorted(cur_paths - prev_paths):
        changes.append(f"+ {path} added")
    for path in sorted(cur_paths & prev_paths):
        cur_methods, prev_methods = current[path], previous[path]
        for method in sorted(set(prev_methods) - set(cur_methods)):
            changes.append(f"- {method} {path} removed")
        for method in sorted(set(cur_methods) - set(prev_methods)):
            changes.append(f"+ {method} {path} added")
        for method in sorted(set(cur_methods) & set(prev_methods)):
            after, before = cur_methods[method], prev_methods[method]
            if after == before:
                continue
            detail: list[str] = []
            for field in ("operationId", "body"):
                if after.get(field) != before.get(field):
                    detail.append(f"{field}: {before.get(field)!r} -> {after.get(field)!r}")
            if after.get("params") != before.get("params"):
                detail.append(f"params: {before.get('params')} -> {after.get('params')}")
            if after.get("responses") != before.get("responses"):
                detail.append(f"responses: {before.get('responses')} -> {after.get('responses')}")
            changes.append(f"~ {method} {path} — " + "; ".join(detail))
    return changes


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

    # Contract facts (params / body / responses) cost one openapi() pass, so
    # only build them when a baseline will actually be written or compared.
    operations = _openapi_operations(app) if (write_baseline or baseline.is_file()) else None

    failures = [
        f"{entry['name']}: registry claims {entry['method']} {entry['path']} "
        f"but FastAPI serves no such route"
        for entry in entries
        if (entry["path"], entry["method"]) not in served
    ]

    print(
        f"   routers: {len(routers_list)} · routes: {len(inventory)} · "
        f"descriptor crossings: {len(entries)}"
        + (f" · operations: {len(operations)}" if operations is not None else "")
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
                        "Route + OpenAPI baseline for scripts/check_contract.py — "
                        "report-only drift signal, regenerated with --write-baseline."
                    ),
                    "routers": len(routers_list),
                    "count": len(inventory),
                    "routes": inventory,
                    "operations": operations or {},
                },
                indent=1,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            f"   wrote baseline {baseline.relative_to(ROOT)} "
            f"({len(inventory)} routes, {len(operations or {})} operations)"
        )
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

            # Contract-level drift: same path+method can still have changed
            # shape. Report-only, like the route diff above.
            stored_ops = stored.get("operations")
            if operations is None:
                pass  # nothing to compare (no baseline ops were requested)
            elif not isinstance(stored_ops, dict) or not stored_ops:
                print(
                    "   ⚠ baseline carries no OpenAPI operations — regenerate with --write-baseline"
                )
            else:
                op_changes = _diff_operations(operations, stored_ops)
                if op_changes:
                    print(f"   ⚠ OpenAPI contract drift vs {baseline.name} (report-only):")
                    for line in op_changes[:12]:
                        print(f"      {line}")
                    if len(op_changes) > 12:
                        print(f"      … +{len(op_changes) - 12} more")
                else:
                    print(f"   ✓ OpenAPI operations match baseline ({len(stored_ops)} operations)")
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
    parser.add_argument(
        "--write-declaration-baseline",
        action="store_true",
        help="record current hand-built declaration counts (the ratchet baseline)",
    )
    parser.add_argument("--json", type=Path, help="write a machine-readable artifact")
    args = parser.parse_args(argv)

    if args.write_declaration_baseline:
        count = write_declaration_baseline()
        sites = scan_handbuilt(SERVER_DIR)
        total = sum(len(found) for found in sites.values())
        print(
            f"   wrote {DECLARATION_BASELINE_PATH.relative_to(ROOT)} "
            f"({count} module(s), {total} hand-built site(s))"
        )

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
