"""
Bulk migration: replace all module-level ``import torch`` with
``from domains.training.slonet_compat import torch`` across
``packages/core-py/domains/``.

Usage::

    python scripts/migrate_torch.py          # dry-run
    python scripts/migrate_torch.py --apply  # write changes
"""

import ast
import argparse
import os
import sys
from pathlib import Path


REPLACEMENT = "from domains.training.slonet_compat import torch"

# Patterns that use an alias — we need to add a local binding
ALIAS_MAP = {
    "import torch.nn as nn": "nn = torch.nn",
    "import torch.nn.functional as F": "F = torch.F",
    "import torch.distributed as dist": "dist = torch.distributed",
    "import torch.multiprocessing as mp": "mp = torch.multiprocessing",
    "import torch.nn.init as init": "init = torch.nn.init",
}

# For `from torch.xxx import Y` patterns, generate local aliases
# mapping: (module, name) -> alias string
FROM_ALIAS_MAP = {
    ("torch.nn.parallel", "DistributedDataParallel"): "torch.nn.parallel.DistributedDataParallel",
    ("torch.cuda.amp", "autocast"): "torch.cuda.amp.autocast",
    ("torch.cuda.amp", "GradScaler"): "torch.cuda.amp.GradScaler",
    ("torch.utils.data", "Dataset"): "torch.utils.data.Dataset",
    ("torch.utils.data", "DataLoader"): "torch.utils.data.DataLoader",
}

# Files that should NOT be migrated (compat shim itself)
EXCLUDE = {"slonet_compat.py", "slonet.py"}


def _is_module_level_torch_import(stmt, lines):
    """Check if a statement is a module-level torch import."""
    if isinstance(stmt, ast.Import):
        return any(
            alias.name == "torch" or alias.name.startswith("torch.")
            for alias in stmt.names
        )
    if isinstance(stmt, ast.ImportFrom):
        return stmt.module and (stmt.module == "torch" or stmt.module.startswith("torch."))
    return False


def _get_text(line_num, lines):
    """Get the original text of a line (1-indexed)."""
    return lines[line_num - 1] if 1 <= line_num <= len(lines) else ""


def migrate_file(filepath, apply=False):
    """Migrate a single file. Returns (changed, issues) tuple."""
    if filepath.name in EXCLUDE:
        return False, []

    with open(filepath) as f:
        original = f.read()

    try:
        tree = ast.parse(original)
    except SyntaxError as e:
        return False, [f"  SKIP {filepath.name}: syntax error ({e})"]

    lines = original.split("\n")

    torch_imports = []
    for node in ast.iter_child_nodes(tree):
        if _is_module_level_torch_import(node, lines):
            text = _get_text(node.lineno, lines)
            torch_imports.append((node.lineno, text, node))

    if not torch_imports:
        return False, []

    # Collect aliases that need local bindings
    needed_aliases = set()
    needs_torch_base = False
    for lineno, text, node in torch_imports:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "torch" and alias.asname is None:
                    needs_torch_base = True
                elif alias.name == "torch" and alias.asname:
                    pass  # unusual, skip
                elif alias.asname:
                    key = f"import {alias.name} as {alias.asname}"
                    if key in ALIAS_MAP:
                        needed_aliases.add(ALIAS_MAP[key])
        elif isinstance(node, ast.ImportFrom):
            # Handle `from torch.xxx import Y` patterns
            for alias in node.names:
                aname = alias.asname or alias.name
                key = (node.module, alias.name)
                if key in FROM_ALIAS_MAP:
                    needed_aliases.add(f"{aname} = {FROM_ALIAS_MAP[key]}")
                elif node.module and (node.module == "torch" or node.module.startswith("torch.")):
                    # Generic: from torch.xxx import Y -> Y = torch.xxx.Y
                    mod_path = node.module
                    if mod_path == "torch":
                        needed_aliases.add(f"{aname} = torch.{alias.name}")
                    else:
                        needed_aliases.add(f"{aname} = {mod_path}.{alias.name}")

    # Check if any code uses bare torch (not just attributes)
    # If there's no `import torch` but only `from torch.xxx import Y`, we still need torch
    if not needs_torch_base:
        for lineno, text, node in torch_imports:
            if isinstance(node, ast.ImportFrom):
                needs_torch_base = True  # Need torch for anything

    # Build replacement lines
    new_lines = list(lines)
    used_lines = set()

    # Remove all torch import lines
    for lineno, text, node in torch_imports:
        # Mark the line for removal
        if node.lineno not in used_lines:
            new_lines[node.lineno - 1] = None  # mark for removal
            used_lines.add(node.lineno)

    # Insert the compat import at the first torch import location
    first_line = min(lineno for lineno, _, _ in torch_imports)
    insert_lines = [REPLACEMENT]
    for alias_line in sorted(needed_aliases):
        if alias_line not in insert_lines:
            insert_lines.append(alias_line)
    # Insert before the first removed line
    new_lines[first_line - 1] = "\n".join(insert_lines)

    # Remove None lines
    new_lines = [l for l in new_lines if l is not None]

    result = "\n".join(new_lines)

    if result == original:
        return False, []

    if apply:
        with open(filepath, "w") as f:
            f.write(result)
        return True, []
    else:
        return True, []


def main():
    parser = argparse.ArgumentParser(description="Migrate torch imports to slonet_compat")
    parser.add_argument("--apply", action="store_true", help="Write changes")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    targets = sorted((root / "domains").rglob("*.py"))

    changed = 0
    errors = []

    for fp in targets:
        c, issues = migrate_file(fp, apply=args.apply)
        if issues:
            for msg in issues:
                print(msg)
        if c:
            rel = fp.relative_to(root.parent.parent)
            print(f"  {'✗' if args.apply else '~'} {rel}")
            changed += 1

    print(f"\n{'Applied' if args.apply else 'Would change'} {changed} files.")
    if errors:
        print(f"Errors: {errors}")


if __name__ == "__main__":
    main()
