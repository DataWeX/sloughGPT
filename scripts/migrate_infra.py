#!/usr/bin/env python3
"""Migrate infrastructure implementations from old location to domain/infrastructure/_internal/."""

import os
import re

OLD_DIR = "packages/core-py/domains/infrastructure"
NEW_DIR = "domain/infrastructure/_internal"

SKIP = {"__init__"}


def convert_imports(content: str) -> str:
    """Rewrite domain.infrastructure._internal.X → same-package reference."""
    # from domain.infrastructure._internal.X import Y → from .X import Y
    content = re.sub(
        r"from domain\.infrastructure\._internal\.(\w+) import ", r"from .\1 import ", content
    )
    # import domain.infrastructure._internal.X as Y → from . import X as Y  (unlikely but safe)
    content = re.sub(
        r"import domain\.infrastructure\._internal\.(\w+) as (\w+)",
        r"from . import \1 as \2",
        content,
    )
    return content


def make_shim(module_name: str) -> str:
    """Create a backward-compat shim file."""
    return f'''"""Backward-compatibility shim."""
from domain.infrastructure._internal.{module_name} import *  # noqa: F401,F403
try:
    from domain.infrastructure._internal.{module_name} import __all__  # noqa: F401
except ImportError:
    pass
import sys as _sys
_mod = _sys.modules[__name__]
_real = _sys.modules.get("domain.infrastructure._internal.{module_name}")
if _real is not None:
    for _k in dir(_real):
        if not _k.startswith("__"):
            setattr(_mod, _k, getattr(_real, _k))
'''


def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    migrated = 0
    skipped = 0

    for fname in sorted(os.listdir(OLD_DIR)):
        if not fname.endswith(".py"):
            continue
        name = fname[:-3]  # strip .py
        if name in SKIP:
            continue

        old_path = os.path.join(OLD_DIR, fname)
        new_path = os.path.join(NEW_DIR, fname)

        # Check if current shim is a simple shim (1-3 lines)
        with open(new_path) as f:
            current = f.read()
        if "Backward-compatibility shim" in current and len(current.splitlines()) <= 5:
            # Simple shim — safe to replace
            pass
        else:
            # Already has content or is complex — skip
            skipped += 1
            print(f"  SKIP (non-trivial): {name}")
            continue

        # Read old implementation
        with open(old_path) as f:
            impl = f.read()

        # Convert imports
        impl = convert_imports(impl)

        # Write implementation to new location
        with open(new_path, "w") as f:
            f.write(impl)

        # Write shim to old location
        with open(old_path, "w") as f:
            f.write(make_shim(name))

        migrated += 1

    print(f"\nMigrated: {migrated}, Skipped: {skipped}")


if __name__ == "__main__":
    main()
