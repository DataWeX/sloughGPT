#!/usr/bin/env python3
"""
Validate example JSON under packages/standards/standards/v1/examples against JSON Schemas (requires jsonschema).

  pip install -e ".[dev]"   # includes jsonschema
  python scripts/validate_standards_schemas.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCHEMA_DIR = _REPO_ROOT / "packages" / "standards" / "standards" / "v1" / "schemas"

# (example path relative to repo, schema filename)
_EXAMPLES: list[tuple[str, str]] = [
    (
        "packages/standards/standards/v1/examples/dataset_manifest.github_code.example.json",
        "dataset_manifest.json",
    ),
    (
        "packages/standards/standards/v1/examples/dataset_manifest.local_txt.example.json",
        "dataset_manifest.json",
    ),
]


def main() -> int:
    try:
        import jsonschema
        from jsonschema import validators
    except ImportError:
        print(
            "Missing jsonschema: pip install -e \".[dev]\" (from repo root) or pip install jsonschema",
            file=sys.stderr,
        )
        return 1

    for rel, schema_name in _EXAMPLES:
        instance_path = _REPO_ROOT / rel
        schema_path = _SCHEMA_DIR / schema_name
        if not instance_path.is_file():
            print(f"SKIP (missing): {rel}", file=sys.stderr)
            continue
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        instance = json.loads(instance_path.read_text(encoding="utf-8"))
        cls = validators.validator_for(schema)
        cls.check_schema(schema)
        cls(schema).validate(instance)
        print(f"OK {rel} -> {schema_name}")

    print("All present examples passed validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
