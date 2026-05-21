#!/usr/bin/env python3
"""
Validate a SloughGPT v1 dataset manifest and optionally print the resolved training text path.

  python scripts/validate_dataset_manifest.py path/to/dataset_manifest.json
  python scripts/validate_dataset_manifest.py path/to/dataset_manifest.json --resolve

Manifest JSON describes corpus layout only; native training ``step_*.pt`` vocabulary is separate
(``docs/policies/CONTRIBUTING.md``, *Checkpoint vocabulary*).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from domains.training.dataset_manifest import (  # noqa: E402
    ManifestError,
    load_manifest,
    resolve_training_data_path,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate v1 dataset manifest JSON.")
    parser.add_argument("manifest", type=Path, help="Path to dataset_manifest.json")
    parser.add_argument(
        "--resolve",
        action="store_true",
        help="Resolve splits.train / input.txt and print the training file path",
    )
    args = parser.parse_args()
    mp = args.manifest
    try:
        m = load_manifest(mp)
        if args.resolve:
            text_path, _ = resolve_training_data_path(mp)
            print(f"OK: dataset_id={m['dataset_id']!r} version={m['version']!r} domain={m.get('domain')!r}")
            print(f"Training text file: {text_path.resolve()}")
        else:
            print(f"OK: dataset_id={m['dataset_id']!r} version={m['version']!r} domain={m.get('domain')!r}")
    except ManifestError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
