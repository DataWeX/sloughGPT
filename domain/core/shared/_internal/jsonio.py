"""JSON file IO helpers."""

from __future__ import annotations

import json
from typing import Any

__all__ = ["load_json", "save_json"]


def load_json(path: str) -> dict:
    """Load a JSON document from *path*."""
    with open(path) as f:
        return json.load(f)


def save_json(data: Any, path: str, indent: int = 2) -> None:
    """Write *data* to *path* as JSON."""
    with open(path, "w") as f:
        json.dump(data, f, indent=indent)
