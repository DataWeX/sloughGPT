"""Small pure operations on dicts, numbers, and config mappings."""

from __future__ import annotations

__all__ = ["clamp", "merge_dicts", "validate_config"]


def merge_dicts(*dicts: dict) -> dict:
    """Shallow-merge dictionaries left to right (later keys win)."""
    result = {}
    for d in dicts:
        result.update(d)
    return result


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Constrain *value* to ``[min_val, max_val]``."""
    return max(min_val, min(max_val, value))


def validate_config(config: dict, required_keys: list[str]) -> bool:
    """True when every key in *required_keys* is present in *config*."""
    return all(key in config for key in required_keys)
