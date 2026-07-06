"""Query matching engine for MogDB.

Supports MongoDB-style query operators::

    {"field": "value"}            # exact match
    {"field": {"$gt": 5}}         # comparison
    {"field": {"$in": [1, 2, 3]}} # membership
    {"field": {"$regex": "^foo"}} # regex
    {"$or": [...]}                # logical
    {"$and": [...]}               # logical
    {"field": {"$exists": True}}  # field existence
"""

import re
from typing import Any, Dict, List


def _compare(value: Any, op: str, expected: Any) -> bool:
    if op == "$eq":
        return value == expected
    if op == "$ne":
        return value != expected
    if op == "$gt":
        return isinstance(value, (int, float)) and value > expected
    if op == "$gte":
        return isinstance(value, (int, float)) and value >= expected
    if op == "$lt":
        return isinstance(value, (int, float)) and value < expected
    if op == "$lte":
        return isinstance(value, (int, float)) and value <= expected
    if op == "$in":
        return isinstance(expected, list) and value in expected
    if op == "$nin":
        return isinstance(expected, list) and value not in expected
    if op == "$regex":
        if isinstance(expected, str):
            return bool(re.search(expected, str(value)))
        return False
    if op == "$exists":
        return (value is not None) == bool(expected)
    return False


def _get_field(doc: Dict[str, Any], field: str) -> Any:
    """Get nested field value via dot-separated path."""
    parts = field.split(".")
    current: Any = doc
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def match_document(doc: Dict[str, Any], query: Dict[str, Any]) -> bool:
    """Return True if *doc* matches the MongoDB-style *query*.

    Operates as an implicit ``$and`` at the top level.
    """
    for field, condition in query.items():
        if field == "$or":
            if not isinstance(condition, list):
                return False
            if not any(match_document(doc, sub) for sub in condition):
                return False
            continue
        if field == "$and":
            if not isinstance(condition, list):
                return False
            if not all(match_document(doc, sub) for sub in condition):
                return False
            continue
        if field == "$nor":
            if not isinstance(condition, list):
                return False
            if any(match_document(doc, sub) for sub in condition):
                return False
            continue

        value = _get_field(doc, field)

        if isinstance(condition, dict) and any(
            k.startswith("$") for k in condition
        ):
            for op, expected in condition.items():
                if not _compare(value, op, expected):
                    return False
        else:
            if value != condition:
                return False

    return True
