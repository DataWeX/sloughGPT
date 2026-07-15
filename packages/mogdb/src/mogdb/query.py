"""Query matching engine for MogDB.

Supports MongoDB-style query operators::

    {"field": "value"}            # exact match
    {"field": {"$gt": 5}}         # comparison
    {"field": {"$in": [1, 2, 3]}} # membership
    {"field": {"$regex": "^foo"}} # regex
    {"field": {"$not": {"$gt": 5}}} # negation
    {"field": {"$type": "string"}}  # type check
    {"field": {"$size": 3}}         # array size
    {"$or": [...]}                  # logical
    {"$and": [...]}                 # logical
    {"$nor": [...]}                 # logical
"""

import re
from typing import Any, Dict, List


_TYPE_MAP = {
    "string": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
    "dict": dict,
    "null": type(None),
    "number": (int, float),
    "object": dict,
    "array": list,
}


def _compare(value: Any, op: str, expected: Any) -> bool:
    """Compare a document field value against a single query operator.

    Returns True if the condition is satisfied, False otherwise.
    Unknown operators return True (ignored) instead of False to avoid
    breaking compound conditions like ``{"$regex": "x", "$options": "i"}``.
    """
    if op == "$eq":
        return value == expected
    if op == "$ne":
        return value != expected
    if op == "$gt":
        try:
            return value > expected
        except TypeError:
            return False
    if op == "$gte":
        try:
            return value >= expected
        except TypeError:
            return False
    if op == "$lt":
        try:
            return value < expected
        except TypeError:
            return False
    if op == "$lte":
        try:
            return value <= expected
        except TypeError:
            return False
    if op == "$in":
        return isinstance(expected, list) and value in expected
    if op == "$nin":
        return isinstance(expected, list) and value not in expected
    if op == "$regex":
        if isinstance(expected, str):
            flags = 0
            # $options is handled at a higher level — this is just the regex itself
            return bool(re.search(expected, str(value)))
        return False
    if op == "$options":
        # Handled alongside $regex at the compound level — always passes
        return True
    if op == "$exists":
        return (value is not None) == bool(expected)
    if op == "$not":
        if isinstance(expected, dict):
            return not _match_ops(value, expected)
        return True
    if op == "$type":
        expected_type = _TYPE_MAP.get(expected)
        if expected_type is None:
            return True
        return isinstance(value, expected_type)
    if op == "$size":
        if isinstance(value, list) and isinstance(expected, int):
            return len(value) == expected
        return False
    # Unknown operators are ignored (return True) so they don't break
    # compound conditions like {"$regex": "x", "$options": "i"}
    return True


def _match_ops(value: Any, ops: Dict[str, Any]) -> bool:
    """Match a value against multiple operators (e.g. ``{"$gt": 5, "$lt": 10}``).

    Handles ``$regex`` + ``$options`` combination.
    """
    # Handle $regex + $options together
    if "$regex" in ops and "$options" in ops:
        pattern = ops["$regex"]
        flags_str = ops["$options"]
        flags = 0
        if "i" in flags_str:
            flags |= re.IGNORECASE
        if "m" in flags_str:
            flags |= re.MULTILINE
        if "s" in flags_str:
            flags |= re.DOTALL
        if isinstance(pattern, str):
            if not re.search(pattern, str(value), flags):
                return False
        else:
            return False
        # Skip $regex and $options in the loop below
        skip = {"$regex", "$options"}
    else:
        skip = set()

    for op, expected in ops.items():
        if op in skip:
            continue
        if not _compare(value, op, expected):
            return False
    return True


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
            if not _match_ops(value, condition):
                return False
        else:
            if value != condition:
                return False

    return True
