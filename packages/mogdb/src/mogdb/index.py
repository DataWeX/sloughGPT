"""Field indexes for MogDB collections.

Provides hash-based indexes for fast equality lookups and sorted indexes
for range queries.
"""

from typing import Any, Dict, List, Optional, Set


class Index:
    """A hash index on a single field for O(1) equality lookups.

    Maintains a ``field_value -> set[doc_id]`` mapping that is updated
    on insert/update/delete.

    Parameters
    ----------
    field:
        The document field (dot-separated for nested) to index.
    unique:
        If True, duplicate field values raise ``ValueError``.
    """

    def __init__(self, field: str, unique: bool = False):
        self.field = field
        self.unique = unique
        self._map: Dict[Any, Set[str]] = {}

    def add(self, doc_id: str, field_value: Any) -> None:
        if self.unique and field_value in self._map:
            existing = self._map[field_value]
            if existing and doc_id not in existing:
                raise ValueError(
                    f"Unique index violation on {self.field!r}: {field_value!r}"
                )
        self._map.setdefault(field_value, set()).add(doc_id)

    def remove(self, doc_id: str, field_value: Any) -> None:
        s = self._map.get(field_value)
        if s:
            s.discard(doc_id)
            if not s:
                del self._map[field_value]

    def update(self, doc_id: str, old_value: Any, new_value: Any) -> None:
        self.remove(doc_id, old_value)
        self.add(doc_id, new_value)

    def lookup(self, field_value: Any) -> List[str]:
        """Return all document IDs matching *field_value*."""
        return list(self._map.get(field_value, set()))

    def clear(self) -> None:
        self._map.clear()


class SortedIndex:
    """A sorted index for range queries on numeric/string fields.

    Maintains a list of ``(field_value, doc_id)`` pairs kept in sorted
    order. Supports ``$gt``, ``$gte``, ``$lt``, ``$lte`` lookups.
    """

    def __init__(self, field: str):
        self.field = field
        self._entries: List[Any] = []

    def add(self, doc_id: str, field_value: Any) -> None:
        import bisect

        pair = (field_value, doc_id)
        idx = bisect.bisect_left(self._entries, pair)
        self._entries.insert(idx, pair)

    def remove(self, doc_id: str, field_value: Any) -> None:
        self._entries = [
            (v, i) for v, i in self._entries if not (v == field_value and i == doc_id)
        ]

    def range(self, gte: Any = None, lte: Any = None) -> List[str]:
        """Return doc IDs where field is in [gte, lte]."""
        if gte is None and lte is None:
            return [i for _, i in self._entries]

        import bisect

        start = (
            bisect.bisect_left(self._entries, (gte, ""))
            if gte is not None
            else 0
        )
        end = (
            bisect.bisect_right(self._entries, (lte, "\uffff"))
            if lte is not None
            else len(self._entries)
        )
        return [i for _, i in self._entries[start:end]]
