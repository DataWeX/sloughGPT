"""board_index — read-side MogDB query index over the kanban ``board.jsonl``.

The board stays a JSONL file: it is the source of truth, hand-editable,
versioned, and what every existing reader (``board show``, the GUI, ``verify``,
``export``) parses. This module adds a *derived* MogDB collection so callers
can run filtered queries (column / priority / assignee / tag / free text)
without re-reading the whole file on every call.

This is the third leg of the MogDB migration (after ``response_tracker`` and
the ``checkpoint_journal``) and follows the same contract:

- **JSONL is authoritative; the index is a read cache.** Nothing here ever
  writes, moves, or rewrites ``board.jsonl``. ``sync()`` reads via
  :meth:`~grib.store.PlannerStore.load_board` — the same defensive,
  column-coercing read ``board show`` renders — so the index can never
  disagree with the rendered board.
- **Reads are freshness-checked** against the source file's ``(mtime, size)``.
  A stale index silently re-syncs before answering, so a card added, moved,
  or deleted behind the journal's back is picked up. Cards absent from the
  file are pruned from the collection.
- **Silent fallback to a file read.** If MogDB is unavailable (not importable,
  path unwritable, corrupt journal) every entry point falls back to reading
  ``board.jsonl`` directly — callers get identical results, just slower.
- **Isolation.** The index lives inside ``board_dir`` (``<board_dir>/index``),
  so a custom ``--board-dir`` (tests, a second workspace) is fully isolated
  from the canonical ``~/.config/kanban`` store.

Filtering split: column / priority / assignee are pushed down to the MogDB
query (plain equality); tag and free-text are applied in Python on the result.
``match_document``'s plain equality does not do array containment (``tags`` is
a list), and a kanban board is tens-to-hundreds of cards where a post-filter
scan is microseconds — so pushing the equality fields down is the "MogDB-first"
half and the rest is an honest Python filter, not a missing index.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from . import config
from .store import PlannerStore

logger = logging.getLogger("grib.board_index")

_INDEX_DIRNAME = "index"
_CARDS_COLLECTION = "cards"
_META_COLLECTION = "meta"
_SOURCE_META_ID = "source"


def filter_cards(
    docs: list[dict[str, Any]],
    *,
    column: str | None = None,
    priority: str | None = None,
    assignee: str | None = None,
    tag: str | None = None,
    text: str | None = None,
) -> list[dict[str, Any]]:
    """Filter card dicts. Shared by the index path and the file-fallback path."""
    out = docs
    if column:
        out = [d for d in out if d.get("column") == column]
    if priority:
        out = [d for d in out if d.get("priority") == priority]
    if assignee:
        out = [d for d in out if d.get("assignee") == assignee]
    if tag:
        out = [d for d in out if tag in (d.get("tags") or [])]
    if text:
        needle = text.lower()
        out = [
            d
            for d in out
            if needle in (d.get("title") or "").lower()
            or needle in (d.get("description") or "").lower()
        ]
    return out


def _public(doc: dict[str, Any]) -> dict[str, Any]:
    """Strip MogDB bookkeeping keys, leaving the card's own fields."""
    return {k: v for k, v in doc.items() if k not in ("_id", "_created", "_updated")}


class BoardIndex:
    """Read-only MogDB query index over one board directory's ``board.jsonl``."""

    def __init__(self, board_dir: str | Path | None = None) -> None:
        self._board_dir = Path(board_dir) if board_dir else config.default_board_dir()
        # notes_dir is pointed at board_dir purely to keep PlannerStore's
        # mkdir side effect inside this board's tree — the index is read-only
        # and must never touch the canonical notes directory.
        self._store = PlannerStore(board_dir=self._board_dir, notes_dir=self._board_dir)
        self._dir = self._board_dir / _INDEX_DIRNAME
        self._db: Any = None
        self._cards: Any = None
        self._meta: Any = None

    @property
    def index_dir(self) -> Path:
        return self._dir

    def _ensure(self):
        """Lazily open the MogDB store. Raises if MogDB is unavailable."""
        if self._db is None:
            from mogdb import MogDB

            self._db = MogDB(str(self._dir))
            self._cards = self._db.collection(_CARDS_COLLECTION)
            self._meta = self._db.collection(_META_COLLECTION)
        return self._cards, self._meta

    # ── Source freshness ────────────────────────────────────────────────────

    def _source_stat(self) -> tuple[float, int] | None:
        src = self._board_dir / "board.jsonl"
        if not src.exists():
            return None
        st = src.stat()
        return (st.st_mtime, st.st_size)

    def _ensure_fresh(self) -> bool:
        """True when the index is usable (already fresh, or just refreshed)."""
        try:
            _, meta = self._ensure()
        except Exception as e:  # noqa: BLE001 — any MogDB failure → file fallback
            logger.debug("board index unavailable: %s", e)
            return False
        want = self._source_stat()
        got = meta.find_one({"_id": _SOURCE_META_ID})
        have = (got.get("mtime"), got.get("size")) if got else None
        if have != want:
            try:
                self.sync()
            except Exception as e:  # noqa: BLE001
                logger.warning("board index sync failed, falling back to file read: %s", e)
                return False
        return True

    # ── Sync ────────────────────────────────────────────────────────────────

    def sync(self) -> int:
        """Rebuild the collection from ``board.jsonl``. Returns card count.

        Stats the source *before* reading, then records that stat: if the file
        changes mid-read, the next query sees a mismatched mtime and re-syncs
        (the safe direction — never "fresh" on a half-read board).
        """
        cards, meta = self._ensure()
        want = self._source_stat()
        board = self._store.load_board()
        docs: list[dict[str, Any]] = []
        for card in board.cards:
            doc = card.to_dict()
            doc["_id"] = card.id  # card id as _id → upsert-by-id
            docs.append(doc)
        ids = [d["_id"] for d in docs]
        for doc in docs:
            if cards.find_one({"_id": doc["_id"]}) is not None:
                cards.find_one_and_replace({"_id": doc["_id"]}, doc)
            else:
                cards.insert_one(doc)
        # Prune anything the file no longer has. $nin with an empty list
        # matches every doc, so an empty board correctly clears the index.
        cards.delete_many({"_id": {"$nin": ids}})
        meta_doc = {
            "_id": _SOURCE_META_ID,
            "mtime": want[0] if want else None,
            "size": want[1] if want else None,
        }
        if meta.find_one({"_id": _SOURCE_META_ID}) is not None:
            meta.find_one_and_replace({"_id": _SOURCE_META_ID}, meta_doc)
        else:
            meta.insert_one(meta_doc)
        logger.debug("indexed %d card(s) from %s", len(docs), self._board_dir)
        return len(docs)

    def count(self) -> int:
        """Number of cards in the index (freshness-checked)."""
        if self._ensure_fresh():
            cards, _ = self._ensure()
            return cards.count()
        return len(self._store.load_board().cards)

    # ── Query ───────────────────────────────────────────────────────────────

    def query(
        self,
        *,
        column: str | None = None,
        priority: str | None = None,
        assignee: str | None = None,
        tag: str | None = None,
        text: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return matching cards as plain dicts (MogDB-backed, file fallback)."""
        if self._ensure_fresh():
            try:
                cards, _ = self._ensure()
                pushed: dict[str, Any] = {}
                if column:
                    pushed["column"] = column
                if priority:
                    pushed["priority"] = priority
                if assignee:
                    pushed["assignee"] = assignee
                docs = cards.find(pushed or None)
                out = [_public(d) for d in filter_cards(docs, tag=tag, text=text)]
                return out[:limit] if limit else out
            except Exception as e:  # noqa: BLE001
                logger.warning("board index query failed, falling back to file read: %s", e)
        return self._query_file(
            column=column,
            priority=priority,
            assignee=assignee,
            tag=tag,
            text=text,
            limit=limit,
        )

    def _query_file(
        self,
        *,
        column: str | None = None,
        priority: str | None = None,
        assignee: str | None = None,
        tag: str | None = None,
        text: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        board = self._store.load_board()
        docs = [c.to_dict() for c in board.cards]
        out = filter_cards(
            docs, column=column, priority=priority, assignee=assignee, tag=tag, text=text
        )
        return out[:limit] if limit else out

    def close(self) -> None:
        if self._db is not None:
            try:
                self._db.close()
            except Exception as e:  # noqa: BLE001
                logger.debug("board index close failed: %s", e)
            self._db = None
            self._cards = None
            self._meta = None
