"""
app_planner.notes_controller — the application service for note operations.

``packages/app-planner`` carries two note dialects:

* ``core.NoteStore``  — ``list_notes / get / create / update / delete / search``
  (legacy; used by ``gui.py`` and the old ``notes`` binary),
* ``store.PlannerStore`` — ``list_notes / get_note / create_note /
  update_note / delete_note / search_notes / sync`` (used by ``cli.py``).

Both are *data layers*: they persist, they do not decide what a valid note is
and they do not decide what a filtered list means. Those decisions were being
re-made at each call site — ``gui.py`` re-implemented title/status/tag
validation in its HTTP handlers and inlined the tag/status/query/limit read
model, while ``cli.py`` passed raw argv straight through. Two implementations
of one contract, which is exactly the divergence ``sync.py`` already had to
absorb by accepting either dialect.

This module is that contract, in one place. Projections (CLI subcommands,
HTTP handlers, the future planner interface) call *here*; the stores stay the
data layer and keep owning file IO.

Design rules:

* **No file IO in this controller.** Delegation only.
* **Dialect-tolerant, not dialect-hiding.** Both stores are supported through
  an explicit branch per operation, so the differences stay visible and
  testable instead of being papered over with ``**kwargs`` forwarding.
* **Fail loudly on fields a dialect cannot store.** ``PlannerStore`` has no
  ``author``; ``update_note`` would silently ignore it. Silent drops are data
  loss, so that is a ``NoteValidationError``.
* **Read model is derived once.** ``list_notes()``/``search()``/``tags()``/
  ``stats()`` are the single implementation every projection reuses. The read
  method is named ``list_notes``, not ``list``: inside a class body a method
  called ``list`` shadows the builtin, and every later ``-> list[...]``
  annotation in that class would then resolve to the method instead of the
  type (mypy ``valid-type``).
"""

from __future__ import annotations

import threading
from datetime import date
from typing import Any, Iterable

from . import config

__all__ = ["NotesController", "NoteValidationError", "NoteStoreError"]

#: Fields a note may carry, per dialect. The controller rejects anything else
#: rather than letting a store drop it silently.
_PLANNER_FIELDS = frozenset({"title", "body", "status", "tags", "sprint", "gh", "assignee"})
_LEGACY_FIELDS = _PLANNER_FIELDS | {"author"}

#: Both stores are asked for far more than the caller wants: the legacy store
#: applies ``limit`` *before* any controller-side query filter, which would
#: truncate the candidate set and make ``query`` silently miss notes.
_STORE_FETCH_LIMIT = 9999

#: Read-model default — matches the GUI's ``/api/notes`` page size.
DEFAULT_LIST_LIMIT = 500


class NoteValidationError(ValueError):
    """A note payload rejected before it reached the store."""


class NoteStoreError(RuntimeError):
    """The store violated the operation's contract after reporting success."""


def _normalise_title(title: Any) -> str:
    if not isinstance(title, str) or not title.strip():
        raise NoteValidationError("title is required")
    return title.strip()


def _normalise_status(status: Any) -> str:
    if not isinstance(status, str) or status not in config.STATUSES:
        raise NoteValidationError(
            f"invalid status: {status!r} (expected one of {', '.join(config.STATUSES)})"
        )
    return status


def _normalise_tags(tags: Any) -> list[str]:
    """Strip, drop empties, de-duplicate — order preserved.

    Order matters: tags render as chips in the GUI and are re-used as card
    tags on sync, so a stable order keeps sync idempotent.
    """
    if tags is None:
        return []
    if isinstance(tags, str):
        tags = [tags]
    if not isinstance(tags, (list, tuple)):
        raise NoteValidationError(f"tags must be a list of strings, got {type(tags).__name__}")
    out: list[str] = []
    for tag in tags:
        if not isinstance(tag, str):
            raise NoteValidationError(f"tags must be strings, got {type(tag).__name__}")
        cleaned = tag.strip()
        if cleaned and cleaned not in out:
            out.append(cleaned)
    return out


def _normalise_limit(limit: Any) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise NoteValidationError(f"limit must be a non-negative integer, got {limit!r}")
    return limit


def _matches(note: Any, query: str) -> bool:
    """Case-insensitive hit across title, body and tags (author when present).

    Superset of both prior implementations: ``gui.py`` matched
    title/tags/body, ``PlannerStore.search_notes`` matched title/body/tags,
    ``NoteStore.search`` also matched author. One rule, all three fields.
    """
    haystacks: Iterable[Any] = (
        getattr(note, "title", ""),
        getattr(note, "body", ""),
        getattr(note, "tags", []),
        getattr(note, "author", ""),
    )
    for haystack in haystacks:
        if isinstance(haystack, str):
            if query in haystack.lower():
                return True
        else:
            if any(query in str(item).lower() for item in haystack):
                return True
    return False


def _date_str(note: Any) -> str:
    """Creation date as ``YYYY-MM-DD``, for both dialects.

    ``core.Note`` exposes ``date_str`` as a property; ``store.Note`` has no
    such property, so fall back to slicing ``created_at``.
    """
    value = getattr(note, "date_str", None)
    if isinstance(value, str) and value:
        return value
    created = getattr(note, "created_at", "") or ""
    return created[:10]


class NotesController:
    """Validates, filters and dispatches note operations to either store.

    Args:
        store: a ``core.NoteStore`` or ``store.PlannerStore``.
        lock: optional guard. ``gui.py`` serialises mutations through a
            process-wide ``threading.Lock``; passing it here keeps the
            controller usable from both the threaded HTTP server and single
            threaded CLI without the controller owning a lock of its own.
    """

    def __init__(self, store: Any, lock: threading.Lock | None = None):
        if hasattr(store, "create_note") and hasattr(store, "list_notes"):
            self._dialect = "planner"
            self._fields = _PLANNER_FIELDS
        elif hasattr(store, "create") and hasattr(store, "list_notes"):
            self._dialect = "legacy"
            self._fields = _LEGACY_FIELDS
        else:
            raise TypeError(
                f"unsupported note store {type(store).__name__}: expected core.NoteStore "
                "or store.PlannerStore"
            )
        self.store = store
        self.lock = lock

    # ── Introspection ─────────────────────────────────────────────────

    @property
    def dialect(self) -> str:
        """``"planner"`` (``PlannerStore``) or ``"legacy"`` (``NoteStore``)."""
        return self._dialect

    @property
    def fields(self) -> frozenset[str]:
        """Writable fields for this store dialect."""
        return self._fields

    # ── Linkage: note status ↔ board column ──────────────────────────

    @staticmethod
    def column_for(note_or_status: Any) -> str:
        """Board column a note belongs in (``config.STATUS_TO_COLUMN``)."""
        status = (
            note_or_status
            if isinstance(note_or_status, str)
            else getattr(note_or_status, "status", "")
        )
        return config.STATUS_TO_COLUMN.get((status or "").lower(), "todo")

    @staticmethod
    def status_for(column: str) -> str:
        """Note status a board column maps back to (``config.COLUMN_TO_STATUS``)."""
        return config.COLUMN_TO_STATUS.get(column, "open")

    # ── Read model ────────────────────────────────────────────────────

    def list_notes(
        self,
        tag: str | None = None,
        status: str | None = None,
        query: str | None = None,
        limit: int = DEFAULT_LIST_LIMIT,
        author: str | None = None,
        sprint: str | None = None,
        today: bool = False,
    ) -> list[Any]:
        """Notes matching *tag*/*status*/*author*/*sprint*/*query*, newest window first.

        Named ``list_notes`` rather than ``list`` so the method does not
        shadow the builtin inside the class body (see module docstring).

        The store does what it can natively (both dialects filter tag/status;
        the legacy one also filters author/sprint/today) and the controller
        applies the rest. ``limit`` is applied **last**, after the query —
        truncating first would hide matching notes behind an unrelated page
        of results.
        """
        limit = _normalise_limit(limit)
        if status is not None:
            _normalise_status(status)

        kwargs: dict[str, Any] = {"limit": _STORE_FETCH_LIMIT}
        if tag is not None:
            kwargs["tag"] = tag
        if status is not None:
            kwargs["status"] = status
        if self._dialect == "legacy":
            if author is not None:
                kwargs["author"] = author
            if sprint is not None:
                kwargs["sprint"] = sprint
            if today:
                kwargs["today"] = True
        elif author or sprint or today:
            raise NoteValidationError(
                f"{self._dialect} store does not support author/sprint/today filters"
            )

        # `store` is Any-typed (either dialect), so bind the result to keep
        # the declared return type honest for mypy.
        notes: list[Any] = self.store.list_notes(**kwargs)

        if query:
            needle = query.strip().lower()
            if needle:
                notes = [n for n in notes if _matches(n, needle)]
        return notes[:limit]

    def search(self, query: str, **filters: Any) -> list[Any]:
        """``list_notes()`` with *query* required — the shape callers expect by name."""
        if not isinstance(query, str) or not query.strip():
            raise NoteValidationError("query is required")
        # `**filters` is Any-typed, so mypy loses the call's return type;
        # re-bind it to keep the declared `list[Any]` honest.
        notes: list[Any] = self.list_notes(query=query, **filters)
        return notes

    def get(self, note_id: str) -> Any | None:
        """Fetch by id. Both dialects resolve unambiguous id prefixes."""
        if not isinstance(note_id, str) or not note_id.strip():
            raise NoteValidationError("note id is required")
        if self._dialect == "planner":
            return self.store.get_note(note_id)
        return self.store.get(note_id)

    def tags(self) -> dict[str, int]:
        """Tag → count across every note (the board's tag chip source)."""
        counts: dict[str, int] = {}
        for note in self.store.list_notes(limit=_STORE_FETCH_LIMIT):
            for tag in getattr(note, "tags", []) or []:
                counts[tag] = counts.get(tag, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

    def stats(self) -> dict[str, Any]:
        """``{total, today, by_status}`` — one store read, not the two the GUI made."""
        notes = self.store.list_notes(limit=_STORE_FETCH_LIMIT)
        today_str = date.today().isoformat()
        by_status: dict[str, int] = {}
        today = 0
        for note in notes:
            status = getattr(note, "status", "") or "open"
            by_status[status] = by_status.get(status, 0) + 1
            if _date_str(note) == today_str:
                today += 1
        return {"total": len(notes), "today": today, "by_status": by_status}

    @staticmethod
    def to_dict(note: Any) -> dict[str, Any]:
        """JSON-ready read model — emits only the fields the note actually has.

        The two ``Note`` dataclasses differ (``author`` is legacy-only), so a
        fixed key set would either 500 on ``None`` or invent empty fields.
        """
        keys = (
            "id",
            "title",
            "body",
            "status",
            "tags",
            "sprint",
            "gh",
            "assignee",
            "author",
            "created_at",
            "updated_at",
        )
        out: dict[str, Any] = {}
        for key in keys:
            try:
                value = getattr(note, key)
            except AttributeError:
                continue
            if key == "tags":
                out[key] = list(value or [])
            elif value is not None:
                out[key] = value
        return out

    # ── Mutations ─────────────────────────────────────────────────────

    def create(
        self,
        title: str,
        body: str = "",
        status: str = "open",
        tags: list[str] | None = None,
        sprint: str = "",
        gh: str = "",
        assignee: str = "",
        author: str = "",
    ) -> Any:
        """Validate then persist a new note. Raises ``NoteValidationError`` first."""
        payload = self._validate(
            {
                "title": title,
                "body": body,
                "status": status,
                "tags": tags,
                "sprint": sprint,
                "gh": gh,
                "assignee": assignee,
                "author": author,
            },
            creating=True,
        )
        # An empty optional field is just "not supplied"; a populated one the
        # dialect cannot store (author on PlannerStore) is rejected rather
        # than dropped, or the caller believes it was saved when it was not.
        unstorable = {k for k, v in payload.items() if k not in self._fields and v}
        if unstorable:
            raise NoteValidationError(
                f"field(s) {', '.join(sorted(unstorable))} are not storable by the "
                f"{self._dialect} store"
            )
        kwargs = {k: v for k, v in payload.items() if k in self._fields}
        if self._dialect == "planner":
            return self._mutate(lambda: self.store.create_note(**kwargs))
        return self._mutate(lambda: self.store.create(**kwargs))

    def update(self, note_id: str, **fields: Any) -> Any | None:
        """Partial update. Returns ``None`` when the note does not exist."""
        if not fields:
            raise NoteValidationError("no fields to update")
        payload = self._validate(fields, creating=False)
        unknown = set(payload) - self._fields
        if unknown:
            raise NoteValidationError(
                f"field(s) {', '.join(sorted(unknown))} are not storable by the "
                f"{self._dialect} store"
            )
        note_id = _require_id(note_id)
        if self._dialect == "planner":
            return self._mutate(lambda: self.store.update_note(note_id, **payload))
        return self._mutate(lambda: self.store.update(note_id, **payload))

    def delete(self, note_id: str) -> bool:
        """Remove a note, then prove it.

        Returns ``True`` only when the note is actually gone; ``False`` when
        nothing matched (callers map that to 404). If the store reports a
        successful delete while the note is still readable, ``NoteStoreError``
        is raised rather than returning ``True``.

        The verification exists because ``PlannerStore`` cannot currently
        express intent to delete: ``_write_notes`` unions the caller's
        snapshot with every id it finds on disk ("a record this snapshot
        never saw" is kept verbatim), and a deliberately removed note is
        indistinguishable from one the snapshot never saw — so it is
        re-appended on write. ``delete_note()`` then returns ``True`` while
        the record survives. Returning that ``True`` would make every
        projection (``notes rm``, a future planner interface) tell the user
        something false, so the seam checks instead of trusting.
        """
        note_id = _require_id(note_id)
        if self._dialect == "planner":
            deleted = self._mutate(lambda: self.store.delete_note(note_id))
        else:
            deleted = self._mutate(lambda: self.store.delete(note_id))
        if not deleted:
            return False
        if self.get(note_id) is not None:
            raise NoteStoreError(
                f"store reported deleting note {note_id!r} but it is still readable — "
                "PlannerStore._write_notes re-appends ids found on disk, so a deletion "
                "cannot survive the union-on-write merge"
            )
        return True

    def sync(self, kanban_store: Any | None = None) -> tuple[int, int, int]:
        """Reconcile notes → board cards. Returns ``(added, updated, total)``.

        ``PlannerStore`` owns both sides and syncs itself. The legacy dialect
        only has the note half, so it needs the board handed in — the same
        contract ``sync.sync_notes_to_board`` already implements.
        """
        if self._dialect == "planner":
            return tuple(self.store.sync())  # type: ignore[return-value]
        if kanban_store is None:
            raise NoteValidationError(
                "sync() on a legacy NoteStore requires the kanban store to sync against"
            )
        from .sync import sync_notes_to_board

        return sync_notes_to_board(self.store, kanban_store)

    # ── Internals ─────────────────────────────────────────────────────

    def _mutate(self, fn: Any) -> Any:
        """Run *fn* under the caller's lock when one was supplied.

        The GUI holds a process-wide lock across a read-modify-write; without
        it here, two requests could interleave a stale read and a write and
        the union-on-write merge in ``PlannerStore`` would have to absorb the
        conflict. Keeping the lock at this seam means CLI callers (no lock)
        and the HTTP server (locked) share one mutation path.
        """
        if self.lock is None:
            return fn()
        with self.lock:
            return fn()

    def _validate(self, fields: dict[str, Any], creating: bool) -> dict[str, Any]:
        """Normalise and reject a payload before it reaches a store."""
        out: dict[str, Any] = {}
        for key, value in fields.items():
            if value is None:
                # None means "leave this field alone" on a partial update.
                continue
            if key == "title":
                out[key] = _normalise_title(value)
            elif key == "status":
                out[key] = _normalise_status(value)
            elif key == "tags":
                out[key] = _normalise_tags(value)
            elif key in ("body", "sprint", "gh", "assignee", "author"):
                if not isinstance(value, str):
                    raise NoteValidationError(f"{key} must be a string, got {type(value).__name__}")
                out[key] = value
            else:
                raise NoteValidationError(f"unknown note field: {key}")
        if creating:
            _normalise_title(out.get("title", ""))
            _normalise_status(out.get("status", "open"))
        return out


def _require_id(note_id: Any) -> str:
    if not isinstance(note_id, str) or not note_id.strip():
        raise NoteValidationError("note id is required")
    return note_id
