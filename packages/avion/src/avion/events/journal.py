"""Event journal — append-only JSONL source of truth for avion events.

One event per line, flushed on every append. Everything else in the event
layer (`EventRecorder`, `StructuredLogger`) is a *view* over this file —
never an independent writer.

Stdlib-only on purpose: avion's logger must work standalone, with no
``domain/`` or third-party logging dependency. Enforced by
``tests/test_stdlib_only.py``.

Invariants:

* **no-clobber** — opened for append; existing history is never truncated.
* **monotonic seq** — sequence numbers continue from the highest value on
  disk when reopened, never restarting at 1.
* **torn-tail safe** — a crash mid-write leaves an incomplete last line.
  Opening appends a newline so later appends stay parseable (the fragment
  is kept, not deleted) and recovers its sequence number by regex so that
  value is never reused.
* **thread-safe** — sequence assignment and the write happen under one
  lock, so concurrent writers cannot interleave or duplicate a seq.
"""

from __future__ import annotations

import json
import os
import re
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TextIO

from avion.events.models import Event, EventType

_SEQ_RE = re.compile(r'"seq"\s*:\s*(\d+)')


def _event_from_dict(d: Any) -> Event | None:
    """Rebuild an Event from a stored line. Returns None for junk lines."""
    if not isinstance(d, dict):
        return None
    raw_type = d.get("type", "")
    try:
        event_type = EventType(str(raw_type))
    except ValueError:
        event_type = EventType.CUSTOM
    return Event(
        type=event_type,
        name=str(d.get("name", "")),
        data=dict(d.get("data") or {}),
        success=bool(d.get("success", True)),
        error=str(d.get("error", "")),
        duration_ms=float(d.get("duration_ms") or 0.0),
        timestamp=float(d.get("timestamp") or 0.0),
        seq=int(d.get("seq") or 0),
    )


class EventJournal:
    """Append-only JSONL journal.

    Usage::

        journal = EventJournal("avion_output/events.jsonl")
        journal.append(event)      # seq assigned, line flushed
        for stored in journal:     # canonical view, survives reopen
            ...
        journal.close()
    """

    def __init__(self, path: str | Path, *, fsync: bool = False):
        self.path = Path(path)
        self._fsync = fsync
        self._lock = threading.Lock()
        self._fh: TextIO | None = None
        self._recovered_seq = self._scan()

    # ── opening ────────────────────────────────────────────────────────

    def _scan(self) -> int:
        """Recover the highest stored seq and isolate a torn final line."""
        if not self.path.exists():
            return 0
        raw = self.path.read_bytes()
        if not raw:
            return 0

        highest = 0
        # A torn tail can hold partial UTF-8; decode leniently so the
        # lines before it still yield their sequence numbers.
        for line in raw.decode("utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except ValueError:
                match = _SEQ_RE.search(line)
                if match:
                    highest = max(highest, int(match.group(1)))
                continue
            if isinstance(parsed, dict) and isinstance(parsed.get("seq"), int):
                highest = max(highest, int(parsed["seq"]))

        if not raw.endswith(b"\n"):
            # Keep the fragment (deleting history is forbidden here) but put
            # a newline after it so the next append starts its own line.
            with open(self.path, "ab") as fh:
                fh.write(b"\n")
                fh.flush()
        return highest

    def _handle(self) -> TextIO:
        """Open for append on first use (lazy: creating a journal writes nothing)."""
        if self._fh is None or self._fh.closed:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "a", encoding="utf-8")
        return self._fh

    # ── writing ────────────────────────────────────────────────────────

    @property
    def seq(self) -> int:
        """Highest sequence number handed out so far."""
        with self._lock:
            return self._recovered_seq

    def append(self, event: Event) -> Event:
        """Assign the next seq, serialize, append and flush the line."""
        with self._lock:
            self._recovered_seq += 1
            event.seq = self._recovered_seq
            line = json.dumps(event.to_dict(), default=str) + "\n"
            fh = self._handle()
            fh.write(line)
            # Flush per event: a reader in another handle/process must see
            # the event as soon as append() returns (durability contract).
            fh.flush()
            if self._fsync:
                os.fsync(fh.fileno())
        return event

    # ── reading (the canonical view) ───────────────────────────────────

    def __iter__(self) -> Iterator[Event]:
        if not self.path.exists():
            return
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                except ValueError:
                    continue  # torn/partial line — never parsed as an event
                event = _event_from_dict(parsed)
                if event is not None:
                    yield event

    def events(self) -> list[Event]:
        """Every stored event, in order."""
        return list(self)

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def close(self) -> None:
        """Flush and close. Appends after close transparently reopen."""
        with self._lock:
            if self._fh is not None and not self._fh.closed:
                self._fh.close()
            self._fh = None

    def __enter__(self) -> EventJournal:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
