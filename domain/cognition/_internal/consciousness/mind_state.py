"""Mind-state substrate — the append-only event log of the inner life.

Records (``episode | dream | reflection | belief_delta``) carry time,
payload, provenance (who wrote: ``experience | dream | reflection |
operator``) and edges, so the inner life is a queryable temporal graph —
the structured information environment that every renderer (router
projections, CLI, agent self-context, tests) reads instead of hand-shaping
its own dicts.

``self_model.json`` stays the state document (the checkpoint, windowed);
this log is the complete timeline — ``query_mind`` renders time. Sync, no
async, no network: importable and testable standalone (PGQ discipline).
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

RECORD_TYPES = ("episode", "dream", "reflection", "belief_delta")
PROVENANCES = ("experience", "dream", "reflection", "operator")


@dataclass
class MindRecord:
    """One fact of the inner life."""

    type: str
    payload: dict[str, Any]
    provenance: str
    t: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    edges: list[dict[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.type not in RECORD_TYPES:
            raise ValueError(f"unknown record type {self.type!r}; valid: {RECORD_TYPES}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "t": self.t,
            "type": self.type,
            "provenance": self.provenance,
            "payload": self.payload,
            "edges": list(self.edges),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MindRecord:
        return cls(
            id=d.get("id") or uuid.uuid4().hex,
            t=float(d.get("t", 0.0)),
            type=d["type"],
            provenance=d.get("provenance", "experience"),
            payload=d.get("payload", {}),
            edges=list(d.get("edges", [])),
        )


class MindLog:
    """Append-only JSONL log at ``{store_path}/mind_log.jsonl``.

    One writer (the engine, single process). Appends heal a torn tail from a
    crashed write (next record never fuses onto a fragment); readers skip
    torn or foreign lines — a bad line must never poison a query.
    """

    FILENAME = "mind_log.jsonl"

    def __init__(self, store_path: str | Path) -> None:
        self.path = Path(store_path) / self.FILENAME

    def append(self, record: MindRecord) -> MindRecord:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record.to_dict(), ensure_ascii=False) + "\n"
        needs_newline = False
        if self.path.exists() and self.path.stat().st_size > 0:
            with self.path.open("rb") as f:
                f.seek(-1, 2)
                needs_newline = f.read(1) != b"\n"
        with self.path.open("a", encoding="utf-8") as f:
            f.write(("\n" if needs_newline else "") + line)
        return record

    def query(
        self,
        *,
        since: float | None = None,
        until: float | None = None,
        record_type: str | None = None,
        limit: int = 100,
    ) -> list[MindRecord]:
        """Chronological records matching the filters (newest ``limit``)."""
        if not self.path.exists():
            return []
        out: list[MindRecord] = []
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = MindRecord.from_dict(json.loads(line))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue  # torn or foreign line: skip, never poison the query
                if since is not None and record.t < since:
                    continue
                if until is not None and record.t > until:
                    continue
                if record_type is not None and record.type != record_type:
                    continue
                out.append(record)
        out.sort(key=lambda r: r.t)
        return out[-limit:] if limit else out

    def latest(self) -> MindRecord | None:
        """The most recently appended record — O(1) byte-tail read.

        The status/stream hot path polls this every tick, so it must not
        scale with log size the way ``query()`` (full scan + sort) does:
        read one bounded window off the end of the file and scan its lines
        newest-first, returning the first that parses. Torn or foreign tail
        lines step aside, mirroring ``query()``'s skip discipline — and a
        line cut by the window boundary fails to parse the same way, so a
        pathological over-long record escalates to one full read rather
        than being silently missed.
        """
        if not self.path.exists() or self.path.stat().st_size == 0:
            return None
        with self.path.open("rb") as f:
            size = f.seek(0, 2)

            def _scan(window: bytes) -> MindRecord | None:
                for raw in reversed(window.split(b"\n")):
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        return MindRecord.from_dict(json.loads(raw))
                    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                        continue
                return None

            f.seek(max(size - 65536, 0))  # 64 KiB tail window: hundreds of records
            found = _scan(f.read())
            if found is None and size > 65536:
                f.seek(0)  # pathological: window entirely torn/foreign/over-long
                found = _scan(f.read())
            return found


def query_mind(
    store_path: str | Path,
    *,
    since: float | None = None,
    until: float | None = None,
    record_type: str | None = None,
    limit: int = 100,
) -> list[MindRecord]:
    """Standalone query — no engine instance, no loop, no network."""
    return MindLog(store_path).query(since=since, until=until, record_type=record_type, limit=limit)
