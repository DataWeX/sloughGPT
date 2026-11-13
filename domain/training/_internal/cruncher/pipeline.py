"""Cruncher pipeline — streaming ingest to immutable training snapshots.

``crunch_stream`` is the primary interface: records flow in one at a time
(live corpus tail, pack import, anything iterable), each is scrubbed,
quality-gated, and deduped against the *entire persisted corpus history* —
not a batch boundary.  ``crunch`` is a batch convenience over the same path.

Sealing writes a new immutable ``v<N>`` via :class:`SnapshotStore`; live
ingest never mutates a published snapshot.  The dedup index state travels
with each snapshot, so the next crunch (next run, next pack) resumes the
corpus memory where the last seal left it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .clean import (
    DedupIndex,
    QualityThresholds,
    quality_ok,
    scrub_pii,
    text_fingerprint,
)
from .snapshot import SnapshotRef, SnapshotStore

logger = logging.getLogger("slo.training.cruncher")


@dataclass
class CrunchResult:
    """Counters for one crunch run (stream or batch)."""

    records_seen: int = 0
    records_kept: int = 0
    dropped_quality: int = 0
    dropped_duplicate: int = 0
    pii_replacements: int = 0
    bytes_in: int = 0
    sources: list[str] = field(default_factory=list)

    @property
    def dedup_rate(self) -> float:
        return self.dropped_duplicate / self.records_seen if self.records_seen else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "records_seen": self.records_seen,
            "records_kept": self.records_kept,
            "dropped_quality": self.dropped_quality,
            "dropped_duplicate": self.dropped_duplicate,
            "pii_replacements": self.pii_replacements,
            "bytes_in": self.bytes_in,
            "dedup_rate": round(self.dedup_rate, 4),
            "sources": list(self.sources),
        }


class Cruncher:
    """Streaming cleaner + immutable snapshotter for one dataset.

    Args:
        dataset_id: identity under the snapshot store.
        store: snapshot store (default: ``data/crunched/``).
        quality: quality-gate thresholds.
        dedup: LSH index; restored from the latest snapshot when None, so
            cross-snapshot dedup is on by default.
    """

    def __init__(
        self,
        dataset_id: str,
        *,
        store: SnapshotStore | None = None,
        quality: QualityThresholds | None = None,
        dedup: DedupIndex | None = None,
    ):
        self._dataset_id = dataset_id
        self._store = store or SnapshotStore()
        self._quality = quality or QualityThresholds()
        self._dedup = dedup if dedup is not None else self._restore_dedup()
        self._buffer: list[str] = []
        self.result = CrunchResult()

    def _restore_dedup(self) -> DedupIndex:
        """Load the dedup index from the latest sealed snapshot, if any."""
        ref = self._store.latest(self._dataset_id)
        if ref is None:
            return DedupIndex()
        state = self._store.load_dedup_state(ref)
        if state is None:
            logger.warning(
                "%s v%d has no dedup state; starting a fresh index",
                self._dataset_id,
                ref.version,
                extra={"tag": "TRAIN"},
            )
            return DedupIndex()
        return DedupIndex.from_state(state)

    def _ingest_record(self, text: str, source: str | None) -> str | None:
        """Clean one record; return trainable text or None if dropped."""
        self.result.records_seen += 1
        self.result.bytes_in += len(text.encode("utf-8"))
        if source and source not in self.result.sources:
            self.result.sources.append(source)

        cleaned, pii_n = scrub_pii(text)
        cleaned = cleaned.strip()
        self.result.pii_replacements += pii_n

        if not quality_ok(cleaned, self._quality):
            self.result.dropped_quality += 1
            return None

        sig = text_fingerprint(cleaned)
        if self._dedup.is_near_duplicate(sig):
            self.result.dropped_duplicate += 1
            return None
        self._dedup.add(sig)
        self.result.records_kept += 1
        self._buffer.append(cleaned)
        return cleaned

    def crunch_stream(
        self,
        records: Iterable[str],
        *,
        source: str | None = None,
    ) -> CrunchResult:
        """Ingest a live stream of raw text records into the pending buffer.

        Each record is cleaned and checked against the full corpus history
        immediately — streaming-safe, no batch boundaries.  Call :meth:`seal`
        to freeze the buffer into an immutable snapshot.
        """
        for record in records:
            if not isinstance(record, str):
                raise TypeError(f"crunch_stream expects str records, got {type(record).__name__}")
            self._ingest_record(record, source)
        return self.result

    def seal(
        self,
        *,
        sources: list[str] | None = None,
        mix: dict[str, float] | None = None,
        domain: str = "general",
    ) -> SnapshotRef:
        """Freeze the pending buffer as the next immutable snapshot.

        Returns:
            SnapshotRef of the sealed version.

        Raises:
            SnapshotError: buffer is empty (nothing to seal).
        """
        if not self._buffer:
            from .snapshot import SnapshotError

            raise SnapshotError(f"nothing to seal for {self._dataset_id!r}: buffer is empty")
        all_sources = list(dict.fromkeys([*(sources or []), *self.result.sources]))
        ref = self._store.seal(
            self._dataset_id,
            self._buffer,
            sources=all_sources,
            mix=mix,
            domain=domain,
            dedup_state=self._dedup.to_state(),
            stats=self.result.as_dict(),
        )
        self._buffer.clear()
        return ref

    def crunch(
        self,
        records: Iterable[str],
        *,
        source: str | None = None,
        sources: list[str] | None = None,
        mix: dict[str, float] | None = None,
        domain: str = "general",
    ) -> SnapshotRef:
        """Batch convenience: ingest everything, then seal immediately."""
        self.crunch_stream(records, source=source)
        return self.seal(sources=sources, mix=mix, domain=domain)


def conversation_records(corpus_path: Path | str) -> Iterable[str]:
    """Yield assistant-reply text records from a ConversationLogger corpus.

    The corpus (``data/api_conversations/corpus.jsonl``) is the live source:
    the model's own API conversations, appended continuously.  Each line is
    ``{"messages": [{"role", "content"}, ...], "meta": {...}}``; we yield the
    assistant turns (the model's outputs — train-≡-serve text).

    Malformed lines are skipped with a warning, never raise — a live tail
    must not die on one bad record.
    """
    path = Path(corpus_path)
    if not path.is_file():
        raise FileNotFoundError(f"conversation corpus not found: {path}")
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("corpus line %d: bad JSON, skipped", lineno, extra={"tag": "TRAIN"})
                continue
            for msg in rec.get("messages", []):
                if msg.get("role") == "assistant" and msg.get("content"):
                    yield str(msg["content"])
