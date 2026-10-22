"""Cruncher — streaming clean→snapshot pipeline for weight-training data.

Turns raw text streams (live conversation corpus, imported packs) into
immutable, versioned training snapshots that feed ``SloughGPTTrainer``
through the existing dataset manifest adapter.  See
``domain/training/_internal/cruncher/pipeline.py`` for the entry points.

Usage::

    from domain.training.cruncher import Cruncher, conversation_records

    cruncher = Cruncher("api-conversations")
    cruncher.crunch_stream(conversation_records("data/api_conversations/corpus.jsonl"))
    ref = cruncher.seal(sources=["api-conversations"])
"""

from __future__ import annotations

from .adapter import resolve_latest, resolve_snapshot, trainer_data_path
from .clean import (
    DedupIndex,
    QualityThresholds,
    estimate_jaccard,
    quality_ok,
    scrub_pii,
    text_fingerprint,
)
from .pipeline import Cruncher, CrunchResult, conversation_records
from .snapshot import SnapshotError, SnapshotRef, SnapshotStore

__all__ = [
    "CrunchResult",
    "Cruncher",
    "DedupIndex",
    "QualityThresholds",
    "SnapshotError",
    "SnapshotRef",
    "SnapshotStore",
    "conversation_records",
    "estimate_jaccard",
    "quality_ok",
    "resolve_latest",
    "resolve_snapshot",
    "scrub_pii",
    "text_fingerprint",
    "trainer_data_path",
]
