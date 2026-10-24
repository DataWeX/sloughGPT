"""ModalitySequence — the ``@stream`` block convention for multi-modal data.

A ``ModalitySequence`` is one stream of encoded data from a single modality
(text, voice, image, binary), rendered as an AML block::

    @stream text
        payload = "The quick brown fox jumps over the lazy dog."
        pair_id = "conv-42"
        source = "wikipedia"

    @stream voice
        payload = [759, 121, 118, 3]
        pair_id = "conv-42"

``pair_id`` links paired sequences (e.g. a voice utterance and its text
transcript) so downstream consumers can align modalities without guessing.
Payload is a string (text modalities) or an inline list of ints (token /
feature id sequences for voice/image/binary).

This module is the format home for the cruncher pipeline: it defines the
shape only. Cleaning, dedup, and snapshotting live in
``domain/training/_internal/cruncher``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aml.parser import parse
from aml.schema import AmlBlock, AmlDocument
from aml.serializer import serialize

#: Modalities the format knows about. Text ships first; voice/image/binary
#: are reserved so adapters can emit them without a format change.
MODALITIES: frozenset[str] = frozenset({"text", "voice", "image", "binary"})

STREAM_TAG = "stream"


@dataclass
class ModalitySequence:
    """One modality's encoded data stream.

    Attributes:
        modality: one of ``MODALITIES``.
        payload: text content (str) or token/feature ids (list[int]).
        pair_id: optional link grouping paired sequences across modalities.
        metadata: extra key/value context (source, language, sample_rate...).
    """

    modality: str
    payload: str | list[int]
    pair_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.modality not in MODALITIES:
            raise ValueError(
                f"unknown modality {self.modality!r}; expected one of {sorted(MODALITIES)}"
            )
        if not isinstance(self.payload, (str, list)):
            raise TypeError(f"payload must be str or list[int], got {type(self.payload).__name__}")
        if isinstance(self.payload, list) and not all(isinstance(x, int) for x in self.payload):
            raise TypeError("payload lists must contain only ints (token/feature ids)")

    def to_block(self) -> AmlBlock:
        """Render as an AML ``@stream`` block."""
        meta: dict[str, Any] = {"payload": self.payload}
        if self.pair_id is not None:
            meta["pair_id"] = self.pair_id
        meta.update(self.metadata)
        return AmlBlock(tag=STREAM_TAG, name=self.modality, metadata=meta)

    @classmethod
    def from_block(cls, block: AmlBlock) -> ModalitySequence:
        """Build from an AML ``@stream`` block.

        Raises:
            ValueError: block tag is not ``stream`` or payload is missing.
        """
        if block.tag != STREAM_TAG:
            raise ValueError(f"expected @{STREAM_TAG} block, got @{block.tag}")
        if block.name not in MODALITIES:
            raise ValueError(
                f"@stream name must be one of {sorted(MODALITIES)}, got {block.name!r}"
            )
        meta = dict(block.metadata)
        if "payload" not in meta:
            raise ValueError("@stream block missing 'payload'")
        payload = meta.pop("payload")
        pair_id = meta.pop("pair_id", None)
        return cls(modality=block.name, payload=payload, pair_id=pair_id, metadata=meta)


def sequences_to_aml(sequences: list[ModalitySequence], *, version: str = "1.0") -> str:
    """Serialize sequences to an AML document string.

    Text payloads must be single-line (the parser is line-oriented); the
    cruncher pipeline collapses newlines during cleaning.  Raises ValueError
    on a multi-line text payload.
    """
    for seq in sequences:
        if isinstance(seq.payload, str) and ("\n" in seq.payload or "\r" in seq.payload):
            raise ValueError(
                f"@stream {seq.modality}: text payload must be single-line; "
                "collapse newlines before serializing"
            )
    doc = AmlDocument(version=version)
    for seq in sequences:
        doc.blocks.append(seq.to_block())
    return serialize(doc)


def aml_to_sequences(source: str) -> list[ModalitySequence]:
    """Parse an AML document string into a list of ``ModalitySequence``."""
    doc = parse(source)
    return [ModalitySequence.from_block(b) for b in doc.by_tag(STREAM_TAG)]
