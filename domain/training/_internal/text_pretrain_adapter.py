"""text-pretrain adapter — sealed AML snapshot → ``BatchArray`` (ADR-002 day-one entry).

The canonical flow through the composition gate (dogfood): records →
``Cruncher`` (dedup → quality) → snapshot → this adapter → ``BatchArray``.
Registration is import-time (fail-fast at the registry boundary); the
trainer's ``data_path`` path stays untouched until the spine trigger
(ADR-002 §4). Never imports the training loop (gate-pinned).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterator

from .adapter_registry import AdapterSpec, register

if TYPE_CHECKING:
    from .cruncher import SnapshotRef


class SnapshotBatchArray:
    """Map-style view over a sealed snapshot's ``train.txt`` (one record per line).

    Re-iterable (reads are pure), sized, random-access — the epochs/shuffle
    collection semantics ``BatchArray`` requires.
    """

    def __init__(self, ref: SnapshotRef) -> None:
        self._lines = Path(ref.train_path).read_text(encoding="utf-8").splitlines()

    def __len__(self) -> int:
        return len(self._lines)

    def __getitem__(self, i: int) -> str:
        return self._lines[i]

    def __iter__(self) -> Iterator[str]:
        return iter(self._lines)


def build(ref: SnapshotRef) -> SnapshotBatchArray:
    """Sealed snapshot → ``BatchArray`` (the adapter's declared out-contract)."""
    return SnapshotBatchArray(ref)


SPEC = AdapterSpec(
    name="text-pretrain",
    task="text-pretrain",
    consumes=("text-records",),
    produces=SnapshotBatchArray,
    build=build,
    module=__name__,
)

register(SPEC)
