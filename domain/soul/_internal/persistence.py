"""QuantumCognitiveEngine state persistence — the mogdb seam.

The engine stays pure logic; this module owns all I/O. Callers cross the
seam explicitly at checkpoints:

    store = QuantumStateStore()
    engine.hydrate(store.load())   # boot: null state + overlay
    store.save(engine)             # checkpoint: durable fields only

Never raises: a missing, corrupt, or unwritable store degrades to the
null state (``load() -> None``, ``save() -> False``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from domain.soul._internal.quantum import QuantumCognitiveEngine

logger = logging.getLogger("slo.soul.persistence")

SCHEMA_VERSION = 1
_STATE_KEY = "quantum-engine"
_COLLECTION = "quantum_state"


class QuantumStateStore:
    """mogdb-backed store for the engine's durable state. Never raises."""

    def __init__(self, db_path: str = "data/soul_quantum.db"):
        self._db_path = db_path
        self._db: Any = None

    def _collection(self) -> Any:
        from mogdb import MogDB

        if self._db is None:
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            self._db = MogDB(self._db_path)
        return self._db.collection(_COLLECTION)

    def save(self, engine: QuantumCognitiveEngine) -> bool:
        """Persist the engine's durable snapshot. ``False`` on any failure."""
        try:
            doc: dict[str, Any] = {
                "key": _STATE_KEY,
                "schema_version": SCHEMA_VERSION,
            }
            doc.update(engine.snapshot())
            col = self._collection()
            if col.find_one({"key": _STATE_KEY}):
                col.update_one({"key": _STATE_KEY}, {"$set": doc})
            else:
                col.insert_one(doc)
            return True
        except Exception as exc:  # noqa: BLE001 — persistence never breaks the engine
            logger.debug("quantum state save failed: %s", exc)
            return False

    def load(self) -> dict[str, Any] | None:
        """Stored state, or ``None`` (null state) on missing/corrupt/foreign schema."""
        try:
            doc = self._collection().find_one({"key": _STATE_KEY})
        except Exception as exc:  # noqa: BLE001 — a broken store is a null state
            logger.debug("quantum state load failed: %s", exc)
            return None
        if not isinstance(doc, dict):
            return None
        if doc.get("schema_version") != SCHEMA_VERSION:
            return None
        return doc
