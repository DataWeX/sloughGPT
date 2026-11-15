"""Tests for quantum state persistence — engine snapshot/hydrate + QuantumStateStore.

Split from test_soul_quantum.py because these tests touch the persistence
seam (mogdb I/O); the pure-logic suite stays I/O-free by contract.
"""

from __future__ import annotations

from domain.soul._internal.persistence import SCHEMA_VERSION, QuantumStateStore
from domain.soul._internal.quantum import QuantumCognitiveEngine


def _engine() -> QuantumCognitiveEngine:
    engine = QuantumCognitiveEngine(coherence=0.7)
    engine.entangle("alpha", "beta")
    engine.entangle("gamma", "delta")
    return engine


class TestSnapshotHydrate:
    """Pure engine half of the seam — no I/O."""

    def test_snapshot_returns_durable_fields_only(self):
        engine = _engine()
        engine.create_superposition(["x", "y"])
        state = engine.snapshot()
        assert state["coherence"] == 0.7
        assert state["decoherence_rate"] == 0.01
        assert state["entangled_pairs"] == [["alpha", "beta"], ["gamma", "delta"]]
        # superposition is per-invocation state — never in the snapshot
        assert "superposition" not in state

    def test_snapshot_is_json_native(self):
        import json

        state = _engine().snapshot()
        assert json.loads(json.dumps(state)) == state

    def test_hydrate_none_is_false_and_keeps_null_state(self):
        engine = QuantumCognitiveEngine()
        assert engine.hydrate(None) is False
        assert engine.coherence == 0.9  # constructor default untouched
        assert engine.entangled_pairs == []

    def test_hydrate_overlays_durable_fields(self):
        fresh = QuantumCognitiveEngine()
        assert fresh.hydrate(_engine().snapshot()) is True
        assert fresh.coherence == 0.7
        assert fresh.entangled_pairs == [("alpha", "beta"), ("gamma", "delta")]

    def test_hydrate_leaves_superposition_empty(self):
        fresh = QuantumCognitiveEngine()
        fresh.hydrate(_engine().snapshot())
        assert fresh.superposition == []

    def test_hydrate_tolerates_invalid_field_values(self):
        fresh = QuantumCognitiveEngine()
        # corrupt-but-dict state: bad types are skipped, never raised
        ok = fresh.hydrate({"coherence": "nan", "entangled_pairs": ["not-a-pair", ["only-one"]]})
        assert ok is True
        assert fresh.coherence == 0.9  # invalid value skipped
        assert fresh.entangled_pairs == []  # malformed pairs dropped


class TestQuantumStateStore:
    """mogdb-backed store half — never raises, null state on any failure."""

    def _store(self, tmp_path) -> QuantumStateStore:
        return QuantumStateStore(db_path=str(tmp_path / "soul_quantum.db"))

    def test_round_trip_durable_fields(self, tmp_path):
        store = self._store(tmp_path)
        assert store.save(_engine()) is True

        state = store.load()
        assert state is not None
        assert state["schema_version"] == SCHEMA_VERSION

        fresh = QuantumCognitiveEngine()
        assert fresh.hydrate(state) is True
        assert fresh.coherence == 0.7
        assert fresh.entangled_pairs == [("alpha", "beta"), ("gamma", "delta")]

    def test_load_missing_store_is_null_state(self, tmp_path):
        assert self._store(tmp_path).load() is None

    def test_save_is_idempotent_single_doc(self, tmp_path):
        store = self._store(tmp_path)
        store.save(_engine())
        store.save(_engine())
        # one logical record: reload must still yield exactly the two pairs
        fresh = QuantumCognitiveEngine()
        fresh.hydrate(store.load())
        assert fresh.entangled_pairs == [("alpha", "beta"), ("gamma", "delta")]

    def test_schema_version_mismatch_is_null_state(self, tmp_path):
        # a doc written by a future schema version must not hydrate:
        # write it first, then let the store open lazily and reject it
        from mogdb import MogDB

        db_path = str(tmp_path / "soul_quantum.db")
        db = MogDB(db_path)
        db.collection("quantum_state").insert_one(
            {
                "key": "quantum-engine",
                "schema_version": SCHEMA_VERSION + 1,
                "coherence": 0.5,
            }
        )
        db.close()

        assert QuantumStateStore(db_path=db_path).load() is None

    def test_unwritable_path_save_false_load_none_never_raise(self, tmp_path):
        blocker = tmp_path / "blocker"
        blocker.write_text("i am a file, not a directory")
        store = QuantumStateStore(db_path=str(blocker / "nested" / "q.db"))
        assert store.save(_engine()) is False  # no exception escapes
        assert store.load() is None

    def test_superposition_never_persisted(self, tmp_path):
        engine = _engine()
        engine.create_superposition(["x", "y", "z"])
        store = self._store(tmp_path)
        store.save(engine)

        state = store.load()
        assert state is not None
        assert "superposition" not in state

        fresh = QuantumCognitiveEngine()
        fresh.hydrate(state)
        assert fresh.superposition == []
