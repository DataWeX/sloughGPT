---
id: 20261105_122337_quantum-engine-state-persistence-snapshothydrate-seam-over-m
status: done
board: 20261105_122337_quantum-engine-state-persistence-snapshothydrate-seam-over-m
title: "Quantum engine state persistence: snapshot/hydrate seam over mogdb (layer first)"
tags: ["soul", "quantum", "mogdb", "persistence"]
sprint: ""
gh: null
---

# Quantum engine state persistence: snapshot/hydrate seam over mogdb (layer first)

## Goal
Persist QuantumCognitiveEngine durable state into mogdb via an explicit snapshot/hydrate seam; boot from null state and overlay persisted data.

## Decision (expert review 2026-11-05, user approved: continue building)
**Layer first.** QuantumCognitiveEngine is instantiated only in tests today (re-exported from domain.soul, no production consumer) — land the persistence layer + tests now; wiring a consumer is a follow-up card when a production instantiation exists.

## Plan (TDD)
1. domain/soul/_internal/persistence.py — QuantumStateStore over a mogdb collection: save(engine) / load() -> dict|None; complex amplitude serialized as [re, im]; schema version field; never-raise (corrupt/missing store -> None = null state).
2. QuantumCognitiveEngine.snapshot() / .hydrate() thin wrappers — durable fields ONLY: entangled_pairs, coherence, decoherence_rate. superposition always boots [] (ephemeral per-invocation state; random phases make cross-restart persistence meaningless).
3. Coupling boundary: quantum.py must NOT import mogdb — store injected/passed at the seam so tests/core-py/test_soul_quantum.py ('pure logic, no mocks') stays I/O-free.
4. Tests: round-trip save/load, corrupt store -> null state, pure path untouched (existing tests green unchanged).

## Owned paths (collision = stop)
domain/soul/_internal/persistence.py (new), domain/soul/_internal/quantum.py, tests/core-py/test_soul_quantum.py or new test_soul_quantum_persistence.py
MUST NOT TOUCH: hd_memory.py hot path, cognitive.py episodic store, packages/mogdb internals, superposition/measure semantics.

## Verification
PYTHONNOUSERSITE=1 PYTHONPATH="$PWD/packages/downcraft:$PWD/packages/core-py:$PWD/apps/api/server" /home/mana/miniconda3/envs/sloughgpt/bin/python -m pytest tests/core-py/test_soul_quantum.py -q -p no:randomly
+ ruff check/format on touched paths + scripts/run_affected.sh green + scripts/benchmark_consciousness_hook.py no regression.

## Definition of done
Tests green incl. new round-trip/corrupt-store cases; gates green; pushed to main; AGENT_SYNC updated.
