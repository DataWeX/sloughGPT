---
id: 20261105_134334_quantum-persistence-consumer-wiring-first-production-instant
status: done
board: 20261105_134334_quantum-persistence-consumer-wiring-first-production-instant
title: "Quantum persistence consumer wiring — first production instantiation + hydrate-at-boot/snapshot-at-checkpoint ruling"
tags: ["soul", "quantum", "mogdb", "follow-up"]
sprint: ""
gh: null
---

# Quantum persistence consumer wiring — first production instantiation + hydrate-at-boot/snapshot-at-checkpoint ruling

## Context
Companion to card 20261105_122337 (quantum persistence layer — landed first). Expert review 2026-11-05 found QuantumCognitiveEngine is instantiated ONLY in tests (re-exported from domain.soul, zero production consumers) — so the layer shipped without a consumer BY DESIGN ("layer first", user approved "continue building").

## Goal
Decide + build the first production consumer: when does the engine actually run, where does hydrate() happen at boot, and where are the snapshot() checkpoints?

## Questions to answer at card time
1. WHO runs QuantumCognitiveEngine in production? (consciousness pipeline? soul/HD memory? nothing yet?) — if nothing yet, this card may rule "no consumer needed until X" and close as a recorded ruling (llm_v1 precedent).
2. If wired: hydrate() at boot (null state + overlay), snapshot() at explicit checkpoints (session boundaries — NEVER inside interfere()/measure() hot path; benchmark_consciousness_hook is the regression gate).
3. Superposition stays ephemeral (never persisted) — v1 schema.

## Owned paths (if building)
domain/soul/_internal/persistence.py, quantum.py, the consumer module, tests.
Gates: run_affected + benchmark_consciousness_hook no-regression + ruff + domain-baseline rehash.

## Decision of the layer card
Layer-first: QuantumStateStore over mogdb (data/soul_quantum.db), snapshot/hydrate seam, schema-version gate, never-raises.
