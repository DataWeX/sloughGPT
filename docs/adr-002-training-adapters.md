# ADR: Training Adapters — Task Artifacts, BatchArray Seam, Composition Gate

## Status

Accepted

## Context

Training data reaches `SloughGPTTrainer` today through loose modules that each hand-roll their own path to the `data_path` argument (`own_dataset_adapter`, `cruncher/adapter`, `espeak_pairs`, `experience_adapter`). Nothing is registered, nothing declares an in/out contract, and nothing fails until a training job is already running. Three prior directions were rejected:

1. **Per-source adapters** — couples the artifact (what is trained on) to its provenance (where it came from); every new source forks adapter code.
2. **One-shot iterators** — epochs N>1 and shuffle both need random access; a stream adapter must either buffer everything (kills streaming) or re-open its source (couples adapter to I/O).
3. **On-demand/manual gates** — composition checks that run only when someone remembers to run them; drift is discovered in CI, not at the boundary.

This ADR supersedes parked card `63116e83` (the AML cruncher design). The AML pipeline itself (`domain/training/_internal/cruncher/`: import → clean → snapshot → adapter) has landed; this records the standing ruling around it.

## Decision

### 1. Adapters are the noun; sources are config

An **adapter** is a per-training-task artifact: `text-pretrain`, `tts-espeak`, `image-caption`, `chat-sft` — never per-source. **Sources** are four closed config kinds (`local`, `http`, `kaggle`, `shell` — operator-tier only). Config names components; it never contains them. A new source is one YAML line, zero rebuild.

### 2. Trainer seam: `BatchArray` — one door for adapters

```python
@runtime_checkable
class BatchArray(Protocol):
    """Map-style batch contract (Dataset semantics, NOT a stream)."""
    def __len__(self) -> int: ...          # sized
    def __getitem__(self, i): ...          # random access → shuffle
    def __iter__(self): ...                # re-iterable → epochs
```

- Every adapter implements `BatchArray`; **adapters never import the loop** (pinned mechanically by the gate).
- **Exactly one seam** on `SloughGPTTrainer`: adapters enter through `BatchArray`, never through any other door.
- **Placement vs the existing `BatchSampler`** (`training_handler.py`): `BatchSampler.get_batch(batch_size)` is the _loop-internal batch-production_ seam (live-feed path, `FeedBatchSampler`); `BatchArray` is the _adapter-facing data_ seam (data in). Two protocols, two roles — "one seam" means adapters have exactly one way in, not that no other protocol exists internally.
- Modality payloads live **inside** the batch, never in the contract.

### 3. Composition gate: automatic, three layers

One gate file (`tests/core-py/test_adapter_composition_gate.py`), not per-adapter tests:

1. **Fail-fast at registration** — `register()` validates the declared in/out contract; a broken adapter raises and never registers.
2. **Static parametrized contract composition** — walks the whole registry asserting declared in/out contracts compose (`test_command_parity.py` shape; static, side-effect-free).
3. **One dynamic synthetic smoke** — a tiny fixture batch through the **real AML pipeline** (dedup → quality → snapshot → `BatchArray`), proving runtime shape. The AML cruncher is the canonical flow through the gate (dogfood).

Rules: 100% registry coverage from day one, **no grandfather baseline**; rides the affected-closure loop and every full-CI run (Test Rule clause 4 — seam gate); synthetic only — never real credentials or datasets.

### 4. Scope

- **In:** weights training only. The AML import → clean → snapshot design rides here.
- **Out:** knowledge store (parked ruling). "Adapter" in this ADR means _training-data adapter_ — distinct from LoRA adapters, user adapters, and transport adapters elsewhere in the docs.
- **Deferred to the first real training task:** the `BatchArray` spine on `SloughGPTTrainer`, the four source kinds, modality payload layouts.
- **Grandfathered:** `espeak_pairs` until `BatchArray` byte-parity; it stays off the registry until then.

## Consequences

**Positive:**

- Artifact/provenance separation: new source = config line, new task = one adapter.
- Broken adapters die at the boot boundary, never mid-epoch.
- One parametrized gate scales O(1) test code to N adapters; no baseline debt from day one.
- The gate proves the real AML path, hermetically, on every CI run.

**Negative:**

- Legacy adapters (`own_dataset`, `experience`, `espeak`) sit outside the registry until the spine trigger — two data paths coexist meanwhile.
- Map-style contract rules out genuinely infinite streams (by design: epochs require collection semantics).

## Registry (day one)

| Adapter                        | Task          | Status                          |
| ------------------------------ | ------------- | ------------------------------- |
| `text_pretrain` (AML pipeline) | text-pretrain | registered — the canonical flow |
