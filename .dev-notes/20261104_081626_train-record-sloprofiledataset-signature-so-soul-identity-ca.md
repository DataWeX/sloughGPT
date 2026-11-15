---
id: 20261104_081626_train-record-sloprofiledataset-signature-so-soul-identity-ca
status: todo
board: 20261104_081626_train-record-sloprofiledataset-signature-so-soul-identity-ca
title: "train: record SloProfile.dataset_signature so .soul identity can prove dataset equivalence"
tags: ["models", "training", "model-loading-v1", "soul-identity", "area:training", "track:now"]
sprint: "model-loading-v1"
gh: null
---

# train: record SloProfile.dataset_signature so .soul identity can prove dataset equivalence

Found while landing card 20261104_053224 (a7769616c): `ModelLoader._try_load_soul` now resolves a native `.soul` by header identity, and two of its tiers depend on metadata that training never writes.

Measured on the 5 real souls in `models/slonet-native/` (all trained on the same `data/datasets/tinyshakespeare/input.txt`): every one records `dataset_signature=""` while `integrity_hash` and `born_at` are populated.

Consequences today:
- `dataset:<signature>` lookups can never match (tier inert).
- Family aliases (`sloughgpt`, `sloughgpt-v1`, `slonet`, `slonet-native`) REFUSE with a candidate listing, because equivalence requires the same base_model AND the same non-empty dataset signature. So `SLO_AUTOLOAD_MODEL=sloughgpt` now fails loudly instead of serving the newest checkpoint. That refusal is intended (it was serving a wrong-artifact guess before), but the family journey stays broken until this lands.

Fix: compute a dataset content hash at training time and set `SloProfile.dataset_signature` before `save_soul(...)` (`domain/training/_internal/train_pipeline.py` ~L2688), so retraining the same model on an unchanged dataset yields the same signature and the artifacts become provably interchangeable. Hash the resolved dataset bytes (or a stable sample + size + digest) — never the path — and keep it stable across runs over the same data.

Note: existing souls cannot be back-filled by the loader — either retrain, or add a migration that recomputes the signature from the recorded `training_dataset` when the bytes are still present (and logs what it rewrote).
