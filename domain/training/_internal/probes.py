"""Static probe evaluation — held-out foreign-corpus measurements.

Probe records are carved out of a corpus at seal time (``build_corpus
--probe``) and written to ``probe.jsonl`` beside the snapshot manifest.
They never enter ``train.txt``, so loss on them measures generalization,
not memorization — the scoreboard for sequential training (forgetting
vs drift).

Sync and host-free: the caller supplies a trainer (model + tokenizer +
config) and decides when to run (stage boundaries).
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("slo.training.probes")


def probe_path_for(manifest_path: str | Path) -> Path:
    """Sibling ``probe.jsonl`` for a sealed manifest (may not exist)."""
    return Path(manifest_path).parent / "probe.jsonl"


def load_probe_texts(path: str | Path) -> list[str]:
    """Read probe records from ``probe.jsonl`` (``{"text": ...}`` or raw lines)."""
    texts: list[str] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                texts.append(line)
                continue
            if isinstance(obj, dict):
                for key in ("text", "content"):
                    value = obj.get(key)
                    if isinstance(value, str) and value.strip():
                        texts.append(value)
                        break
            elif isinstance(obj, str) and obj.strip():
                texts.append(obj)
    return texts


def evaluate_probe(
    trainer: Any,
    probe_path: str | Path,
    *,
    num_batches: int = 5,
) -> dict[str, Any] | None:
    """Average model loss on a probe file.

    Mirrors ``SloughGPTTrainer.evaluate`` (numpy batch in, loss out) but on
    an *external* corpus tokenized with the trainer's own tokenizer/vocab —
    unknown characters are dropped, never mapped to a wrong id, so the
    number stays comparable across stages sharing one vocab.

    Returns ``None`` when the probe is missing/empty or too short for one
    block (nothing honest to measure).
    """
    texts = load_probe_texts(probe_path)
    if not texts:
        return None

    tokenizer = getattr(trainer, "tokenizer", None)
    stoi = getattr(trainer, "stoi", None) or {}
    if tokenizer is not None:
        ids: list[int] = []
        for text in texts:
            ids.extend(int(i) for i in tokenizer.encode(text))
    else:
        ids = [int(stoi[c]) for c in "".join(texts) if c in stoi]

    data = np.asarray(ids, dtype=np.int64)
    block = int(trainer.config.block_size)
    if len(data) <= block + 1:
        logger.debug("probe %s too small (%d tokens) — skipped", probe_path, len(data))
        return None

    model = trainer.training_model
    model.eval()
    batch_size = int(trainer.config.batch_size)
    total = 0.0
    steps = 0
    for _ in range(num_batches):
        idx = np.random.randint(0, len(data) - block, size=batch_size)
        offsets = np.arange(block)
        x = data[idx[:, None] + offsets]
        y = data[idx[:, None] + offsets + 1]
        _, loss = model(x, y)
        value = float(loss.item())
        if np.isfinite(value):
            total += value
            steps += 1
    if steps == 0:
        return None
    avg = total / steps
    return {
        "probe_loss": avg,
        "probe_ppl": float(math.exp(avg)),
        "records": len(texts),
        "tokens": int(len(data)),
    }
