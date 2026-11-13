"""Goal 18: long-run pretraining — crash→resume continuity + durable metrics.

Track B needs an unattended months-long run to survive process death:

1. A train segment reaches ``max_steps`` and leaves a resumable checkpoint.
2. A fresh ``SloughGPTTrainer`` with ``resume=True`` continues from that
   checkpoint — ``global_step`` does not restart at 0 — and finishes at the
   raised budget with finite loss.
3. ``cancel_event`` stops the whole run (not just the current epoch) and still
   leaves a final checkpoint.
4. ``LongRunRecorder`` streams progress dicts to append-only JSONL that
   survives across segments (readable after a "crash").

No fourth training loop: everything goes through ``SloughGPTTrainer.train``.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pytest

from domain.training._internal.long_run import LongRunRecorder
from domain.training._internal.train_pipeline import CheckpointManager, SloughGPTTrainer

_CYCLE = "abcdefghijkl"
_CORPUS = "".join(_CYCLE[i % len(_CYCLE)] for i in range(2201))


def _write_corpus(tmp_path: Path) -> str:
    p = tmp_path / "corpus.txt"
    p.write_text(_CORPUS, encoding="ascii")
    return str(p)


def _trainer(data_path: str, tmp_path: Path, *, max_steps: int, epochs: int = 200) -> SloughGPTTrainer:
    return SloughGPTTrainer(
        data_path=data_path,
        n_embed=32,
        n_layer=1,
        n_head=2,
        block_size=16,
        batch_size=8,
        epochs=epochs,
        max_steps=max_steps,
        lr=2e-3,
        warmup_steps=5,
        checkpoint_dir=str(tmp_path / "ckpts"),
        checkpoint_interval=4,
        log_interval=9999,
        eval_interval=9999,
        device="cpu",
        soul_name="goal18-longrun",
    )


def test_long_run_recorder_appends_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    rec = LongRunRecorder(path)
    assert rec.count == 0
    assert rec.read_all() == []

    rec.record(step=1, train_loss=2.5)
    rec.record(step=2, train_loss=1.5, eval_loss=1.6)
    assert rec.count == 2
    assert path.exists()

    rows = rec.read_all()
    assert len(rows) == 2
    assert rows[0]["step"] == 1
    assert rows[1]["train_loss"] == 1.5
    assert "ts" in rows[0] and "ts" in rows[1]

    # Durable across a new process handle (crash→reopen)
    rec2 = LongRunRecorder(path)
    assert len(rec2.read_all()) == 2
    rec2.record(step=3, train_loss=0.5)
    assert len(rec2.read_all()) == 3
    # Valid JSONL only
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            json.loads(line)


@pytest.mark.timeout(240)
def test_crash_resume_continues_global_step(tmp_path: Path) -> None:
    """Segment A stops at max_steps; segment B resume=True reaches a higher budget."""
    np.random.seed(0)
    data_path = _write_corpus(tmp_path)
    ckpt_dir = tmp_path / "ckpts"
    metrics_path = tmp_path / "metrics.jsonl"
    rec = LongRunRecorder(metrics_path)

    # Segment A: budget 12
    t1 = _trainer(data_path, tmp_path, max_steps=12)
    events_a: list[dict] = []

    def on_a(info: dict) -> None:
        events_a.append(dict(info))
        rec.record(**info)

    r1 = t1.train(on_progress=on_a)
    step_a = int(r1["global_step"])
    assert step_a == 12, r1
    assert ckpt_dir.is_dir()
    assert CheckpointManager(str(ckpt_dir)).latest_valid_path() is not None

    # Segment B: fresh trainer, resume from latest, higher budget
    t2 = _trainer(data_path, tmp_path, max_steps=24)
    events_b: list[dict] = []

    def on_b(info: dict) -> None:
        events_b.append(dict(info))
        rec.record(**info)

    r2 = t2.train(resume=True, on_progress=on_b)
    step_b = int(r2["global_step"])
    assert step_b == 24, r2
    # Restored, not restarted
    assert t2.global_step >= step_a
    assert events_b, "expected progress on resumed segment"
    first_live = next(e for e in events_b if e.get("global_step"))
    assert int(first_live["global_step"]) >= step_a

    # Losses finite throughout
    losses = [e["train_loss"] for e in events_a + events_b if e.get("train_loss") is not None]
    assert losses
    assert all(np.isfinite(x) for x in losses)

    # JSONL survived both segments (progress uses global_step, not step)
    rows = rec.read_all()
    assert len(rows) >= 2
    steps_seen = [
        r["global_step"] for r in rows if r.get("global_step") is not None
    ]
    assert steps_seen, rows
    assert max(steps_seen) >= step_a
    assert metrics_path.stat().st_size > 0
    assert any(s == step_a for s in steps_seen)


@pytest.mark.timeout(240)
def test_cancel_stops_run_and_leaves_checkpoint(tmp_path: Path) -> None:
    """cancel_event ends the whole run (not just the epoch) with a final checkpoint."""
    np.random.seed(0)
    data_path = _write_corpus(tmp_path)
    trainer = _trainer(data_path, tmp_path, max_steps=10_000, epochs=50)
    cancel = threading.Event()
    seen = {"steps": 0}

    def on_progress(info: dict) -> None:
        if info.get("train_loss") is not None:
            seen["steps"] += 1
            if seen["steps"] >= 3:
                cancel.set()

    result = trainer.train(on_progress=on_progress, cancel_event=cancel)
    # Cancelled well before the 10k budget
    assert int(result["global_step"]) < 10_000
    assert int(result["global_step"]) >= 1
    # Final save ran → resumable artifact exists
    latest = CheckpointManager(str(tmp_path / "ckpts")).latest_valid_path()
    assert latest is not None

    # Resume from cancelled state and continue
    t2 = _trainer(data_path, tmp_path, max_steps=int(result["global_step"]) + 8)
    r2 = t2.train(resume=True)
    assert int(r2["global_step"]) > int(result["global_step"])
    assert np.isfinite(float(r2.get("final_loss") or 1.0))
