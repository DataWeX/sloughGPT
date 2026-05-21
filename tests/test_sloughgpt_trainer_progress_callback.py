"""``SloughGPTTrainer.train(on_progress=...)`` reports step and loss for API / UI consumers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest


def test_sloughgpt_trainer_on_progress_receives_steps_and_loss(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("abcdefgh" * 80, encoding="utf-8")

    from domains.training.train_pipeline import SloughGPTTrainer

    events: List[Dict[str, Any]] = []

    def on_progress(info: Dict[str, Any]) -> None:
        events.append(dict(info))

    trainer = SloughGPTTrainer(
        data_path=str(corpus),
        n_embed=48,
        n_layer=1,
        n_head=2,
        block_size=12,
        batch_size=2,
        epochs=1,
        max_steps=5,
        device="cpu",
        use_mixed_precision=False,
        checkpoint_dir=str(tmp_path / "ck"),
        checkpoint_interval=100_000,
        log_interval=1,
        eval_interval=1_000_000,
    )
    trainer.train(on_progress=on_progress)

    assert events, "expected at least one on_progress callback"
    train_events = [e for e in events if e.get("train_loss") is not None]
    assert train_events, "expected train_loss in some events"
    last = train_events[-1]
    assert last["global_step"] >= 1
    assert 0 <= last["progress_percent"] <= 99
    assert last["epoch"] == 1
    assert "learning_rate" in last
