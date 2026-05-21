"""Smoke test for domains.training.train_pipeline.SloughGPTTrainer (CLI / API driver)."""

from pathlib import Path

import pytest


def test_sloughgpt_trainer_runs_short_cpu_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Mirrors ``cli.py train`` local path: tiny data, CPU, few steps."""
    monkeypatch.chdir(tmp_path)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("abcdefgh" * 80, encoding="utf-8")

    from domains.training.train_pipeline import SloughGPTTrainer

    trainer = SloughGPTTrainer(
        data_path=str(corpus),
        n_embed=48,
        n_layer=1,
        n_head=2,
        block_size=12,
        batch_size=2,
        epochs=1,
        max_steps=3,
        device="cpu",
        use_mixed_precision=False,
        checkpoint_dir=str(tmp_path / "ckpt"),
        checkpoint_interval=100_000,
    )
    assert trainer.vocab_size == 8
    result = trainer.train()
    assert "global_step" in result
    assert result["global_step"] > 0
