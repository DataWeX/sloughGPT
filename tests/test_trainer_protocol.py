"""``TrainerProtocol`` structural typing and :func:`run_trainer_async`."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest


def test_sloughgpt_trainer_is_trainer_protocol_instance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("abcdefgh" * 80, encoding="utf-8")

    from domains.training.train_pipeline import SloughGPTTrainer
    from domains.training.trainer_protocol import TrainerProtocol

    trainer = SloughGPTTrainer(
        data_path=str(corpus),
        n_embed=48,
        n_layer=1,
        n_head=2,
        block_size=12,
        batch_size=2,
        epochs=1,
        max_steps=2,
        device="cpu",
        use_mixed_precision=False,
        checkpoint_dir=str(tmp_path / "ckpt"),
        checkpoint_interval=100_000,
    )
    assert isinstance(trainer, TrainerProtocol)


def test_run_trainer_async_matches_train(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("abcdefgh" * 80, encoding="utf-8")

    from domains.training.train_pipeline import SloughGPTTrainer
    from domains.training.trainer_protocol import run_trainer_async

    trainer = SloughGPTTrainer(
        data_path=str(corpus),
        n_embed=48,
        n_layer=1,
        n_head=2,
        block_size=12,
        batch_size=2,
        epochs=1,
        max_steps=2,
        device="cpu",
        use_mixed_precision=False,
        checkpoint_dir=str(tmp_path / "ckpt"),
        checkpoint_interval=100_000,
    )
    sync = trainer.train()
    trainer2 = SloughGPTTrainer(
        data_path=str(corpus),
        n_embed=48,
        n_layer=1,
        n_head=2,
        block_size=12,
        batch_size=2,
        epochs=1,
        max_steps=2,
        device="cpu",
        use_mixed_precision=False,
        checkpoint_dir=str(tmp_path / "ckpt2"),
        checkpoint_interval=100_000,
    )
    async def _run() -> dict:
        return await run_trainer_async(trainer2)

    async_result = asyncio.run(_run())
    assert async_result.keys() == sync.keys()
    assert async_result["global_step"] == sync["global_step"]
