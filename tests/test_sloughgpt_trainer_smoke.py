"""Smoke test for domains.training.train_pipeline.SloughGPTTrainer (CLI / API driver)."""

from pathlib import Path

import pytest


def test_sloughgpt_trainer_runs_short_cpu_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mirrors ``cli.py train`` local path: tiny data, CPU, few steps."""
    monkeypatch.chdir(tmp_path)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("abcdefghij" * 64, encoding="utf-8")

    from domain.training._internal.train_pipeline import SloughGPTTrainer

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
        checkpoint_dir=str(tmp_path / "ckpt"),
        checkpoint_interval=100_000,
    )
    assert trainer.vocab_size == 10
    result = trainer.train()
    assert "global_step" in result
    assert result["global_step"] > 0


def test_save_strips_trailing_soul_extension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``save()`` takes a path WITHOUT extension; a caller-supplied trailing
    ``.soul`` must not double up — the job record says ``X.soul``, so the file
    on disk has to be exactly ``X.soul`` (was written as ``X.soul.soul``)."""
    monkeypatch.chdir(tmp_path)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("abcdefghij" * 64, encoding="utf-8")

    from domain.training._internal.train_pipeline import SloughGPTTrainer

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
        checkpoint_dir=str(tmp_path / "ckpt"),
        checkpoint_interval=100_000,
    )
    out = tmp_path / "model_trained.soul"
    trainer.save(str(out))
    assert out.exists()
    assert not Path(str(out) + ".soul").exists()

    stem_out = tmp_path / "model_stem"
    trainer.save(str(stem_out))
    assert Path(str(stem_out) + ".soul").exists()
    assert not stem_out.exists()
