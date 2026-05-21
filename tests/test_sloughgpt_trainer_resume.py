"""Resume coverage for ``domains.training.train_pipeline.SloughGPTTrainer``."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch


def test_sloughgpt_trainer_resume_weights_only_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Weights-only ``model_state_dict`` + ``training_info`` (``train_sloughgpt``-style export)."""
    monkeypatch.chdir(tmp_path)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("abcdefgh" * 80, encoding="utf-8")

    from domains.training.train_pipeline import SloughGPTTrainer

    seed = SloughGPTTrainer(
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
        checkpoint_dir=str(tmp_path / "ck_a"),
        checkpoint_interval=100_000,
    )
    seed.train()

    wpath = tmp_path / "weights.pt"
    torch.save(
        {
            "model_state_dict": seed.model.state_dict(),
            "training_info": {
                "vocab_size": seed.vocab_size,
                "n_embed": seed.config.n_embed,
                "n_layer": seed.config.n_layer,
                "n_head": seed.config.n_head,
                "block_size": seed.config.block_size,
            },
        },
        wpath,
    )

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
        checkpoint_dir=str(tmp_path / "ck_b"),
        checkpoint_interval=100_000,
    )
    trainer.train(resume=True, resume_path=str(wpath))
    assert trainer.global_step > 0


def test_sloughgpt_trainer_resume_full_trainer_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Resume from native ``step_*.pt`` continues past the saved global step."""
    monkeypatch.chdir(tmp_path)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("abcdefgh" * 80, encoding="utf-8")

    from domains.training.train_pipeline import SloughGPTTrainer

    t1 = SloughGPTTrainer(
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
        checkpoint_dir=str(tmp_path / "ck_run1"),
        checkpoint_interval=100_000,
    )
    t1.train()
    ck_dir = Path(t1.config.checkpoint_dir)
    written = sorted(ck_dir.glob("step_*.pt"))
    assert written, "expected at least one step_*.pt from training"
    path = written[-1]
    saved_step = int(path.stem.split("_")[1])

    bundle = torch.load(path, map_location="cpu", weights_only=False)
    assert "stoi" in bundle and "itos" in bundle and "chars" in bundle
    assert len(bundle["stoi"]) == t1.vocab_size
    assert bundle["chars"] == [t1.itos[i] for i in range(t1.vocab_size)]

    t2 = SloughGPTTrainer(
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
        checkpoint_dir=str(tmp_path / "ck_run2"),
        checkpoint_interval=100_000,
    )
    t2.train(resume=True, resume_path=str(path))
    assert t2.global_step == 5
    assert saved_step <= 3


def test_sloughgpt_trainer_resume_latest_in_checkpoint_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``train(resume=True, resume_path=None)`` loads newest ``step_*.pt`` in ``checkpoint_dir``."""
    monkeypatch.chdir(tmp_path)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("abcdefgh" * 80, encoding="utf-8")

    from domains.training.train_pipeline import SloughGPTTrainer

    shared_ck = tmp_path / "shared_ckpt"
    t1 = SloughGPTTrainer(
        data_path=str(corpus),
        n_embed=48,
        n_layer=1,
        n_head=2,
        block_size=12,
        batch_size=2,
        epochs=1,
        max_steps=3,
        lr=3e-4,
        device="cpu",
        use_mixed_precision=False,
        checkpoint_dir=str(shared_ck),
        checkpoint_interval=100_000,
    )
    t1.train()

    t2 = SloughGPTTrainer(
        data_path=str(corpus),
        n_embed=48,
        n_layer=1,
        n_head=2,
        block_size=12,
        batch_size=2,
        epochs=1,
        max_steps=5,
        lr=3e-4,
        device="cpu",
        use_mixed_precision=False,
        checkpoint_dir=str(shared_ck),
        checkpoint_interval=100_000,
    )
    t2.train(resume=True, resume_path=None)
    assert t2.global_step == 5
    assert t2.config.learning_rate == pytest.approx(3e-4)
