"""Resume / checkpoint coverage for train_sloughgpt.train_sloughgpt."""

from pathlib import Path

import pytest
import torch

from domains.models import SloughGPTModel
from train_sloughgpt import prepare_data, train_sloughgpt


def test_train_sloughgpt_resume_weights_only(tmp_path: Path) -> None:
    data = tmp_path / "tiny.txt"
    data.write_text("abcd" * 40, encoding="utf-8")
    ckpt = tmp_path / "weights.pt"

    _, vocab_size, stoi, itos = prepare_data(data, block_size=8)
    model = SloughGPTModel(
        vocab_size=vocab_size, n_embed=32, n_layer=1, n_head=2, block_size=8
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "training_info": {
                "vocab_size": vocab_size,
                "n_embed": 32,
                "n_layer": 1,
                "n_head": 2,
                "block_size": 8,
            },
            "stoi": stoi,
            "itos": itos,
        },
        ckpt,
    )

    m2, stoi2, itos2 = train_sloughgpt(
        data_path=str(data),
        n_embed=32,
        n_layer=1,
        n_head=2,
        block_size=8,
        batch_size=2,
        epochs=1,
        max_steps=1,
        device="cpu",
        resume_from=str(ckpt),
        save_format="torch",
        save_path=str(tmp_path / "out_run"),
        checkpoint_interval=0,
    )
    assert m2 is not None
    assert stoi2 == stoi
    assert itos2 == itos


def test_train_sloughgpt_full_resume_from_periodic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Periodic checkpoints land in ``models/`` under cwd; isolate with chdir."""
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "corpus.txt"
    data.write_text("mnop" * 200, encoding="utf-8")

    train_sloughgpt(
        data_path=str(data),
        n_embed=40,
        n_layer=1,
        n_head=2,
        block_size=12,
        batch_size=2,
        epochs=1,
        max_steps=2,
        device="cpu",
        save_format="torch",
        save_path=str(tmp_path / "first_run"),
        checkpoint_interval=1,
    )
    ckpts = sorted(Path("models").glob("checkpoint_step_*.pt"))
    assert ckpts, "expected models/checkpoint_step_*.pt"

    train_sloughgpt(
        data_path=str(data),
        n_embed=40,
        n_layer=1,
        n_head=2,
        block_size=12,
        batch_size=2,
        epochs=1,
        max_steps=1,
        device="cpu",
        resume_from=str(ckpts[-1]),
        save_format="torch",
        save_path=str(tmp_path / "second_run"),
        checkpoint_interval=0,
    )
