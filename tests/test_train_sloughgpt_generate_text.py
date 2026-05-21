"""Smoke tests for train_sloughgpt.generate_text (device alignment with model)."""

from pathlib import Path

import pytest
import torch

from domains.models import SloughGPTModel
from train_sloughgpt import generate_text, prepare_data


@pytest.fixture
def tiny_text_path(tmp_path: Path) -> Path:
    p = tmp_path / "corpus.txt"
    p.write_text("abcdefghijklmnopqrstuvwxyz" * 8, encoding="utf-8")
    return p


def test_generate_text_runs_and_decodes(tiny_text_path: Path) -> None:
    _, vocab_size, stoi, itos = prepare_data(tiny_text_path, block_size=32)
    model = SloughGPTModel(
        vocab_size=vocab_size,
        n_embed=64,
        n_layer=2,
        n_head=4,
        block_size=32,
    )
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = model.to(device)

    out = generate_text(model, stoi, itos, prompt="abc", max_new_tokens=8, temperature=0.9)
    assert isinstance(out, str)
    assert len(out) >= len("abc")
