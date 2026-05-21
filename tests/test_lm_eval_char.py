"""``domains.training.lm_eval_char`` — SloughGPT char-LM perplexity on a text file.

Vocabulary resolution and warnings: ``docs/policies/CONTRIBUTING.md`` (*Checkpoint vocabulary*).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
import torch

from domains.models import SloughGPTModel
from domains.training.lm_eval_char import evaluate_sloughgpt_char_lm, main


def test_eval_sloughgpt_char_lm_smoke(tmp_path: Path) -> None:
    chars = sorted(list("abc\n"))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    m = SloughGPTModel(
        vocab_size=len(chars),
        n_embed=16,
        n_layer=1,
        n_head=2,
        block_size=8,
    )
    bundle = {
        "model_state_dict": m.state_dict(),
        "training_info": {
            "vocab_size": len(chars),
            "n_embed": 16,
            "n_layer": 1,
            "n_head": 2,
            "block_size": 8,
        },
        "stoi": stoi,
        "itos": itos,
    }
    ckpt = tmp_path / "tiny.ckpt"
    torch.save(bundle, ckpt)
    data = tmp_path / "eval.txt"
    data.write_text("abc\n" * 40, encoding="utf-8")

    out = evaluate_sloughgpt_char_lm(str(ckpt), str(data), device="cpu")
    assert out["num_token_positions"] > 0
    assert torch.isfinite(torch.tensor(out["mean_loss"]))
    assert torch.isfinite(torch.tensor(out["perplexity"]))
    assert out["perplexity"] < 1e9
    assert out["vocab_size"] == len(chars)
    assert not out["warnings"]


def test_lm_eval_char_main_json(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chars = sorted(list("abc\n"))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    m = SloughGPTModel(
        vocab_size=len(chars),
        n_embed=16,
        n_layer=1,
        n_head=2,
        block_size=8,
    )
    bundle = {
        "model_state_dict": m.state_dict(),
        "training_info": {
            "vocab_size": len(chars),
            "n_embed": 16,
            "n_layer": 1,
            "n_head": 2,
            "block_size": 8,
        },
        "stoi": stoi,
        "itos": itos,
    }
    ckpt = tmp_path / "tiny2.ckpt"
    torch.save(bundle, ckpt)
    data = tmp_path / "ev2.txt"
    data.write_text("abc\n" * 40, encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lm_eval_char",
            "--checkpoint",
            str(ckpt),
            "--data",
            str(data),
            "--json",
        ],
    )
    main()
    cap = capsys.readouterr().out.strip()
    payload = json.loads(cap)
    assert payload["num_token_positions"] > 0
    assert "mean_loss" in payload
    assert math.isfinite(payload["mean_loss"])
