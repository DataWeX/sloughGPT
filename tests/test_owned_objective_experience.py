"""Goals 13+14: owned objective + owned data wired into the single TrainingLoop.

Acceptance:
  * ``OwnedObjective`` combines with the primary loss when experience metadata
    is present and collapses to pure next-token when it is not.
  * ``data_path="experience"`` loads owned chat/feedback into ``prepare_data``.
  * ``SloughGPTTrainer`` streams ``ExperienceSampler`` batches and applies the
    owned objective inside ``train_step`` (no fourth loop — scatter gate stays
    green because the backward/step pairing lives only in train_pipeline).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from domain.training._internal.experience_adapter import (
    ExperienceSampler,
    is_experience_source,
    load_experience_pairs,
    load_experience_text,
)
from domain.training._internal.owned_objective import OwnedObjective, OwnedObjectiveConfig


class _FakeLoss:
    """Minimal Tensor-like stand-in: .item() and * scalar."""

    def __init__(self, value: float) -> None:
        self._value = float(value)

    def item(self) -> float:
        return self._value

    def __mul__(self, other: float) -> _FakeLoss:
        return _FakeLoss(self._value * float(other))

    def __rmul__(self, other: float) -> _FakeLoss:
        return self.__mul__(other)


def test_is_experience_source() -> None:
    assert is_experience_source("experience")
    assert is_experience_source("experience:/tmp/corpus.jsonl")
    assert not is_experience_source("feed:http://x")
    assert not is_experience_source("corpus.txt")
    assert not is_experience_source(None)
    assert not is_experience_source(["a", "b"])


def test_load_experience_text_synthetic_seed_validates() -> None:
    from domain.training._internal.train_pipeline import validate_training_data

    text, pairs = load_experience_text()
    assert len(pairs) >= 10
    validate_training_data(text)
    assert len(set(text)) >= 10


def test_load_experience_pairs_from_jsonl(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    rows = [
        {"user_msg": "ping", "assistant_msg": "pong from owned data"},
        {"prompt": "hello", "completion": "world from prompt-completion form"},
        {"user_msg": "", "assistant_msg": "skip empty"},
    ]
    corpus.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    pairs = load_experience_pairs(corpus_path=corpus)
    assert len(pairs) == 2
    assert pairs[0]["user_msg"] == "ping"


def test_owned_objective_without_metadata_is_identity() -> None:
    obj = OwnedObjective(OwnedObjectiveConfig())
    primary = _FakeLoss(2.0)
    combined, metrics = obj.compute(None, np.zeros(1), np.zeros(1), primary, metadata=None)
    assert combined.item() == pytest.approx(2.0)
    assert metrics["primary"] == pytest.approx(2.0)
    assert "combined" not in metrics


def test_owned_objective_scales_loss_with_metadata() -> None:
    obj = OwnedObjective(OwnedObjectiveConfig(tool_success_weight=0.5, memory_weight=0.25))
    primary = _FakeLoss(2.0)
    metadata = {"tool_success": [1, 1, 0, 1], "memory_match": [1, 1, 1, 1]}
    combined, metrics = obj.compute(None, np.zeros(4), np.zeros(4), primary, metadata=metadata)
    # tool rate 0.75 → scale (1 - 0.5*0.75)=0.625; memory rate 1 → *(1-0.25)=0.75
    expected = 2.0 * (1 - 0.5 * 0.75) * (1 - 0.25)
    assert combined.item() == pytest.approx(expected)
    assert metrics["tool_success_rate"] == pytest.approx(0.75)
    assert metrics["memory_match_rate"] == pytest.approx(1.0)
    assert combined.item() < primary.item()


def test_experience_sampler_batch_shapes() -> None:
    stoi = {c: i for i, c in enumerate("abcdefghijklmnopqrstuvwxyz ")}
    sampler = ExperienceSampler(stoi=stoi, block_size=16, seed=7)
    stats = sampler.stats()
    assert stats["pairs"] >= 2
    assert stats["tokens"] > 16
    x, y = sampler.get_batch(4)
    assert x.shape == (4, 16)
    assert y.shape == (4, 16)
    # y is x shifted by one along the flat sequence — check via re-encode
    assert x.dtype == np.int32


def test_prepare_data_experience_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    # Isolate from any real home cache by pointing loaders at an empty tree
    # and only the synthetic seed (default project corpora absent under tmp).
    from domain.training._internal import experience_adapter as ea

    monkeypatch.setattr(ea, "_load_chat_corpus", lambda path=None: [])
    monkeypatch.setattr(ea, "_load_feedback_pairs", lambda path=None: [])

    from domain.training._internal.train_pipeline import prepare_data

    data, vocab_size, stoi, itos = prepare_data("experience", block_size=16)
    assert vocab_size >= 10
    assert data.dtype == np.int64
    assert len(data) > 16
    assert len(stoi) == vocab_size


def test_trainer_runs_experience_data_with_owned_objective(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: data_path=experience + use_owned_objective through train()."""
    monkeypatch.chdir(tmp_path)
    from domain.training._internal import experience_adapter as ea

    monkeypatch.setattr(ea, "_load_chat_corpus", lambda path=None: [])
    monkeypatch.setattr(ea, "_load_feedback_pairs", lambda path=None: [])

    from domain.training._internal.train_pipeline import SloughGPTTrainer

    trainer = SloughGPTTrainer(
        data_path="experience",
        n_embed=32,
        n_layer=1,
        n_head=2,
        block_size=16,
        batch_size=4,
        epochs=1,
        max_steps=3,
        device="cpu",
        checkpoint_dir=str(tmp_path / "ckpt"),
        checkpoint_interval=100_000,
        log_interval=9999,
        eval_interval=9999,
        use_owned_objective=True,
        owned_tool_weight=0.3,
        owned_memory_weight=0.2,
    )
    assert trainer._owned_objective is not None
    assert trainer._experience_sampler is not None
    assert trainer._owned_metadata is not None and "memory_match" in trainer._owned_metadata

    # One train_step surfaces owned metrics (metadata present → scaled loss)
    metrics = trainer.train_step()
    assert np.isfinite(metrics["raw_loss"])
    assert "owned_primary" in metrics
    assert "owned_memory_match_rate" in metrics
    assert "owned_combined" in metrics
    assert metrics["owned_combined"] < metrics["raw_loss"] + 1e-9

    result = trainer.train()
    assert result["global_step"] > 0


def test_trainer_regular_data_owned_objective_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without experience metadata, owned objective is a pure next-term passthrough."""
    monkeypatch.chdir(tmp_path)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("abcdefghij" * 64, encoding="utf-8")

    from domain.training._internal.train_pipeline import SloughGPTTrainer

    trainer = SloughGPTTrainer(
        data_path=str(corpus),
        n_embed=32,
        n_layer=1,
        n_head=2,
        block_size=16,
        batch_size=4,
        epochs=1,
        max_steps=2,
        device="cpu",
        checkpoint_dir=str(tmp_path / "ckpt"),
        checkpoint_interval=100_000,
        log_interval=9999,
        eval_interval=9999,
        use_owned_objective=True,
    )
    assert trainer._owned_objective is not None
    assert trainer._experience_sampler is None
    assert trainer._owned_metadata is None

    metrics = trainer.train_step()
    assert np.isfinite(metrics["raw_loss"])
    assert "owned_primary" in metrics
    assert "owned_memory_match_rate" not in metrics
    assert metrics["owned_primary"] == pytest.approx(metrics["raw_loss"])
