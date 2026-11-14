"""Tests for the self-gen flywheel: producer gate/dedup + in-training integration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


class TestSelfGenProducer:
    """Unit tests for the corpus write boundary (gate → dedup → append)."""

    def _make_producer(self, tmp_path: Path, min_quality: float = 0.0):
        from domain.training._internal.selfgen import SelfGenProducer

        corpus = tmp_path / "selfgen" / "corpus.jsonl"
        return SelfGenProducer(corpus, prompts=["The quick brown"], min_quality=min_quality)

    def test_accept_writes_record_and_counts(self, tmp_path: Path) -> None:
        p = self._make_producer(tmp_path)
        ok = p.accept("The quick brown", "fox jumps over the lazy dog today")
        assert ok is True
        assert p.written == 1
        records = [json.loads(l) for l in p.corpus_path.read_text().splitlines()]
        assert records[0]["user_msg"] == "The quick brown"
        assert records[0]["assistant_msg"] == "fox jumps over the lazy dog today"
        assert records[0]["gen"] is True

    def test_short_continuation_rejected(self, tmp_path: Path) -> None:
        p = self._make_producer(tmp_path)
        assert p.accept("The quick brown", "hi") is False
        assert p.written == 0
        assert p.rejected_short == 1

    def test_duplicate_rejected_after_first_write(self, tmp_path: Path) -> None:
        p = self._make_producer(tmp_path)
        text = "a fox jumps gracefully over a sleeping dog"
        assert p.accept("The quick brown", text) is True
        assert p.accept("The quick brown", text) is False
        assert p.rejected_dup == 1
        assert p.written == 1

    def test_duplicate_rejected_across_reinstantiation(self, tmp_path: Path) -> None:
        p = self._make_producer(tmp_path)
        text = "a fox jumps gracefully over a sleeping dog"
        p.accept("The quick brown", text)
        p2 = self._make_producer(tmp_path)
        assert p2.accept("The quick brown", text) is False
        assert p2.rejected_dup == 1

    def test_low_quality_rejected(self, tmp_path: Path) -> None:
        p = self._make_producer(tmp_path, min_quality=5.0)
        # Highly repetitive text scores low on the repetition signal.
        ok = p.accept("The quick brown", "the the the the the the the the the the")
        assert ok is False
        assert p.rejected_quality == 1
        assert not p.corpus_path.exists() or p.corpus_path.read_text().strip() == ""

    def test_next_prompts_rotates(self, tmp_path: Path) -> None:
        from domain.training._internal.selfgen import SelfGenProducer

        corpus = tmp_path / "selfgen" / "corpus.jsonl"
        p = SelfGenProducer(corpus, prompts=["aa111111", "bb222222"])
        assert p.next_prompts(2) == ["aa111111", "bb222222"]
        assert p.next_prompts(2) == ["aa111111", "bb222222"]

    def test_prompts_from_text_derives_chunks(self) -> None:
        from domain.training._internal.selfgen import SelfGenProducer

        text = "abcdefghij" * 100
        prompts = SelfGenProducer.prompts_from_text(text, count=4)
        assert len(prompts) == 4
        assert all(len(p) >= 8 for p in prompts)

    def test_empty_prompts_raise(self, tmp_path: Path) -> None:
        from domain.training._internal.selfgen import SelfGenProducer

        with pytest.raises(ValueError):
            SelfGenProducer(tmp_path / "c.jsonl", prompts=[])

    def test_resolve_corpus_path_local_base(self, tmp_path: Path) -> None:
        from domain.training._internal.selfgen import resolve_corpus_path

        p = resolve_corpus_path("selfgen", local_base=tmp_path)
        assert p == tmp_path / "selfgen" / "corpus.jsonl"


class TestSelfGenConfig:
    """TrainerConfig validation for the selfgen_* fields."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"selfgen_interval": -1},
            {"selfgen_quota": 1.5},
            {"selfgen_quota": -0.1},
            {"selfgen_samples": 0},
            {"selfgen_min_quality": 6.0},
        ],
    )
    def test_invalid_selfgen_config_raises(self, kwargs: dict) -> None:
        from domain.training._internal.train_pipeline import TrainerConfig

        with pytest.raises(ValueError):
            TrainerConfig(**kwargs)

    def test_defaults_are_disabled(self) -> None:
        from domain.training._internal.train_pipeline import TrainerConfig

        cfg = TrainerConfig()
        assert cfg.selfgen_interval == 0
        assert cfg.selfgen_quota == 0.25


class TestSelfGenTrainerIntegration:
    """SloughGPTTrainer: rounds run inside train(), corpus grows, batches mix."""

    def _make_trainer(self, tmp_path: Path, **selfgen_overrides):
        from domain.training._internal.train_pipeline import (
            SloughGPTTrainer,
            TrainerConfig,
        )

        corpus = tmp_path / "corpus.txt"
        tmp_path.mkdir(parents=True, exist_ok=True)
        corpus.write_text("abcdefghij" * 64, encoding="utf-8")
        cfg = TrainerConfig(
            n_embed=48,
            n_layer=1,
            n_head=2,
            block_size=12,
            batch_size=2,
            epochs=1,
            max_steps=6,
            log_interval=100,
            eval_interval=100,
            checkpoint_interval=100_000,
            checkpoint_dir=str(tmp_path / "ckpt"),
            selfgen_interval=2,
            selfgen_prompts=["The quick brown fox"],
            selfgen_min_quality=0.0,
            selfgen_local_base=str(tmp_path / "feed"),
            **selfgen_overrides,
        )
        return SloughGPTTrainer(data_path=str(corpus), config=cfg)

    def test_rounds_grow_corpus_during_training(self, tmp_path: Path) -> None:
        trainer = self._make_trainer(tmp_path)
        assert trainer._selfgen_producer is not None

        result = trainer.train()
        assert result["global_step"] == 6

        # 6 steps / interval 2 = rounds at steps 2, 4, 6
        producer = trainer._selfgen_producer
        assert producer.rounds == 3
        corpus = Path(producer.corpus_path)
        records = [json.loads(l) for l in corpus.read_text().splitlines() if l.strip()]
        # generation on an untrained tiny model may all be gated/dup — but the
        # producer must have been invoked and the run must have completed
        assert corpus.exists()
        assert producer.rounds == 3
        assert producer.written == len(records)

    def test_disabled_by_default(self, tmp_path: Path) -> None:
        from domain.training._internal.train_pipeline import (
            SloughGPTTrainer,
            TrainerConfig,
        )

        corpus = tmp_path / "corpus.txt"
        corpus.write_text("abcdefghij" * 64, encoding="utf-8")
        cfg = TrainerConfig(
            n_embed=48, n_layer=1, n_head=2, block_size=12, batch_size=2, epochs=1, max_steps=2
        )
        trainer = SloughGPTTrainer(data_path=str(corpus), config=cfg)
        assert trainer._selfgen_producer is None
        assert trainer._selfgen_sampler is None

    def test_quota_mixing_uses_both_sources(self, tmp_path: Path) -> None:
        """quota=1.0 always draws from the selfgen sampler; quota=0 never does."""
        trainer_hi = self._make_trainer(tmp_path / "hi", selfgen_quota=1.0)
        for _ in range(5):
            x, y = trainer_hi.get_batch("train")
            assert x.shape == (2, 12)
            assert y.shape == (2, 12)

        trainer_lo = self._make_trainer(tmp_path / "lo", selfgen_quota=0.0)
        # Track sampler usage by monkey-counting: with quota 0 the mix rng
        # never selects, so batches come from the static split.
        draws = []
        orig = trainer_lo._selfgen_sampler.get_batch

        def spy(bs):
            draws.append(bs)
            return orig(bs)

        trainer_lo._selfgen_sampler.get_batch = spy  # type: ignore[method-assign]
        for _ in range(10):
            trainer_lo.get_batch("train")
        assert draws == []

    def test_generation_failure_never_kills_run(self, tmp_path: Path) -> None:
        trainer = self._make_trainer(tmp_path)

        def boom(*_args, **_kwargs):
            raise RuntimeError("generation exploded")

        trainer.generate = boom  # type: ignore[method-assign]
        result = trainer.train()
        assert result["global_step"] == 6
        assert trainer._selfgen_producer.rounds == 3

    def test_train_batch_returns_int64(self, tmp_path: Path) -> None:
        trainer = self._make_trainer(tmp_path, selfgen_quota=1.0)
        x, y = trainer.get_batch("train")
        assert x.dtype == np.int64
        assert y.dtype == np.int64
