"""Unit tests for ``training.router._sloughgpt_trainer_kwds`` (HTTP → SloughGPTTrainer)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SERVER_DIR = _REPO_ROOT / "apps" / "api" / "server"
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))


def test_maps_learning_rate_scheduler_and_device() -> None:
    from training.router import _sloughgpt_trainer_kwds

    snap = {
        "n_embed": 64,
        "n_layer": 2,
        "n_head": 2,
        "block_size": 32,
        "batch_size": 8,
        "epochs": 1,
        "learning_rate": 0.02,
        "max_steps": 10,
        "log_interval": 5,
        "eval_interval": 20,
        "scheduler": "linear",
        "device": "cpu",
        "dropout": 0.2,
        "weight_decay": 0.05,
        "gradient_accumulation_steps": 2,
        "max_grad_norm": 2.0,
        "warmup_steps": 0,
        "min_lr": 1e-6,
        "use_lora": True,
        "lora_rank": 16,
        "lora_alpha": 32,
        "checkpoint_dir": "ck",
        "checkpoint_interval": 99,
        "save_best_only": True,
        "max_checkpoints": 3,
    }
    k = _sloughgpt_trainer_kwds(snap)
    assert k["lr"] == 0.02
    assert k["scheduler_type"] == "linear"
    assert k["device"] == "cpu"
    assert k["dropout"] == 0.2
    assert k["weight_decay"] == 0.05
    assert k["gradient_accumulation_steps"] == 2
    assert k["max_grad_norm"] == 2.0
    assert "use_mixed_precision" not in k
    assert "mixed_precision_dtype" not in k
    assert k["warmup_steps"] == 0
    assert k["min_lr"] == 1e-6
    assert k["use_lora"] is True
    assert k["lora_rank"] == 16
    assert k["lora_alpha"] == 32
    assert k["checkpoint_dir"] == "ck"
    assert k["checkpoint_interval"] == 99
    assert k["save_best_only"] is True
    assert k["max_checkpoints"] == 3


def test_device_none_when_missing() -> None:
    from training.router import _sloughgpt_trainer_kwds

    k = _sloughgpt_trainer_kwds({"learning_rate": 1e-3})
    assert k["device"] is None


# ---------------------------------------------------------------------------
# BPE tokenizer wiring — algo='bpe' trains a cached TokenTree (card:
# "char-level tokenization inflates total_steps")
# ---------------------------------------------------------------------------

_CORPUS = (
    "the quick brown fox jumps over the lazy dog and runs across the meadow "
    "again and again while the dog sleeps soundly in the warm sun all day long. "
) * 20


def _write_corpus(tmp_path, text=_CORPUS, name="corpus.txt") -> str:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return str(p)


class TestBpeTokenizerWiring:
    def test_bpe_algo_returns_token_tree(self, tmp_path, monkeypatch) -> None:
        from training.helpers import _sloughgpt_trainer_kwds

        from domain.training._internal.token_tree import TokenTree

        monkeypatch.setenv("SLO_TOKEN_TREE_CACHE", str(tmp_path / "cache"))
        path = _write_corpus(tmp_path)
        k = _sloughgpt_trainer_kwds({"algo": "bpe"}, data_path=path)
        assert isinstance(k["tokenizer"], TokenTree)
        ids = k["tokenizer"].encode("the quick brown fox")
        assert ids and all(isinstance(i, int) for i in ids)

    @pytest.mark.parametrize("algo", [None, "char", "unigram", "wordpiece", ""])
    def test_legacy_algos_get_no_tokenizer_key(self, tmp_path, monkeypatch, algo) -> None:
        from training.helpers import _sloughgpt_trainer_kwds

        monkeypatch.setenv("SLO_TOKEN_TREE_CACHE", str(tmp_path / "cache"))
        path = _write_corpus(tmp_path)
        k = _sloughgpt_trainer_kwds({"algo": algo}, data_path=path)
        assert "tokenizer" not in k

    def test_bpe_without_data_path_degrades_to_legacy(self) -> None:
        from training.helpers import _sloughgpt_trainer_kwds

        k = _sloughgpt_trainer_kwds({"algo": "bpe"})
        assert "tokenizer" not in k

    def test_tokenizer_is_cached_on_disk(self, tmp_path, monkeypatch) -> None:
        from training.helpers import _sloughgpt_trainer_kwds

        from domain.training._internal.token_tree import TokenTree

        monkeypatch.setenv("SLO_TOKEN_TREE_CACHE", str(tmp_path / "cache"))
        path = _write_corpus(tmp_path)
        first = _sloughgpt_trainer_kwds({"algo": "bpe"}, data_path=path)["tokenizer"]

        def _boom(*_a, **_k):
            raise AssertionError("cache miss — tokenizer retrained")

        monkeypatch.setattr(TokenTree, "train", _boom)
        second = _sloughgpt_trainer_kwds({"algo": "bpe"}, data_path=path)["tokenizer"]
        assert isinstance(second, TokenTree)
        assert second.encode("the quick brown fox") == first.encode("the quick brown fox")

    def test_same_corpus_reuses_cache_and_new_corpus_retrains(
        self, tmp_path, monkeypatch
    ) -> None:
        from training.helpers import _sloughgpt_trainer_kwds

        from domain.training._internal.token_tree import TokenTree

        monkeypatch.setenv("SLO_TOKEN_TREE_CACHE", str(tmp_path / "cache"))
        trains = {"n": 0}
        orig_train = TokenTree.train

        def _spy(self, *args, **kwargs):
            trains["n"] += 1
            return orig_train(self, *args, **kwargs)

        monkeypatch.setattr(TokenTree, "train", _spy)

        path_a = _write_corpus(tmp_path, "alpha beta gamma delta " * 50, "a.txt")
        path_b = _write_corpus(tmp_path, "zeta eta theta iota kappa " * 50, "b.txt")
        _sloughgpt_trainer_kwds({"algo": "bpe"}, data_path=path_a)
        assert trains["n"] == 1
        _sloughgpt_trainer_kwds({"algo": "bpe"}, data_path=path_b)
        assert trains["n"] == 2  # different content hash -> retrain
        _sloughgpt_trainer_kwds({"algo": "bpe"}, data_path=path_a)
        assert trains["n"] == 2  # same content -> cache hit

    def test_sample_corpus_caps_and_is_deterministic(self) -> None:
        from training.helpers import _sample_corpus

        text = "abcdefghij" * 100_000  # 1,000,000 chars
        sample = _sample_corpus(text, 400_000)
        assert 0 < len(sample) <= 400_000
        assert sample == _sample_corpus(text, 400_000)
        assert sample == text[::3]  # even spread, deterministic stride
        short = "short text"
        assert _sample_corpus(short, 400_000) == short
