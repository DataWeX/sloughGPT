"""Goal 12: owned architecture — native .soul train → .slnc serve, no HF.

Acceptance:
  * Tiny native train writes a self-contained ``.soul`` (config + tokenizer).
  * ``soul_to_slnc`` compiles native key layout without HuggingFace.
  * ``from_soul`` rebuilds RoPE + RMSNorm (not abs-pos + LayerNorm).
  * ``from_slnc`` serves with the embedded tokenizer and generates text.
  * No MorphTokenizer / HF download is required on the path.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

DATA_TEXT = (
    "the quick brown fox jumps over the lazy dog and runs across the meadow "
    "again and again while the dog sleeps soundly in the warm sun all day "
    "long. " * 8
)


def _tiny_config(tmp_path, **overrides):
    from domain.training._internal.train_pipeline import TrainerConfig

    cfg = TrainerConfig(
        vocab_size=0,
        n_embed=16,
        n_layer=1,
        n_head=2,
        block_size=8,
        dropout=0.0,
        batch_size=2,
        epochs=1,
        max_steps=3,
        gradient_accumulation_steps=1,
        checkpoint_dir=str(tmp_path / "ckpts"),
        log_interval=1,
        eval_interval=1000,
        checkpoint_interval=1000,
        warmup_steps=1,
        min_lr=1e-5,
        max_checkpoints=2,
        scheduler_type="cosine",
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


@pytest.fixture
def trained_soul(tmp_path: Path) -> str:
    """Train a tiny native model and save a self-contained .soul."""
    from domain.training._internal.train_pipeline import SloughGPTTrainer

    corpus = tmp_path / "corpus.txt"
    corpus.write_text(DATA_TEXT, encoding="utf-8")
    trainer = SloughGPTTrainer(str(corpus), config=_tiny_config(tmp_path))
    out = str(tmp_path / "goal12")
    trainer.save(out)
    soul_path = out + ".soul"
    assert os.path.exists(soul_path), f"missing .soul: {soul_path}"
    return soul_path


class TestFromSoulNativeArch:
    def test_from_soul_builds_rope_rmsnorm(self, trained_soul: str) -> None:
        from domain.inference._internal.slonet_provider import SloNetChatProvider

        provider = SloNetChatProvider.from_soul(trained_soul, model_id="goal12-soul")
        model = provider._model
        assert model.pos_emb is None, "native soul must use RoPE (no abs pos_emb)"
        assert model.blocks[0].attn.use_rope is True
        norm = model.layers[-2]
        assert type(norm).__name__ == "SloRMSNorm", f"expected RMSNorm, got {type(norm)}"
        assert provider._tokenizer is not None, "soul must carry an embedded tokenizer"
        ids = provider._tokenizer.encode("hello")
        assert isinstance(ids, list) and len(ids) > 0

    def test_from_soul_generate_runs(self, trained_soul: str) -> None:
        from domain.inference._internal.slonet_provider import SloNetChatProvider

        provider = SloNetChatProvider.from_soul(trained_soul, model_id="goal12-gen")
        out = provider._model.generate(np.array([[1, 2, 3]], dtype=np.int64), max_new_tokens=4)
        arr = out.data if hasattr(out, "data") else out
        assert arr.shape[-1] >= 3


class TestSoulToSlnc:
    def test_soul_to_slnc_native_keys_no_hf(self, trained_soul: str, tmp_path: Path) -> None:
        from domain.infrastructure._internal.soul_to_slnc import soul_to_slnc

        slnc_path = tmp_path / "goal12.slnc"
        result = soul_to_slnc(trained_soul, slnc_path)
        assert os.path.exists(result)
        assert Path(result).suffix == ".slnc"

        from domain.infrastructure._internal.slnc.parser import SLNCParser

        parser = SLNCParser(str(slnc_path))
        config = parser.config
        weights = parser.get_weights_dict()
        # Native key layout preserved — no HF remapping.
        assert "tok_emb.weight" in weights
        assert "blocks.0.attn.q_proj.weight" in weights
        assert "lm_head.weight" in weights
        # Architecture flags for rebuild (explicit + HF-style keys)
        assert config.get("use_rope") is True
        assert config.get("norm_type") == "rms_norm"
        assert config.get("position_embedding_type") == "rope"
        # Embedded tokenizer so from_slnc never needs HF (main uses slo_tokenizer).
        tok = config.get("slo_tokenizer")
        assert isinstance(tok, dict), f"missing slo_tokenizer in config: {list(config)}"
        assert tok.get("type") in ("char", "token_tree")

    def test_from_slnc_embedded_tokenizer_and_generate(
        self, trained_soul: str, tmp_path: Path
    ) -> None:
        from domain.inference._internal.slonet_provider import SloNetChatProvider
        from domain.infrastructure._internal.soul_to_slnc import soul_to_slnc

        slnc_path = soul_to_slnc(trained_soul, tmp_path / "serve.slnc")

        # model_id is NOT an HF id — embedded tokenizer must cover it.
        provider = SloNetChatProvider.from_slnc(slnc_path, model_id="goal12-native")
        assert provider._tokenizer is not None, "embedded tokenizer missing"
        assert provider._tokenizer.encode("hello")

        model = provider._model
        assert model.pos_emb is None, "from_slnc must rebuild RoPE for native"
        assert type(model.layers[-2]).__name__ == "SloRMSNorm"

        out = model.generate(np.array([[1, 2, 3]], dtype=np.int64), max_new_tokens=4)
        arr = out.data if hasattr(out, "data") else out
        assert arr.shape[-1] >= 3

    def test_arch_config_native_weight_map(self, trained_soul: str) -> None:
        from domain.inference._internal.slo_format import load_soul

        _, state_dict = load_soul(trained_soul)
        from domain.infrastructure._internal.arch_config import build_arch

        arch = build_arch("sloughgpt", {"model_type": "sloughgpt"}, set(state_dict))
        assert arch.transpose_weights is False
        assert arch.activation == "swiglu"
        assert arch.positional in ("rope", "absolute")
        # Native identity: file keys are the model params.
        assert arch.weight_map.get("embed.token") == "tok_emb.weight"
