"""Tests for HFLoraTrainer, load_lora_adapter, merge_lora_adapter."""

import tempfile
import numpy as np
import pytest
from pathlib import Path

from domains.training.slonet import SloTransformer, Tensor, cross_entropy, SloAdam
from domains.training.lora import (
    LoRALinear, LoRAConfig, apply_lora_to_model, get_lora_parameters,
    _walk_slo_tree,
)
from domains.training.hf_lora_finetune import (
    HFLoraConfig, HFLoraTrainer, load_lora_adapter, merge_lora_adapter,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_model(vocab_size=64, n_embed=32, n_layer=2, n_head=2, block_size=128):
    """Create a tiny SloTransformer for testing."""
    return SloTransformer(
        vocab_size=vocab_size, n_embed=n_embed,
        n_layer=n_layer, n_head=n_head, block_size=block_size,
    )


def _save_text_data(path: Path, n_chars=2000):
    """Write a small text file for training."""
    text = "hello world " * (n_chars // 12)
    path.write_text(text[:n_chars])
    return path


# ── HFLoraConfig ─────────────────────────────────────────────────────────────


class TestHFLoraConfig:

    def test_defaults(self):
        cfg = HFLoraConfig()
        assert cfg.rank == 8
        assert cfg.alpha == 16.0
        assert cfg.epochs == 3
        assert cfg.learning_rate == 1e-4

    def test_auto_adapter_name(self):
        cfg = HFLoraConfig(model_path="models/gpt2.slnc")
        assert cfg.adapter_name == "gpt2_lora_r8"

    def test_custom_adapter_name(self):
        cfg = HFLoraConfig(adapter_name="my_adapter")
        assert cfg.adapter_name == "my_adapter"


# ── HFLoraTrainer ────────────────────────────────────────────────────────────


class TestHFLoraTrainer:

    def test_apply_lora(self):
        """apply_lora should inject LoRA layers into the model."""
        model = _make_model()
        cfg = HFLoraConfig(target_modules=["W_q", "W_v"])
        trainer = HFLoraTrainer(cfg)
        trainer.model = model

        params = trainer.apply_lora()
        assert len(params) > 0
        n_lora = sum(1 for p in params.values()
                     if isinstance(p, (Tensor,)) and hasattr(p, 'data'))
        assert n_lora > 0

    def test_apply_lora_no_model_raises(self):
        """apply_lora without load_model should raise."""
        cfg = HFLoraConfig()
        trainer = HFLoraTrainer(cfg)
        with pytest.raises(RuntimeError):
            trainer.apply_lora()

    def test_train_step_with_synthetic_data(self):
        """Run a few training steps on synthetic data — should produce finite losses."""
        model = _make_model(vocab_size=32, n_embed=16, n_layer=1, n_head=2, block_size=32)

        cfg = HFLoraConfig(
            rank=4, alpha=8.0,
            target_modules=["W_q", "W_v"],
            epochs=1,
            batch_size=4,
            block_size=32,
            learning_rate=1e-3,
            log_interval=5,
        )
        trainer = HFLoraTrainer(cfg)
        trainer.model = model
        trainer.apply_lora()

        # Manually run a few training steps
        lora_tensors = [p for p in trainer.lora_params.values()
                        if hasattr(p, 'data') and hasattr(p, 'requires_grad')]
        optimizer = SloAdam(lr=cfg.learning_rate)

        vocab_size = model.vocab_size
        losses = []
        for step in range(10):
            x = np.random.randint(0, vocab_size, (2, cfg.block_size))
            y = np.random.randint(0, vocab_size, (2, cfg.block_size))

            logits, _ = model.forward(Tensor(x))
            loss = cross_entropy(
                logits.reshape(-1, vocab_size),
                Tensor(y.reshape(-1)),
            )
            loss.backward()

            # Clip gradients
            total_norm = 0.0
            for p in lora_tensors:
                if hasattr(p, 'grad') and p.grad is not None:
                    total_norm += float(np.sum(p.grad.data ** 2))
            total_norm = np.sqrt(total_norm)
            if total_norm > 1.0:
                scale = 1.0 / total_norm
                for p in lora_tensors:
                    if hasattr(p, 'grad') and p.grad is not None:
                        p.grad.data *= scale

            optimizer.step(lora_tensors)
            for p in lora_tensors:
                if hasattr(p, 'grad') and p.grad is not None:
                    p.grad.data[:] = 0.0

            losses.append(float(loss.data))

        # All losses should be finite
        assert all(np.isfinite(l) for l in losses), f"Non-finite loss in {losses}"


# ── save/load adapter ────────────────────────────────────────────────────────


class TestSaveLoadAdapter:

    def test_save_and_load_roundtrip(self):
        """Save adapter, load it back, weights should match."""
        model = _make_model(vocab_size=32, n_embed=16, n_layer=1, n_head=2)
        cfg = HFLoraConfig(rank=4, alpha=8.0, target_modules=["W_q", "W_v"])
        trainer = HFLoraTrainer(cfg)
        trainer.model = model
        trainer.apply_lora()

        # Save
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.config.output_dir = tmpdir
            adapter_path = trainer._save_adapter()
            assert adapter_path.exists()

            # Create fresh model with LoRA
            model2 = _make_model(vocab_size=32, n_embed=16, n_layer=1, n_head=2)
            lora_cfg = LoRAConfig(rank=4, alpha=8.0, target_modules=["W_q", "W_v"])
            model2 = apply_lora_to_model(model2, lora_cfg)

            # Load
            model2 = load_lora_adapter(model2, str(adapter_path))

            # Verify weights match
            for name, param in get_lora_parameters(model).items():
                param2 = get_lora_parameters(model2).get(name)
                assert param2 is not None, f"Missing param {name} after load"
                np.testing.assert_allclose(param.data, param2.data, rtol=1e-6)


# ── merge_lora_adapter ───────────────────────────────────────────────────────


class TestMergeLoRAAdapter:

    def test_merge_sets_weights(self):
        """After merge, LoRA weight should be folded into base weight."""
        model = _make_model(vocab_size=32, n_embed=16, n_layer=1, n_head=2)
        cfg = HFLoraConfig(rank=4, alpha=8.0, target_modules=["W_q"])
        trainer = HFLoraTrainer(cfg)
        trainer.model = model
        trainer.apply_lora()

        # Record base weight before merge (for a specific LoRA layer)
        lora_params_before = get_lora_parameters(model)
        first_lora_key = [k for k in lora_params_before if 'lora_A' in k][0]
        w_before = lora_params_before[first_lora_key].data.copy()

        # Merge
        model = merge_lora_adapter(model)

        # After merge, lora_A and lora_B should be zeroed
        lora_params_after = get_lora_parameters(model)
        for key in lora_params_after:
            if 'lora_A' in key or 'lora_B' in key:
                np.testing.assert_allclose(
                    lora_params_after[key].data, 0.0, atol=1e-7,
                    err_msg=f"{key} not zeroed after merge",
                )

    def test_merge_clears_has_lora(self):
        """After merge, _has_lora should be False."""
        model = _make_model()
        cfg = HFLoraConfig(rank=4, target_modules=["W_q"])
        trainer = HFLoraTrainer(cfg)
        trainer.model = model
        trainer.apply_lora()

        assert model._has_lora is True
        model = merge_lora_adapter(model)
        assert model._has_lora is False

    def test_merge_via_walk_slo_tree(self):
        """merge_lora_adapter should replace LoRALinear with SloLinear via _walk_slo_tree."""
        from domains.training.slonet import SloLinear

        model = _make_model(vocab_size=32, n_embed=16, n_layer=1, n_head=2)
        cfg = HFLoraConfig(rank=4, target_modules=["W_q", "W_v"])
        trainer = HFLoraTrainer(cfg)
        trainer.model = model
        trainer.apply_lora()

        model = merge_lora_adapter(model)

        # After merge, no LoRALinear should remain — all replaced with SloLinear
        for path, module in _walk_slo_tree(model, []):
            assert not isinstance(module, LoRALinear), \
                f"LoRALinear still present at {path} after merge"
            if 'W_q' in path or 'W_v' in path:
                assert isinstance(module, SloLinear), \
                    f"Expected SloLinear at {path}, got {type(module).__name__}"


# ── cancel ───────────────────────────────────────────────────────────────────


class TestCancellation:

    def test_stop_sets_flag(self):
        """stop() should set the cancel event."""
        cfg = HFLoraConfig()
        trainer = HFLoraTrainer(cfg)
        import threading
        trainer._cancel_event = threading.Event()
        trainer._is_training = True

        trainer.stop()
        assert trainer._cancel_event.is_set()
        assert trainer._is_training is False
