"""Tests for HFLoraTrainer, load_lora_adapter, merge_lora_adapter."""

import tempfile
import numpy as np
import pytest
from pathlib import Path

from domains.training.slonet import SloTransformer, Tensor, cross_entropy, SloAdam
from domains.training.lora import (
    LoRALinear, LoRAConfig, apply_lora_to_model, get_lora_parameters,
    _walk_slo_tree, count_lora_parameters,
)
from domains.training.hf_lora_finetune import (
    HFLoraConfig, HFLoraTrainer, load_lora_adapter, merge_lora_adapter,
    _LoRADataset,
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

    def test_auto_adapter_name_with_rank(self):
        cfg = HFLoraConfig(model_path="models/model.slnc", rank=16)
        assert cfg.adapter_name == "model_lora_r16"

    def test_default_target_modules(self):
        cfg = HFLoraConfig()
        assert cfg.target_modules == ["W_q", "W_k", "W_v", "W_o"]

    def test_custom_target_modules(self):
        cfg = HFLoraConfig(target_modules=["W_q", "W_v"])
        assert cfg.target_modules == ["W_q", "W_v"]

    def test_dropout_default(self):
        cfg = HFLoraConfig()
        assert cfg.dropout == 0.0

    def test_weight_decay_default(self):
        cfg = HFLoraConfig()
        assert cfg.weight_decay == 0.01

    def test_warmup_steps_default(self):
        cfg = HFLoraConfig()
        assert cfg.warmup_steps == 0

    def test_grad_clip_default(self):
        cfg = HFLoraConfig()
        assert cfg.grad_clip == 1.0

    def test_grad_accumulation_default(self):
        cfg = HFLoraConfig()
        assert cfg.grad_accumulation_steps == 1

    def test_output_dir_default(self):
        cfg = HFLoraConfig()
        assert cfg.output_dir == "models"

    def test_log_interval_default(self):
        cfg = HFLoraConfig()
        assert cfg.log_interval == 10

    def test_progress_callback_default(self):
        cfg = HFLoraConfig()
        assert cfg.progress_callback is None

    def test_cancel_event_default(self):
        cfg = HFLoraConfig()
        assert cfg._cancel_event is None

    def test_block_size_default(self):
        cfg = HFLoraConfig()
        assert cfg.block_size == 128

    def test_batch_size_default(self):
        cfg = HFLoraConfig()
        assert cfg.batch_size == 8

    def test_model_path_stored(self):
        cfg = HFLoraConfig(model_path="/path/to/model.slnc")
        assert cfg.model_path == "/path/to/model.slnc"

    def test_data_path_stored(self):
        cfg = HFLoraConfig(data_path="/path/to/data.txt")
        assert cfg.data_path == "/path/to/data.txt"

    def test_auto_adapter_name_path_edge(self):
        cfg = HFLoraConfig(model_path="simple.txt")
        assert cfg.adapter_name == "simple_lora_r8"

    def test_all_fields_stored(self):
        cfg = HFLoraConfig(
            model_path="m", data_path="d", rank=2, alpha=4.0,
            dropout=0.1, target_modules=["W_q"],
            epochs=1, batch_size=2, block_size=64,
            learning_rate=0.01, weight_decay=0.1, warmup_steps=10,
            grad_clip=0.5, grad_accumulation_steps=4,
            output_dir="/tmp", adapter_name="custom",
            log_interval=5,
        )
        assert cfg.model_path == "m"
        assert cfg.data_path == "d"
        assert cfg.rank == 2
        assert cfg.alpha == 4.0
        assert cfg.dropout == 0.1
        assert cfg.target_modules == ["W_q"]
        assert cfg.epochs == 1
        assert cfg.batch_size == 2
        assert cfg.block_size == 64
        assert cfg.learning_rate == 0.01
        assert cfg.weight_decay == 0.1
        assert cfg.warmup_steps == 10
        assert cfg.grad_clip == 0.5
        assert cfg.grad_accumulation_steps == 4
        assert cfg.output_dir == "/tmp"
        assert cfg.adapter_name == "custom"
        assert cfg.log_interval == 5


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

        assert all(np.isfinite(l) for l in losses), f"Non-finite loss in {losses}"

    def test_trainer_init(self):
        cfg = HFLoraConfig()
        trainer = HFLoraTrainer(cfg)
        assert trainer.config is cfg
        assert trainer.model is None
        assert trainer.lora_params == {}

    def test_is_training_default(self):
        cfg = HFLoraConfig()
        trainer = HFLoraTrainer(cfg)
        assert trainer.is_training is False

    def test_stop_sets_flag(self):
        import threading
        cfg = HFLoraConfig()
        cfg._cancel_event = threading.Event()
        trainer = HFLoraTrainer(cfg)
        trainer._is_training = True
        trainer.stop()
        assert cfg._cancel_event.is_set()
        assert trainer._is_training is False

    def test_apply_lora_returns_dict(self):
        model = _make_model()
        cfg = HFLoraConfig(target_modules=["W_q"])
        trainer = HFLoraTrainer(cfg)
        trainer.model = model
        params = trainer.apply_lora()
        assert isinstance(params, dict)

    def test_apply_lora_populates_lora_params(self):
        model = _make_model()
        cfg = HFLoraConfig(target_modules=["W_q"])
        trainer = HFLoraTrainer(cfg)
        trainer.model = model
        trainer.apply_lora()
        assert len(trainer.lora_params) > 0

    def test_lora_params_keys_contain_lora(self):
        model = _make_model()
        cfg = HFLoraConfig(target_modules=["W_q"])
        trainer = HFLoraTrainer(cfg)
        trainer.model = model
        trainer.apply_lora()
        for key in trainer.lora_params:
            assert "lora" in key.lower()

    def test_model_set_after_apply(self):
        model = _make_model()
        cfg = HFLoraConfig(target_modules=["W_q"])
        trainer = HFLoraTrainer(cfg)
        trainer.model = model
        trainer.apply_lora()
        assert trainer.model is not None

    def test_stop_without_cancel_event(self):
        cfg = HFLoraConfig()
        trainer = HFLoraTrainer(cfg)
        trainer._is_training = True
        trainer.stop()
        assert trainer._is_training is False

    def test_training_thread_default_none(self):
        cfg = HFLoraConfig()
        trainer = HFLoraTrainer(cfg)
        assert trainer._training_thread is None


# ── save/load adapter ────────────────────────────────────────────────────────


class TestSaveLoadAdapter:

    def test_save_and_load_roundtrip(self):
        """Save adapter, load it back, weights should match."""
        model = _make_model(vocab_size=32, n_embed=16, n_layer=1, n_head=2)
        cfg = HFLoraConfig(rank=4, alpha=8.0, target_modules=["W_q", "W_v"])
        trainer = HFLoraTrainer(cfg)
        trainer.model = model
        trainer.apply_lora()

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.config.output_dir = tmpdir
            adapter_path = trainer._save_adapter()
            assert adapter_path.exists()

            model2 = _make_model(vocab_size=32, n_embed=16, n_layer=1, n_head=2)
            lora_cfg = LoRAConfig(rank=4, alpha=8.0, target_modules=["W_q", "W_v"])
            model2 = apply_lora_to_model(model2, lora_cfg)

            model2 = load_lora_adapter(model2, str(adapter_path))

            for name, param in get_lora_parameters(model).items():
                param2 = get_lora_parameters(model2).get(name)
                assert param2 is not None, f"Missing param {name} after load"
                np.testing.assert_allclose(param.data, param2.data, rtol=1e-6)

    def test_save_creates_npz(self):
        model = _make_model(vocab_size=32, n_embed=16, n_layer=1, n_head=2)
        cfg = HFLoraConfig(rank=4, target_modules=["W_q"])
        trainer = HFLoraTrainer(cfg)
        trainer.model = model
        trainer.apply_lora()
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.config.output_dir = tmpdir
            path = trainer._save_adapter()
            assert path.suffix == ".npz"

    def test_save_metadata_includes_config(self):
        model = _make_model(vocab_size=32, n_embed=16, n_layer=1, n_head=2)
        cfg = HFLoraConfig(rank=4, alpha=8.0, target_modules=["W_q"])
        trainer = HFLoraTrainer(cfg)
        trainer.model = model
        trainer.apply_lora()
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.config.output_dir = tmpdir
            path = trainer._save_adapter()
            data = np.load(str(path))
            assert "_config/rank" in data
            assert int(data["_config/rank"][0]) == 4
            assert "_config/alpha" in data
            assert float(data["_config/alpha"][0]) == 8.0

    def test_save_file_is_valid_npz(self):
        model = _make_model(vocab_size=32, n_embed=16, n_layer=1, n_head=2)
        cfg = HFLoraConfig(rank=4, target_modules=["W_q"])
        trainer = HFLoraTrainer(cfg)
        trainer.model = model
        trainer.apply_lora()
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.config.output_dir = tmpdir
            path = trainer._save_adapter()
            data = np.load(str(path))
            assert len(data.files) > 0

    def test_adapter_name_in_filename(self):
        model = _make_model(vocab_size=32, n_embed=16, n_layer=1, n_head=2)
        cfg = HFLoraConfig(rank=4, target_modules=["W_q"], adapter_name="test_adapter")
        trainer = HFLoraTrainer(cfg)
        trainer.model = model
        trainer.apply_lora()
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.config.output_dir = tmpdir
            path = trainer._save_adapter()
            assert "test_adapter" in path.name

    def test_load_nonexistent_adapter(self):
        model = _make_model(vocab_size=32, n_embed=16, n_layer=1, n_head=2)
        cfg = LoRAConfig(rank=4, target_modules=["W_q"])
        model = apply_lora_to_model(model, cfg)
        with pytest.raises(Exception):
            load_lora_adapter(model, "/nonexistent/path.npz")

    def test_save_load_preserves_rank(self):
        model = _make_model(vocab_size=32, n_embed=16, n_layer=1, n_head=2)
        cfg = HFLoraConfig(rank=8, target_modules=["W_q"])
        trainer = HFLoraTrainer(cfg)
        trainer.model = model
        trainer.apply_lora()
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.config.output_dir = tmpdir
            path = trainer._save_adapter()
            data = np.load(str(path))
            assert int(data["_config/rank"][0]) == 8

    def test_save_load_multiple_modules(self):
        model = _make_model(vocab_size=32, n_embed=16, n_layer=1, n_head=2)
        cfg = HFLoraConfig(rank=4, target_modules=["W_q", "W_v", "W_k"])
        trainer = HFLoraTrainer(cfg)
        trainer.model = model
        trainer.apply_lora()
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.config.output_dir = tmpdir
            path = trainer._save_adapter()
            data = np.load(str(path))
            assert int(data["_config/target_modules"][0]) == 3

    def test_save_creates_output_dir(self):
        model = _make_model(vocab_size=32, n_embed=16, n_layer=1, n_head=2)
        cfg = HFLoraConfig(rank=4, target_modules=["W_q"])
        trainer = HFLoraTrainer(cfg)
        trainer.model = model
        trainer.apply_lora()
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.config.output_dir = str(Path(tmpdir) / "nested" / "dir")
            path = trainer._save_adapter()
            assert path.exists()


# ── merge_lora_adapter ───────────────────────────────────────────────────────


class TestMergeLoRAAdapter:

    def test_merge_sets_weights(self):
        """After merge, LoRA weight should be folded into base weight."""
        model = _make_model(vocab_size=32, n_embed=16, n_layer=1, n_head=2)
        cfg = HFLoraConfig(rank=4, alpha=8.0, target_modules=["W_q"])
        trainer = HFLoraTrainer(cfg)
        trainer.model = model
        trainer.apply_lora()

        lora_params_before = get_lora_parameters(model)
        first_lora_key = [k for k in lora_params_before if 'lora_A' in k][0]
        w_before = lora_params_before[first_lora_key].data.copy()

        model = merge_lora_adapter(model)

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

        for path, module in _walk_slo_tree(model, []):
            assert not isinstance(module, LoRALinear), \
                f"LoRALinear still present at {path} after merge"
            if 'W_q' in path or 'W_v' in path:
                assert isinstance(module, SloLinear), \
                    f"Expected SloLinear at {path}, got {type(module).__name__}"

    def test_merge_returns_model(self):
        model = _make_model()
        cfg = HFLoraConfig(rank=4, target_modules=["W_q"])
        trainer = HFLoraTrainer(cfg)
        trainer.model = model
        trainer.apply_lora()
        result = merge_lora_adapter(model)
        assert result is model

    def test_merge_multiple_modules(self):
        from domains.training.slonet import SloLinear
        model = _make_model(vocab_size=32, n_embed=16, n_layer=1, n_head=2)
        cfg = HFLoraConfig(rank=4, target_modules=["W_q", "W_v", "W_k"])
        trainer = HFLoraTrainer(cfg)
        trainer.model = model
        trainer.apply_lora()
        model = merge_lora_adapter(model)
        for path, module in _walk_slo_tree(model, []):
            assert not isinstance(module, LoRALinear)

    def test_merge_then_inference(self):
        model = _make_model(vocab_size=32, n_embed=16, n_layer=1, n_head=2)
        cfg = HFLoraConfig(rank=4, target_modules=["W_q"])
        trainer = HFLoraTrainer(cfg)
        trainer.model = model
        trainer.apply_lora()
        model = merge_lora_adapter(model)
        x = np.random.randint(0, 32, (1, 16))
        logits, _ = model.forward(Tensor(x))
        assert logits is not None

    def test_merge_base_weight_approximately_unchanged(self):
        model = _make_model(vocab_size=32, n_embed=16, n_layer=1, n_head=2)
        # Record base weight before LoRA
        from domains.training.slonet import SloLinear
        w_before = None
        for path, module in _walk_slo_tree(model, []):
            if 'W_q' in path and isinstance(module, SloLinear):
                w_before = module.weight.data.copy()
                break

        cfg = HFLoraConfig(rank=4, alpha=0.01, target_modules=["W_q"])
        trainer = HFLoraTrainer(cfg)
        trainer.model = model
        trainer.apply_lora()
        model = merge_lora_adapter(model)

        for path, module in _walk_slo_tree(model, []):
            if 'W_q' in path and isinstance(module, SloLinear):
                # With small alpha, merged weight should be close to original
                np.testing.assert_allclose(
                    module.weight.data, w_before, atol=0.1,
                    err_msg=f"W_q weight changed significantly after merge"
                )
                break


# ── cancel ───────────────────────────────────────────────────────────────────


class TestCancellation:

    def test_stop_sets_flag(self):
        """stop() should set the cancel event."""
        import threading
        cfg = HFLoraConfig()
        cfg._cancel_event = threading.Event()
        trainer = HFLoraTrainer(cfg)
        trainer._is_training = True

        trainer.stop()
        assert cfg._cancel_event.is_set()
        assert trainer._is_training is False

    def test_stop_without_event(self):
        cfg = HFLoraConfig()
        trainer = HFLoraTrainer(cfg)
        trainer._is_training = True
        trainer.stop()
        assert trainer._is_training is False

    def test_stop_idempotent(self):
        import threading
        cfg = HFLoraConfig()
        cfg._cancel_event = threading.Event()
        trainer = HFLoraTrainer(cfg)
        trainer._is_training = True
        trainer.stop()
        trainer.stop()
        assert cfg._cancel_event.is_set()
        assert trainer._is_training is False

    def test_stop_sets_config_event(self):
        import threading
        cfg = HFLoraConfig()
        cfg._cancel_event = threading.Event()
        trainer = HFLoraTrainer(cfg)
        trainer._is_training = True
        trainer.stop()
        assert cfg._cancel_event.is_set()


# ── _LoRADataset ─────────────────────────────────────────────────────────────


class TestLoRADataset:
    def test_len(self):
        data = np.arange(100, dtype=np.int64)
        ds = _LoRADataset(data, block_size=10)
        assert len(ds) == 90

    def test_getitem_shapes(self):
        data = np.arange(100, dtype=np.int64)
        ds = _LoRADataset(data, block_size=10)
        x, y = ds[0]
        assert x.shape == (10,)
        assert y.shape == (10,)

    def test_getitem_y_shifted(self):
        data = np.arange(100, dtype=np.int64)
        ds = _LoRADataset(data, block_size=10)
        x, y = ds[0]
        np.testing.assert_array_equal(y, data[1:11])

    def test_getitem_nonzero_idx(self):
        data = np.arange(100, dtype=np.int64)
        ds = _LoRADataset(data, block_size=10)
        x, y = ds[5]
        np.testing.assert_array_equal(x, data[5:15])
        np.testing.assert_array_equal(y, data[6:16])

    def test_len_too_short(self):
        data = np.arange(5, dtype=np.int64)
        ds = _LoRADataset(data, block_size=10)
        assert len(ds) == 0

    def test_len_exact_block_size(self):
        data = np.arange(10, dtype=np.int64)
        ds = _LoRADataset(data, block_size=10)
        assert len(ds) == 0

    def test_len_one_more_than_block(self):
        data = np.arange(11, dtype=np.int64)
        ds = _LoRADataset(data, block_size=10)
        assert len(ds) == 1

    def test_converts_list_to_array(self):
        ds = _LoRADataset([1, 2, 3, 4, 5], block_size=2)
        assert isinstance(ds.data, np.ndarray)

    def test_preserves_existing_array(self):
        arr = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        ds = _LoRADataset(arr, block_size=2)
        assert ds.data is arr

    def test_block_size_one(self):
        data = np.arange(10, dtype=np.int64)
        ds = _LoRADataset(data, block_size=1)
        x, y = ds[0]
        assert x.shape == (1,)
        assert y.shape == (1,)
        assert y[0] == data[1]

    def test_getitem_types(self):
        data = np.arange(100, dtype=np.int64)
        ds = _LoRADataset(data, block_size=10)
        x, y = ds[0]
        assert x.dtype == np.int64
        assert y.dtype == np.int64


# ── count_lora_parameters ───────────────────────────────────────────────────


class TestCountLoraParameters:
    def test_count_after_apply(self):
        model = _make_model()
        cfg = LoRAConfig(rank=4, target_modules=["W_q"])
        model = apply_lora_to_model(model, cfg)
        count = count_lora_parameters(model)
        assert count > 0

    def test_count_is_integer(self):
        model = _make_model()
        cfg = LoRAConfig(rank=4, target_modules=["W_q"])
        model = apply_lora_to_model(model, cfg)
        count = count_lora_parameters(model)
        assert isinstance(count, int)

    def test_count_increases_with_rank(self):
        model1 = _make_model()
        model2 = _make_model()
        cfg1 = LoRAConfig(rank=2, target_modules=["W_q"])
        cfg2 = LoRAConfig(rank=8, target_modules=["W_q"])
        model1 = apply_lora_to_model(model1, cfg1)
        model2 = apply_lora_to_model(model2, cfg2)
        assert count_lora_parameters(model2) > count_lora_parameters(model1)

    def test_count_increases_with_more_modules(self):
        model1 = _make_model()
        model2 = _make_model()
        cfg1 = LoRAConfig(rank=4, target_modules=["W_q"])
        cfg2 = LoRAConfig(rank=4, target_modules=["W_q", "W_v"])
        model1 = apply_lora_to_model(model1, cfg1)
        model2 = apply_lora_to_model(model2, cfg2)
        # More target modules means more LoRA layers
        lora1 = [k for k in get_lora_parameters(model1) if "lora" in k]
        lora2 = [k for k in get_lora_parameters(model2) if "lora" in k]
        assert len(lora2) > len(lora1)
