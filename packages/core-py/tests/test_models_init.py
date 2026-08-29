"""Tests for domains.models — ModelInterface, ModelLoader, SloughGPTModel."""

import tempfile
from pathlib import Path

import numpy as np
import pytest


class TestModelLoader:
    def test_register_and_load_external(self):
        from domains.models import ModelLoader, ModelInterface

        class DummyModel:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        ModelLoader.register("test_dummy", DummyModel)
        result = ModelLoader._load_external_model("test_dummy", {"foo": "bar"})
        assert isinstance(result, DummyModel)
        assert result.kwargs["foo"] == "bar"

    def test_register_loader_dispatch(self):
        from domains.models import ModelLoader

        def my_loader(path, device, **kwargs):
            return {"path": path, "device": device}

        ModelLoader.register_loader(".myext", my_loader)
        result = ModelLoader.load("/tmp/test.myext")
        assert result["path"] == "/tmp/test.myext"
        assert result["device"] == "cpu"

    def test_unknown_external_model_raises(self):
        from domains.models import ModelLoader
        with pytest.raises(ValueError, match="Unknown model_type"):
            ModelLoader._load_external_model("nonexistent_type", {})

    def test_gguf_without_llama_cpp_raises(self):
        from domains.models import ModelLoader
        try:
            import llama_cpp
            pytest.skip("llama-cpp-python installed, can't test ImportError")
        except ImportError:
            with pytest.raises(NotImplementedError, match="llama-cpp-python"):
                ModelLoader._load_gguf("/tmp/model.gguf", device="cpu")


class TestSloughGPTModel:
    @pytest.fixture
    def small_model(self):
        from domains.models import SloughGPTModel
        return SloughGPTModel(
            vocab_size=256, n_embed=32, n_layer=2, n_head=2,
            block_size=16, max_seq_len=64,
        )

    def test_config(self, small_model):
        cfg = small_model.config()
        assert cfg["vocab_size"] == 256
        assert cfg["n_embed"] == 32
        assert cfg["n_layer"] == 2
        assert cfg["n_head"] == 2
        assert cfg["model_type"] == "sloughgpt"

    def test_num_parameters(self, small_model):
        assert small_model.num_parameters() > 0

    def test_get_model_size_mb(self, small_model):
        size = small_model.get_model_size_mb()
        assert size > 0

    def test_state_dict_has_config(self, small_model):
        sd = small_model.state_dict()
        assert "config" in sd
        assert sd["config"]["vocab_size"] == 256

    def test_state_dict_load_state_dict_roundtrip(self, small_model):
        sd = small_model.state_dict()
        from domains.models import SloughGPTModel
        new_model = SloughGPTModel(
            vocab_size=256, n_embed=32, n_layer=2, n_head=2,
            block_size=16, max_seq_len=64,
        )
        new_model.load_state_dict(sd)
        # Verify params match
        for (k1, p1), (k2, p2) in zip(
            small_model.state_dict().items(),
            new_model.state_dict().items(),
        ):
            if k1 == "config":
                continue
            assert k1 == k2
            np.testing.assert_array_equal(p1.data, p2.data)

    def test_to_device(self, small_model):
        result = small_model.to("cuda")
        assert result is small_model
        assert small_model._device == "cuda"

    def test_eval(self, small_model):
        result = small_model.eval()
        assert result is small_model

    def test_train_mode(self, small_model):
        result = small_model.train_mode()
        assert result is small_model

    def test_clear_kv_cache(self, small_model):
        small_model.clear_kv_cache()  # should not raise

    def test_freeze_embeddings(self, small_model):
        small_model.freeze_embeddings()
        assert small_model.layers[0].weight.requires_grad is False

    def test_forward(self, small_model):
        input_ids = np.array([[1, 2, 3, 4]], dtype=np.int32)
        logits, loss = small_model.forward(input_ids)
        assert logits is not None
        # No targets → loss is None
        assert loss is None

    def test_forward_with_targets(self, small_model):
        input_ids = np.array([[1, 2, 3, 4]], dtype=np.int32)
        targets = np.array([[2, 3, 4, 5]], dtype=np.int32)
        logits, loss = small_model.forward(input_ids, targets=targets)
        assert logits is not None
        assert loss is not None

    def test_generate(self, small_model):
        input_ids = np.array([[1, 2, 3]], dtype=np.int32)
        result = small_model.generate(input_ids, max_new_tokens=5)
        assert result.ndim == 2
        assert result.shape[0] == 1
        assert result.shape[1] > 3  # generated at least some tokens


class TestAliases:
    def test_rmsnorm_alias(self):
        from domains.models import RMSNorm
        from domains.training.slonet import SloRMSNorm
        assert RMSNorm is SloRMSNorm

    def test_attention_alias(self):
        from domains.models import SloughGPTAttention
        from domains.training.slonet import SloMultiHeadAttention
        assert SloughGPTAttention is SloMultiHeadAttention

    def test_block_alias(self):
        from domains.models import SloughGPTBlock
        from domains.training.slonet import SloTransformerBlock
        assert SloughGPTBlock is SloTransformerBlock

    def test_swiglu_alias(self):
        from domains.models import SwiGLU
        from domains.training.slonet import SloFeedForward
        assert SwiGLU is SloFeedForward

    def test_rotate_half_callable(self):
        from domains.models import rotate_half
        assert callable(rotate_half)

    def test_apply_rotary_pos_emb_callable(self):
        from domains.models import apply_rotary_pos_emb
        assert callable(apply_rotary_pos_emb)
