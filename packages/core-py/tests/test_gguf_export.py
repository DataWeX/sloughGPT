"""Tests for gguf_export.py — pure logic, no mocks."""

import numpy as np
import pytest

from domains.training.gguf_export import (
    ARCHITECTURE_MAPPINGS,
    QUANTIZATION_TYPES,
    MOBILE_RECOMMENDED,
    GGUFExportConfig,
    TensorMapping,
    SloughGPTMapping,
    LLaMAMapping,
    MistralMapping,
    GPT2Mapping,
    OPTMapping,
    FalconMapping,
    GPTNeoXMapping,
    BloomMapping,
    PhiMapping,
    GemmaMapping,
    QwenMapping,
    DeepseekMapping,
    YiMapping,
    _as_float16,
    detect_architecture,
    register_architecture,
    count_layers,
    get_block_mapping,
    estimate_memory_requirements,
    list_available_quantizations,
    list_supported_architectures,
)


# ── GGUFExportConfig ──────────────────────────────────────────────


class TestGGUFExportConfig:
    def test_defaults(self):
        cfg = GGUFExportConfig()
        assert cfg.model_name == "sloughgpt"
        assert cfg.model_version == "1.0"
        assert cfg.quantization == "Q4_K_M"
        assert cfg.use_gpu is False
        assert cfg.n_ctx == 2048
        assert cfg.rope_freq_base == 10000.0
        assert cfg.rope_freq_scale == 1.0
        assert cfg.architecture is None

    def test_custom_values(self):
        cfg = GGUFExportConfig(
            model_name="mymodel",
            model_version="2.0",
            quantization="F16",
            use_gpu=True,
            n_ctx=4096,
            rope_freq_base=50000.0,
            rope_freq_scale=0.5,
            architecture="llama",
        )
        assert cfg.model_name == "mymodel"
        assert cfg.model_version == "2.0"
        assert cfg.quantization == "F16"
        assert cfg.use_gpu is True
        assert cfg.n_ctx == 4096
        assert cfg.rope_freq_base == 50000.0
        assert cfg.rope_freq_scale == 0.5
        assert cfg.architecture == "llama"


# ── _as_float16 ───────────────────────────────────────────────────


class TestAsFloat16:
    def test_numpy_float32(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = _as_float16(arr)
        assert result is not None
        assert result.dtype == np.float16
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_numpy_int(self):
        arr = np.array([1, 2, 3], dtype=np.int32)
        result = _as_float16(arr)
        assert result is not None
        assert result.dtype == np.float16

    def test_numpy_object_returns_none(self):
        arr = np.array(["a", "b"], dtype=object)
        result = _as_float16(arr)
        assert result is None

    def test_slonet_tensor_like(self):
        class FakeTensor:
            def __init__(self):
                self.data = np.array([1.0, 2.0], dtype=np.float32)

        result = _as_float16(FakeTensor())
        assert result is not None
        assert result.dtype == np.float16
        np.testing.assert_array_equal(result, [1.0, 2.0])

    def test_non_convertible_returns_none(self):
        class CantConvert:
            pass
        result = _as_float16(CantConvert())
        assert result is None

    def test_string_raises(self):
        with pytest.raises(ValueError):
            _as_float16("not a tensor")

    def test_list_input(self):
        result = _as_float16([1.0, 2.0])
        assert result is not None
        assert result.dtype == np.float16

    def test_torch_tensor_like(self):
        class FakeTorch:
            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return np.array([5.0, 6.0], dtype=np.float32)

        result = _as_float16(FakeTorch())
        assert result is not None
        assert result.dtype == np.float16


# ── count_layers ───────────────────────────────────────────────────


class TestCountLayers:
    def test_count_sloughgpt(self):
        sd = {f"blocks.{i}.norm1.weight": np.zeros(1) for i in range(6)}
        assert count_layers(sd, "blocks.") == 6

    def test_count_llama(self):
        sd = {f"model.layers.{i}.self_attn.q_proj.weight": np.zeros(1) for i in range(12)}
        assert count_layers(sd, "model.layers.") == 12

    def test_no_blocks(self):
        sd = {"tok_emb.weight": np.zeros(1)}
        assert count_layers(sd, "blocks.") == 0

    def test_non_contiguous(self):
        sd = {
            "blocks.0.norm1.weight": np.zeros(1),
            "blocks.5.norm1.weight": np.zeros(1),
        }
        assert count_layers(sd, "blocks.") == 6


# ── detect_architecture ───────────────────────────────────────────


class TestDetectArchitecture:
    def test_sloughgpt(self):
        sd = {
            "tok_emb.weight": np.zeros(1),
            "norm.weight": np.zeros(1),
            "mlp.w1.weight": np.zeros(1),
        }
        m = detect_architecture(sd)
        assert isinstance(m, SloughGPTMapping)

    def test_llama(self):
        sd = {
            "model.embed_tokens.weight": np.zeros(1),
            "model.norm.weight": np.zeros(1),
            "model.layers.0.self_attn.q_proj.weight": np.zeros(1),
        }
        m = detect_architecture(sd)
        assert isinstance(m, LLaMAMapping)

    def test_mistral(self):
        sd = {
            "model.embed_tokens.weight": np.zeros(1),
            "input_layernorm.weight": np.zeros(1),
        }
        m = detect_architecture(sd)
        assert isinstance(m, MistralMapping)

    def test_gpt2(self):
        sd = {
            "wte.weight": np.zeros(1),
            "ln_f.weight": np.zeros(1),
            "h.0.ln_1.weight": np.zeros(1),
        }
        m = detect_architecture(sd)
        assert isinstance(m, GPT2Mapping)

    def test_falcon(self):
        sd = {
            "transformer.word_embeddings.weight": np.zeros(1),
            "transformer.h.0.ln_attn.weight": np.zeros(1),
        }
        m = detect_architecture(sd)
        assert isinstance(m, FalconMapping)

    def test_unknown_falls_back_to_sloughgpt(self):
        sd = {"random_tensor.weight": np.zeros(1)}
        m = detect_architecture(sd)
        assert isinstance(m, SloughGPTMapping)


# ── register_architecture ─────────────────────────────────────────


class TestRegisterArchitecture:
    def test_register_and_retrieve(self):
        class DummyMapping(TensorMapping):
            def __init__(self):
                super().__init__("dummy", "llama")

            def get_tensor_map(self):
                return {}

            def get_block_prefix(self):
                return "layers."

            def has_rope(self):
                return False

            def has_position_embeddings(self):
                return True

        register_architecture("dummy", DummyMapping())
        assert "dummy" in ARCHITECTURE_MAPPINGS
        assert isinstance(ARCHITECTURE_MAPPINGS["dummy"], DummyMapping)
        # cleanup
        del ARCHITECTURE_MAPPINGS["dummy"]


# ── Mapping classes ────────────────────────────────────────────────


class TestMappingBase:
    def test_base_raises(self):
        m = TensorMapping("base", "llama")
        with pytest.raises(NotImplementedError):
            m.get_tensor_map()
        with pytest.raises(NotImplementedError):
            m.get_block_prefix()
        with pytest.raises(NotImplementedError):
            m.has_rope()
        with pytest.raises(NotImplementedError):
            m.has_position_embeddings()

    def test_base_special_tensors_default(self):
        m = TensorMapping("base", "llama")
        assert m.get_special_tensors() == {}

    def test_base_fused_qkv_default(self):
        m = TensorMapping("base", "llama")
        assert m.get_fused_qkv_keys() == []


class TestSloughGPTMapping:
    def test_tensor_map(self):
        m = SloughGPTMapping()
        tm = m.get_tensor_map()
        assert tm["tok_emb.weight"] == "token_embd.weight"
        assert tm["lm_head.weight"] == "output.weight"
        assert tm["norm.weight"] == "output_norm.weight"

    def test_block_prefix(self):
        assert SloughGPTMapping().get_block_prefix() == "blocks."

    def test_has_rope(self):
        assert SloughGPTMapping().has_rope() is True

    def test_no_position_embeddings(self):
        assert SloughGPTMapping().has_position_embeddings() is False

    def test_special_tensors(self):
        st = SloughGPTMapping().get_special_tensors()
        assert "rope.cos" in st
        assert "rope.sin" in st

    def test_block_mapping(self):
        bm = SloughGPTMapping().get_block_mapping(2)
        assert "blocks.0.norm1.weight" in bm
        assert bm["blocks.0.norm1.weight"] == "blk.0.attn_norm.weight"
        assert "blocks.1.mlp.w1.weight" in bm
        assert bm["blocks.1.mlp.w1.weight"] == "blk.1.ffn_gate.weight"
        # Should not contain block 2
        assert "blocks.2.norm1.weight" not in bm


class TestLLaMAMapping:
    def test_tensor_map(self):
        m = LLaMAMapping()
        tm = m.get_tensor_map()
        assert tm["model.embed_tokens.weight"] == "token_embd.weight"

    def test_block_prefix(self):
        assert LLaMAMapping().get_block_prefix() == "model.layers."

    def test_block_mapping(self):
        bm = LLaMAMapping().get_block_mapping(1)
        assert "model.layers.0.input_layernorm.weight" in bm
        assert "model.layers.0.self_attn.q_proj.weight" in bm
        assert bm["model.layers.0.self_attn.q_proj.weight"] == "blk.0.attn_q.weight"


class TestMistralMapping:
    def test_tensor_map(self):
        tm = MistralMapping().get_tensor_map()
        assert "model.embed_tokens.weight" in tm

    def test_block_prefix(self):
        assert MistralMapping().get_block_prefix() == "model.layers."


class TestGPT2Mapping:
    def test_tensor_map(self):
        tm = GPT2Mapping().get_tensor_map()
        assert tm["wte.weight"] == "token_embd.weight"
        assert tm["ln_f.weight"] == "output_norm.weight"

    def test_block_prefix(self):
        assert GPT2Mapping().get_block_prefix() == "h."

    def test_no_rope(self):
        assert GPT2Mapping().has_rope() is False

    def test_has_position_embeddings(self):
        assert GPT2Mapping().has_position_embeddings() is True

    def test_block_mapping(self):
        bm = GPT2Mapping().get_block_mapping(1)
        assert "h.0.ln_1.weight" in bm
        assert "h.0.attn.c_attn.weight" in bm


class TestOPTMapping:
    def test_tensor_map(self):
        tm = OPTMapping().get_tensor_map()
        assert tm["model.decoder.final_layer_norm.weight"] == "output_norm.weight"

    def test_block_prefix(self):
        assert OPTMapping().get_block_prefix() == "model.decoder.layers."


class TestFalconMapping:
    def test_tensor_map(self):
        tm = FalconMapping().get_tensor_map()
        assert "transformer.word_embeddings.weight" in tm

    def test_fused_qkv(self):
        fk = FalconMapping().get_fused_qkv_keys()
        assert len(fk) == 1
        assert fk[0][0] == "query_key_value.weight"

    def test_block_mapping(self):
        bm = FalconMapping().get_block_mapping(1)
        assert "transformer.h.0.self_attention.query_key_value.weight" in bm
        assert "transformer.h.0.self_attention.dense.weight" in bm


class TestGPTNeoXMapping:
    def test_tensor_map(self):
        tm = GPTNeoXMapping().get_tensor_map()
        assert "embed_in.weight" in tm
        assert "embed_out.weight" in tm

    def test_fused_qkv(self):
        fk = GPTNeoXMapping().get_fused_qkv_keys()
        assert len(fk) == 1


class TestBloomMapping:
    def test_tensor_map(self):
        tm = BloomMapping().get_tensor_map()
        assert "word_embeddings.weight" in tm

    def test_fused_qkv(self):
        fk = BloomMapping().get_fused_qkv_keys()
        assert len(fk) == 1


class TestPhiMapping:
    def test_tensor_map(self):
        tm = PhiMapping().get_tensor_map()
        assert "model.embed_tokens.weight" in tm
        assert "model.final_layernorm.weight" in tm

    def test_block_prefix(self):
        assert PhiMapping().get_block_prefix() == "model.h."

    def test_has_rope(self):
        assert PhiMapping().has_rope() is True


class TestGemmaMapping:
    def test_tensor_map(self):
        tm = GemmaMapping().get_tensor_map()
        assert "model.embed_tokens.weight" in tm

    def test_gguf_type(self):
        assert GemmaMapping().gguf_type == "gemma"


class TestQwenMapping:
    def test_tensor_map(self):
        tm = QwenMapping().get_tensor_map()
        assert "transformer.wte.weight" in tm

    def test_block_prefix(self):
        assert QwenMapping().get_block_prefix() == "transformer.h."


class TestDeepseekMapping:
    def test_tensor_map(self):
        tm = DeepseekMapping().get_tensor_map()
        assert "model.embed_tokens.weight" in tm


class TestYiMapping:
    def test_tensor_map(self):
        tm = YiMapping().get_tensor_map()
        assert "model.embed_tokens.weight" in tm


# ── get_block_mapping ──────────────────────────────────────────────


class TestGetBlockMapping:
    def test_default_sloughgpt(self):
        bm = get_block_mapping(n_layers=3)
        assert "blocks.0.norm1.weight" in bm
        assert "blocks.2.mlp.w3.weight" in bm

    def test_with_model_none(self):
        bm = get_block_mapping(model=None, n_layers=2)
        assert len(bm) > 0


# ── estimate_memory_requirements ───────────────────────────────────


class TestEstimateMemory:
    def test_basic_estimate(self):
        result = estimate_memory_requirements(
            vocab_size=256, n_layer=6, n_embed=128, n_ctx=512, quantization="Q4_K_M"
        )
        assert "model_mb" in result
        assert "kv_cache_mb" in result
        assert "total_mb" in result
        assert result["total_mb"] > 0
        assert result["total_mb"] == result["model_mb"] + result["kv_cache_mb"]

    def test_f32_larger_than_q4(self):
        f32 = estimate_memory_requirements(256, 6, 128, 512, "F32")
        q4 = estimate_memory_requirements(256, 6, 128, 512, "Q4_K_M")
        assert f32["model_mb"] > q4["model_mb"]

    def test_unknown_quantization_uses_default(self):
        result = estimate_memory_requirements(256, 6, 128, 512, "UNKNOWN")
        assert result["model_mb"] > 0

    def test_kv_cache_scales_with_n_ctx(self):
        r1 = estimate_memory_requirements(256, 6, 128, 256, "F16")
        r2 = estimate_memory_requirements(256, 6, 128, 512, "F16")
        assert r2["kv_cache_mb"] > r1["kv_cache_mb"]


# ── list_available_quantizations ───────────────────────────────────


class TestListQuantizations:
    def test_returns_all(self):
        quants = list_available_quantizations()
        assert len(quants) == len(QUANTIZATION_TYPES)

    def test_tuple_format(self):
        for name, desc, mobile in list_available_quantizations():
            assert isinstance(name, str)
            assert isinstance(desc, str)
            assert isinstance(mobile, bool)

    def test_mobile_recommended_flag(self):
        for name, desc, mobile in list_available_quantizations():
            if name in MOBILE_RECOMMENDED:
                assert mobile is True


# ── list_supported_architectures ───────────────────────────────────


class TestListArchitectures:
    def test_returns_all(self):
        archs = list_supported_architectures()
        assert "sloughgpt" in archs
        assert "llama" in archs
        assert "gpt2" in archs
        assert "falcon" in archs
        assert "phi" in archs
        assert "gemma" in archs
        assert "qwen" in archs
        assert "deepseek" in archs
        assert "yi" in archs
