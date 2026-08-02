"""Tests for domains/training/gguf_export.py."""

import os
import subprocess
import sys
import types

import numpy as np
import pytest

from domains.training.gguf_export import (
    ARCHITECTURE_MAPPINGS,
    BloomMapping,
    DeepseekMapping,
    FalconMapping,
    GPT2Mapping,
    GPTNeoXMapping,
    GGUFExportConfig,
    GemmaMapping,
    LLaMAMapping,
    MOBILE_RECOMMENDED,
    MistralMapping,
    OPTMapping,
    PhiMapping,
    QUANTIZATION_TYPES,
    QwenMapping,
    SloughGPTMapping,
    TensorMapping,
    YiMapping,
    _as_float16,
    count_layers,
    detect_architecture,
    estimate_memory_requirements,
    export_to_gguf,
    export_to_gguf_fp16,
    export_to_gguf_q4_k_m,
    get_block_mapping,
    get_model_info_gguf,
    get_tensor_mapping,
    list_available_quantizations,
    list_supported_architectures,
    quantize_gguf,
    register_architecture,
)


class _FakeGGUFWriter:
    instances = []

    def __init__(self, path, arch):
        self.path = path
        self.arch = arch
        self.calls = []
        self.tensors = []
        _FakeGGUFWriter.instances.append(self)

    def add_tensor(self, name, data):
        self.tensors.append((name, np.asarray(data)))

    def __getattr__(self, name):
        def _record(*args):
            self.calls.append((name,) + tuple(args))

        return _record


class _FakeGGUFReader:
    def __init__(self, path):
        self.path = path


class _FakeGGUF(types.ModuleType):
    def __init__(self, name="gguf"):
        super().__init__(name)
        self.GGUFWriter = _FakeGGUFWriter
        self.GGUFReader = _FakeGGUFReader


@pytest.fixture
def fake_gguf(monkeypatch):
    _FakeGGUFWriter.instances.clear()
    monkeypatch.setitem(sys.modules, "gguf", _FakeGGUF())
    return _FakeGGUFWriter.instances


class _StubModel:
    def __init__(self, state_dict, config=None):
        self._sd = state_dict
        self._config = config
        self.eval_called = 0

    def state_dict(self):
        return self._sd

    def eval(self):
        self.eval_called += 1
        return self


def _sd(**items):
    sd = {}
    for name, shape in items.items():
        sd[name] = np.random.RandomState(0).randn(*shape).astype(np.float32)
    return sd


_SLOUGHT_SD = _sd(
    **{
        "tok_emb.weight": (32, 16),
        "lm_head.weight": (32, 16),
        "norm.weight": (16,),
        "blocks.0.norm1.weight": (16,),
        "blocks.0.norm2.weight": (16,),
        "blocks.0.attn.q_proj.weight": (16, 16),
        "blocks.0.attn.k_proj.weight": (16, 16),
        "blocks.0.attn.v_proj.weight": (16, 16),
        "blocks.0.attn.o_proj.weight": (16, 16),
        "blocks.0.mlp.w1.weight": (16, 16),
        "blocks.0.mlp.w2.weight": (16, 16),
        "blocks.0.mlp.w3.weight": (16, 16),
    }
)


def _slought_model():
    return _StubModel(
        _SLOUGHT_SD,
        config={"vocab_size": 32, "n_embed": 16, "n_head": 2, "n_kv_head": 2},
    )


def _get_call(writer, name):
    for call in writer.calls:
        if call[0] == name:
            return call[1:]
    return None


def _tensor_names(writer):
    return [name for name, _ in writer.tensors]


class TestGGUFExportConfig:
    def test_defaults(self):
        c = GGUFExportConfig()
        assert c.model_name == "sloughgpt"
        assert c.model_version == "1.0"
        assert c.quantization == "Q4_K_M"
        assert c.use_gpu is False
        assert c.n_ctx == 2048
        assert c.rope_freq_base == 10000.0
        assert c.rope_freq_scale == 1.0
        assert c.architecture is None

    def test_custom(self):
        c = GGUFExportConfig(
            model_name="mine", model_version="2.0", quantization="Q8_0",
            use_gpu=True, n_ctx=4096, rope_freq_base=500000.0,
            rope_freq_scale=0.5, architecture="llama",
        )
        assert c.model_name == "mine"
        assert c.model_version == "2.0"
        assert c.quantization == "Q8_0"
        assert c.use_gpu is True
        assert c.n_ctx == 4096
        assert c.rope_freq_base == 500000.0
        assert c.rope_freq_scale == 0.5
        assert c.architecture == "llama"


class TestTensorMappingBase:
    def test_abstract_methods_raise(self):
        m = TensorMapping("base", "llama")
        assert m.name == "base"
        assert m.gguf_type == "llama"
        with pytest.raises(NotImplementedError):
            m.get_tensor_map()
        with pytest.raises(NotImplementedError):
            m.get_block_prefix()
        with pytest.raises(NotImplementedError):
            m.has_rope()
        with pytest.raises(NotImplementedError):
            m.has_position_embeddings()

    def test_defaults(self):
        m = TensorMapping("base", "llama")
        assert m.get_special_tensors() == {}
        assert m.get_block_mapping(3) == {}
        assert m.get_fused_qkv_keys() == []


MAPPING_CASES = [
    (SloughGPTMapping, "sloughgpt", "llama", "blocks.", True, False, False, ["rope.cos", "rope.sin"]),
    (LLaMAMapping, "llama", "llama", "model.layers.", True, False, False, []),
    (MistralMapping, "mistral", "mistral", "model.layers.", True, False, False, []),
    (GPT2Mapping, "gpt2", "gpt2", "h.", False, True, False, []),
    (OPTMapping, "opt", "llama", "model.decoder.layers.", False, True, False, []),
    (FalconMapping, "falcon", "llama", "transformer.h.", False, True, True, []),
    (GPTNeoXMapping, "gpt_neox", "llama", "layers.", False, True, True, []),
    (BloomMapping, "bloom", "llama", "h.", False, True, True, []),
    (PhiMapping, "phi", "llama", "model.h.", True, False, False, []),
    (GemmaMapping, "gemma", "gemma", "model.layers.", True, False, False, []),
    (QwenMapping, "qwen", "qwen", "transformer.h.", True, False, False, []),
    (DeepseekMapping, "deepseek", "llama", "model.layers.", True, False, False, []),
    (YiMapping, "yi", "llama", "model.layers.", True, False, False, []),
]


class TestMappings:
    @pytest.mark.parametrize(
        "cls,name,gguf_type,prefix,rope,pos_emb,fused,special",
        MAPPING_CASES,
    )
    def test_identity(self, cls, name, gguf_type, prefix, rope, pos_emb, fused, special):
        m = cls()
        assert m.name == name
        assert m.gguf_type == gguf_type
        assert m.get_block_prefix() == prefix
        assert m.has_rope() is rope
        assert m.has_position_embeddings() is pos_emb
        assert bool(m.get_fused_qkv_keys()) is fused
        assert m.get_special_tensors() == {k: k for k in special}

    @pytest.mark.parametrize("cls,name,gguf_type,prefix,rope,pos_emb,fused,special", MAPPING_CASES)
    def test_block_mapping_count(self, cls, name, gguf_type, prefix, rope, pos_emb, fused, special):
        m = cls()
        mapping = m.get_block_mapping(2)
        layer_keys = [k for k in mapping if k.startswith(prefix)]
        assert len(layer_keys) >= 1
        assert len(mapping) == len(layer_keys)

    @pytest.mark.parametrize("cls,name,gguf_type,prefix,rope,pos_emb,fused,special", MAPPING_CASES)
    def test_block_mapping_targets_use_block_index(self, cls, name, gguf_type, prefix, rope, pos_emb, fused, special):
        m = cls()
        mapping = m.get_block_mapping(1)
        for src, dst in mapping.items():
            assert src.startswith(prefix)
            assert ".0." in dst

    def test_tensor_map_top_level_keys(self):
        assert "tok_emb.weight" in SloughGPTMapping().get_tensor_map()
        assert "token_embd.weight" in SloughGPTMapping().get_tensor_map().values()
        assert "model.embed_tokens.weight" in LLaMAMapping().get_tensor_map()
        assert "wte.weight" in GPT2Mapping().get_tensor_map()
        assert "transformer.word_embeddings.weight" in FalconMapping().get_tensor_map()

    @pytest.mark.parametrize("cls,name,gguf_type,prefix,rope,pos_emb,fused,special", MAPPING_CASES)
    def test_get_tensor_map_maps_to_output_norm(self, cls, name, gguf_type, prefix, rope, pos_emb, fused, special):
        values = cls().get_tensor_map().values()
        assert "output_norm.weight" in values
        assert "token_embd.weight" in values

    def test_fused_qkv_architectures(self):
        assert FalconMapping().get_fused_qkv_keys() == [("query_key_value.weight", "falcon")]
        assert GPTNeoXMapping().get_fused_qkv_keys() == [("query_key_value.weight", "gpt_neox")]
        assert BloomMapping().get_fused_qkv_keys() == [("query_key_value.weight", "bloom")]
        assert SloughGPTMapping().get_fused_qkv_keys() == []


class TestAsFloat16:
    def test_numpy_array(self):
        arr = np.random.RandomState(0).randn(4, 4).astype(np.float32)
        out = _as_float16(arr)
        assert out.dtype == np.float16
        assert out.shape == (4, 4)

    def test_slonet_tensor(self):
        class _T:
            data = np.random.RandomState(0).randn(3).astype(np.float32)

        out = _as_float16(_T())
        assert out.dtype == np.float16

    def test_torch_like(self):
        class _Torch:
            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return np.random.RandomState(0).randn(2).astype(np.float32)

        out = _as_float16(_Torch())
        assert out.dtype == np.float16

    def test_plain_sequence(self):
        out = _as_float16([1.0, 2.0, 3.0])
        assert out.dtype == np.float16
        assert list(out) == [1.0, 2.0, 3.0]

    def test_object_dtype_returns_none(self):
        arr = np.array([object()])
        assert _as_float16(arr) is None

    def test_list_of_objects_returns_none(self):
        assert _as_float16([object()]) is None

    def test_unconvertible_returns_none(self):
        class _Bad:
            def __array__(self, dtype=None, copy=None):
                raise ValueError("cannot convert")

        assert _as_float16(_Bad()) is None


DETECT_CASES = [
    (_SLOUGHT_SD, SloughGPTMapping),
    (_sd(**{"model.embed_tokens.weight": (1,), "model.norm.weight": (1,),
            "model.layers.0.self_attn.q_proj.weight": (1,)}), LLaMAMapping),
    (_sd(**{"wte.weight": (1,), "ln_f.weight": (1,), "h.0.ln_1.weight": (1,)}), GPT2Mapping),
    (_sd(**{"model.embed_tokens.weight": (1,), "model.decoder.layers.0.fc1.weight": (1,),
            "model.decoder.final_layer_norm.weight": (1,)}), OPTMapping),
    (_sd(**{"transformer.word_embeddings.weight": (1,), "transformer.h.0.self_attention.dense.weight": (1,)}),
     FalconMapping),
    (_sd(**{"embed_in.weight": (1,), "final_layer_norm.weight": (1,),
            "layers.0.attention.dense.weight": (1,)}), GPTNeoXMapping),
    (_sd(**{"word_embeddings.weight": (1,), "ln_f.weight": (1,),
            "h.0.self_attention.dense.weight": (1,)}), BloomMapping),
]


class TestDetectArchitecture:
    @pytest.mark.parametrize("state_dict,expected", DETECT_CASES)
    def test_detect(self, state_dict, expected):
        assert isinstance(detect_architecture(state_dict), expected)

    def test_no_match_defaults_to_sloughgpt(self):
        m = detect_architecture({"totally.unknown.weight": 1})
        assert isinstance(m, SloughGPTMapping)

    def test_mistral_shape_detects_as_llama(self):
        sd = _sd(**{"model.embed_tokens.weight": (1,), "model.layers.0.input_layernorm.weight": (1,),
                    "model.norm.weight": (1,)})
        assert isinstance(detect_architecture(sd), LLaMAMapping)


class TestRegisterArchitecture:
    def test_register_custom(self, monkeypatch):
        class Custom(TensorMapping):
            def __init__(self):
                super().__init__("custom", "llama")

            def get_tensor_map(self):
                return {"a": "b"}

            def get_block_prefix(self):
                return "blk."

            def has_rope(self):
                return True

            def has_position_embeddings(self):
                return False

        monkeypatch.setitem(ARCHITECTURE_MAPPINGS, "custom", Custom())
        assert "custom" in ARCHITECTURE_MAPPINGS
        assert isinstance(ARCHITECTURE_MAPPINGS["custom"], Custom)

    def test_register_function(self, monkeypatch):
        class Custom(TensorMapping):
            def __init__(self):
                super().__init__("custom2", "llama")

            def get_tensor_map(self):
                return {"a": "b"}

            def get_block_prefix(self):
                return "blk."

            def has_rope(self):
                return True

            def has_position_embeddings(self):
                return False

        register_architecture("custom2", Custom())
        try:
            assert "custom2" in ARCHITECTURE_MAPPINGS
            assert isinstance(ARCHITECTURE_MAPPINGS["custom2"], Custom)
        finally:
            del ARCHITECTURE_MAPPINGS["custom2"]


class TestGetTensorMapping:
    def test_returns_combined_map(self):
        mapping = get_tensor_mapping(_slought_model())
        assert mapping["tok_emb.weight"] == "token_embd.weight"
        assert mapping["lm_head.weight"] == "output.weight"
        assert mapping["norm.weight"] == "output_norm.weight"
        assert mapping["blocks.0.attn.q_proj.weight"] == "blk.0.attn_q.weight"
        assert mapping["blocks.0.mlp.w3.weight"] == "blk.0.ffn_up.weight"
        assert "blocks.1.attn_q.weight" not in mapping.values()


class TestCountLayers:
    def test_non_layers_prefix(self):
        sd = {"h.0.ln_1.weight": 1, "h.1.ln_1.weight": 1}
        assert count_layers(sd, "h.") == 2

    def test_layers_prefix(self):
        sd = {
            "model.layers.0.input_layernorm.weight": 1,
            "model.layers.2.post_attention_layernorm.weight": 1,
        }
        assert count_layers(sd, "model.layers.") == 3

    def test_no_matching_keys(self):
        assert count_layers({"x.weight": 1}, "h.") == 0


class TestGetBlockMapping:
    def test_with_model(self):
        mapping = get_block_mapping(_slought_model())
        assert mapping["blocks.0.attn.q_proj.weight"] == "blk.0.attn_q.weight"

    def test_without_model_uses_default(self):
        mapping = get_block_mapping()
        assert mapping["blocks.0.attn.q_proj.weight"] == "blk.0.attn_q.weight"
        assert "blocks.0.attn.o_proj.weight" in mapping


class TestExportToGGUF:
    def test_import_error_without_gguf(self):
        sys.modules.pop("gguf", None)
        with pytest.raises(ImportError, match="gguf not installed"):
            export_to_gguf(_slought_model(), "/tmp/none.gguf")

    def test_export_sloughgpt(self, fake_gguf, tmp_path):
        out = tmp_path / "model.gguf"
        model = _slought_model()
        result = export_to_gguf(model, str(out), config=GGUFExportConfig())
        writer = fake_gguf[-1]

        assert result == str(out)
        assert model.eval_called == 1
        assert _get_call(writer, "add_name") == ("sloughgpt",)
        assert _get_call(writer, "add_vocab_size") == (32,)
        assert _get_call(writer, "add_context_length") == (2048,)
        assert _get_call(writer, "add_embedding_length") == (16,)
        assert _get_call(writer, "add_block_count") == (1,)
        assert _get_call(writer, "add_head_count") == (2,)
        assert _get_call(writer, "add_head_count_kv") == (2,)
        assert _get_call(writer, "add_feed_forward_length") == (64,)
        assert _get_call(writer, "add_rope_freq_base") == (10000.0,)
        assert _get_call(writer, "add_tokenizer_model") == ("llama",)
        assert _get_call(writer, "add_add_bos_token") == (False,)
        assert _get_call(writer, "add_add_eos_token") == (False,)
        assert _get_call(writer, "add_add_space_prefix") == (False,)
        assert _get_call(writer, "flush") == ()

        names = _tensor_names(writer)
        assert "token_embd.weight" in names
        assert "output.weight" in names
        assert "output_norm.weight" in names
        assert "blk.0.attn_q.weight" in names
        assert "blk.0.ffn_gate.weight" in names

    def test_export_falls_back_to_model_attrs(self, fake_gguf, tmp_path):
        model = _StubModel(_SLOUGHT_SD, config=None)
        out = tmp_path / "m.gguf"
        export_to_gguf(model, str(out))
        writer = fake_gguf[-1]
        assert _get_call(writer, "add_vocab_size") == (256,)
        assert _get_call(writer, "add_head_count") == (8,)

    def test_unknown_architecture_raises(self, fake_gguf, tmp_path):
        out = tmp_path / "m.gguf"
        with pytest.raises(ValueError, match="Unknown architecture"):
            export_to_gguf(_slought_model(), str(out), config=GGUFExportConfig(architecture="nope"))

    def test_explicit_architecture_uses_it(self, fake_gguf, tmp_path):
        sd = _sd(**{
            "transformer.word_embeddings.weight": (32, 16),
            "lm_head.weight": (32, 16),
            "transformer.ln_f.weight": (16,),
            "transformer.h.0.self_attention.query_key_value.weight": (6, 16),
            "transformer.h.0.self_attention.dense.weight": (16, 16),
        })
        model = _StubModel(sd, config={"vocab_size": 32, "n_embed": 16, "n_head": 2, "n_kv_head": 2})
        out = tmp_path / "falcon.gguf"
        export_to_gguf(model, str(out), config=GGUFExportConfig(architecture="falcon"))
        writer = fake_gguf[-1]
        names = _tensor_names(writer)
        assert "blk.0.attn_q.weight" in names
        assert "blk.0.attn_k.weight" in names
        assert "blk.0.attn_v.weight" in names
        shapes = {name: data.shape for name, data in writer.tensors}
        assert shapes["blk.0.attn_q.weight"] == (2, 16)
        assert shapes["blk.0.attn_k.weight"] == (2, 16)
        assert shapes["blk.0.attn_v.weight"] == (2, 16)

    def test_tokenizer_vocab(self, fake_gguf, tmp_path):
        class _Tok:
            def get_vocab(self):
                return {"a": 0, "b": 1}

            def decode(self, ids):
                return chr(ids[0] + 97)

        out = tmp_path / "m.gguf"
        export_to_gguf(_slought_model(), str(out), tokenizer=_Tok())
        writer = fake_gguf[-1]
        token_list = _get_call(writer, "add_token_list")[0]
        assert token_list == ["a", "b"]

    def test_tokenizer_without_get_vocab_uses_chr(self, fake_gguf, tmp_path):
        out = tmp_path / "m.gguf"
        export_to_gguf(_slought_model(), str(out), tokenizer=object())
        writer = fake_gguf[-1]
        token_list = _get_call(writer, "add_token_list")[0]
        assert len(token_list) == 32
        assert token_list[0] == chr(0)

    def test_tokenizer_without_decode(self, fake_gguf, tmp_path):
        class _Tok:
            def get_vocab(self):
                return {"a": 0, "b": 1}

        out = tmp_path / "m.gguf"
        export_to_gguf(_slought_model(), str(out), tokenizer=_Tok())
        writer = fake_gguf[-1]
        token_list = _get_call(writer, "add_token_list")[0]
        assert token_list == ["\x00", "\x01"]

    def test_no_tokenizer_uses_chr_encoding(self, fake_gguf, tmp_path):
        model = _StubModel(
            _SLOUGHT_SD,
            config={"vocab_size": 301, "n_embed": 16, "n_head": 2, "n_kv_head": 2},
        )
        out = tmp_path / "m.gguf"
        export_to_gguf(model, str(out))
        writer = fake_gguf[-1]
        token_list = _get_call(writer, "add_token_list")[0]
        assert len(token_list) == 301
        assert token_list[0] == chr(0)
        assert token_list[255] == chr(255)
        assert token_list[300] == "<0x12C>"

    def test_unconvertible_tensor_skipped(self, fake_gguf, tmp_path):
        sd = dict(_SLOUGHT_SD)
        sd["junk.weight"] = np.array([object()])
        model = _StubModel(sd, config={"vocab_size": 32, "n_embed": 16, "n_head": 2, "n_kv_head": 2})
        out = tmp_path / "m.gguf"
        export_to_gguf(model, str(out))
        writer = fake_gguf[-1]
        assert "junk.weight" not in _tensor_names(writer)

    def test_parent_dir_created(self, fake_gguf, tmp_path):
        out = tmp_path / "deep" / "nested" / "m.gguf"
        result = export_to_gguf(_slought_model(), str(out))
        assert os.path.isdir(str(tmp_path / "deep" / "nested"))
        assert result == str(out)


class TestExportWrappers:
    def test_fp16(self, fake_gguf, tmp_path):
        out = tmp_path / "m.gguf"
        result = export_to_gguf_fp16(_slought_model(), str(out))
        assert result == str(out)
        assert _get_call(fake_gguf[-1], "add_description") == ("sloughgpt exported with F16",)

    def test_q4_k_m(self, fake_gguf, tmp_path):
        out = tmp_path / "m.gguf"
        result = export_to_gguf_q4_k_m(_slought_model(), str(out))
        assert result == str(out)
        assert _get_call(fake_gguf[-1], "add_description") == ("sloughgpt exported with Q4_K_M",)


class TestQuantizeGGUF:
    def test_binary_not_found_returns_input(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda _name: None)
        inp = str(tmp_path / "in.gguf")
        assert quantize_gguf(inp, str(tmp_path / "out.gguf")) == inp

    def test_success_returns_output(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/llama-quantize")

        class _Result:
            returncode = 0
            stderr = ""

        monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Result())
        inp = str(tmp_path / "in.gguf")
        out = str(tmp_path / "out.gguf")
        assert quantize_gguf(inp, out) == out

    def test_failure_returns_input(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/llama-quantize")

        class _Result:
            returncode = 1
            stderr = "boom"

        monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Result())
        inp = str(tmp_path / "in.gguf")
        assert quantize_gguf(inp, str(tmp_path / "out.gguf")) == inp

    def test_exception_returns_input(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/llama-quantize")
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
        inp = str(tmp_path / "in.gguf")
        assert quantize_gguf(inp, str(tmp_path / "out.gguf")) == inp


class TestGetModelInfoGGUF:
    def test_info(self, fake_gguf, tmp_path):
        path = tmp_path / "m.gguf"
        path.write_bytes(b"x" * (3 * 1024 * 1024))
        info = get_model_info_gguf(str(path))
        assert info["path"] == str(path)
        assert info["file_size_mb"] == 3.0

    def test_error_returns_empty(self, tmp_path):
        assert get_model_info_gguf(str(tmp_path / "missing.gguf")) == {}


class TestEstimateMemory:
    def test_quantization_diffs(self):
        f32 = estimate_memory_requirements(1000, 4, 128, 512, "F32")
        q4 = estimate_memory_requirements(1000, 4, 128, 512, "Q4_K_M")
        assert f32["model_mb"] > q4["model_mb"]
        assert f32["total_mb"] > q4["total_mb"]
        assert f32["kv_cache_mb"] == q4["kv_cache_mb"]
        for key in ("model_mb", "kv_cache_mb", "total_mb"):
            assert f32[key] > 0

    def test_unknown_quantization_uses_default(self):
        a = estimate_memory_requirements(1000, 4, 128, 512, "NOPE")
        b = estimate_memory_requirements(1000, 4, 128, 512, "Q4_K_M")
        assert a["model_mb"] == b["model_mb"]

    def test_smaller_quant_smaller_model(self):
        q8 = estimate_memory_requirements(1000, 4, 128, 512, "Q8_0")
        q2 = estimate_memory_requirements(1000, 4, 128, 512, "Q2_K")
        assert q2["model_mb"] < q8["model_mb"]


class TestLists:
    def test_available_quantizations(self):
        items = list_available_quantizations()
        assert len(items) == len(QUANTIZATION_TYPES)
        for name, desc, rec in items:
            assert QUANTIZATION_TYPES[name] == desc
            assert rec == (name in MOBILE_RECOMMENDED)

    def test_supported_architectures(self):
        archs = list_supported_architectures()
        for expected in ("sloughgpt", "llama", "gpt2", "falcon", "qwen", "bloom"):
            assert expected in archs
