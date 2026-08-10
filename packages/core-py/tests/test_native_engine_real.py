"""Wave N/O: real-component coverage for native/* C engine + CTransformProvider.

Every test uses real programmatic inputs only:
- a hand-built real ``.slnc`` file (per the spec in ``slnc/spec.py``)
- real tiny transformer weights mapped through ``map_slnc_to_native``
- the compiled ``libtransformer_forward.so`` via ``bindings.load_lib()``
- the real shared tokenizer via ``domains.inference.tokenizer``

No mocks, no stubs, no third-party installs.
"""

import asyncio
import ctypes
import json
import struct
import zlib

import numpy as np
import pytest

from domains.infrastructure.slnc.spec import compute_header_size
from domains.inference.native import bindings as B
from domains.inference.native.engine import (
    NativeEngine,
    NativeTransformerProvider,
    _detect_model_type,
    _format_chat_gpt2,
    _format_chat_llama,
    _format_chat_qwen,
    format_chat,
    get_engine,
    sample_token,
)
from domains.inference.native.weight_mapper import map_slnc_to_native
from domains.inference.ct_provider import CTransformProvider

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

L = 2
D = 16
NH = 4
NKV = 2
HD = 4
FF = 32
V = 512
MAX_POS = 64


def _config(**overrides):
    cfg = {
        "architectures": ["Qwen2ForCausalLM"],
        "num_hidden_layers": L,
        "hidden_size": D,
        "num_attention_heads": NH,
        "num_key_value_heads": NKV,
        "intermediate_size": FF,
        "vocab_size": V,
        "max_position_embeddings": MAX_POS,
        "rope_theta": 10000.0,
        "eos_token_id": 2,
    }
    cfg.update(overrides)
    return cfg


def _weights(seed: int = 7):
    rng = np.random.RandomState(seed)
    w = {}

    def r(*shape):
        return rng.randn(*shape).astype(np.float32)

    w["model.embed_tokens.weight"] = r(V, D)
    for i in range(L):
        p = f"model.layers.{i}"
        w[f"{p}.input_layernorm.weight"] = r(D)
        w[f"{p}.self_attn.q_proj.weight"] = r(NH * HD, D)
        w[f"{p}.self_attn.q_proj.bias"] = r(NH * HD)
        w[f"{p}.self_attn.k_proj.weight"] = r(NKV * HD, D)
        w[f"{p}.self_attn.k_proj.bias"] = r(NKV * HD)
        w[f"{p}.self_attn.v_proj.weight"] = r(NKV * HD, D)
        w[f"{p}.self_attn.v_proj.bias"] = r(NKV * HD)
        w[f"{p}.self_attn.o_proj.weight"] = r(NH * HD, D)
        w[f"{p}.self_attn.o_proj.bias"] = r(D)
        w[f"{p}.post_attention_layernorm.weight"] = r(D)
        w[f"{p}.mlp.gate_proj.weight"] = r(D, FF)
        w[f"{p}.mlp.up_proj.weight"] = r(D, FF)
        w[f"{p}.mlp.down_proj.weight"] = r(FF, D)
    w["model.norm.weight"] = r(D)
    w["model.lm_head.weight"] = r(V, D)
    return w


def _build_slnc(path, weights, config):
    json_bytes = json.dumps(config, sort_keys=True).encode()
    header_size = compute_header_size(json_bytes)
    prepared = [(name, np.ascontiguousarray(arr, dtype=np.float32))
                for name, arr in weights.items()]
    table_size = sum(4 + len(name.encode()) + 8 + 4 + 4 + arr.ndim * 4 + 4 + 4
                     for name, arr in prepared)
    data_offset = header_size + table_size
    table = b""
    entries = []
    cur = data_offset
    for name, arr in prepared:
        ndim = arr.ndim
        nb = name.encode()
        table += struct.pack("<I", len(nb)) + nb
        table += struct.pack("<Q", cur)
        table += struct.pack("<I", arr.nbytes)
        table += struct.pack("<I", ndim)
        table += struct.pack(f"<{ndim}I", *arr.shape)
        table += struct.pack("<I", 0)
        table += struct.pack("<I", zlib.crc32(arr.tobytes()) & 0xFFFFFFFF)
        cur += arr.nbytes
        entries.append(arr)
    meta = struct.pack(
        "<10I",
        L, D, NH, FF, V, MAX_POS,
        L, MAX_POS, len(weights), data_offset,
    ) + b"\x00" * 24
    with open(path, "wb") as f:
        f.write(b"SLNC")
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<I", 0))
        f.write(meta)
        f.write(struct.pack("<I", len(json_bytes)))
        f.write(json_bytes)
        pad = header_size - (4 + 4 + 4 + 64 + 4 + len(json_bytes))
        f.write(b"\x00" * pad)
        f.write(table)
        for arr in entries:
            f.write(arr.tobytes())


def _loaded_engine(seed: int = 7, **cfg_overrides):
    engine = NativeEngine()
    engine.load_from_slnc(_weights(seed), _config(**cfg_overrides), seq_capacity=16)
    return engine


class _FakeTokenizer:
    """Deterministic stand-in for a real tokenizer in wiring tests."""

    def __init__(self, vocab_size=V, stop_ids=(2,), template=True):
        self.vocab_size = vocab_size
        self._stop = tuple(stop_ids)
        self._template = template

    def encode(self, text: str) -> list:
        return [ord(c) for c in text[:8]]

    def decode(self, ids: list) -> str:
        return "".join(chr(i) if 32 <= i < 127 else "?" for i in ids)

    def apply_chat_template(self, messages: list) -> str:
        return "CHAT:" + "|".join(f"{m['role']}={m['content']}" for m in messages) + ":ASST"

    def chat_stop_ids(self):
        return self._stop


class TestBindings:
    def test_load_lib_returns_c_library(self):
        lib = B.load_lib()
        assert hasattr(lib, "transformer_forward_step")
        assert hasattr(lib, "transformer_load_weights")
        assert hasattr(lib, "transformer_kv_cache_init")
        assert lib.transformer_forward_step.restype is not None

    def test_load_lib_is_cached(self):
        assert B.load_lib() is B.load_lib()

    def test_lib_structs_have_expected_fields(self):
        lib = B.load_lib()
        assert [f[0] for f in lib._Config._fields_] == [
            "n_layers", "hidden_dim", "n_heads", "n_kv_heads", "head_dim",
            "ff_dim", "vocab_size", "block_size", "rope_base", "rope_theta",
        ]
        assert "data" in [f[0] for f in lib._Weights._fields_]
        assert "k" in [f[0] for f in lib._KVCache._fields_]

    def test_load_lib_via_env_var(self, monkeypatch):
        monkeypatch.setenv("MAN_TRANSFORMER_LIB", B.load_lib()._name)
        monkeypatch.setattr(B, "_LIB", None)
        assert B.load_lib() is not None


class TestWeightMapper:
    def test_qwen_style_layout(self):
        w = _weights()
        flat, info = map_slnc_to_native(w, L, D, NH, NKV, HD, FF, V, is_qwen_style=True)
        layer_size = info["layer_size"]
        expected = V * D + layer_size * L + D + V * D
        assert info["total_floats"] == expected
        assert len(flat) == expected
        assert info["layers"][0]["offset"] == V * D
        assert info["layers"][1]["offset"] == V * D + layer_size

    def test_qwen_style_slice_correctness(self):
        w = _weights()
        flat, info = map_slnc_to_native(w, L, D, NH, NKV, HD, FF, V, is_qwen_style=True)
        np.testing.assert_allclose(
            flat[0:V * D], w["model.embed_tokens.weight"].ravel(), rtol=1e-6, atol=0
        )
        l0 = info["layers"][0]["offset"]
        np.testing.assert_allclose(
            flat[l0:l0 + D], w["model.layers.0.input_layernorm.weight"].ravel(),
            rtol=1e-6, atol=0,
        )
        norm_off = V * D + info["layer_size"] * L
        np.testing.assert_allclose(
            flat[norm_off:norm_off + D], w["model.norm.weight"].ravel(), rtol=1e-6, atol=0
        )
        head_off = norm_off + D
        np.testing.assert_allclose(
            flat[head_off:head_off + V * D], w["model.lm_head.weight"].ravel(),
            rtol=1e-6, atol=0,
        )

    def test_gpt2_style_naming(self):
        rng = np.random.RandomState(3)
        w = {"wte.weight": rng.randn(V, D).astype(np.float32)}
        for i in range(L):
            w[f"h.{i}.input_layernorm.weight"] = rng.randn(D).astype(np.float32)
            w[f"h.{i}.post_attention_layernorm.weight"] = rng.randn(D).astype(np.float32)
            w[f"h.{i}.mlp.gate_proj.weight"] = rng.randn(D, FF).astype(np.float32)
            w[f"h.{i}.mlp.up_proj.weight"] = rng.randn(D, FF).astype(np.float32)
            w[f"h.{i}.mlp.down_proj.weight"] = rng.randn(FF, D).astype(np.float32)
        w["ln_f.weight"] = rng.randn(D).astype(np.float32)
        flat, info = map_slnc_to_native(w, L, D, NH, NKV, HD, FF, V, is_qwen_style=False)
        assert info["total_floats"] == V * D + info["layer_size"] * L + D + V * D
        l0 = info["layers"][0]["offset"]
        np.testing.assert_allclose(
            flat[l0:l0 + D], w["h.0.input_layernorm.weight"].ravel(), rtol=1e-6, atol=0
        )
        ff_off = l0 + D + D*(NH*HD) + NH*HD + D*(NKV*HD) + NKV*HD + D*(NKV*HD) + NKV*HD + NH*HD*D + D + D
        np.testing.assert_allclose(
            flat[ff_off:ff_off + D * FF], w["h.0.mlp.gate_proj.weight"].ravel(), rtol=1e-6, atol=0
        )
        head_off = V * D + info["layer_size"] * L + D
        np.testing.assert_allclose(
            flat[head_off:head_off + V * D], w["wte.weight"].ravel(), rtol=1e-6, atol=0
        )

    def test_missing_biases_default_to_zeros(self):
        w = {"model.embed_tokens.weight": np.random.RandomState(1).randn(V, D).astype(np.float32)}
        flat, info = map_slnc_to_native(w, L, D, NH, NKV, HD, FF, V, is_qwen_style=True)
        assert info["total_floats"] == V * D + info["layer_size"] * L + D + V * D
        layers_end = V * D + info["layer_size"] * L + D
        assert np.allclose(flat[V * D:layers_end], 0.0)
        np.testing.assert_allclose(flat[layers_end:], flat[:V * D], rtol=1e-6, atol=0)

    def test_empty_tensors_default_to_zeros(self):
        flat, info = map_slnc_to_native({}, L, D, NH, NKV, HD, FF, V, is_qwen_style=True)
        assert info["total_floats"] == V * D + info["layer_size"] * L + D + V * D
        assert np.allclose(flat, 0.0)

    def test_explicit_mlp_biases_written(self):
        rng = np.random.RandomState(5)
        w = {"model.embed_tokens.weight": rng.randn(V, D).astype(np.float32)}
        for i in range(L):
            p = f"model.layers.{i}"
            w[f"{p}.input_layernorm.weight"] = rng.randn(D).astype(np.float32)
            w[f"{p}.mlp.gate_proj.weight"] = rng.randn(D, FF).astype(np.float32)
            w[f"{p}.mlp.gate_proj.bias"] = rng.randn(FF).astype(np.float32)
            w[f"{p}.mlp.up_proj.weight"] = rng.randn(D, FF).astype(np.float32)
            w[f"{p}.mlp.up_proj.bias"] = rng.randn(FF).astype(np.float32)
            w[f"{p}.mlp.down_proj.weight"] = rng.randn(FF, D).astype(np.float32)
            w[f"{p}.mlp.down_proj.bias"] = rng.randn(D).astype(np.float32)
        w["model.norm.weight"] = rng.randn(D).astype(np.float32)
        flat, info = map_slnc_to_native(w, L, D, NH, NKV, HD, FF, V, is_qwen_style=True)
        l0 = info["layers"][0]["offset"]
        gate_b_off = l0 + D + D*(NH*HD) + NH*HD + D*(NKV*HD) + NKV*HD + D*(NKV*HD) + NKV*HD + NH*HD*D + D + D + D*FF
        np.testing.assert_allclose(
            flat[gate_b_off:gate_b_off + FF],
            w["model.layers.0.mlp.gate_proj.bias"].ravel(), rtol=1e-6, atol=0,
        )
        down_b_off = l0 + D + D*(NH*HD) + NH*HD + D*(NKV*HD) + NKV*HD + D*(NKV*HD) + NKV*HD + NH*HD*D + D + D + D*FF + FF + D*FF + FF + FF*D
        np.testing.assert_allclose(
            flat[down_b_off:down_b_off + D],
            w["model.layers.0.mlp.down_proj.bias"].ravel(), rtol=1e-6, atol=0,
        )


class TestSamplingAndFormatting:
    def test_sample_token_greedy_when_cold(self):
        logits = np.array([0.1, 0.9, 0.5, -3.0], dtype=np.float32)
        assert sample_token(logits, temperature=0.0, top_p=0.9, top_k=50) == 1

    def test_sample_token_default_rng(self):
        logits = np.array([0.1, 0.9, 0.5, -3.0], dtype=np.float32)
        assert isinstance(sample_token(logits, temperature=1.0, top_p=1.0, top_k=0), int)

    def test_sample_token_top_k_restricts_support(self):
        logits = np.array([0.0, 100.0, 99.0, -100.0, -100.0], dtype=np.float32)
        rng = np.random.default_rng(0)
        drawn = {sample_token(logits, temperature=1.0, top_p=1.0, top_k=2, rng=rng)
                 for _ in range(200)}
        assert drawn <= {1, 2}

    def test_sample_token_top_p_restricts_support(self):
        logits = np.array([0.0, 100.0, 99.0, -100.0, -100.0], dtype=np.float32)
        rng = np.random.default_rng(0)
        drawn = {sample_token(logits, temperature=1.0, top_p=0.5, top_k=0, rng=rng)
                 for _ in range(200)}
        assert drawn <= {1, 2}

    def test_sample_token_survives_non_finite_logits(self):
        logits = np.array([np.nan, np.inf, -np.inf, 0.5], dtype=np.float32)
        tok = sample_token(logits, temperature=0.7, top_p=0.9, top_k=10)
        assert isinstance(tok, int) and 0 <= tok < len(logits)

    def test_detect_model_type(self):
        assert _detect_model_type({"architectures": ["Qwen2ForCausalLM"]}) == "qwen2"
        assert _detect_model_type({"architectures": ["LlamaForCausalLM"]}) == "llama"
        assert _detect_model_type({"architectures": ["MistralForCausalLM"]}) == "llama"
        assert _detect_model_type({"architectures": ["PhiForCausalLM"]}) == "llama"
        assert _detect_model_type({"architectures": ["GPT2LMHeadModel"]}) == "gpt2"
        assert _detect_model_type({"architectures": ["OpenAIGPTLMHeadModel"]}) == "gpt2"
        assert _detect_model_type({"architectures": ["AnythingElse"]}) == "qwen2"

    def test_format_chat_qwen_with_system(self):
        out = format_chat([{"role": "user", "content": "hi"}], "qwen2", system="be brief")
        assert out == "<|im_start|>system\nbe brief<|im_end|><|im_start|>user\nhi<|im_end|><|im_start|>assistant\n"

    def test_format_chat_llama_first_user_with_system(self):
        out = _format_chat_llama([{"role": "user", "content": "hello"}], system="S")
        assert out == "[INST] <<SYS>>\nS\n<</SYS>>\n\nhello [/INST] "

    def test_format_chat_llama_multi_turn(self):
        msgs = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
        out = _format_chat_llama(msgs)
        assert out == "[INST] a [/INST] b </s>"

    def test_format_chat_gpt2(self):
        msgs = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
        out = _format_chat_gpt2(msgs, system="ignored")
        assert out == "User: a\nAssistant: b\nAssistant:"

    def test_format_chat_dispatch_llama(self):
        msgs = [{"role": "user", "content": "hi"}]
        assert format_chat(msgs, "llama", "") == "[INST] hi [/INST] "
        with_sys = format_chat(msgs, "llama", system="be terse")
        assert "be terse" in with_sys and with_sys.endswith("hi [/INST] ")

    def test_format_chat_dispatch_gpt2(self):
        msgs = [{"role": "user", "content": "hi"}]
        assert format_chat(msgs, "gpt2", "") == "User: hi\nAssistant:"

    def test_format_chat_dispatch_default(self):
        msgs = [{"role": "user", "content": "hi"}]
        assert format_chat(msgs, "unknown-model", "") == "User: hi\nAssistant:"


class TestNativeEngine:
    def test_load_and_generate_greedy(self):
        engine = _loaded_engine()
        assert engine.loaded
        text = engine.generate([{"role": "user", "content": "hi there"}], max_tokens=8, temperature=0)
        assert isinstance(text, str) and len(text) > 0

    def test_load_and_generate_sampled(self):
        engine = _loaded_engine()
        text = engine.generate([{"role": "user", "content": "hi"}], max_tokens=8,
                               temperature=0.7, top_p=0.9, top_k=10)
        assert isinstance(text, str) and len(text) > 0

    def test_generate_stream_yields_pieces(self):
        engine = _loaded_engine()
        pieces = list(engine.generate_stream([{"role": "user", "content": "hi"}], max_tokens=8,
                                             temperature=0.0))
        assert len(pieces) > 0
        assert all(isinstance(p, str) and len(p) >= 0 for p in pieces)

    def test_generate_stream_respects_eos(self):
        engine = _loaded_engine()
        pieces = list(engine.generate_stream([{"role": "user", "content": "hi"}], max_tokens=32,
                                             temperature=0.0))
        assert len(pieces) <= 32

    def test_unloaded_engine_generate_raises(self):
        engine = NativeEngine()
        with pytest.raises(RuntimeError):
            engine.generate([{"role": "user", "content": "hi"}])

    def test_unloaded_engine_generate_stream_raises(self):
        engine = NativeEngine()
        with pytest.raises(RuntimeError):
            list(engine.generate_stream([{"role": "user", "content": "hi"}]))

    def test_reset_cache(self):
        engine = _loaded_engine()
        engine.generate([{"role": "user", "content": "hi"}], max_tokens=4, temperature=0)
        engine.reset_cache()
        assert engine.loaded

    def test_tokenize_roundtrip(self):
        engine = _loaded_engine()
        ids = engine._tokenize_simple("hello")
        assert isinstance(ids, list) and all(isinstance(t, int) for t in ids)
        text = engine._detokenize_simple(ids)
        assert isinstance(text, str)

    def test_detokenize_maps_non_ascii_to_placeholder(self, monkeypatch):
        import domains.inference.tokenizer as T
        monkeypatch.setattr(T, "get_tokenizer",
                            lambda: (_ for _ in ()).throw(RuntimeError("no tokenizer")))
        engine = _loaded_engine()
        assert engine._detokenize_simple([0, 255]) == "??"

    def test_get_engine_singleton(self):
        assert get_engine() is get_engine()

    def test_load_info(self):
        engine = _loaded_engine()
        info = engine.load_from_slnc(_weights(), _config(), seq_capacity=16)
        assert info == {"model_type": "qwen2", "layers": L, "hidden": D}

    def test_load_weights_failure_raises(self, monkeypatch):
        engine = NativeEngine()
        monkeypatch.setattr(engine._lib, "transformer_load_weights",
                            lambda *a, **k: 1)
        with pytest.raises(RuntimeError, match="transformer_load_weights"):
            engine.load_from_slnc(_weights(), _config(), seq_capacity=16)

    def test_generate_stops_at_eos(self, monkeypatch):
        engine = _loaded_engine()
        real_step = engine._lib.transformer_forward_step

        def _step_picks_eos(weights, cache, tok, pos, logits_ptr):
            buf = np.ctypeslib.as_array(
                ctypes.cast(logits_ptr, ctypes.POINTER(ctypes.c_float)),
                shape=(V,),
            )
            buf[:] = -1.0
            buf[_config()["eos_token_id"]] = 10.0

        monkeypatch.setattr(engine._lib, "transformer_forward_step", _step_picks_eos)
        assert engine.generate([{"role": "user", "content": "hi"}],
                               max_tokens=8, temperature=0.0) == ""

    def test_generate_stream_stops_at_eos(self, monkeypatch):
        engine = _loaded_engine()

        def _step_picks_eos(weights, cache, tok, pos, logits_ptr):
            buf = np.ctypeslib.as_array(
                ctypes.cast(logits_ptr, ctypes.POINTER(ctypes.c_float)),
                shape=(V,),
            )
            buf[:] = -1.0
            buf[_config()["eos_token_id"]] = 10.0

        monkeypatch.setattr(engine._lib, "transformer_forward_step", _step_picks_eos)
        pieces = list(engine.generate_stream([{"role": "user", "content": "hi"}],
                                             max_tokens=8, temperature=0.0))
        assert pieces == []

    def test_tokenize_uses_real_tokenizer(self):
        from domains.inference.tokenizer import get_tokenizer
        engine = _loaded_engine()
        assert engine._tokenize_simple("hi") == get_tokenizer().encode("hi")

    def test_detokenize_uses_real_tokenizer(self):
        from domains.inference.tokenizer import get_tokenizer
        engine = _loaded_engine()
        ids = engine._tokenize_simple("hi")
        assert engine._detokenize_simple(ids) == get_tokenizer().decode(ids)

    def test_tokenize_fallback_when_tokenizer_missing(self, monkeypatch):
        import domains.inference.tokenizer as T
        monkeypatch.setattr(T, "get_tokenizer",
                            lambda: (_ for _ in ()).throw(RuntimeError("no tokenizer")))
        engine = _loaded_engine()
        assert engine._tokenize_simple("hi") == list("hi".encode("utf-8"))

    def test_detokenize_fallback_when_tokenizer_missing(self, monkeypatch):
        import domains.inference.tokenizer as T
        monkeypatch.setattr(T, "get_tokenizer",
                            lambda: (_ for _ in ()).throw(RuntimeError("no tokenizer")))
        engine = _loaded_engine()
        assert engine._detokenize_simple([0, 255]) == "??"


class TestTokenizerWiring:
    def test_set_tokenizer_routes_encode_decode(self):
        engine = _loaded_engine()
        fake = _FakeTokenizer()
        engine.set_tokenizer(fake)
        assert engine._tokenize_simple("hi") == [ord("h"), ord("i")]
        assert engine._detokenize_simple([72, 105]) == "Hi"

    def test_set_tokenizer_none_restores_fallback(self):
        from domains.inference.tokenizer import get_tokenizer
        engine = _loaded_engine()
        engine.set_tokenizer(_FakeTokenizer())
        engine.set_tokenizer(None)
        assert engine._tokenize_simple("hi") == get_tokenizer().encode("hi")

    def test_load_from_slnc_with_tokenizer_sets_stop_ids(self):
        engine = NativeEngine()
        engine.load_from_slnc(_weights(), _config(eos_token_id=999),
                              tokenizer=_FakeTokenizer(stop_ids=(2,)))
        assert engine._tokenizer is not None
        assert engine._stop_ids() == {2}

    def test_generate_stops_at_tokenizer_stop_id(self, monkeypatch):
        engine = NativeEngine()
        engine.load_from_slnc(_weights(), _config(eos_token_id=999),
                              tokenizer=_FakeTokenizer(stop_ids=(2,)))

        def _step_picks_stop(weights, cache, tok, pos, logits_ptr):
            buf = np.ctypeslib.as_array(
                ctypes.cast(logits_ptr, ctypes.POINTER(ctypes.c_float)),
                shape=(V,),
            )
            buf[:] = -1.0
            buf[2] = 10.0

        monkeypatch.setattr(engine._lib, "transformer_forward_step", _step_picks_stop)
        assert engine.generate([{"role": "user", "content": "hi"}],
                               max_tokens=8, temperature=0.0) == ""

    def test_build_prompt_prefers_apply_chat_template(self):
        engine = _loaded_engine()
        engine.set_tokenizer(_FakeTokenizer())
        prompt = engine._build_prompt([{"role": "user", "content": "hi"}], system="sys")
        assert prompt == "CHAT:system=sys|user=hi:ASST"

    def test_build_prompt_falls_back_to_format_chat(self):
        engine = _loaded_engine()
        assert engine._build_prompt([{"role": "user", "content": "hi"}], system="sys") == \
            format_chat([{"role": "user", "content": "hi"}], "qwen2", "sys")

    def test_sample_masks_beyond_tokenizer_vocab(self):
        engine = _loaded_engine()
        engine.set_tokenizer(_FakeTokenizer(vocab_size=4))
        logits = np.full(V, -1.0, dtype=np.float32)
        logits[200] = 10.0
        logits[0] = 5.0
        tok = engine._sample(logits, 0.0, 0.9, 50, np.random.default_rng())
        assert tok == 0

    def test_load_from_slnc_hf_model_id_uses_real_tokenizer(self):
        from domains.infrastructure.morph_tokenizer import MorphTokenizer
        try:
            real = MorphTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        except FileNotFoundError:
            pytest.skip("Qwen tokenizer not cached locally")
        engine = NativeEngine()
        engine.load_from_slnc(_weights(), _config(), hf_model_id="Qwen/Qwen2.5-0.5B-Instruct")
        assert engine._tokenizer is not None
        assert engine._tokenize_simple("Hello world") == [9707, 1879]
        assert engine._stop_ids() == {int(i) for i in real.chat_stop_ids()}

    def test_from_slnc_file_roundtrip(self, tmp_path):
        slnc_path = str(tmp_path / "tiny.slnc")
        _build_slnc(slnc_path, _weights(), _config())
        engine = NativeEngine.from_slnc_file(slnc_path, seq_capacity=16)
        assert engine.loaded
        assert engine._model_type == "qwen2"
        out = engine.generate([{"role": "user", "content": "hi"}], max_tokens=8,
                              temperature=0.0)
        assert isinstance(out, str)

    def test_hf_id_from_slnc_path(self):
        from domains.inference.native.engine import _hf_id_from_slnc_path
        assert _hf_id_from_slnc_path(
            "/x/hf-cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/a/model.slnc"
        ) == "Qwen/Qwen2.5-0.5B-Instruct"
        assert _hf_id_from_slnc_path("/tmp/model.slnc") is None


class TestRealModelEndToEnd:
    SLNC = ("models/hf-cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/model.slnc")

    def _available_mem_kb(self) -> int:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1])
        except OSError:
            return 0
        return 0

    @pytest.mark.slow
    @pytest.mark.integration
    def test_real_qwen_slnc_native_generation(self, tmp_path):
        from pathlib import Path
        repo_root = Path(__file__).resolve().parents[3]
        slnc = repo_root / self.SLNC
        if not slnc.exists():
            pytest.skip("real Qwen .slnc not present on disk")
        if self._available_mem_kb() < 5 * 1024 * 1024:
            pytest.skip("less than 5GB available memory")

        engine = NativeEngine.from_slnc_file(str(slnc), seq_capacity=512)
        assert engine.loaded
        assert engine._model_type == "qwen2"
        assert engine._tokenizer is not None
        assert 151643 in engine._stop_ids()

        out = engine.generate([{"role": "user", "content": "Hello"}],
                              max_tokens=16, temperature=0.0)
        assert isinstance(out, str) and len(out) > 0
        assert "?" not in out


class TestNativeProviderWiring:
    @pytest.fixture(autouse=True)
    def _clean_registries(self):
        import domains.models.provider as mod
        mod._providers.clear()
        mod._processors.clear()
        yield
        mod._providers.clear()
        mod._processors.clear()

    def test_setup_providers_native_slnc_path(self, tmp_path):
        from domains.models.provider import setup_providers, get_provider
        slnc_path = str(tmp_path / "tiny.slnc")
        _build_slnc(slnc_path, _weights(), _config())
        setup_providers(native_slnc_path=slnc_path)
        default = get_provider("default")
        assert default.metadata["text_provider"] == "native-c"
        provider = get_provider("native-c")
        assert provider is not None
        assert provider.metadata["type"] == "native-c"
        assert provider.metadata["loaded"] is True

    def test_setup_providers_missing_slnc_degrades_gracefully(self):
        from domains.models.provider import setup_providers, get_provider
        setup_providers(native_slnc_path="/nonexistent/model.slnc")
        default = get_provider("default")
        assert default.metadata["text_provider"] is None
        assert get_provider("native-c") is None


class TestNativeTransformerProvider:
    def test_provider_identity_and_capabilities(self):
        engine = _loaded_engine()
        provider = NativeTransformerProvider(engine, model_id="native-test")
        assert provider.model_id == "native-test"
        assert provider.capabilities.chat is True
        assert provider.capabilities.streaming is True
        assert provider.embed("anything") == []
        meta = provider.metadata
        assert meta["model_id"] == "native-test"
        assert meta["type"] == "native-c"
        assert meta["loaded"] is True
        assert meta["vocab_size"] == V
        assert meta["layers"] == L

    def test_provider_chat(self):
        engine = _loaded_engine()
        provider = NativeTransformerProvider(engine)
        text = asyncio.run(provider.chat([{"role": "user", "content": "hi"}], max_tokens=8))
        assert isinstance(text, str) and len(text) > 0

    def test_provider_chat_stream(self):
        engine = _loaded_engine()
        provider = NativeTransformerProvider(engine)

        async def _collect():
            return [p async for p in provider.chat_stream(
                [{"role": "user", "content": "hi"}], max_tokens=8)]

        pieces = asyncio.run(_collect())
        assert len(pieces) > 0

    def test_provider_chat_stream_honors_cancel(self):
        engine = _loaded_engine()
        provider = NativeTransformerProvider(engine)
        import threading

        event = threading.Event()
        event.set()

        async def _collect():
            return [p async for p in provider.chat_stream(
                [{"role": "user", "content": "hi"}], max_tokens=8, cancel_event=event)]

        assert asyncio.run(_collect()) == []

    def test_provider_chat_stream_propagates_generation_error(self):
        engine = NativeEngine()
        provider = NativeTransformerProvider(engine)

        async def _collect():
            return [p async for p in provider.chat_stream(
                [{"role": "user", "content": "hi"}], max_tokens=8)]

        assert asyncio.run(_collect()) == []


class TestCTransformProvider:
    def test_from_slnc_real_file(self, tmp_path):
        slnc_path = str(tmp_path / "tiny.slnc")
        _build_slnc(slnc_path, _weights(), _config())
        provider = CTransformProvider.from_slnc(slnc_path, seq_capacity=16)
        assert provider._engine.loaded
        out = provider.generate("hi", max_tokens=8)
        assert isinstance(out, str) and len(out) > 0
        meta = provider.metadata()
        assert meta["architecture"] == "NativeEngine"
        assert meta["n_layer"] == L
        assert meta["vocab_size"] == V
        assert meta["engine"] == "c"
        assert meta["has_tokenizer"] is True

    def test_from_slnc_roundtrip_slnc_parser(self, tmp_path):
        from domains.infrastructure.slnc.parser import SLNCParser
        slnc_path = str(tmp_path / "tiny.slnc")
        w = _weights()
        _build_slnc(slnc_path, w, _config())
        parser = SLNCParser(slnc_path)
        assert parser.config["architectures"] == ["Qwen2ForCausalLM"]
        loaded = parser.get_weights_dict()
        assert set(w.keys()) == set(loaded.keys())
        np.testing.assert_allclose(
            loaded["model.embed_tokens.weight"],
            w["model.embed_tokens.weight"], rtol=1e-6, atol=0,
        )

    def test_tokenize_detokenize_uses_real_tokenizer(self):
        from domains.inference.tokenizer import get_tokenizer
        engine = _loaded_engine()
        provider = CTransformProvider(engine)
        assert provider._tokenizer is not None
        ids = provider.tokenize("hi")
        assert ids == get_tokenizer().encode("hi")
        assert provider.detokenize(ids) == get_tokenizer().decode(ids)

    def test_init_tolerates_missing_tokenizer(self, monkeypatch):
        import domains.inference.tokenizer as T
        monkeypatch.setattr(T, "get_tokenizer",
                            lambda: (_ for _ in ()).throw(RuntimeError("no tokenizer")))
        provider = CTransformProvider(_loaded_engine())
        assert provider._tokenizer is None
        assert provider.metadata()["has_tokenizer"] is False

    def test_tokenize_detokenize_fallback(self):
        engine = _loaded_engine()
        provider = CTransformProvider(engine)
        provider._tokenizer = None
        ids = provider.tokenize("hi")
        assert isinstance(ids, list) and all(isinstance(t, int) for t in ids)
        assert isinstance(provider.detokenize(ids), str)

    def test_metadata_from_config(self):
        engine = _loaded_engine()
        provider = CTransformProvider(engine, model_id="c-tiny")
        meta = provider.metadata()
        assert meta["model_id"] == "c-tiny"
        assert meta["n_embed"] == D
        assert meta["n_head"] == NH
