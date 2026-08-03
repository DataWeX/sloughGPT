"""
Wave F tests for SloNetChatProvider — end-to-end .slnc loading, quantization,
async chat paths, streaming, and provider plumbing.

FEATURE: slonet-wave-f — Covers from_slnc (plain + quantized), chat/chat_stream
(server + builtin), _build_prompt, _load_tokenizer, generate_with_logprobs
seed branch, safetensors BF16 loader, fused-QKV bias split, and module helpers.
DO NOT DELETE.
"""
import asyncio
import json
import struct
import zlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from domains.inference.slonet_provider import (
    SloNetChatProvider,
    _get_slo_layernorm,
    _split_fused_qkv,
    convert_hf_to_slonet,
)
from domains.infrastructure.slnc.spec import (
    compute_header_size,
    compute_tensor_entry_size,
)

N_EMBED, N_LAYER, N_HEAD, VOCAB, N_POS, N_INNER = 16, 1, 2, 50, 32, 64


class FakeTokenizer:
    """Deterministic tokenizer stub for provider tests."""

    vocab_size = VOCAB
    eos_token_id = 2

    def encode(self, text):
        return [1, 2, 3]

    def decode(self, ids):
        return "".join(chr(97 + (int(i) % 26)) for i in ids)

    def apply_chat_template(self, messages):
        return messages[-1].get("content", "")


def _make_gpt2_tensors(rng):
    """Build a minimal GPT-2-style tensor dict for a .slnc file."""
    t = {}
    t["wte.weight"] = rng.standard_normal((VOCAB, N_EMBED)).astype(np.float32)
    t["wpe.weight"] = rng.standard_normal((N_POS, N_EMBED)).astype(np.float32)
    t["ln_f.weight"] = np.ones(N_EMBED, dtype=np.float32)
    t["ln_f.bias"] = np.zeros(N_EMBED, dtype=np.float32)
    for i in range(N_LAYER):
        p = f"h.{i}"
        t[f"{p}.ln_1.weight"] = np.ones(N_EMBED, dtype=np.float32)
        t[f"{p}.ln_1.bias"] = np.zeros(N_EMBED, dtype=np.float32)
        t[f"{p}.attn.c_attn.weight"] = rng.standard_normal((N_EMBED, 3 * N_EMBED)).astype(np.float32)
        t[f"{p}.attn.c_attn.bias"] = rng.standard_normal((3 * N_EMBED,)).astype(np.float32)
        t[f"{p}.attn.c_proj.weight"] = rng.standard_normal((N_EMBED, N_EMBED)).astype(np.float32)
        t[f"{p}.attn.c_proj.bias"] = np.zeros(N_EMBED, dtype=np.float32)
        t[f"{p}.ln_2.weight"] = np.ones(N_EMBED, dtype=np.float32)
        t[f"{p}.ln_2.bias"] = np.zeros(N_EMBED, dtype=np.float32)
        t[f"{p}.mlp.c_fc.weight"] = rng.standard_normal((N_EMBED, N_INNER)).astype(np.float32)
        t[f"{p}.mlp.c_fc.bias"] = np.zeros(N_INNER, dtype=np.float32)
        t[f"{p}.mlp.c_proj.weight"] = rng.standard_normal((N_INNER, N_EMBED)).astype(np.float32)
        t[f"{p}.mlp.c_proj.bias"] = np.zeros(N_EMBED, dtype=np.float32)
    return t


def _slnc_config():
    return {
        "vocab_size": VOCAB,
        "n_embd": N_EMBED,
        "n_head": N_HEAD,
        "n_layer": N_LAYER,
        "n_positions": N_POS,
        "n_inner": N_INNER,
        "hidden_act": "gelu",
        "layer_norm_type": "layer_norm",
    }


def build_slnc(path, config, tensors):
    """Write a valid .slnc file mirroring SLNCCompiler layout exactly."""
    json_bytes = json.dumps(config, sort_keys=True).encode()
    header_size = compute_header_size(json_bytes)
    entries = []
    for name, arr in tensors.items():
        data = arr.astype(np.float32).tobytes()
        ndim = arr.ndim
        crc = zlib.crc32(data) & 0xFFFFFFFF
        entries.append((name, data, ndim, tuple(arr.shape), crc))
    formula_table_size = sum(
        compute_tensor_entry_size(e[2], len(e[0].encode())) for e in entries
    )
    data_offset = (header_size + formula_table_size + 63) & ~63
    cur = data_offset
    offs = []
    for name, data, ndim, shape, crc in entries:
        offs.append((name, cur, len(data), ndim, shape, crc))
        cur += len(data)
    with open(path, "wb") as f:
        f.write(b"SLNC")
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<I", 0))
        f.write(struct.pack(
            "<10I", N_LAYER, N_EMBED, N_HEAD, N_INNER, VOCAB, N_POS,
            N_LAYER, N_POS, len(tensors), data_offset,
        ))
        f.write(b"\x00" * 24)
        f.write(struct.pack("<I", len(json_bytes)))
        f.write(json_bytes)
        f.write(b"\x00" * (header_size - (12 + 64 + 4 + len(json_bytes))))
        for name, off, size, ndim, shape, crc in offs:
            nb = name.encode()
            f.write(struct.pack("<I", len(nb)))
            f.write(nb)
            f.write(struct.pack("<Q", off))
            f.write(struct.pack("<I", size))
            f.write(struct.pack("<I", ndim))
            f.write(struct.pack("<%dI" % ndim, *shape))
            f.write(struct.pack("<I", 0))
            f.write(struct.pack("<I", crc))
        table_end = f.tell()
        f.write(b"\x00" * (data_offset - table_end))
        for name, data, ndim, shape, crc in entries:
            f.write(data)


@pytest.fixture
def slnc_path(tmp_path):
    path = tmp_path / "tiny.slnc"
    build_slnc(path, _slnc_config(), _make_gpt2_tensors(np.random.default_rng(0)))
    return path


@pytest.fixture
def real_provider(slnc_path):
    """Provider loaded from a real .slnc file with a stubbed tokenizer."""
    with patch.object(
        SloNetChatProvider, "_load_tokenizer", return_value=FakeTokenizer()
    ):
        provider = SloNetChatProvider.from_slnc(str(slnc_path), model_id="gpt2")
    return provider


class TestModuleHelpers:
    """Direct coverage for module-level helpers."""

    def test_get_slo_layernorm(self):
        ln = _get_slo_layernorm()
        assert callable(ln)

    def test_split_fused_qkv_weight(self):
        arr = np.random.randn(N_EMBED, 3 * N_EMBED).astype(np.float32)
        out = _split_fused_qkv(
            "h.0.attn.c_attn.weight", arr, N_EMBED, N_LAYER, {}
        )
        assert "blocks.0.attn.q_proj.weight" in out
        assert out["blocks.0.attn.q_proj.weight"].shape == (N_EMBED, N_EMBED)

    def test_split_fused_qkv_bias(self):
        arr = np.random.randn(3 * N_EMBED).astype(np.float32)
        out = _split_fused_qkv("h.0.attn.c_attn.bias", arr, N_EMBED, N_LAYER, {})
        assert "blocks.0.attn.q_proj.bias" in out
        assert out["blocks.0.attn.q_proj.bias"].shape == (N_EMBED,)

    def test_split_fused_qkv_bad_layer_key(self):
        arr = np.random.randn(N_EMBED, 3 * N_EMBED).astype(np.float32)
        out = _split_fused_qkv("h.x.attn.c_attn.weight", arr, N_EMBED, N_LAYER, {})
        assert out == {}


class TestConstructorAndSafetensors:
    """Constructor guard and BF16 safetensors loader."""

    def test_init_raises_typeerror(self):
        with pytest.raises(TypeError):
            SloNetChatProvider("gpt2")

    def _write_safetensors(self, path, blocks):
        raws = {}
        for name, (dtype, arr) in blocks.items():
            if dtype == "BF16":
                raws[name] = (arr.view(np.uint32) >> 16).astype(np.uint16).tobytes()
            else:
                raws[name] = arr.tobytes()
        header = {}
        offset = 0
        for name, (dtype, arr) in blocks.items():
            header[name] = {
                "dtype": dtype,
                "shape": list(arr.shape),
                "data_offsets": [offset, offset + len(raws[name])],
            }
            offset += len(raws[name])
        hdr_bytes = json.dumps(header).encode()
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", len(hdr_bytes)))
            f.write(hdr_bytes)
            for name, raw in raws.items():
                f.write(raw)

    def test_load_safetensors_bf16_all_dtypes(self, tmp_path):
        path = tmp_path / "w.bin"
        f32 = np.random.randn(2, 3).astype(np.float32)
        bf16 = np.random.randn(4).astype(np.float32)
        f16 = np.random.randn(5).astype(np.float16)
        i32 = np.random.randint(0, 9, size=(3,)).astype(np.int32)
        blocks = {
            "a.f32": ("F32", f32),
            "a.bf16": ("BF16", bf16),
            "a.f16": ("F16", f16),
            "a.i32": ("I32", i32),
        }
        self._write_safetensors(path, blocks)
        weights = SloNetChatProvider._load_safetensors_bf16(str(path))
        assert weights["a.f32"].shape == (2, 3)
        assert weights["a.bf16"].shape == (4,)
        assert np.allclose(weights["a.bf16"], bf16, atol=1e-2)
        assert weights["a.f16"].dtype == np.float32
        assert weights["a.i32"].shape == (3,)

    def test_load_safetensors_skips_metadata(self, tmp_path):
        path = tmp_path / "w2.bin"
        arr = np.random.randn(2).astype(np.float32)
        header = {
            "__metadata__": {"format": "pt"},
            "x.weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
        }
        hdr_bytes = json.dumps(header).encode()
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", len(hdr_bytes)))
            f.write(hdr_bytes)
            f.write(arr.tobytes())
        weights = SloNetChatProvider._load_safetensors_bf16(str(path))
        assert "x.weight" in weights


class TestFromSlnc:
    """End-to-end .slnc loading and provider plumbing."""

    def test_from_slnc_generate_and_metadata(self, real_provider):
        text = real_provider.generate("hello", max_tokens=8, temperature=0.5)
        assert isinstance(text, str) and len(text) > 0
        meta = real_provider.metadata()
        assert meta["model_id"] == "gpt2"
        assert meta["architecture"] == "SloTransformer"
        assert meta["quantized"] is False
        assert meta["has_tokenizer"] is True

    def test_properties_and_server_plumbing(self, real_provider):
        assert real_provider.model_id == "gpt2"
        caps = real_provider.capabilities
        assert caps.chat is True and caps.streaming is True

        assert real_provider.get_server() is None
        fake_server = MagicMock()
        real_provider.set_server(fake_server)
        assert real_provider.get_server() is fake_server

    def test_quantization_report_unquantized(self, real_provider):
        assert real_provider.quantization_report() == {"quantized": False}

    def test_embed_and_token_utils(self, real_provider):
        emb = real_provider.embed("hello")
        assert emb.shape == (N_EMBED,)
        assert real_provider.count_tokens("hello") == 3
        assert real_provider.tokenize("hello") == [1, 2, 3]
        assert isinstance(real_provider.detokenize([1, 2]), str)

    def test_generate_max_new_tokens_alias(self, real_provider):
        out = real_provider.generate("hello", max_new_tokens=4)
        assert isinstance(out, str)

    def test_from_slnc_quantize_fresh(self, slnc_path):
        with patch.object(
            SloNetChatProvider, "_load_tokenizer", return_value=FakeTokenizer()
        ):
            provider = SloNetChatProvider.from_slnc(
                str(slnc_path), model_id="gpt2", quantize=True
            )
        report = provider.quantization_report()
        assert report["quantized"] is True
        assert report["bits"] == 8
        assert report["mode"] == "symmetric"
        assert "summary" in report and "per_tensor" in report
        text = provider.generate("hello", max_tokens=4)
        assert isinstance(text, str)
        assert slnc_path.with_suffix(".slnc.quant.npz").exists()
        assert slnc_path.with_suffix(".slnc.quant.json").exists()

    def test_from_slnc_quantize_prequantized(self, slnc_path):
        with patch.object(
            SloNetChatProvider, "_load_tokenizer", return_value=FakeTokenizer()
        ):
            SloNetChatProvider.from_slnc(str(slnc_path), quantize=True)
        npz_path = slnc_path.with_suffix(".slnc.quant.npz")
        json_path = slnc_path.with_suffix(".slnc.quant.json")
        assert npz_path.exists()
        json_path.unlink()
        with patch.object(
            SloNetChatProvider, "_load_tokenizer", return_value=FakeTokenizer()
        ):
            provider = SloNetChatProvider.from_slnc(str(slnc_path), quantize=True)
        assert provider.quantization_report()["quantized"] is True
        assert provider.generate("hello", max_tokens=4)

    def test_from_slnc_quantize_metadata_only(self, slnc_path):
        with patch.object(
            SloNetChatProvider, "_load_tokenizer", return_value=FakeTokenizer()
        ):
            SloNetChatProvider.from_slnc(str(slnc_path), quantize=True)
        npz_path = slnc_path.with_suffix(".slnc.quant.npz")
        json_path = slnc_path.with_suffix(".slnc.quant.json")
        npz_path.unlink()
        assert json_path.exists()
        with patch.object(
            SloNetChatProvider, "_load_tokenizer", return_value=FakeTokenizer()
        ):
            provider = SloNetChatProvider.from_slnc(str(slnc_path), quantize=True)
        assert provider.quantization_report()["quantized"] is True
        assert npz_path.exists()

    def test_from_slnc_rope_config(self, tmp_path):
        cfg = _slnc_config()
        cfg["rope_theta"] = 10000.0
        cfg["rms_norm_eps"] = 1e-5
        cfg["num_key_value_heads"] = N_HEAD
        path = tmp_path / "rope.slnc"
        build_slnc(path, cfg, _make_gpt2_tensors(np.random.default_rng(1)))
        with patch.object(
            SloNetChatProvider, "_load_tokenizer", return_value=FakeTokenizer()
        ):
            provider = SloNetChatProvider.from_slnc(str(path), model_id="qwen")
        assert provider.model_id == "qwen"
        assert provider.generate("hi", max_tokens=4)

    def test_from_slnc_resource_manager_applies(self, slnc_path):
        rm = MagicMock()
        with patch(
            "domains.infrastructure.resource_manager.get_resource_manager",
            return_value=rm,
        ), patch.object(
            SloNetChatProvider, "_load_tokenizer", return_value=FakeTokenizer()
        ):
            provider = SloNetChatProvider.from_slnc(str(slnc_path))
        rm.apply_blas_env.assert_called_once()
        rm.apply_compute_limits.assert_called_once()
        assert provider.model_id == "gpt2"

    def test_from_slnc_resource_manager_error_tolerated(self, slnc_path):
        with patch(
            "domains.infrastructure.resource_manager.get_resource_manager",
            side_effect=RuntimeError("boom"),
        ), patch.object(
            SloNetChatProvider, "_load_tokenizer", return_value=FakeTokenizer()
        ):
            provider = SloNetChatProvider.from_slnc(str(slnc_path))
        assert provider.generate("hi", max_tokens=4)


class TestBuildPromptAndTokenizer:
    """_build_prompt branch coverage and tokenizer loading."""

    def _provider_with_tokenizer(self, tokenizer):
        provider = SloNetChatProvider.__new__(SloNetChatProvider)
        provider._tokenizer = tokenizer
        return provider

    def test_prompt_empty(self):
        provider = self._provider_with_tokenizer(FakeTokenizer())
        assert provider._build_prompt([]) == ""
        assert provider._build_prompt(None) == ""

    def test_prompt_string(self):
        provider = self._provider_with_tokenizer(FakeTokenizer())
        assert provider._build_prompt("hello world") == "hello world"

    def test_prompt_list_of_strings(self):
        provider = self._provider_with_tokenizer(FakeTokenizer())
        assert provider._build_prompt(["a", "b", "c"]) == "c"

    def test_prompt_list_of_dicts_with_template(self):
        provider = self._provider_with_tokenizer(FakeTokenizer())
        messages = [{"role": "user", "content": "hi"}]
        assert provider._build_prompt(messages) == "hi"

    def test_prompt_list_of_dicts_no_template(self):
        class NoTemplateTokenizer:
            vocab_size = VOCAB
            eos_token_id = 2

            def encode(self, text):
                return [1, 2, 3]

            def decode(self, ids):
                return "".join(chr(97 + (int(i) % 26)) for i in ids)

        provider = self._provider_with_tokenizer(NoTemplateTokenizer())
        messages = [{"role": "user", "content": "fallback"}]
        assert provider._build_prompt(messages) == "fallback"

    def test_prompt_unknown_items_no_template(self):
        class NoTemplateTokenizer:
            vocab_size = VOCAB
            eos_token_id = 2

            def encode(self, text):
                return [1, 2, 3]

            def decode(self, ids):
                return "".join(chr(97 + (int(i) % 26)) for i in ids)

        provider = self._provider_with_tokenizer(NoTemplateTokenizer())
        assert provider._build_prompt([None, 7]) == ""

    def test_load_tokenizer_success(self, real_provider):
        with patch(
            "domains.infrastructure.morph_tokenizer.MorphTokenizer.from_pretrained",
            return_value=FakeTokenizer(),
        ) as mock_load:
            tok = real_provider._load_tokenizer(Path("."), _slnc_config())
        assert tok is not None
        mock_load.assert_called_once()

    def test_load_tokenizer_failure_raises(self, real_provider):
        with patch(
            "domains.infrastructure.morph_tokenizer.MorphTokenizer.from_pretrained",
            side_effect=RuntimeError("boom"),
        ):
            with pytest.raises(RuntimeError):
                real_provider._load_tokenizer(Path("."), _slnc_config())


class TestChatPaths:
    """Async chat/chat_stream — server delegation and builtin pipeline."""

    def test_chat_sync_no_server(self, real_provider):
        result = asyncio.run(real_provider.chat("hi", max_tokens=4))
        assert isinstance(result, str) and len(result) > 0

    def test_chat_with_server(self, real_provider):
        class FakeServer:
            async def generate(self, prompt, **kwargs):
                return "from-server"

        real_provider.set_server(FakeServer())
        result = asyncio.run(real_provider.chat("hi", max_tokens=4))
        assert result == "from-server"

    def test_chat_stream_server(self, real_provider):
        class FakeServer:
            async def generate_stream(self, prompt, **kwargs):
                yield "a"
                yield "b"

        real_provider.set_server(FakeServer())

        async def collect():
            return [t async for t in real_provider.chat_stream("hi", max_tokens=2)]

        assert asyncio.run(collect()) == ["a", "b"]

    def test_chat_stream_builtin(self, real_provider):
        async def collect():
            return [t async for t in real_provider.chat_stream("hi", max_tokens=4)]

        toks = asyncio.run(collect())
        assert isinstance(toks, list) and len(toks) > 0
        assert all(isinstance(t, str) for t in toks)

    def test_chat_stream_builtin_empty_decode(self, real_provider):
        real_provider._tokenizer.decode = lambda ids: ""
        async def collect():
            return [t async for t in real_provider.chat_stream("hi", max_tokens=3)]

        assert asyncio.run(collect()) == []

    def test_generate_with_logprobs_seed(self, real_provider):
        text, logprobs = real_provider.generate_with_logprobs(
            "hello", max_tokens=3, seed=42
        )
        assert len(logprobs) == 3
        assert all("token_id" in e and "logprob" in e for e in logprobs)
        assert isinstance(text, str)


class TestConverterEdgeCases:
    """Converter branches not hit by the main from_slnc path."""

    def test_convert_llama_style_swiglu(self):
        n_embed, n_layer = 64, 2
        sd = {
            "model.embed_tokens.weight": np.random.randn(1000, n_embed).astype(np.float32),
            "model.norm.weight": np.ones(n_embed, dtype=np.float32),
        }
        for i in range(n_layer):
            p = f"model.layers.{i}"
            sd[f"{p}.input_layernorm.weight"] = np.ones(n_embed, dtype=np.float32)
            sd[f"{p}.self_attn.q_proj.weight"] = np.random.randn(n_embed, n_embed).astype(np.float32)
            sd[f"{p}.self_attn.k_proj.weight"] = np.random.randn(n_embed, n_embed).astype(np.float32)
            sd[f"{p}.self_attn.v_proj.weight"] = np.random.randn(n_embed, n_embed).astype(np.float32)
            sd[f"{p}.self_attn.o_proj.weight"] = np.random.randn(n_embed, n_embed).astype(np.float32)
            sd[f"{p}.post_attention_layernorm.weight"] = np.ones(n_embed, dtype=np.float32)
            sd[f"{p}.mlp.gate_proj.weight"] = np.random.randn(4 * n_embed, n_embed).astype(np.float32)
            sd[f"{p}.mlp.up_proj.weight"] = np.random.randn(4 * n_embed, n_embed).astype(np.float32)
            sd[f"{p}.mlp.down_proj.weight"] = np.random.randn(n_embed, 4 * n_embed).astype(np.float32)

        result = convert_hf_to_slonet(sd, n_layer=n_layer)
        assert "tok_emb.weight" in result
        for i in range(n_layer):
            assert f"blocks.{i}.ff.w1.weight" in result
            assert f"blocks.{i}.ff.w2.weight" in result
            assert f"blocks.{i}.ff.w3.weight" in result
