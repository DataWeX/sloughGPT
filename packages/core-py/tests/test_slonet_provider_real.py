"""Wave M: real-component coverage for SloNetChatProvider.

Every test uses real programmatic inputs only:
- a hand-built real ``.slnc`` file (per the spec in ``slnc/spec.py``)
- a real ``tokenizer.json`` served from a temp ``HF_HOME`` cache layout
- a real ``SloTransformer`` / ``MorphTokenizer`` / ``SloNetServer``

No mocks, no stubs, no third-party installs.
"""

import asyncio
import ctypes
import json
import os
import struct
import zlib

import numpy as np
import pytest

from domains.infrastructure.slnc.spec import compute_header_size
from domains.inference.slonet_provider import (
    SloNetChatProvider,
    _get_slo_layernorm,
    _split_fused_qkv,
)

# QuantEngine on tiny random weights can hit float32 scale overflows; these
# RuntimeWarnings come from real quantization math, not test failures.
pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

N_LAYER, N_EMBD, N_HEAD, N_INNER, VOCAB, NPOS = 2, 32, 4, 128, 64, 64
_CHARS = "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?"


def _hf_weights(seed: int = 0):
    rng = np.random.RandomState(seed)
    w = {}
    for i in range(N_LAYER):
        w[f"h.{i}.ln_1.weight"] = rng.randn(N_EMBD).astype(np.float32)
        w[f"h.{i}.ln_1.bias"] = rng.randn(N_EMBD).astype(np.float32)
        w[f"h.{i}.attn.c_attn.weight"] = rng.randn(N_EMBD, 3 * N_EMBD).astype(np.float32)
        w[f"h.{i}.attn.c_attn.bias"] = rng.randn(3 * N_EMBD).astype(np.float32)
        w[f"h.{i}.attn.c_proj.weight"] = rng.randn(N_EMBD, N_EMBD).astype(np.float32)
        w[f"h.{i}.attn.c_proj.bias"] = rng.randn(N_EMBD).astype(np.float32)
        w[f"h.{i}.ln_2.weight"] = rng.randn(N_EMBD).astype(np.float32)
        w[f"h.{i}.ln_2.bias"] = rng.randn(N_EMBD).astype(np.float32)
        w[f"h.{i}.mlp.c_fc.weight"] = rng.randn(N_EMBD, N_INNER).astype(np.float32)
        w[f"h.{i}.mlp.c_fc.bias"] = rng.randn(N_INNER).astype(np.float32)
        w[f"h.{i}.mlp.c_proj.weight"] = rng.randn(N_INNER, N_EMBD).astype(np.float32)
        w[f"h.{i}.mlp.c_proj.bias"] = rng.randn(N_EMBD).astype(np.float32)
    w["wte.weight"] = rng.randn(VOCAB, N_EMBD).astype(np.float32)
    w["wpe.weight"] = rng.randn(NPOS, N_EMBD).astype(np.float32)
    w["ln_f.weight"] = rng.randn(N_EMBD).astype(np.float32)
    w["ln_f.bias"] = rng.randn(N_EMBD).astype(np.float32)
    return w


def _build_slnc(path, weights, config):
    json_bytes = json.dumps(config, sort_keys=True).encode()
    header_size = compute_header_size(json_bytes)
    table = b""
    entries = []
    cur = header_size
    for name, arr in weights.items():
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        nb = name.encode()
        ndim = arr.ndim
        crc = zlib.crc32(arr.tobytes()) & 0xFFFFFFFF
        table += struct.pack("<I", len(nb)) + nb
        table += struct.pack("<Q", cur)
        table += struct.pack("<I", arr.nbytes)
        table += struct.pack("<I", ndim)
        table += struct.pack(f"<{ndim}I", *arr.shape)
        table += struct.pack("<I", 0)
        table += struct.pack("<I", crc)
        cur += arr.nbytes
        entries.append(arr)
    data_offset = header_size + len(table)
    meta = struct.pack(
        "<10I",
        N_LAYER, N_EMBD, N_HEAD, N_INNER, VOCAB, NPOS,
        N_LAYER, NPOS, len(weights), data_offset,
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


def _build_tokenizer(hf_home, model_id="gpt2"):
    vocab = {ch: i for i, ch in enumerate(_CHARS)}
    eos_id = len(vocab)
    vocab["<|endoftext|>"] = eos_id
    tok = {
        "model": {
            "type": "BPE",
            "vocab": vocab,
            "merges": ["h e", "he l", "hel lo", "w o"],
            "eos_token_id": eos_id,
        },
        "pre_tokenizer": {"type": "ByteLevel"},
    }
    slug = model_id.replace("/", "--")
    d = os.path.join(hf_home, "hub", f"models--{slug}", "snapshots", "deadbeef")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "tokenizer.json"), "w") as f:
        json.dump(tok, f)
    return eos_id


@pytest.fixture
def slnc(tmp_path):
    config = {
        "architectures": ["GPT2LMHeadModel"],
        "n_embd": N_EMBD,
        "n_head": N_HEAD,
        "n_layer": N_LAYER,
        "vocab_size": VOCAB,
        "n_inner": N_INNER,
        "n_positions": NPOS,
        "n_ctx": NPOS,
        "hidden_act": "gelu",
        "layer_norm_type": "layer_norm",
    }
    path = tmp_path / "tiny.slnc"
    _build_slnc(path, _hf_weights(), config)
    return str(path)


@pytest.fixture
def provider(slnc, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    _build_tokenizer(str(tmp_path))
    return SloNetChatProvider.from_slnc(slnc, model_id="gpt2")


# ── Construction ──────────────────────────────────────────────────────────────


def test_from_slnc_real_components(provider):
    assert provider.model_id == "gpt2"
    assert provider.capabilities.chat is True
    assert provider.capabilities.streaming is True
    assert provider.capabilities.embedding is False
    assert provider.quantization_report() == {"quantized": False}
    assert provider.get_server() is None


def test_from_slnc_generate_embed_tokens(provider):
    text = provider.generate("hello", max_tokens=8, temperature=0.0)
    assert isinstance(text, str) and text
    emb = provider.embed("hello")
    assert emb.shape == (N_EMBD,)
    assert emb.dtype == np.float32
    assert provider.tokenize("ab") == [provider.tokenize("a")[0], provider.tokenize("b")[0]]
    assert provider.detokenize(provider.tokenize("a")) == "a"
    assert provider.count_tokens("hello world") > 0


def test_from_slnc_max_new_tokens_overrides(provider):
    text = provider.generate("hi", max_new_tokens=3, temperature=0.0)
    assert isinstance(text, str)


def test_from_slnc_metadata(provider):
    md = provider.metadata()
    assert md["model_id"] == "gpt2"
    assert md["vocab_size"] == VOCAB


def test_init_raises_typeerror():
    with pytest.raises(TypeError):
        SloNetChatProvider("gpt2")


def test_get_slo_layernorm_caches_class():
    from domains.training.slonet import SloLayerNorm
    assert _get_slo_layernorm() is SloLayerNorm
    assert _get_slo_layernorm() is SloLayerNorm


# ── _split_fused_qkv edge paths ───────────────────────────────────────────────


def test_split_fused_qkv_non_integer_layer_returns_empty():
    arr = np.zeros((N_EMBD, 3 * N_EMBD), dtype=np.float32)
    assert _split_fused_qkv("h.abc.attn.c_attn.weight", arr, N_EMBD, N_LAYER, {}) == {}


def test_split_fused_qkv_unknown_key_returns_empty():
    arr = np.zeros((N_EMBD, 3 * N_EMBD), dtype=np.float32)
    assert _split_fused_qkv("not.a.qkv.key", arr, N_EMBD, N_LAYER, {}) == {}


# ── _load_safetensors_bf16 ────────────────────────────────────────────────────


def _write_safetensors(path):
    bf16 = np.array([0x3F80, 0xC000], dtype=np.uint16).tobytes()
    f32 = np.array([1.5, -2.5], dtype=np.float32).tobytes()
    f16 = np.array([1.25, -3.0], dtype=np.float16).tobytes()
    unknown = np.array([1.0, 2.0], dtype=np.float32).tobytes()
    header = {
        "__metadata__": {"format": "pt"},
        "bf16_tensor": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]},
        "f32_tensor": {"dtype": "F32", "shape": [2], "data_offsets": [4, 12]},
        "f16_tensor": {"dtype": "F16", "shape": [2], "data_offsets": [12, 16]},
        "unknown_tensor": {"dtype": "I8", "shape": [2], "data_offsets": [16, 24]},
    }
    hdr = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hdr)))
        f.write(hdr)
        f.write(bf16 + f32 + f16 + unknown)


def test_load_safetensors_bf16_real_file(tmp_path):
    path = tmp_path / "model.safetensors"
    _write_safetensors(path)
    weights = SloNetChatProvider._load_safetensors_bf16(path)
    np.testing.assert_allclose(weights["bf16_tensor"], [1.0, -2.0], rtol=1e-6)
    np.testing.assert_allclose(weights["f32_tensor"], [1.5, -2.5], rtol=1e-6)
    np.testing.assert_allclose(weights["f16_tensor"], [1.25, -3.0], rtol=1e-6)
    np.testing.assert_allclose(weights["unknown_tensor"], [1.0, 2.0], rtol=1e-6)
    assert "__metadata__" not in weights


# ── _build_prompt / _load_tokenizer ───────────────────────────────────────────


def test_build_prompt_empty_and_legacy(provider):
    assert provider._build_prompt([]) == ""
    assert provider._build_prompt("hi") == "hi"
    assert provider._build_prompt(["first", "second"]) == "second"


def test_build_prompt_uses_chat_template(provider):
    prompt = provider._build_prompt([{"role": "user", "content": "hello"}])
    assert "hello" in prompt


def test_load_tokenizer_missing_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    inst = SloNetChatProvider.__new__(SloNetChatProvider)
    inst._hf_model_id = "this-model-does-not-exist-12345"
    with pytest.raises(RuntimeError):
        inst._load_tokenizer(str(tmp_path), {})


# ── server attach + chat/chat_stream ─────────────────────────────────────────


def test_set_get_server(provider):
    server = provider.to_server(enable_warmup=False)
    provider.set_server(server)
    assert provider.get_server() is server
    provider.set_server(None)
    assert provider.get_server() is None


def test_chat_server_attached(provider):
    server = provider.to_server(enable_warmup=False)
    provider.set_server(server)
    result = asyncio.run(provider.chat([{"role": "user", "content": "hello"}], max_tokens=5))
    assert isinstance(result, str) and result


def test_chat_stream_server_attached(provider):
    server = provider.to_server(enable_warmup=False)
    provider.set_server(server)

    async def _collect():
        return [t async for t in provider.chat_stream(
            [{"role": "user", "content": "hello"}], max_tokens=5)]

    tokens = asyncio.run(_collect())
    assert isinstance(tokens, list) and tokens


def test_chat_no_server_uses_sync_thread(provider):
    result = asyncio.run(provider.chat([{"role": "user", "content": "hello"}], max_tokens=5))
    assert isinstance(result, str) and result


def test_generate_sync_direct(provider):
    text = provider._generate_sync([{"role": "user", "content": "hello"}], max_tokens=5)
    assert isinstance(text, str)


def test_chat_stream_no_server_real_streaming(provider):
    async def _collect():
        return [t async for t in provider.chat_stream(
            [{"role": "user", "content": "hello"}], max_tokens=5)]

    tokens = asyncio.run(_collect())
    assert isinstance(tokens, list) and tokens


# ── generation features ───────────────────────────────────────────────────────


def test_generate_with_logprobs_seed(provider):
    text, lps = provider.generate_with_logprobs("hello", max_tokens=4, seed=1)
    assert isinstance(text, str)
    assert len(lps) == 4
    assert all("token_id" in e and "logprob" in e for e in lps)


def test_generate_with_stop_list(provider):
    text = provider.generate_with_stop("hello", max_tokens=20, stop=["o"], temperature=0.0)
    assert isinstance(text, str)


def test_generate_batch_real(provider):
    results = provider.generate_batch(["a", "b"], max_tokens=3, temperature=0.0)
    assert len(results) == 2


# ── quantization via real QuantEngine ─────────────────────────────────────────


@pytest.fixture
def quantized_provider(slnc, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    _build_tokenizer(str(tmp_path))
    return SloNetChatProvider.from_slnc(slnc, model_id="gpt2", quantize=True)


def test_quantize_fresh_and_report(quantized_provider):
    report = quantized_provider.quantization_report()
    assert report["quantized"] is True
    assert report["bits"] == 8
    assert report["mode"] == "symmetric"
    assert report["summary"] is not None
    assert quantized_provider.generate("hi", max_tokens=3, temperature=0.0)


def test_quantize_reload_prequantized_npz(slnc, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    _build_tokenizer(str(tmp_path))
    first = SloNetChatProvider.from_slnc(slnc, model_id="gpt2", quantize=True)
    assert first.quantization_report()["quantized"] is True
    second = SloNetChatProvider.from_slnc(slnc, model_id="gpt2", quantize=True)
    assert second.quantization_report()["quantized"] is True


def test_quantize_reload_metadata_only(slnc, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    _build_tokenizer(str(tmp_path))
    SloNetChatProvider.from_slnc(slnc, model_id="gpt2", quantize=True)
    npz = tmp_path / "tiny.slnc.quant.npz"
    json_meta = tmp_path / "tiny.slnc.quant.json"
    assert npz.exists() and json_meta.exists()
    npz.unlink()
    reloaded = SloNetChatProvider.from_slnc(slnc, model_id="gpt2", quantize=True)
    assert reloaded.quantization_report()["quantized"] is True
