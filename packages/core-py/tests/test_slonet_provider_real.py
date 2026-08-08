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

# Quantine on tiny random weights can hit float32 scale overflows; these
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


# ── quantization via real Quantine ─────────────────────────────────────────


# These tests exercise the quantization *mechanics* (fresh quantize, npz
# reload, metadata reload). from_slnc() now skips quantization when the AVX2
# int8 GEMM kernel is unavailable, so we pin HAS_AVX2=True to force the
# quantized path regardless of the host machine.


@pytest.fixture
def avx2_available(monkeypatch):
    monkeypatch.setattr("domains.infrastructure.quant_core.wrapper.HAS_AVX2", True)


@pytest.fixture
def quantized_provider(slnc, tmp_path, monkeypatch, avx2_available):
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


def test_quantize_reload_prequantized_npz(slnc, tmp_path, monkeypatch, avx2_available):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    _build_tokenizer(str(tmp_path))
    first = SloNetChatProvider.from_slnc(slnc, model_id="gpt2", quantize=True)
    assert first.quantization_report()["quantized"] is True
    second = SloNetChatProvider.from_slnc(slnc, model_id="gpt2", quantize=True)
    assert second.quantization_report()["quantized"] is True


def test_quantize_reload_metadata_only(slnc, tmp_path, monkeypatch, avx2_available):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    _build_tokenizer(str(tmp_path))
    SloNetChatProvider.from_slnc(slnc, model_id="gpt2", quantize=True)
    npz = tmp_path / "tiny.slnc.quant.npz"
    json_meta = tmp_path / "tiny.slnc.quant.json"
    assert npz.exists() and json_meta.exists()
    npz.unlink()
    reloaded = SloNetChatProvider.from_slnc(slnc, model_id="gpt2", quantize=True)
    assert reloaded.quantization_report()["quantized"] is True


def test_fused_gemm_generation_bit_identical(quantized_provider, monkeypatch):
    """Fused [q;k;v] / [w1;w3] GEMMs must not change generated output.

    ``generate_numpy`` merges same-input quantized projections into single C
    calls; each fused output row depends only on its own weight/scale/bias and
    the shared per-token activation scale, so it is bit-identical to the
    unfused path. Monkeypatching ``_fuse_quant_weights`` to None forces the
    unfused inlined loop for a pure-numerics comparison.
    """
    model = quantized_provider._model
    tok = quantized_provider._tokenizer
    ids = np.array(tok.encode("hello world"), dtype=np.int64).flatten()

    def gen():
        return model.generate_numpy(ids, max_new_tokens=5, temperature=0.0, eos_token=-1)

    fused = gen()
    packs = [(b.attn._fused_qkv(), b.ff._fused_gate_up()) for b in model.blocks]
    assert any(p[0] is not None or p[1] is not None for p in packs), "fusion packs not built"

    monkeypatch.setattr("domains.training.slonet._fuse_quant_weights", lambda *a, **k: None)
    unfused = gen()
    np.testing.assert_array_equal(fused, unfused)


def test_quantize_skipped_without_avx2(slnc, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setattr("domains.infrastructure.quant_core.wrapper.HAS_AVX2", False)
    _build_tokenizer(str(tmp_path))
    provider = SloNetChatProvider.from_slnc(slnc, model_id="gpt2", quantize=True)
    assert provider.quantization_report() == {"quantized": False}
    assert provider.generate("hi", max_tokens=3, temperature=0.0)


# ── lazy_from_slnc (server autoload deferred-weight path) ─────────────────────


@pytest.fixture
def lazy_provider(slnc, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    _build_tokenizer(str(tmp_path))
    return SloNetChatProvider.lazy_from_slnc(slnc, model_id="gpt2")


def test_lazy_from_slnc_header_only_deferred(lazy_provider):
    """Construction reads only the .slnc header — no weights resident."""
    assert lazy_provider._model is None
    assert lazy_provider._parser is None
    assert lazy_provider._loaded is False
    md = lazy_provider.metadata()
    assert md["model_id"] == "gpt2"
    assert md["vocab_size"] == VOCAB
    assert md["total_params"] > 0
    assert lazy_provider.tokenize("hello")  # tokenizer is eager
    stats = lazy_provider.session_stats()
    assert stats["active_sessions"] == 0


def test_lazy_from_slnc_generate_matches_eager(slnc, tmp_path, monkeypatch, lazy_provider):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    _build_tokenizer(str(tmp_path))
    eager = SloNetChatProvider.from_slnc(slnc, model_id="gpt2")
    assert lazy_provider._model is None
    lazy_text = lazy_provider.generate("hello", max_tokens=8, temperature=0.0)
    eager_text = eager.generate("hello", max_tokens=8, temperature=0.0)
    assert isinstance(lazy_text, str) and lazy_text
    assert lazy_text == eager_text
    assert lazy_provider._loaded is True
    assert lazy_provider._model is not None
    assert lazy_provider.metadata()["lazy"] is True


def test_lazy_from_slnc_release_and_reload(lazy_provider):
    text = lazy_provider.generate("hi", max_tokens=4, temperature=0.0)
    assert lazy_provider._model is not None
    assert lazy_provider.release_model() is True
    assert lazy_provider._model is None
    assert lazy_provider.session_stats()["active_sessions"] == 0
    again = lazy_provider.generate("hi", max_tokens=4, temperature=0.0)
    assert again == text
    assert lazy_provider._model is not None


def test_lazy_from_slnc_release_before_load(lazy_provider):
    assert lazy_provider.release_model() is False
    text = lazy_provider.generate("hi", max_tokens=4, temperature=0.0)
    assert text


def test_lazy_from_slnc_chat_stream_cross_turn_kv(lazy_provider):
    async def _collect(session_id):
        return [t async for t in lazy_provider.chat_stream(
            [{"role": "user", "content": "hello"}], max_tokens=4, session_id=session_id)]

    asyncio.run(_collect("s1"))
    stats = lazy_provider.session_stats()
    assert stats["active_sessions"] == 1
    asyncio.run(_collect("s1"))
    stats = lazy_provider.session_stats()
    assert stats["active_sessions"] == 1
    assert stats["cached_tokens"] > 0
    asyncio.run(_collect("s2"))
    assert lazy_provider.session_stats()["active_sessions"] == 2
    assert lazy_provider.clear_session("s1") is True
    assert lazy_provider.session_stats()["active_sessions"] == 1
    assert lazy_provider.clear_all_sessions() == 1
