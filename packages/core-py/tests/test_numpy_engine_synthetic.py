"""Synthetic-model tests for NumpyEngine — no cached HF models required.

Covers the weight-loading, constructor, generation, and streaming paths that
the model-dependent fixtures (``gpt2``/``qwen2``) skip when models are not
cached locally and ``safetensors`` is unavailable. Builds a tiny GPT-2 shaped
model entirely from generated numpy arrays + a hand-written ``.safetensors``
file and a minimal ``tokenizer.json`` inside a fake HuggingFace cache.
"""

import json
import struct
import sys
import types

import numpy as np
import pytest

from domains.infrastructure.numpy_engine import KVCache, NumpyEngine, _load_weights
from domains.infrastructure.morph_tokenizer import MorphTokenizer

VOCAB = 64
N_EMBD = 16
N_HEAD = 4
N_LAYER = 2
N_CTX = 32
MODEL_ID = "tiny-test"
EOS = 63

CHARS = "abcdefghijklmnopqrstuvwxyz" + "".join(chr(i) for i in range(32, 32 + (VOCAB - 26)))


def _config():
    return {
        "architectures": ["GPT2LMHeadModel"],
        "vocab_size": VOCAB,
        "n_embd": N_EMBD,
        "n_head": N_HEAD,
        "n_layer": N_LAYER,
        "n_positions": N_CTX,
        "n_ctx": N_CTX,
        "n_inner": N_EMBD * 4,
        "_name_or_path": MODEL_ID,
    }


def _weights():
    rng = np.random.default_rng(0)
    w = {
        "wte.weight": rng.standard_normal((VOCAB, N_EMBD)).astype(np.float32),
        "wpe.weight": rng.standard_normal((N_CTX, N_EMBD)).astype(np.float32),
    }
    for i in range(N_LAYER):
        w[f"h.{i}.ln_1.weight"] = rng.standard_normal(N_EMBD).astype(np.float32) + 1.0
        w[f"h.{i}.ln_1.bias"] = rng.standard_normal(N_EMBD).astype(np.float32)
        w[f"h.{i}.attn.c_attn.weight"] = rng.standard_normal((N_EMBD, 3 * N_EMBD)).astype(np.float32)
        w[f"h.{i}.attn.c_attn.bias"] = rng.standard_normal(3 * N_EMBD).astype(np.float32)
        w[f"h.{i}.attn.c_proj.weight"] = rng.standard_normal((N_EMBD, N_EMBD)).astype(np.float32)
        w[f"h.{i}.attn.c_proj.bias"] = rng.standard_normal(N_EMBD).astype(np.float32)
        w[f"h.{i}.ln_2.weight"] = rng.standard_normal(N_EMBD).astype(np.float32) + 1.0
        w[f"h.{i}.ln_2.bias"] = rng.standard_normal(N_EMBD).astype(np.float32)
        w[f"h.{i}.mlp.c_fc.weight"] = rng.standard_normal((N_EMBD, N_EMBD * 4)).astype(np.float32)
        w[f"h.{i}.mlp.c_fc.bias"] = rng.standard_normal(N_EMBD * 4).astype(np.float32)
        w[f"h.{i}.mlp.c_proj.weight"] = rng.standard_normal((N_EMBD * 4, N_EMBD)).astype(np.float32)
        w[f"h.{i}.mlp.c_proj.bias"] = rng.standard_normal(N_EMBD).astype(np.float32)
    w["ln_f.weight"] = rng.standard_normal(N_EMBD).astype(np.float32) + 1.0
    w["ln_f.bias"] = rng.standard_normal(N_EMBD).astype(np.float32)
    return w


def _tokenizer():
    return MorphTokenizer(vocab={c: i for i, c in enumerate(CHARS)},
                          merges=[], eos_token_id=EOS, byte_level=False)


def _write_safetensors(path, tensors, dtype="F32"):
    header = {"__metadata__": {"format": "pt"}}
    blobs = {}
    offset = 0
    for name, arr in tensors.items():
        f32 = np.asarray(arr, dtype=np.float32)
        if dtype == "BF16":
            raw = (f32.view(np.uint32) >> 16).astype(np.uint16).tobytes()
        elif dtype == "F16":
            raw = f32.astype(np.float16).tobytes()
        else:
            raw = f32.tobytes()
        header[name] = {"dtype": dtype, "shape": list(f32.shape),
                        "data_offsets": [offset, offset + len(raw)]}
        blobs[name] = raw
        offset += len(raw)
    header_bytes = json.dumps(header).encode()
    while len(header_bytes) % 8 != 0:
        header_bytes += b" "
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for raw in blobs.values():
            f.write(raw)


def _read_safetensors(path):
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        out = {}
        for key, info in header.items():
            if key.startswith("__"):
                continue
            start, end = info["data_offsets"]
            f.seek(8 + header_len + start)
            raw = f.read(end - start)
            shape = info["shape"]
            dtype = info["dtype"]
            if dtype == "BF16":
                u16 = np.frombuffer(raw, dtype=np.uint16)
                f32 = np.zeros(len(u16), dtype=np.float32)
                f32.view(np.uint32)[:] = u16.astype(np.uint32) << 16
                out[key] = f32.reshape(shape)
            elif dtype == "F16":
                out[key] = np.frombuffer(raw, dtype=np.float16).reshape(shape).astype(np.float32)
            else:
                out[key] = np.frombuffer(raw, dtype=np.float32).reshape(shape)
        return out


def _write_tokenizer_json(path):
    with open(path, "w") as f:
        json.dump({"model": {"vocab": {c: i for i, c in enumerate(CHARS)},
                             "merges": [], "eos_token_id": EOS}},
                  f)


class _FakeSafeFile:
    def __init__(self, path):
        self._tensors = _read_safetensors(path)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def keys(self):
        return list(self._tensors.keys())

    def get_tensor(self, key):
        return self._tensors[key]


def _install_fake_safetensors(monkeypatch, fail=False):
    mod = types.ModuleType("safetensors")

    def safe_open(path, framework="numpy"):
        if fail:
            raise RuntimeError("fake safe_open failure")
        return _FakeSafeFile(path)

    mod.safe_open = safe_open
    monkeypatch.setitem(sys.modules, "safetensors", mod)


@pytest.fixture
def hf_cache(tmp_path, monkeypatch):
    hub = tmp_path / "hub"
    snap = hub / f"models--{MODEL_ID}" / "snapshots" / "abc123"
    snap.mkdir(parents=True, exist_ok=True)
    with open(snap / "config.json", "w") as f:
        json.dump(_config(), f)
    _write_safetensors(snap / "model.safetensors", _weights(), dtype="F32")
    _write_tokenizer_json(snap / "tokenizer.json")
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    return hub


class TestLoadWeights:
    """Coverage for _load_weights (mmap + bfloat16-fallback + config discovery)."""

    def test_load_mmap_success(self, hf_cache, monkeypatch):
        _install_fake_safetensors(monkeypatch, fail=False)
        config, weights = _load_weights(MODEL_ID)
        assert config["vocab_size"] == VOCAB
        assert "wte.weight" in weights
        assert weights["wte.weight"].shape == (VOCAB, N_EMBD)

    def test_load_config_at_model_root(self, tmp_path, monkeypatch):
        _install_fake_safetensors(monkeypatch, fail=False)
        model_dir = tmp_path / "hub" / f"models--{MODEL_ID}"
        model_dir.mkdir(parents=True)
        with open(model_dir / "config.json", "w") as f:
            json.dump(_config(), f)
        _write_safetensors(model_dir / "model.safetensors", _weights(), dtype="F32")
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        config, weights = _load_weights(MODEL_ID)
        assert config["n_embd"] == N_EMBD
        assert len(weights) == len(_weights())

    def test_load_fallback_f32(self, hf_cache, monkeypatch):
        _install_fake_safetensors(monkeypatch, fail=True)
        config, weights = _load_weights(MODEL_ID)
        assert weights["wte.weight"].shape == (VOCAB, N_EMBD)

    def test_load_fallback_f16(self, hf_cache, monkeypatch):
        _install_fake_safetensors(monkeypatch, fail=True)
        hub = hf_cache
        st = hub / f"models--{MODEL_ID}" / "snapshots" / "abc123" / "model.safetensors"
        _write_safetensors(st, _weights(), dtype="F16")
        _, weights = _load_weights(MODEL_ID)
        assert weights["wte.weight"].dtype == np.float32

    def test_load_fallback_bf16(self, hf_cache, monkeypatch):
        _install_fake_safetensors(monkeypatch, fail=True)
        hub = hf_cache
        st = hub / f"models--{MODEL_ID}" / "snapshots" / "abc123" / "model.safetensors"
        _write_safetensors(st, _weights(), dtype="BF16")
        _, weights = _load_weights(MODEL_ID)
        assert weights["wte.weight"].dtype == np.float32

    def test_load_missing_model_dir(self):
        with pytest.raises(FileNotFoundError):
            _load_weights("nonexistent/model-xyz")

    def test_load_missing_safetensors(self, tmp_path, monkeypatch):
        model_dir = tmp_path / "hub" / f"models--{MODEL_ID}"
        model_dir.mkdir(parents=True)
        with open(model_dir / "config.json", "w") as f:
            json.dump(_config(), f)
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        with pytest.raises(FileNotFoundError):
            _load_weights(MODEL_ID)

    def test_load_missing_config(self, tmp_path, monkeypatch):
        snap = tmp_path / "hub" / f"models--{MODEL_ID}" / "snapshots" / "abc123"
        snap.mkdir(parents=True)
        _write_safetensors(snap / "model.safetensors", _weights(), dtype="F32")
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        with pytest.raises(FileNotFoundError):
            _load_weights(MODEL_ID)


class TestKVCacheGet:
    def test_get_returns_tuple_after_update(self):
        cache = KVCache(n_layers=2)
        k = np.zeros((4, 2, 8), dtype=np.float32)
        v = np.zeros((4, 2, 8), dtype=np.float32)
        cache.update(0, k, v)
        result = cache.get(0)
        assert result is not None
        k_cat, v_cat = result
        assert k_cat.shape == (4, 2, 8)
        assert v_cat.shape == (4, 2, 8)

    def test_get_returns_none_before_update(self):
        cache = KVCache(n_layers=2)
        assert cache.get(1) is None
        assert cache.seq_len == 0


class TestNumpyEngineConstructor:
    def test_no_compression_stores_raw(self):
        engine = NumpyEngine(config=_config(), weights=_weights(), compress=False)
        assert len(engine._raw_weights) == len(_weights())
        assert not engine._compressed_weights

    def test_get_weight_raw_fallback_and_cache_hit(self):
        engine = NumpyEngine(config=_config(), weights=_weights(), compress=False)
        w1 = engine._get_weight("wte.weight")
        w2 = engine._get_weight("wte.weight")
        assert np.array_equal(w1, w2)
        assert engine._cache.get("wte.weight") is not None

    def test_get_weight_keyerror(self):
        engine = NumpyEngine(config=_config(), weights=_weights(), compress=False)
        with pytest.raises(KeyError):
            engine._get_weight("does.not.exist")

    def test_forward_returns_logits(self):
        engine = NumpyEngine(config=_config(), weights=_weights(),
                             tokenizer=_tokenizer(), compress=False)
        logits = engine._forward([0, 1, 2])
        assert logits.shape == (VOCAB,)

    def test_forward_with_kv_cache(self):
        engine = NumpyEngine(config=_config(), weights=_weights(),
                             tokenizer=_tokenizer(), compress=False)
        cache = KVCache(engine.arch.n_layers)
        logits = engine._forward([0, 1, 2], kv_cache=cache, start_pos=0)
        assert logits.shape == (VOCAB,)
        assert cache.seq_len == 3

    def test_small_weight_stored_raw(self):
        weights = {"tiny": np.ones((4, 4), dtype=np.float32)}
        engine = NumpyEngine(config=_config(), weights=weights, compress=True)
        assert "tiny" in engine._raw_weights

    def test_info_compression_ratio_zero_safe(self):
        engine = NumpyEngine(config=_config(), weights={}, compress=False)
        info = engine.info()
        assert info["compression_ratio"] == 0.0

    def test_compression_linear_centroids(self):
        smooth = np.linspace(0, 1, 4000, dtype=np.float32).reshape(80, 50)
        engine = NumpyEngine(config=_config(), weights={"smooth.linear": smooth},
                             compress=True, n_clusters=16)
        cw = engine._compressed_weights["smooth.linear"]
        assert cw.centroid_fn == "linear"
        decomp = engine._get_weight("smooth.linear")
        assert decomp.shape == (80, 50)
        assert engine._cache.get("smooth.linear") is not None

    def test_get_weight_decompresses_compressed(self):
        engine = NumpyEngine(config=_config(), weights=_weights(),
                             compress=True, n_clusters=16)
        name = "h.0.attn.c_attn.weight"
        w = engine._get_weight(name)
        assert w.shape == (N_EMBD, 3 * N_EMBD)
        assert engine._cache.get(name) is not None

    def test_get_weight_via_model_tree(self, tmp_path):
        from domains.infrastructure.point_compressor import ModelTree, PointLibrary
        library = PointLibrary(name="tiny-lib", storage_dir=tmp_path / "lib")
        tree = ModelTree(MODEL_ID, library, n_clusters=16)
        engine = NumpyEngine(config=_config(), weights=_weights(),
                             compress=False, model_tree=tree)
        w = engine._get_weight("wte.weight")
        assert w.shape == (VOCAB, N_EMBD)


class TestFromPretrained:
    def test_default_tokenizer(self, hf_cache, monkeypatch):
        _install_fake_safetensors(monkeypatch, fail=False)
        engine = NumpyEngine.from_pretrained(MODEL_ID)
        assert engine.tokenizer is not None
        assert engine.vocab_size == VOCAB

    def test_explicit_tokenizer(self, hf_cache, monkeypatch):
        _install_fake_safetensors(monkeypatch, fail=False)
        engine = NumpyEngine.from_pretrained(MODEL_ID, tokenizer=_tokenizer())
        assert engine.vocab_size == VOCAB

    def test_use_points(self, hf_cache, monkeypatch, tmp_path):
        _install_fake_safetensors(monkeypatch, fail=False)
        from domains.infrastructure.point_compressor import PointLibrary
        library = PointLibrary(name="tiny-lib", storage_dir=tmp_path / "lib")
        engine = NumpyEngine.from_pretrained(MODEL_ID, tokenizer=_tokenizer(),
                                             use_points=True, library=library)
        assert engine._model_tree is not None
        assert library.stats()["total_points"] > 0

    def test_use_points_default_library(self, hf_cache, monkeypatch):
        _install_fake_safetensors(monkeypatch, fail=False)
        engine = NumpyEngine.from_pretrained(MODEL_ID, tokenizer=_tokenizer(),
                                             use_points=True)
        assert engine._model_tree is not None
        assert engine._model_tree.library.name == MODEL_ID.replace("/", "_")


class TestFromSlnc:
    def test_from_slnc_with_tokenizer(self, tmp_path):
        from domains.infrastructure.slnc.compiler import SLNCCompiler
        slnc = tmp_path / "tiny.slnc"
        SLNCCompiler().compile_from_dict(_config(), _weights(), str(slnc))
        engine = NumpyEngine.from_slnc(str(slnc), tokenizer=_tokenizer())
        assert engine._parser is not None
        assert engine.vocab_size == VOCAB
        assert engine._forward([1, 2, 3]).shape == (VOCAB,)

    def test_from_slnc_default_tokenizer(self, hf_cache, tmp_path):
        from domains.infrastructure.slnc.compiler import SLNCCompiler
        slnc = tmp_path / "tiny.slnc"
        SLNCCompiler().compile_from_dict(_config(), _weights(), str(slnc))
        engine = NumpyEngine.from_slnc(str(slnc))
        assert engine.tokenizer is not None


class TestGenerate:
    def test_generate_no_tokenizer(self):
        engine = NumpyEngine(config=_config(), weights=_weights(), compress=False)
        with pytest.raises(RuntimeError):
            engine.generate("hello")

    def test_generate_greedy_kv_cache(self):
        engine = NumpyEngine(config=_config(), weights=_weights(),
                             tokenizer=_tokenizer(), compress=False)
        result = engine.generate("hello", max_new_tokens=5, temperature=0.0, top_k=5)
        assert isinstance(result, str)
        assert result.startswith("hello")

    def test_generate_no_kv_cache(self):
        engine = NumpyEngine(config=_config(), weights=_weights(),
                             tokenizer=_tokenizer(), compress=False)
        result = engine.generate("hello", max_new_tokens=5, temperature=0.0,
                                 use_kv_cache=False)
        assert isinstance(result, str)

    def test_generate_sampling(self):
        engine = NumpyEngine(config=_config(), weights=_weights(),
                             tokenizer=_tokenizer(), compress=False)
        result = engine.generate("hello", max_new_tokens=5, temperature=1.0, top_k=5)
        assert isinstance(result, str)

    def test_generate_eos_stops(self, monkeypatch):
        engine = NumpyEngine(config=_config(), weights=_weights(),
                             tokenizer=_tokenizer(), compress=False)
        logits = np.full(VOCAB, -1e9, dtype=np.float32)
        logits[EOS] = 10.0
        monkeypatch.setattr(engine, "_forward",
                            lambda token_ids, kv_cache=None, start_pos=0: logits)
        result = engine.generate("hello", max_new_tokens=10, temperature=0.0)
        assert result == "hello"
        assert engine._kv_cache is None


class TestGenerateStream:
    async def test_stream_no_tokenizer(self):
        engine = NumpyEngine(config=_config(), weights=_weights(), compress=False)
        with pytest.raises(RuntimeError):
            async for _ in engine.generate_stream("hello"):
                pass

    async def test_stream_yields_tokens(self):
        engine = NumpyEngine(config=_config(), weights=_weights(),
                             tokenizer=_tokenizer(), compress=False)
        tokens = []
        async for t in engine.generate_stream("hello", max_new_tokens=3, temperature=0.0):
            tokens.append(t)
        assert len(tokens) == 3
        assert all(isinstance(t, str) for t in tokens)
        assert engine._kv_cache is None

    async def test_stream_sampling(self):
        engine = NumpyEngine(config=_config(), weights=_weights(),
                             tokenizer=_tokenizer(), compress=False)
        tokens = []
        async for t in engine.generate_stream("hello", max_new_tokens=2,
                                              temperature=1.0, top_k=5):
            tokens.append(t)
        assert len(tokens) == 2

    async def test_stream_eos_stops(self, monkeypatch):
        engine = NumpyEngine(config=_config(), weights=_weights(),
                             tokenizer=_tokenizer(), compress=False)
        logits = np.full(VOCAB, -1e9, dtype=np.float32)
        logits[EOS] = 10.0
        monkeypatch.setattr(engine, "_forward",
                            lambda token_ids, kv_cache=None, start_pos=0: logits)
        tokens = []
        async for t in engine.generate_stream("hello", max_new_tokens=10, temperature=0.0):
            tokens.append(t)
        assert tokens == []
        assert engine._kv_cache is None
