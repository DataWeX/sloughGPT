"""
Tests for slonet_provider — tokenizers, config, provider wiring, generation logic.

FEATURE: slonet-provider-tests — pure logic only, no external API mocks.
"""
import math
import struct
import threading
import time

import numpy as np
import pytest

from domains.inference.slonet_provider import (
    SloNetChatProvider,
    _CharTokenizer,
    _TreeTokenizer,
    _split_fused_qkv,
    _ARCH_TO_SLONET_SHARED,
    _ARCH_TO_SLONET_SWIGLU,
    _ARCH_TO_SLONET_GELU,
    convert_hf_to_slonet,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

class MockTokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.eos_token_id = 2
        self.pad_token_id = 0
        self.chat_stop_ids = lambda: []

    def encode(self, text):
        return [abs(hash(c)) % self.vocab_size for c in text[:20]]

    def decode(self, token_ids):
        return "".join(chr(65 + (tid % 26)) for tid in token_ids)

    def apply_chat_template(self, messages):
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        parts.append("assistant:")
        return "\n".join(parts)


class MockModel:
    def __init__(self, vocab_size=1000, max_seq_len=512):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self._config = {"n_embd": 64, "n_head": 4, "n_layer": 2, "vocab_size": vocab_size}
        self._counter = 0

    @property
    def layers(self):
        return [MagicMock()] * 4

    def parameters(self):
        for layer in self.layers:
            if hasattr(layer, "weight"):
                yield layer.weight

    def forward_pass(self, input_ids):
        from domains.inference.forward_pass import ForwardPassResult
        batch, seq_len = input_ids.shape
        logits = np.random.randn(batch, seq_len, self.vocab_size)
        return ForwardPassResult(logits=logits, engine="mock")

    def generate_numpy_stream(self, input_ids, max_new_tokens=50, eos_token=0,
                               temperature=1.0, top_k=None, top_p=None,
                               repetition_penalty=1.0, extra_stop_ids=None, kv_state=None,
                               return_logprobs=False):
        stop_ids = {eos_token} | set(extra_stop_ids or ())
        for i in range(max_new_tokens):
            tok = (self._counter + i) % self.vocab_size
            self._counter += 1
            if tok in stop_ids:
                break
            if return_logprobs:
                yield tok, np.random.randn(self.vocab_size)
            else:
                yield tok

    def generate_numpy(self, input_ids, max_new_tokens=50, temperature=1.0,
                        top_k=None, top_p=None, repetition_penalty=1.0,
                        eos_token=0, extra_stop_ids=None, kv_state=None):
        tokens = list(self.generate_numpy_stream(
            input_ids, max_new_tokens, eos_token, temperature, top_k, top_p,
            repetition_penalty, extra_stop_ids,
        ))
        if not tokens:
            return input_ids
        return np.array([tokens], dtype=np.int64)


from unittest.mock import MagicMock


def _make_provider(**overrides):
    p = SloNetChatProvider.__new__(SloNetChatProvider)
    p._hf_model_id = overrides.get("_hf_model_id", "test-model")
    p._model_id = overrides.get("_model_id", "test-model")
    p._device = "cpu"
    p._model = overrides.get("_model", MockModel())
    p._tokenizer = overrides.get("_tokenizer", MockTokenizer())
    p._quant_engine = overrides.get("_quant_engine", None)
    p._parser = overrides.get("_parser", None)
    p._slnc_path = overrides.get("_slnc_path", "/tmp/fake.slnc")
    p._kv_states = {}
    p._kv_last_access = {}
    p._kv_ttl = 3600.0
    p._kv_max_sessions = overrides.get("_kv_max_sessions", 64)
    p._kv_lock = threading.Lock()
    p._server = None
    return p


# ═══════════════════════════════════════════════════════════════════════════════
# _CharTokenizer
# ═══════════════════════════════════════════════════════════════════════════════

class TestCharTokenizer:
    def test_encode_single_chars(self):
        tok = _CharTokenizer({"a": 1, "b": 2, "c": 3}, {1: "a", 2: "b", 3: "c"})
        assert tok.encode("abc") == [1, 2, 3]

    def test_encode_unknown_char_returns_zero(self):
        tok = _CharTokenizer({"a": 1}, {1: "a"})
        assert tok.encode("z") == [0]

    def test_encode_empty_string(self):
        tok = _CharTokenizer({"a": 1}, {1: "a"})
        assert tok.encode("") == []

    def test_decode_returns_string(self):
        tok = _CharTokenizer({"a": 1, "b": 2}, {1: "a", 2: "b"})
        assert tok.decode([1, 2]) == "ab"

    def test_decode_unknown_id_returns_empty(self):
        tok = _CharTokenizer({"a": 1}, {1: "a"})
        assert tok.decode([99]) == ""

    def test_eos_token_id_defaults_to_newline(self):
        tok = _CharTokenizer({"a": 1, "\n": 5}, {1: "a", 5: "\n"})
        assert tok.eos_token_id == 5

    def test_eos_token_id_zero_when_no_newline(self):
        tok = _CharTokenizer({"a": 1}, {1: "a"})
        assert tok.eos_token_id == 0

    def test_pad_token_id_is_zero(self):
        tok = _CharTokenizer({"a": 1}, {1: "a"})
        assert tok.pad_token_id == 0

    def test_vocab_size(self):
        tok = _CharTokenizer({"a": 1, "b": 5}, {1: "a", 5: "b"})
        assert tok.vocab_size == 6  # max(1,5)+1

    def test_vocab_size_empty(self):
        tok = _CharTokenizer({}, {})
        assert tok.vocab_size == 0

    def test_apply_chat_template_single_message(self):
        tok = _CharTokenizer({"a": 1}, {1: "a"})
        result = tok.apply_chat_template([{"role": "user", "content": "hi"}])
        assert result == "user: hi\nassistant:"

    def test_apply_chat_template_multiple_messages(self):
        tok = _CharTokenizer({"a": 1}, {1: "a"})
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "bye"},
        ]
        result = tok.apply_chat_template(msgs)
        assert "user: hi" in result
        assert "assistant: hello" in result
        assert "user: bye" in result
        assert result.endswith("assistant:")

    def test_apply_chat_template_empty_messages(self):
        tok = _CharTokenizer({"a": 1}, {1: "a"})
        result = tok.apply_chat_template([])
        assert result == "assistant:"

    def test_apply_chat_template_ignores_extra_kwargs(self):
        tok = _CharTokenizer({"a": 1}, {1: "a"})
        result = tok.apply_chat_template([{"role": "user", "content": "x"}], do_sample=True)
        assert "user: x" in result

    def test_json_int_keys_normalised(self):
        """JSON serializes int keys as strings — tokenizer normalizes them."""
        tok = _CharTokenizer({"a": 1}, {"1": "a"})
        assert tok.decode([1]) == "a"

    def test_roundtrip(self):
        tok = _CharTokenizer({"h": 1, "i": 2}, {1: "h", 2: "i"})
        ids = tok.encode("hi")
        text = tok.decode(ids)
        assert text == "hi"


# ═══════════════════════════════════════════════════════════════════════════════
# _TreeTokenizer
# ═══════════════════════════════════════════════════════════════════════════════

class TestTreeTokenizer:
    def test_eos_and_pad_from_tree(self):
        tree = MagicMock()
        tree.eos_id = 99
        tree.pad_id = 7
        tok = _TreeTokenizer(tree)
        assert tok.eos_token_id == 99
        assert tok.pad_token_id == 7

    def test_vocab_size_delegates(self):
        tree = MagicMock()
        tree.vocab_size = 500
        tok = _TreeTokenizer(tree)
        assert tok.vocab_size == 500

    def test_encode_delegates(self):
        tree = MagicMock()
        tree.encode.return_value = [10, 20, 30]
        tok = _TreeTokenizer(tree)
        assert tok.encode("hello") == [10, 20, 30]
        tree.encode.assert_called_once_with("hello")

    def test_decode_delegates(self):
        tree = MagicMock()
        tree.decode.return_value = "decoded"
        tok = _TreeTokenizer(tree)
        assert tok.decode([1, 2, 3]) == "decoded"
        tree.decode.assert_called_once_with([1, 2, 3])

    def test_apply_chat_template_format(self):
        tree = MagicMock()
        tok = _TreeTokenizer(tree)
        msgs = [{"role": "user", "content": "ask"}]
        result = tok.apply_chat_template(msgs)
        assert result == "user: ask\nassistant:"

    def test_apply_chat_template_empty(self):
        tree = MagicMock()
        tok = _TreeTokenizer(tree)
        result = tok.apply_chat_template([])
        assert result == "assistant:"


# ═══════════════════════════════════════════════════════════════════════════════
# _split_fused_qkv
# ═══════════════════════════════════════════════════════════════════════════════

class TestSplitFusedQKV:
    def test_weight_split(self):
        n_embed = 64
        fused = np.random.randn(n_embed, 3 * n_embed).astype(np.float32)
        result = _split_fused_qkv("h.0.attn.c_attn.weight", fused, n_embed, 2, {})
        assert "blocks.0.attn.q_proj.weight" in result
        assert "blocks.0.attn.k_proj.weight" in result
        assert "blocks.0.attn.v_proj.weight" in result
        # Transpose: (64, 192) → (192, 64) → split into 3 × (64, 64)
        assert result["blocks.0.attn.q_proj.weight"].shape == (n_embed, n_embed)

    def test_bias_split(self):
        n_embed = 64
        fused_bias = np.random.randn(3 * n_embed).astype(np.float32)
        result = _split_fused_qkv("h.0.attn.c_attn.bias", fused_bias, n_embed, 2, {})
        assert "blocks.0.attn.q_proj.bias" in result
        assert "blocks.0.attn.k_proj.bias" in result
        assert "blocks.0.attn.v_proj.bias" in result
        assert result["blocks.0.attn.q_proj.bias"].shape == (n_embed,)

    def test_layer_index_extracted_correctly(self):
        n_embed = 32
        fused = np.random.randn(n_embed, 3 * n_embed).astype(np.float32)
        result = _split_fused_qkv("h.3.attn.c_attn.weight", fused, n_embed, 4, {})
        assert "blocks.3.attn.q_proj.weight" in result
        assert "blocks.0.attn.q_proj.weight" not in result

    def test_invalid_key_returns_empty(self):
        result = _split_fused_qkv("some.random.key", np.zeros(10), 64, 2, {})
        assert result == {}

    def test_multi_digit_layer_index(self):
        n_embed = 16
        fused = np.random.randn(n_embed, 3 * n_embed).astype(np.float32)
        result = _split_fused_qkv("h.10.attn.c_attn.weight", fused, n_embed, 12, {})
        assert "blocks.10.attn.q_proj.weight" in result


# ═══════════════════════════════════════════════════════════════════════════════
# SloNetChatProvider.__init__
# ═══════════════════════════════════════════════════════════════════════════════

class TestInit:
    def test_direct_init_raises_typeerror(self):
        with pytest.raises(TypeError, match="removed"):
            SloNetChatProvider("gpt2")

    def test_direct_init_with_kwargs_raises(self):
        with pytest.raises(TypeError):
            SloNetChatProvider(hf_model_id="gpt2")


# ═══════════════════════════════════════════════════════════════════════════════
# Provider configuration & properties
# ═══════════════════════════════════════════════════════════════════════════════

class TestProviderConfig:
    def test_model_id_property(self):
        p = _make_provider(_model_id="my-model")
        assert p.model_id == "my-model"

    def test_capabilities(self):
        p = _make_provider()
        caps = p.capabilities
        assert caps.chat is True
        assert caps.streaming is True
        assert caps.embedding is False
        assert caps.vision is False
        assert caps.functions is False

    def test_server_default_none(self):
        p = _make_provider()
        assert p.get_server() is None

    def test_set_server(self):
        p = _make_provider()
        mock_server = MagicMock()
        p.set_server(mock_server)
        assert p.get_server() is mock_server

    def test_to_server_returns_server_instance(self):
        from domains.infrastructure.slonet_server import SloNetServer
        p = _make_provider()
        server = p.to_server()
        assert isinstance(server, SloNetServer)

    def test_to_server_with_guard(self):
        from domains.infrastructure.slonet_server import SloNetServer
        p = _make_provider()
        guard = MagicMock()
        server = p.to_server(process_guard=guard)
        assert server._process_guard is guard

    def test_to_server_with_lazy_lock_sets_factory(self):
        from domains.infrastructure.slonet_server import SloNetServer
        p = _make_provider()
        p._lazy_lock = threading.Lock()
        server = p.to_server()
        assert server._lazy_model_factory is not None


# ═══════════════════════════════════════════════════════════════════════════════
# _build_prompt
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildPrompt:
    def test_empty_messages(self):
        p = _make_provider()
        assert p._build_prompt([]) == ""

    def test_none_messages(self):
        p = _make_provider()
        assert p._build_prompt(None) == ""

    def test_string_messages(self):
        p = _make_provider()
        assert p._build_prompt("hello world") == "hello world"

    def test_list_of_strings_takes_last(self):
        p = _make_provider()
        assert p._build_prompt(["first", "second"]) == "second"

    def test_list_of_dicts_uses_chat_template(self):
        p = _make_provider()
        msgs = [{"role": "user", "content": "hi"}]
        result = p._build_prompt(msgs)
        assert "user: hi" in result

    def test_list_of_dicts_fallback_to_last_content(self):
        """When tokenizer has no apply_chat_template, falls back to last message."""
        p = _make_provider()
        p._tokenizer = MagicMock(spec=[])  # no apply_chat_template
        msgs = [{"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}]
        result = p._build_prompt(msgs)
        assert result == "answer"


# ═══════════════════════════════════════════════════════════════════════════════
# num_parameters
# ═══════════════════════════════════════════════════════════════════════════════

class TestNumParameters:
    def test_returns_from_meta_when_available(self):
        p = _make_provider()
        p._meta = {"total_params": 12345}
        assert p.num_parameters() == 12345

    def test_returns_zero_when_no_model_no_meta(self):
        p = _make_provider()
        p._model = None
        p._meta = None
        assert p.num_parameters() == 0

    def test_counts_model_params_when_no_meta(self):
        p = _make_provider()
        p._meta = None
        layer = MagicMock()
        layer.weight = MagicMock()
        layer.weight.data = MagicMock()
        layer.weight.data.size = 500
        p._model = MagicMock()
        p._model.parameters.return_value = [layer.weight]
        assert p.num_parameters() == 500


# ═══════════════════════════════════════════════════════════════════════════════
# quantization_report
# ═══════════════════════════════════════════════════════════════════════════════

class TestQuantizationReport:
    def test_not_quantized(self):
        p = _make_provider()
        report = p.quantization_report()
        assert report == {"quantized": False}

    def test_quantized(self):
        engine = MagicMock()
        engine.summary.return_value = {"bits": 8, "mode": "symmetric", "avg_cosine_sim": 0.99}
        engine.error_report.return_value = [{"tensor": "w", "mse": 0.01}]
        p = _make_provider(_quant_engine=engine)
        report = p.quantization_report()
        assert report["quantized"] is True
        assert report["bits"] == 8
        assert report["mode"] == "symmetric"
        assert len(report["per_tensor"]) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Session management (KV cache)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSessionManagement:
    def test_session_stats_empty(self):
        p = _make_provider()
        stats = p.session_stats()
        assert stats["active_sessions"] == 0
        assert stats["max_sessions"] == 64
        assert stats["ttl_seconds"] == 3600.0
        assert stats["cached_tokens"] == 0
        assert stats["oldest_session_age"] == 0.0

    def test_clear_session_nonexistent_returns_false(self):
        p = _make_provider()
        assert p.clear_session("nonexistent") is False

    def test_clear_session_existent_returns_true(self):
        p = _make_provider()
        p._kv_states["s1"] = MagicMock()
        p._kv_last_access["s1"] = time.monotonic()
        assert p.clear_session("s1") is True
        assert "s1" not in p._kv_states
        assert "s1" not in p._kv_last_access

    def test_clear_all_sessions(self):
        p = _make_provider()
        p._kv_states["s1"] = MagicMock()
        p._kv_states["s2"] = MagicMock()
        p._kv_last_access["s1"] = time.monotonic()
        p._kv_last_access["s2"] = time.monotonic()
        n = p.clear_all_sessions()
        assert n == 2
        assert len(p._kv_states) == 0
        assert len(p._kv_last_access) == 0

    def test_clear_all_sessions_empty_returns_zero(self):
        p = _make_provider()
        assert p.clear_all_sessions() == 0

    def test_session_stats_with_states(self):
        p = _make_provider()
        mock_state = MagicMock()
        mock_state.kv_len = 50
        p._kv_states["s1"] = mock_state
        p._kv_last_access["s1"] = time.monotonic()
        stats = p.session_stats()
        assert stats["active_sessions"] == 1
        assert stats["cached_tokens"] == 50

    def test_session_stats_with_tuple_kv_len(self):
        p = _make_provider()
        mock_state = MagicMock()
        mock_state.kv_len = (10, 20)
        p._kv_states["s1"] = mock_state
        p._kv_last_access["s1"] = time.monotonic()
        stats = p.session_stats()
        assert stats["cached_tokens"] == 30

    def test_evict_lru_session(self):
        p = _make_provider()
        p._kv_max_sessions = 2
        now = time.monotonic()
        p._kv_last_access["old"] = now - 100
        p._kv_last_access["mid"] = now - 50
        p._kv_states["old"] = MagicMock()
        p._kv_states["mid"] = MagicMock()
        # _evict_lru_session is called AFTER new session is inserted by
        # _resolve_session_kv, so we need > max_sessions already present.
        # With 2 states and max=2, len == max → no eviction. Set max=1 so
        # that 2 states > max triggers eviction of "old" (LRU).
        p._kv_max_sessions = 1
        p._kv_lock.acquire()
        try:
            p._evict_lru_session("mid")  # "mid" is kept; "old" is evicted
        finally:
            p._kv_lock.release()
        assert "old" not in p._kv_states
        assert "mid" in p._kv_states

    def test_evict_lru_under_cap_does_nothing(self):
        p = _make_provider()
        p._kv_max_sessions = 10
        p._kv_states["s1"] = MagicMock()
        p._kv_last_access["s1"] = time.monotonic()
        p._kv_lock.acquire()
        try:
            p._evict_lru_session("s2")
        finally:
            p._kv_lock.release()
        assert "s1" in p._kv_states

    def test_evict_stale_sessions(self):
        p = _make_provider()
        p._kv_ttl = 1.0
        p._kv_states["stale"] = MagicMock()
        p._kv_last_access["stale"] = time.monotonic() - 10.0  # 10s ago
        p._kv_states["fresh"] = MagicMock()
        p._kv_last_access["fresh"] = time.monotonic()
        p._evict_stale_sessions()
        assert "stale" not in p._kv_states
        assert "fresh" in p._kv_states


# ═══════════════════════════════════════════════════════════════════════════════
# _load_tokenizer
# ═══════════════════════════════════════════════════════════════════════════════

class TestLoadTokenizer:
    def test_raises_runtime_error_on_failure(self):
        p = _make_provider()
        with pytest.raises(RuntimeError):
            p._load_tokenizer("/nonexistent_path", {})


# ═══════════════════════════════════════════════════════════════════════════════
# _get_model
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetModel:
    def test_returns_model_when_set(self):
        model = MockModel()
        p = _make_provider(_model=model)
        assert p._get_model() is model

    def test_raises_runtime_error_when_no_lazy_lock(self):
        p = _make_provider()
        p._model = None
        p._lazy_lock = None
        with pytest.raises(RuntimeError, match="not loaded"):
            p._get_model()


# ═══════════════════════════════════════════════════════════════════════════════
# generate / generate_with_stop / generate_with_logprobs
# ═══════════════════════════════════════════════════════════════════════════════

class TestGenerate:
    def test_generate_returns_string(self):
        p = _make_provider()
        result = p.generate("hello", max_tokens=3)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_max_new_tokens_alias(self):
        p = _make_provider()
        result = p.generate("hi", max_new_tokens=3)
        assert isinstance(result, str)

    def test_generate_with_stop_no_stop(self):
        p = _make_provider()
        result = p.generate_with_stop("hello", max_tokens=5)
        assert len(result) > 0

    def test_generate_with_stop_string(self):
        p = _make_provider()
        p._tokenizer.decode = lambda ids: "ABCHELLOWORLD"
        result = p.generate_with_stop("hello", max_tokens=10, stop="HELLO")
        assert "HELLO" not in result

    def test_generate_with_stop_list(self):
        p = _make_provider()
        p._tokenizer.decode = lambda ids: "ABCSTOPMEEND"
        result = p.generate_with_stop("hello", max_tokens=10, stop=["STOPME", "END"])
        assert "STOPME" not in result
        assert "END" not in result

    def test_generate_with_stop_none(self):
        p = _make_provider()
        result = p.generate_with_stop("hello", max_tokens=5, stop=None)
        assert len(result) > 0

    def test_generate_with_logprobs_structure(self):
        p = _make_provider()
        text, logprobs = p.generate_with_logprobs("hello", max_tokens=3)
        assert isinstance(text, str)
        assert isinstance(logprobs, list)
        for entry in logprobs:
            assert "token_id" in entry
            assert "token" in entry
            assert "logprob" in entry
            assert "position" in entry

    def test_generate_with_logprobs_positions(self):
        p = _make_provider()
        _, logprobs = p.generate_with_logprobs("hello", max_tokens=5)
        for i, entry in enumerate(logprobs):
            assert entry["position"] == i

    def test_generate_batch(self):
        p = _make_provider()
        results = p.generate_batch(["a", "b", "c"], max_tokens=3)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, str)

    def test_generate_batch_empty(self):
        p = _make_provider()
        assert p.generate_batch([], max_tokens=3) == []


# ═══════════════════════════════════════════════════════════════════════════════
# tokenize / detokenize / count_tokens
# ═══════════════════════════════════════════════════════════════════════════════

class TestTokenize:
    def test_tokenize_returns_int_list(self):
        p = _make_provider()
        tokens = p.tokenize("hello")
        assert isinstance(tokens, list)
        assert all(isinstance(t, int) for t in tokens)

    def test_detokenize_returns_string(self):
        p = _make_provider()
        text = p.detokenize([1, 2, 3])
        assert isinstance(text, str)

    def test_count_tokens_positive(self):
        p = _make_provider()
        assert p.count_tokens("hello world") > 0


# ═══════════════════════════════════════════════════════════════════════════════
# metadata
# ═══════════════════════════════════════════════════════════════════════════════

class TestMetadata:
    def test_metadata_has_required_keys(self):
        p = _make_provider()
        meta = p.metadata()
        for key in ["model_id", "architecture", "total_params", "vocab_size",
                     "max_seq_len", "device", "quantized", "has_tokenizer"]:
            assert key in meta, f"Missing: {key}"

    def test_metadata_model_id(self):
        p = _make_provider(_model_id="x")
        assert p.metadata()["model_id"] == "x"

    def test_metadata_architecture(self):
        p = _make_provider()
        assert p.metadata()["architecture"] == "SloTransformer"

    def test_metadata_cached(self):
        p = _make_provider()
        m1 = p.metadata()
        m2 = p.metadata()
        assert m1 is not m2  # returns a copy
        assert m1 == m2


# ═══════════════════════════════════════════════════════════════════════════════
# release_model
# ═══════════════════════════════════════════════════════════════════════════════

class TestReleaseModel:
    def test_eager_provider_returns_false(self):
        p = _make_provider()
        p._lazy_lock = None
        assert p.release_model() is False

    def test_lazy_provider_no_model_returns_false(self):
        p = _make_provider()
        p._lazy_lock = threading.Lock()
        p._model = None
        p._parser = None
        assert p.release_model() is False

    def test_lazy_provider_releases(self):
        p = _make_provider()
        p._lazy_lock = threading.Lock()
        p._model = MockModel()
        p._parser = MagicMock()
        assert p.release_model() is True
        assert p._model is None
        assert p._parser is None


# ═══════════════════════════════════════════════════════════════════════════════
# Mapping tables sanity
# ═══════════════════════════════════════════════════════════════════════════════

class TestArchMappings:
    def test_shared_has_expected_keys(self):
        for key in ["embed.token", "embed.pos", "layers.{i}.q.weight", "final_norm.weight"]:
            assert key in _ARCH_TO_SLONET_SHARED

    def test_swiglu_has_gate_and_up(self):
        assert "layers.{i}.ffn.gate.weight" in _ARCH_TO_SLONET_SWIGLU
        assert "layers.{i}.ffn.up.weight" in _ARCH_TO_SLONET_SWIGLU

    def test_gelu_has_up_only(self):
        assert "layers.{i}.ffn.up.weight" in _ARCH_TO_SLONET_GELU
        assert "layers.{i}.ffn.gate.weight" not in _ARCH_TO_SLONET_GELU

    def test_none_targets_are_dropped(self):
        assert _ARCH_TO_SLONET_SHARED["layers.{i}.attn_norm.bias"] is None
        assert _ARCH_TO_SLONET_SHARED["layers.{i}.qkv.weight"] is None
        assert _ARCH_TO_SLONET_SHARED["final_norm.bias"] is None


# ═══════════════════════════════════════════════════════════════════════════════
# convert_hf_to_slonet — additional
# ═══════════════════════════════════════════════════════════════════════════════

class TestConvertHfToSlonet:
    def test_lm_head_tied_to_tok_emb(self):
        n_embed, n_layer = 32, 1
        sd = {
            "wte.weight": np.random.randn(100, n_embed).astype(np.float32),
            "ln_f.weight": np.ones(n_embed, dtype=np.float32),
            f"h.0.ln_1.weight": np.ones(n_embed, dtype=np.float32),
            f"h.0.ln_1.bias": np.zeros(n_embed, dtype=np.float32),
            f"h.0.attn.c_attn.weight": np.random.randn(n_embed, 3 * n_embed).astype(np.float32),
            f"h.0.attn.c_attn.bias": np.random.randn(3 * n_embed).astype(np.float32),
            f"h.0.attn.c_proj.weight": np.random.randn(n_embed, n_embed).astype(np.float32),
            f"h.0.attn.c_proj.bias": np.zeros(n_embed, dtype=np.float32),
            f"h.0.ln_2.weight": np.ones(n_embed, dtype=np.float32),
            f"h.0.ln_2.bias": np.zeros(n_embed, dtype=np.float32),
            f"h.0.mlp.c_fc.weight": np.random.randn(n_embed, 4 * n_embed).astype(np.float32),
            f"h.0.mlp.c_fc.bias": np.zeros(4 * n_embed, dtype=np.float32),
            f"h.0.mlp.c_proj.weight": np.random.randn(4 * n_embed, n_embed).astype(np.float32),
            f"h.0.mlp.c_proj.bias": np.zeros(n_embed, dtype=np.float32),
        }
        result = convert_hf_to_slonet(sd, n_layer=1)
        np.testing.assert_array_equal(result["tok_emb.weight"], result["lm_head.weight"])

    def test_unknown_architecture_still_works(self):
        n_embed, n_layer = 32, 1
        sd = {
            "wte.weight": np.random.randn(100, n_embed).astype(np.float32),
            "ln_f.weight": np.ones(n_embed, dtype=np.float32),
            f"h.0.ln_1.weight": np.ones(n_embed, dtype=np.float32),
            f"h.0.ln_1.bias": np.zeros(n_embed, dtype=np.float32),
            f"h.0.attn.c_attn.weight": np.random.randn(n_embed, 3 * n_embed).astype(np.float32),
            f"h.0.attn.c_attn.bias": np.random.randn(3 * n_embed).astype(np.float32),
            f"h.0.attn.c_proj.weight": np.random.randn(n_embed, n_embed).astype(np.float32),
            f"h.0.attn.c_proj.bias": np.zeros(n_embed, dtype=np.float32),
            f"h.0.ln_2.weight": np.ones(n_embed, dtype=np.float32),
            f"h.0.ln_2.bias": np.zeros(n_embed, dtype=np.float32),
            f"h.0.mlp.c_fc.weight": np.random.randn(n_embed, 4 * n_embed).astype(np.float32),
            f"h.0.mlp.c_fc.bias": np.zeros(4 * n_embed, dtype=np.float32),
            f"h.0.mlp.c_proj.weight": np.random.randn(4 * n_embed, n_embed).astype(np.float32),
            f"h.0.mlp.c_proj.bias": np.zeros(n_embed, dtype=np.float32),
        }
        result = convert_hf_to_slonet(sd, n_layer=1, config={"architectures": ["UnknownArch"]})
        assert isinstance(result, dict)
        assert "tok_emb.weight" in result

    def test_w3_synth_bias_ones_for_gelu(self):
        """GELU synthesis: w3.bias should be ones."""
        n_embed, n_layer = 32, 1
        sd = {
            "wte.weight": np.random.randn(100, n_embed).astype(np.float32),
            "ln_f.weight": np.ones(n_embed, dtype=np.float32),
            f"h.0.ln_1.weight": np.ones(n_embed, dtype=np.float32),
            f"h.0.ln_1.bias": np.zeros(n_embed, dtype=np.float32),
            f"h.0.attn.c_attn.weight": np.random.randn(n_embed, 3 * n_embed).astype(np.float32),
            f"h.0.attn.c_attn.bias": np.random.randn(3 * n_embed).astype(np.float32),
            f"h.0.attn.c_proj.weight": np.random.randn(n_embed, n_embed).astype(np.float32),
            f"h.0.attn.c_proj.bias": np.zeros(n_embed, dtype=np.float32),
            f"h.0.ln_2.weight": np.ones(n_embed, dtype=np.float32),
            f"h.0.ln_2.bias": np.zeros(n_embed, dtype=np.float32),
            f"h.0.mlp.c_fc.weight": np.random.randn(n_embed, 4 * n_embed).astype(np.float32),
            f"h.0.mlp.c_fc.bias": np.zeros(4 * n_embed, dtype=np.float32),
            f"h.0.mlp.c_proj.weight": np.random.randn(4 * n_embed, n_embed).astype(np.float32),
            f"h.0.mlp.c_proj.bias": np.zeros(n_embed, dtype=np.float32),
        }
        result = convert_hf_to_slonet(sd, n_layer=1)
        np.testing.assert_array_equal(
            result["blocks.0.ff.w3.bias"],
            np.ones(n_embed * 4, dtype=np.float32),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Backward compat alias
# ═══════════════════════════════════════════════════════════════════════════════

class TestBackwardCompat:
    def test_alias_exists(self):
        from domains.inference.slonet_provider import SlonetChatProvider
        assert SlonetChatProvider is SloNetChatProvider
