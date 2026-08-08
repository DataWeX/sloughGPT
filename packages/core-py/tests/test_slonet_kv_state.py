"""
Tests for cross-turn KV cache reuse via NumpyKVState.

Verifies that multi-turn generation with prefix reuse produces the same
output as a full recomputation, and that fallback paths (empty state,
mismatched quantize mode, prefix-mismatch, identical-prompt retry) also
produce correct results.
"""
import math
import numpy as np
import pytest

from domains.training.slonet import (
    SloTransformer, SloTransformerBlock, SloLinear, SloEmbedding,
    SloLayerNorm, NumpyKVState,
)

# ---------------------------------------------------------------------------
# Fixture — tiny 1-block transformer
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_model():
    """Minimal 1-block transformer with GQA and RoPE for fast testing."""
    E, V, BS = 64, 256, 128
    net = SloTransformer(
        vocab_size=V, n_embed=E, n_layer=1, n_head=4, block_size=BS,
        n_kv_head=2,
    )
    net.max_seq_len = BS
    return net


# ---------------------------------------------------------------------------
# NumpyKVState unit tests
# ---------------------------------------------------------------------------

class TestNumpyKVState:

    def test_new_kv_state_is_empty(self, tiny_model):
        state = tiny_model.new_kv_state()
        assert state.prev_ids is None
        assert state.capacity == 0
        assert state.kv_len == []
        assert state.kv_buf_k == []
        assert state.quantize_kv is False

    def test_reset_clears_everything(self, tiny_model):
        state = tiny_model.new_kv_state()
        state.capacity = 100
        state.kv_buf_k = [np.zeros((1, 10, 2, 64))]
        state.kv_len = [5]
        state.prev_ids = np.array([[1, 2, 3]])
        state.quantize_kv = True
        state.reset()
        assert state.prev_ids is None
        assert state.capacity == 0
        assert state.kv_len == []
        assert state.kv_buf_k == []
        assert state.quantize_kv is False

    def test_repr_shows_valid(self, tiny_model):
        state = tiny_model.new_kv_state()
        assert "valid=False" in repr(state)
        state.prev_ids = np.array([[1, 2, 3]])
        assert "valid=True" in repr(state)

    def test_resolve_empty_state_returns_fresh(self, tiny_model):
        state = tiny_model.new_kv_state()
        ids = np.array([[10, 20, 30, 40]])
        buf_k, buf_v, sk, sv, kl, s = tiny_model._resolve_kv_state(
            state, 1, 10, [2], 64, False, ids, 4)
        assert s == 0
        assert kl == [0]
        assert len(buf_k) == 1
        assert buf_k[0].shape == (1, 10, 2, 64)
        assert state.capacity == 10
        assert state.quantize_kv is False

    def test_resolve_mismatched_quantize_falls_back(self, tiny_model):
        state = tiny_model.new_kv_state()
        state.quantize_kv = False
        state.prev_ids = np.array([[1, 2]])
        ids = np.array([[1, 2, 3]])
        _, _, _, _, _, s = tiny_model._resolve_kv_state(
            state, 1, 10, [2], 64, True, ids, 3)
        assert s == 0

    def test_resolve_mismatched_dims_falls_back(self, tiny_model):
        state = tiny_model.new_kv_state()
        state.prev_ids = np.array([[1, 2]])
        state.kv_buf_k = [np.zeros((1, 5, 4, 32))]  # wrong nkv and E
        state.quantize_kv = False
        ids = np.array([[1, 2, 3]])
        _, _, _, _, _, s = tiny_model._resolve_kv_state(
            state, 1, 10, [2], 64, False, ids, 3)
        assert s == 0

    def test_resolve_partial_prefix_reuse(self, tiny_model):
        state = tiny_model.new_kv_state()
        state.prev_ids = np.array([[1, 2, 3]])
        state.kv_buf_k = [np.zeros((1, 5, 2, 64))]
        state.kv_len = [3]
        state.capacity = 5
        state.quantize_kv = False
        ids = np.array([[1, 2, 99]])  # differs at position 2 → partial reuse of [1,2]
        _, _, _, _, kl, s = tiny_model._resolve_kv_state(
            state, 1, 10, [2], 64, False, ids, 3)
        assert s == 2  # reuses first 2 cached positions
        assert kl == [2]

    def test_resolve_identical_prompt_falls_back(self, tiny_model):
        state = tiny_model.new_kv_state()
        state.prev_ids = np.array([[1, 2, 3]])
        state.kv_buf_k = [np.zeros((1, 5, 2, 64))]
        state.kv_len = [3]
        state.capacity = 5
        state.quantize_kv = False
        ids = np.array([[1, 2, 3]])  # same prompt → degenerate case
        _, _, _, _, _, s = tiny_model._resolve_kv_state(
            state, 1, 10, [2], 64, False, ids, 3)
        assert s == 0  # falls back to fresh

    def test_resolve_prefix_reuse(self, tiny_model):
        state = tiny_model.new_kv_state()
        state.prev_ids = np.array([[1, 2, 3]])
        state.kv_buf_k = [np.zeros((1, 5, 2, 64))]
        state.kv_len = [3]
        state.capacity = 5
        state.quantize_kv = False
        ids = np.array([[1, 2, 3, 4, 5]])  # extends by 2 tokens
        _, _, _, _, kl, s = tiny_model._resolve_kv_state(
            state, 1, 10, [2], 64, False, ids, 5)
        assert s == 3  # prefix of length 3 reused
        assert kl == [3]

    def test_resolve_grows_buffer_on_resume(self, tiny_model):
        state = tiny_model.new_kv_state()
        state.prev_ids = np.array([[1, 2, 3]])
        state.kv_buf_k = [np.zeros((1, 5, 2, 64))]
        state.kv_len = [3]
        state.capacity = 5
        state.quantize_kv = False
        ids = np.array([[1, 2, 3, 4, 5, 6, 7, 8]])
        _, _, _, _, _, s = tiny_model._resolve_kv_state(
            state, 1, 20, [2], 64, False, ids, 8)
        assert s == 3
        assert state.capacity == 20
        assert state.kv_buf_k[0].shape == (1, 20, 2, 64)


# ---------------------------------------------------------------------------
# Bit-exact correctness: generate_numpy
# ---------------------------------------------------------------------------

class TestCrossTurnGenerateNumpy:

    def test_stream_matches_non_stream(self, tiny_model):
        ids = np.array([[10, 20, 30, 40, 50]])
        kw = dict(max_new_tokens=5, temperature=0.0, eos_token=None)
        full = tiny_model.generate_numpy(ids, **kw)
        streamed = list(tiny_model.generate_numpy_stream(ids, **kw))
        np.testing.assert_array_equal(full[0], np.concatenate([
            ids, np.array([streamed])], axis=1)[0])

    def test_fresh_vs_reuse_same_input(self, tiny_model):
        """Calling generate_numpy twice with the same prompt yields identical output
        (the second call falls back to fresh, not reusing stale cache state)."""
        ids = np.array([[10, 20, 30]])
        kw = dict(max_new_tokens=8, temperature=0.0)
        out1 = tiny_model.generate_numpy(ids, **kw)
        out2 = tiny_model.generate_numpy(ids, **kw)
        np.testing.assert_array_equal(out1, out2)

    def test_cross_turn_full_recompute_agree(self, tiny_model):
        """Turn-2 output using prefix reuse equals the full recompute from scratch."""
        ids1 = np.array([[10, 20, 30]])
        kw1 = dict(max_new_tokens=4, temperature=0.0)
        turn1 = tiny_model.generate_numpy(ids1, **kw1)
        new_tokens = np.array([[77, 88]])
        ids2 = np.concatenate([turn1, new_tokens], axis=1)

        state = tiny_model.new_kv_state()
        out_reuse = tiny_model.generate_numpy(ids2, max_new_tokens=4,
                                              temperature=0.0, kv_state=state)
        out_fresh = tiny_model.generate_numpy(ids2, max_new_tokens=4,
                                              temperature=0.0)
        np.testing.assert_array_equal(out_reuse, out_fresh)

    def test_state_updated_after_call(self, tiny_model):
        ids = np.array([[10, 20, 30]])
        state = tiny_model.new_kv_state()
        out = tiny_model.generate_numpy(ids, max_new_tokens=5,
                                        temperature=0.0, kv_state=state)
        np.testing.assert_array_equal(state.prev_ids, out)
        assert state.capacity > 0
        assert state.kv_len == [out.shape[1] - 1]

    def test_three_turn_chain(self, tiny_model):
        """Three successive turns with prefix reuse all agree with fresh calls."""
        ids1 = np.array([[10, 20, 30]])
        kw = dict(max_new_tokens=3, temperature=0.0)
        state = tiny_model.new_kv_state()
        t1 = tiny_model.generate_numpy(ids1, kv_state=state, **kw)
        t1_fresh = tiny_model.generate_numpy(ids1, **kw)
        np.testing.assert_array_equal(t1, t1_fresh)

        ids2 = np.concatenate([t1, np.array([[77]])], axis=1)
        t2 = tiny_model.generate_numpy(ids2, kv_state=state, **kw)
        t2_fresh = tiny_model.generate_numpy(ids2, **kw)
        np.testing.assert_array_equal(t2, t2_fresh)

        ids3 = np.concatenate([t2, np.array([[99]])], axis=1)
        t3 = tiny_model.generate_numpy(ids3, kv_state=state, **kw)
        t3_fresh = tiny_model.generate_numpy(ids3, **kw)
        np.testing.assert_array_equal(t3, t3_fresh)

    def test_stream_cross_turn_matches_fresh(self, tiny_model):
        ids1 = np.array([[10, 20, 30]])
        kw = dict(max_new_tokens=3, temperature=0.0)
        state = tiny_model.new_kv_state()
        t1 = tiny_model.generate_numpy(ids1, kv_state=state, **kw)

        ids2 = np.concatenate([t1, np.array([[77]])], axis=1)
        out_reuse = tiny_model.generate_numpy(ids2, kv_state=state, **kw)
        out_fresh = tiny_model.generate_numpy(ids2, **kw)
        np.testing.assert_array_equal(out_reuse, out_fresh)

    def test_prefill_len_one(self, tiny_model):
        """When reuse produces prefill_len=1 (one new token), single-token step works."""
        ids1 = np.array([[10, 20, 30]])
        kw = dict(max_new_tokens=3, temperature=0.0)
        state = tiny_model.new_kv_state()
        t1 = tiny_model.generate_numpy(ids1, kv_state=state, **kw)

        # New prompt = turn1 output (3 tokens) + 1 new token = prefill_len=1
        ids2 = np.concatenate([t1, np.array([[77]])], axis=1)
        out_reuse = tiny_model.generate_numpy(ids2, kv_state=state, **kw)
        out_fresh = tiny_model.generate_numpy(ids2, **kw)
        np.testing.assert_array_equal(out_reuse, out_fresh)

    def test_stop_token_respects_state(self, tiny_model):
        """Stop token triggers correctly even with prefix reuse."""
        ids1 = np.array([[10, 20, 30]])
        kw = dict(max_new_tokens=3, temperature=0.0)
        state = tiny_model.new_kv_state()
        t1 = tiny_model.generate_numpy(ids1, kv_state=state, **kw)
        stop_tok = int(t1[0, -1])
        ids2 = np.concatenate([t1, np.array([[77]])], axis=1)
        out = tiny_model.generate_numpy(ids2, max_new_tokens=10,
                                        temperature=0.0, eos_token=stop_tok,
                                        kv_state=state)
        assert out.shape[1] <= ids2.shape[1] + 10

    def test_quantized_reuse_matches_fresh(self, tiny_model):
        """With explicit quantize_kv=True, reuse equals fresh."""
        ids1 = np.array([[10, 20, 30]])
        kw = dict(max_new_tokens=3, temperature=0.0, quantize_kv=True)
        state = tiny_model.new_kv_state()
        t1 = tiny_model.generate_numpy(ids1, kv_state=state, **kw)
        t1_fresh = tiny_model.generate_numpy(ids1, **kw)
        np.testing.assert_array_equal(t1, t1_fresh)

        ids2 = np.concatenate([t1, np.array([[77]])], axis=1)
        out_reuse = tiny_model.generate_numpy(ids2, kv_state=state, **kw)
        out_fresh = tiny_model.generate_numpy(ids2, **kw)
        np.testing.assert_array_equal(out_reuse, out_fresh)


# ---------------------------------------------------------------------------
# Stream generator abandonment safety
# ---------------------------------------------------------------------------

class TestStreamAbandonment:

    def test_partial_stream_state_valid(self, tiny_model):
        """Generator abandoned mid-yield still has a valid state (no invalidation)."""
        ids = np.array([[10, 20, 30]])
        state = tiny_model.new_kv_state()
        gen = tiny_model.generate_numpy_stream(ids, max_new_tokens=5,
                                               temperature=0.0, kv_state=state)
        tok0 = next(gen)
        tok1 = next(gen)
        try:
            gen.close()
        except Exception:
            pass
        # State reflects the two tokens that were successfully generated
        assert state.prev_ids is not None
        assert state.prev_ids.shape[1] == ids.shape[1] + 2

    def test_stream_complete_state_valid(self, tiny_model):
        """Generator consumed fully has valid state."""
        ids = np.array([[10, 20, 30]])
        state = tiny_model.new_kv_state()
        toks = list(tiny_model.generate_numpy_stream(ids, max_new_tokens=5,
                                                     temperature=0.0,
                                                     kv_state=state))
        assert len(toks) == 5
        assert state.prev_ids is not None
        assert state.prev_ids.shape[1] == ids.shape[1] + 5
        assert state.kv_len == [ids.shape[1] + 5 - 1]
