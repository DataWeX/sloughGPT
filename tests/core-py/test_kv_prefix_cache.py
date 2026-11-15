"""Cross-session KV prefix sharing (card 20261102_023531, "Rung A").

The claim this file has to defend is narrow and total: **a prefix hit is
token-exact**. K/V is a pure function of the token ids and their positions, so
there is no threshold, no similarity, and no wrong-answer class here — unlike
the semantic response tier, where similarity may propose but must never
decide. So the tests are about identity, not quality:

  - a hit resumes, and the output is bit-identical to a cold run
  - a different prompt never resumes (the ``2+3`` / ``4+12`` rule, at the K/V
    tier rather than the answer tier)
  - a snapshot never aliases the session state it was taken from
  - dims / quantize mismatches are refused, because a snapshot taken under
    other weights is not the same computation
"""

from __future__ import annotations

import numpy as np
import pytest

from domain.infrastructure._internal.kv_cache.prefix_cache import (
    KVSnapshot,
    PrefixCache,
)
from domain.training._internal.slonet import SloTransformer

BLOCK = 64
VOCAB, EMBED, BLOCKS = 256, 64, 128


@pytest.fixture
def tiny_model() -> SloTransformer:
    net = SloTransformer(
        vocab_size=VOCAB,
        n_embed=EMBED,
        n_layer=1,
        n_head=4,
        block_size=BLOCKS,
        n_kv_head=2,
    )
    net.max_seq_len = BLOCKS
    return net


def _state(model, tokens: list[int], *, quantized: bool = False):
    """Run one generation so the state holds a filled, harvested prefix."""
    ids = np.array([tokens], dtype=np.int64)
    state = model.new_kv_cache()
    model.generate_numpy(
        ids, max_new_tokens=4, temperature=0.0, kv_state=state, quantize_kv=quantized
    )
    return state


def _fake_buffers(tokens: int, layers: int = 2, nkv: int = 2, head_dim: int = 8):
    k = [
        np.arange(tokens, dtype=np.float32)
        .reshape(1, tokens, 1, 1)
        .repeat(nkv, 2)
        .repeat(head_dim, 3)
    ]
    v = [x + 1000.0 for x in k]
    return k * layers, v * layers


# ---------------------------------------------------------------------------
# PrefixCache — the table itself
# ---------------------------------------------------------------------------


class TestPrefixCacheStore:
    def test_stores_block_aligned_prefix(self):
        cache = PrefixCache(max_entries=4, max_tokens=1024, min_tokens=BLOCK, block=BLOCK)
        ids = np.arange(100, dtype=np.int64).reshape(1, 100)
        k, v = _fake_buffers(100)
        assert cache.store(ids, k, v, filled=100) is True
        # 100 tokens cut down to the last complete block.
        assert len(cache) == 1
        assert cache.get_stats()["cached_tokens"] == BLOCK

    def test_refuses_less_than_one_block(self):
        cache = PrefixCache(min_tokens=BLOCK, block=BLOCK)
        ids = np.arange(BLOCK - 1, dtype=np.int64).reshape(1, BLOCK - 1)
        k, v = _fake_buffers(BLOCK - 1)
        assert cache.store(ids, k, v, filled=BLOCK - 1) is False
        assert len(cache) == 0
        assert cache.get_stats()["rejected"] == 1

    def test_refuses_buffers_shorter_than_filled(self):
        cache = PrefixCache(min_tokens=BLOCK, block=BLOCK)
        ids = np.arange(200, dtype=np.int64).reshape(1, 200)
        k, v = _fake_buffers(BLOCK)
        assert cache.store(ids, k, v, filled=200) is False

    def test_snapshot_does_not_alias_the_source(self):
        """The live state is mutated in place; a snapshot must be a copy."""
        cache = PrefixCache(min_tokens=BLOCK, block=BLOCK)
        ids = np.arange(BLOCK, dtype=np.int64).reshape(1, BLOCK)
        k, v = _fake_buffers(BLOCK)
        cache.store(ids, k, v, filled=BLOCK)
        before = k[0][:, :BLOCK].copy()

        k[0][:, :BLOCK] = -1.0  # next turn overwrites the session's buffers

        snap, matched = cache.match(ids)
        assert matched == BLOCK
        np.testing.assert_array_equal(snap.k[0], before)


class TestPrefixCacheMatch:
    def test_exact_match(self):
        cache = PrefixCache(min_tokens=BLOCK, block=BLOCK)
        ids = np.arange(BLOCK, dtype=np.int64).reshape(1, BLOCK)
        k, v = _fake_buffers(BLOCK)
        cache.store(ids, k, v, filled=BLOCK)
        snap, matched = cache.match(ids)
        assert matched == BLOCK
        assert isinstance(snap, KVSnapshot)

    def test_different_tokens_never_match(self):
        """The K/V analogue of ``2+3`` vs ``4+12``: same length, different ids."""
        cache = PrefixCache(min_tokens=BLOCK, block=BLOCK)
        k, v = _fake_buffers(BLOCK)
        cache.store(np.full((1, BLOCK), 7, dtype=np.int64), k, v, filled=BLOCK)
        assert cache.match(np.full((1, BLOCK), 8, dtype=np.int64)) is None

    def test_shared_prefix_matches_only_the_shared_part(self):
        """The headline case: same system prompt, different user turn, other session."""
        cache = PrefixCache(min_tokens=BLOCK, block=BLOCK)
        shared = np.arange(BLOCK, dtype=np.int64)
        k, v = _fake_buffers(BLOCK + 30)
        cache.store(
            np.concatenate([shared, np.arange(500, 530)]).reshape(1, -1), k, v, filled=BLOCK + 30
        )

        query = np.concatenate([shared, np.arange(900, 940)]).reshape(1, -1)
        snap, matched = cache.match(query)
        assert matched == BLOCK, "must resume at the block boundary, not at the divergence"
        np.testing.assert_array_equal(snap.ids[0, :BLOCK], shared)

    def test_short_query_does_not_match_a_longer_snapshot(self):
        """A query shorter than the stored prefix aligns to its own blocks only."""
        cache = PrefixCache(min_tokens=BLOCK, block=BLOCK)
        k, v = _fake_buffers(BLOCK * 2)
        cache.store(np.arange(BLOCK * 2, dtype=np.int64).reshape(1, -1), k, v, filled=BLOCK * 2)

        snap, matched = cache.match(np.arange(BLOCK, dtype=np.int64).reshape(1, -1))
        assert matched == BLOCK
        assert snap.filled == BLOCK * 2

    def test_query_below_min_tokens_is_a_miss(self):
        cache = PrefixCache(min_tokens=BLOCK, block=BLOCK)
        assert cache.match(np.arange(3, dtype=np.int64).reshape(1, 3)) is None

    def test_stats_count_hits_and_misses(self):
        cache = PrefixCache(min_tokens=BLOCK, block=BLOCK)
        ids = np.arange(BLOCK, dtype=np.int64).reshape(1, BLOCK)
        k, v = _fake_buffers(BLOCK)
        cache.store(ids, k, v, filled=BLOCK)
        cache.match(ids)
        cache.match(np.full((1, BLOCK), 3, dtype=np.int64))
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        # Normalized like ResponseCache/SemanticCache: percent string + raw number.
        assert stats["hit_rate"] == "50.0%"
        assert stats["hit_rate_num"] == 0.5


class TestPrefixCacheEviction:
    def test_entry_cap_evicts_least_recently_used(self):
        cache = PrefixCache(max_entries=2, max_tokens=10_000, min_tokens=BLOCK, block=BLOCK)
        k, v = _fake_buffers(BLOCK)
        for base in (100, 200, 300):
            cache.store(
                np.arange(base, base + BLOCK, dtype=np.int64).reshape(1, -1), k, v, filled=BLOCK
            )
        assert len(cache) == 2
        assert cache.match(np.arange(100, 100 + BLOCK, dtype=np.int64).reshape(1, -1)) is None
        assert cache.get_stats()["evictions"] == 1

    def test_token_budget_evicts(self):
        cache = PrefixCache(max_entries=8, max_tokens=BLOCK * 2, min_tokens=BLOCK, block=BLOCK)
        k, v = _fake_buffers(BLOCK)
        for base in (0, BLOCK, BLOCK * 2):
            cache.store(
                np.arange(base, base + BLOCK, dtype=np.int64).reshape(1, -1), k, v, filled=BLOCK
            )
        assert cache.get_stats()["cached_tokens"] <= BLOCK * 2
        assert len(cache) == 2

    def test_snapshot_larger_than_the_whole_budget_is_rejected(self):
        cache = PrefixCache(max_entries=4, max_tokens=BLOCK, min_tokens=BLOCK, block=BLOCK)
        k, v = _fake_buffers(BLOCK * 2)
        assert (
            cache.store(np.arange(BLOCK * 2, dtype=np.int64).reshape(1, -1), k, v, filled=BLOCK * 2)
            is False
        )


class TestStatsSchema:
    """Every cache answers the status bar with the same core vocabulary.

    The two march in step so a dashboard (or the router facade) can render
    ``enabled / entries / max_entries / hits / misses / hit_rate / evictions``
    without knowing which cache produced the dict. ``hit_rate`` is a percent
    string everywhere — the pre-existing ResponseCache/SemanticCache contract.
    """

    REQUIRED = {"enabled", "entries", "max_entries", "hits", "misses", "hit_rate", "evictions"}

    def test_prefix_cache_exposes_the_shared_fields(self):
        assert self.REQUIRED <= set(PrefixCache().get_stats())

    def test_response_cache_exposes_the_shared_fields(self):
        from domain.inference import ResponseCache

        assert self.REQUIRED <= set(ResponseCache().get_stats())

    def test_semantic_cache_exposes_the_shared_fields(self):
        from domain.inference._internal.semantic_cache import SemanticCache

        assert self.REQUIRED <= set(SemanticCache().get_stats())


# ---------------------------------------------------------------------------
# Model integration — the part that can actually produce wrong text
# ---------------------------------------------------------------------------


class TestCrossSessionResume:
    def test_second_session_with_the_same_prompt_resumes(self, tiny_model):
        """Two different KV states, same prompt: the second reuses the prefix."""
        tokens = list(range(1, 1 + BLOCK + 20))
        cold = tiny_model.generate_numpy(
            np.array([tokens], dtype=np.int64), max_new_tokens=4, temperature=0.0
        )

        first = tiny_model.new_kv_cache()
        tiny_model.generate_numpy(
            np.array([tokens], dtype=np.int64), max_new_tokens=4, temperature=0.0, kv_state=first
        )
        assert tiny_model.prefix_cache_stats()["stores"] >= 1

        # A DIFFERENT state, as a different client / reconnecting session would have.
        second = tiny_model.new_kv_cache()
        adopted = tiny_model.generate_numpy(
            np.array([tokens], dtype=np.int64), max_new_tokens=4, temperature=0.0, kv_state=second
        )

        assert (
            tiny_model.prefix_cache_stats()["hits"]
            + tiny_model.prefix_cache_stats()["partial_hits"]
            >= 1
        )
        np.testing.assert_array_equal(adopted, cold), "a prefix hit must be bit-identical"

    def test_different_prompt_never_resumes(self, tiny_model):
        prompts = [
            np.array([list(range(1, 1 + BLOCK))], dtype=np.int64),
            np.array([list(range(100, 100 + BLOCK))], dtype=np.int64),
        ]
        outs = [tiny_model.generate_numpy(p, max_new_tokens=4, temperature=0.0) for p in prompts]
        assert tiny_model.prefix_cache_stats()["hits"] == 0
        assert tiny_model.prefix_cache_stats()["partial_hits"] == 0
        np.testing.assert_array_equal(
            outs[0], tiny_model.generate_numpy(prompts[0], max_new_tokens=4, temperature=0.0)
        )

    def test_quantize_mismatch_is_refused(self, tiny_model):
        """int8 K/V is not float32 K/V — never transplant across modes."""
        tokens = list(range(1, 1 + BLOCK + 20))
        ids = np.array([tokens], dtype=np.int64)
        first = tiny_model.new_kv_cache()
        tiny_model.generate_numpy(
            ids, max_new_tokens=4, temperature=0.0, kv_state=first, quantize_kv=True
        )
        stats_before = tiny_model.prefix_cache_stats()

        second = tiny_model.new_kv_cache()
        out = tiny_model.generate_numpy(
            ids, max_new_tokens=4, temperature=0.0, kv_state=second, quantize_kv=False
        )
        cold = tiny_model.generate_numpy(ids, max_new_tokens=4, temperature=0.0, quantize_kv=False)

        assert tiny_model.prefix_cache_stats()["partial_hits"] == stats_before["partial_hits"]
        np.testing.assert_array_equal(out, cold)

    def test_quantized_prefix_hit_is_bit_identical(self, tiny_model):
        """int8 K/V is what quantized production models use — pin its identity too."""
        tokens = list(range(1, 1 + BLOCK + 20))
        ids = np.array([tokens], dtype=np.int64)
        cold = tiny_model.generate_numpy(ids, max_new_tokens=4, temperature=0.0, quantize_kv=True)
        assert tiny_model.prefix_cache_stats()["stores"] >= 1

        first = tiny_model.new_kv_cache()
        adopted = tiny_model.generate_numpy(
            ids, max_new_tokens=4, temperature=0.0, kv_state=first, quantize_kv=True
        )
        assert (
            tiny_model.prefix_cache_stats()["hits"]
            + tiny_model.prefix_cache_stats()["partial_hits"]
            >= 1
        )
        np.testing.assert_array_equal(adopted, cold)

    def test_disabled_cache_changes_nothing(self, tiny_model, monkeypatch):
        tiny_model._prefix_cache = None
        tokens = list(range(1, 1 + BLOCK + 20))
        ids = np.array([tokens], dtype=np.int64)
        warm = tiny_model.generate_numpy(ids, max_new_tokens=4, temperature=0.0)
        again = tiny_model.generate_numpy(ids, max_new_tokens=4, temperature=0.0)
        np.testing.assert_array_equal(warm, again)
        assert tiny_model.prefix_cache_stats() == {}

    def test_loading_weights_drops_the_table(self, tiny_model):
        tokens = list(range(1, 1 + BLOCK + 20))
        tiny_model.generate_numpy(
            np.array([tokens], dtype=np.int64), max_new_tokens=4, temperature=0.0
        )
        assert len(tiny_model._prefix_cache) >= 1
        tiny_model.load_state_dict({k: v for k, v in tiny_model.state_dict().items()})
        assert len(tiny_model._prefix_cache) == 0

    def test_stream_path_also_populates_the_table(self, tiny_model):
        tokens = list(range(1, 1 + BLOCK + 20))
        list(
            tiny_model.generate_numpy_stream(
                np.array([tokens], dtype=np.int64), max_new_tokens=4, temperature=0.0
            )
        )
        assert tiny_model.prefix_cache_stats()["stores"] >= 1
