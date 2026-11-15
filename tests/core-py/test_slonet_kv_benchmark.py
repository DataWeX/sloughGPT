"""Cross-turn KV cache benchmark — measures token savings and speedup.

Runs multi-turn conversations with and without KV cache reuse,
comparing wall-clock time and recomputed tokens.

Usage:
    pytest tests/test_slonet_kv_benchmark.py -v -s
"""

import time

import numpy as np
import pytest


@pytest.fixture
def tiny_model():
    """Minimal SloTransformer with GQA for benchmarking."""
    from domain.training._internal.slonet import SloTransformer

    model = SloTransformer(
        vocab_size=256,
        n_embed=64,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        intermediate_size=128,
        block_size=128,
        max_seq_len=128,
        use_rope=True,
        dropout=0.0,
        tie_weights=True,
    )
    rng = np.random.RandomState(42)
    for p in model.parameters():
        p.data = rng.randn(*p.shape).astype(np.float32) * 0.02
    return model


class TestCrossTurnBenchmark:
    """Benchmark cross-turn KV cache vs fresh generation."""

    def _generate_fresh(self, model, input_ids, max_new_tokens, temperature=0.0):
        """Generate without KV cache — full recompute every turn."""
        return model.generate_numpy(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            kv_state=None,
        )

    def _generate_cached(self, model, state, input_ids, max_new_tokens, temperature=0.0):
        """Generate with KV cache — prefix reuse across turns."""
        return model.generate_numpy(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            kv_state=state,
        )

    def _record_recomputes(self, model, monkeypatch):
        """Capture ``(prompt_len, start)`` for every generate call.

        ``_resolve_kv_state`` already computes the resume point; recomputed
        tokens = ``prompt_len - start`` is the exact work a cache skips.
        That is the property a cache benchmark must gate on — wall clock on a
        1-block/64-dim model is single-digit ms where scheduler noise dwarfs
        the signal (observed flake: 0.62x while the cache was not slower).
        Observation only: the real method runs unchanged, we just read its
        return.
        """
        records: list[tuple[int, int]] = []
        orig = model._resolve_kv_state

        def spy(*args, **kwargs):
            out = orig(*args, **kwargs)
            prompt_len = kwargs.get("prompt_len") if "prompt_len" in kwargs else args[7]
            records.append((int(prompt_len), int(out[-1])))
            return out

        monkeypatch.setattr(model, "_resolve_kv_state", spy)
        return records

    def test_two_turn_speedup(self, tiny_model, monkeypatch):
        """Two-turn conversation: the cache's property is WORK SKIPPED, not ms.

        Gates on recomputed tokens — exact and free — instead of wall clock:
        a 1-block/64-dim model finishes a turn in single-digit ms, so a clock
        threshold measures scheduler noise (observed 0.62x in a combined run
        while the cache was in fact not slower). Timings are still printed
        for humans (run with -s); they are just never asserted on.
        """
        m = tiny_model
        records = self._record_recomputes(m, monkeypatch)
        ids1 = np.array([[10, 20, 30, 40, 50]])
        max_new = 10

        t0 = time.perf_counter()
        r1 = self._generate_fresh(m, ids1, max_new)
        t1_fresh = time.perf_counter() - t0

        ids2 = np.concatenate([r1, np.array([[60]])], axis=1)

        t0 = time.perf_counter()
        r2_fresh = self._generate_fresh(m, ids2, max_new)
        t2_fresh = time.perf_counter() - t0

        state = m.new_kv_state()
        t0 = time.perf_counter()
        r1c = self._generate_cached(m, state, ids1, max_new)
        t1_cached = time.perf_counter() - t0

        ids2c = np.concatenate([r1c, np.array([[60]])], axis=1)

        t0 = time.perf_counter()
        r2_cached = self._generate_cached(m, state, ids2c, max_new)
        t2_cached = time.perf_counter() - t0

        # Full-prompt repeat into the same state: every token is already
        # cached, so exactly one token is recomputed (card 20261102_012822 —
        # this used to re-prefill the entire prompt).
        t0 = time.perf_counter()
        self._generate_cached(m, state, ids1, max_new)
        t_repeat = time.perf_counter() - t0

        total_fresh = t1_fresh + t2_fresh
        total_cached = t1_cached + t2_cached
        speedup = total_fresh / total_cached if total_cached > 0 else float("inf")

        print(f"\n{'=' * 60}")
        print("Two-Turn KV Cache Benchmark")
        print(f"{'=' * 60}")
        print(f"Turn 1 (fresh):   {t1_fresh * 1000:8.2f} ms  |  {r1.shape[1]} tokens out")
        print(f"Turn 1 (cached):  {t1_cached * 1000:8.2f} ms  |  {r1c.shape[1]} tokens out")
        print(f"Turn 2 (fresh):   {t2_fresh * 1000:8.2f} ms  |  {r2_fresh.shape[1]} tokens out")
        print(f"Turn 2 (cached):  {t2_cached * 1000:8.2f} ms  |  {r2_cached.shape[1]} tokens out")
        print(f"Repeat (cached):  {t_repeat * 1000:8.2f} ms")
        print(f"{'─' * 60}")
        print(f"Total fresh:      {total_fresh * 1000:8.2f} ms")
        print(f"Total cached:     {total_cached * 1000:8.2f} ms")
        print(f"Speedup:          {speedup:8.2f}x")
        print(f"{'=' * 60}")

        # Exact work per call: (prompt_len, start) in call order —
        # fresh turn1, fresh turn2, cached turn1 (cold state), cached turn2,
        # full-prompt repeat.
        recomputed = [prompt_len - start for prompt_len, start in records]
        fresh_t1, fresh_t2, cached_t1, cached_t2, repeat = recomputed
        prompt1_len = int(ids1.shape[1])
        prompt2_len = int(ids2.shape[1])

        assert len(records) == 5, f"expected 5 generate calls, saw {len(records)}"
        # No cache (kv_state=None) re-prefills everything, every time.
        assert fresh_t1 == prompt1_len, f"fresh turn1 recomputed {fresh_t1}, want {prompt1_len}"
        assert fresh_t2 == prompt2_len, f"fresh turn2 recomputed {fresh_t2}, want {prompt2_len}"
        # A cold state is a full prefill — the cache pays once, then skips.
        assert cached_t1 == prompt1_len, (
            f"cold cached turn1 recomputed {cached_t1}, want {prompt1_len}"
        )
        # Turn 2 recomputes only the appended token plus the previous turn's
        # last generated token (never cached: KV for it is never computed).
        # Everything else — the whole shared prefix — comes from the cache.
        assert cached_t2 == 2, (
            f"cached turn2 recomputed {cached_t2}, want 2 (append + uncached last)"
        )
        assert cached_t2 < fresh_t2, "cache must skip work vs the fresh path"
        # Full-prompt match resumes at prompt_len - 1: exactly one token.
        assert repeat == 1, f"full-prompt repeat recomputed {repeat}, want 1"

    def test_three_turn_cumulative_speedup(self, tiny_model, monkeypatch):
        """Three-turn conversation: cumulative recomputed tokens must stay flat.

        Turn 1 prefills once; every later cached turn recomputes only the
        append (+ the one never-cached last token), while the fresh path
        re-prefills the whole growing prompt each turn. Timings are printed,
        never asserted (same scheduler-noise trap as test_two_turn_speedup).
        """
        m = tiny_model
        records = self._record_recomputes(m, monkeypatch)
        ids = np.array([[10, 20, 30]])
        max_new = 8

        fresh_times = []
        cached_times = []
        state = m.new_kv_state()

        for turn in range(3):
            t0 = time.perf_counter()
            self._generate_fresh(m, ids, max_new)
            fresh_times.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            r_cached = self._generate_cached(m, state, ids, max_new)
            cached_times.append(time.perf_counter() - t0)

            ids = np.concatenate([r_cached, np.array([[70 + turn]])], axis=1)

        total_fresh = sum(fresh_times)
        total_cached = sum(cached_times)
        speedup = total_fresh / total_cached if total_cached > 0 else float("inf")

        print(f"\n{'=' * 60}")
        print("Three-Turn Cumulative KV Cache Benchmark")
        print(f"{'=' * 60}")
        for i in range(3):
            print(f"Turn {i + 1} fresh:  {fresh_times[i] * 1000:8.2f} ms")
            print(f"Turn {i + 1} cached: {cached_times[i] * 1000:8.2f} ms")
        print(f"{'─' * 60}")
        print(f"Total fresh:   {total_fresh * 1000:8.2f} ms")
        print(f"Total cached:  {total_cached * 1000:8.2f} ms")
        print(f"Speedup:       {speedup:8.2f}x")
        print(f"{'=' * 60}")

        # Calls interleave fresh/cached per turn: [f1, c1, f2, c2, f3, c3].
        recomputed = [prompt_len - start for prompt_len, start in records]
        fresh_recomputed = recomputed[0::2]
        cached_recomputed = recomputed[1::2]
        prompt_lens = [records[i][0] for i in range(0, len(records), 2)]

        assert len(records) == 6, f"expected 6 generate calls, saw {len(records)}"
        # Fresh re-prefills the whole (growing) prompt every turn.
        assert fresh_recomputed == prompt_lens, (
            f"fresh recomputed {fresh_recomputed}, want full prompts {prompt_lens}"
        )
        # Cached: one prefill up front, then append-only recomputes.
        assert cached_recomputed == [prompt_lens[0], 2, 2], (
            f"cached recomputed {cached_recomputed}, want [{prompt_lens[0]}, 2, 2]"
        )
        # Cumulative work must keep growing apart, not just win once.
        assert sum(cached_recomputed) * 2 < sum(fresh_recomputed), (
            f"cumulative cached {sum(cached_recomputed)} must be less than half "
            f"of fresh {sum(fresh_recomputed)}"
        )

    def test_stream_matches_batch(self, tiny_model):
        """Streaming path produces same token count as batch path."""
        m = tiny_model
        ids = np.array([[10, 20, 30, 40]])
        max_new = 6

        state_b = m.new_kv_state()
        r_batch = self._generate_cached(m, state_b, ids, max_new)

        state_s = m.new_kv_state()
        stream_tokens = list(
            m.generate_numpy_stream(
                ids,
                max_new_tokens=max_new,
                temperature=0.0,
                kv_state=state_s,
            )
        )

        batch_new_tokens = r_batch.shape[1] - ids.shape[1]
        print(f"\nStream vs Batch: batch_new={batch_new_tokens} stream={len(stream_tokens)} tokens")

        assert batch_new_tokens == len(stream_tokens), (
            f"Token count mismatch: batch_new={batch_new_tokens} stream={len(stream_tokens)}"
        )

    def test_cache_state_evolution(self, tiny_model):
        """Verify kv_len grows correctly across turns."""
        m = tiny_model
        ids = np.array([[10, 20, 30]])
        state = m.new_kv_state()

        for turn in range(4):
            r = self._generate_cached(m, state, ids, max_new_tokens=5)
            ids = np.concatenate([r, np.array([[90 + turn]])], axis=1)

            kv_len = state.kv_len[0]
            print(
                f"Turn {turn + 1}: ids_len={ids.shape[1]}, kv_len={kv_len}, output_len={r.shape[1]}"
            )

        assert state.kv_len[0] > 20, f"kv_len too small after 4 turns: {state.kv_len[0]}"

    def test_compute_savings(self, tiny_model):
        """Verify KV cache reduces the number of tokens processed by the transformer."""
        m = tiny_model
        ids = np.array([[10, 20, 30, 40, 50, 60, 70, 80]])
        max_new = 10

        fresh_tokens_processed = 0
        fresh_ids = ids.copy()
        for turn in range(3):
            r = self._generate_fresh(m, fresh_ids, max_new)
            fresh_tokens_processed += fresh_ids.shape[1] + r.shape[1]
            fresh_ids = np.concatenate([r, np.array([[90 + turn]])], axis=1)

        cached_tokens_processed = 0
        cached_ids = ids.copy()
        state = m.new_kv_state()
        for turn in range(3):
            r = self._generate_cached(m, state, cached_ids, max_new)
            if turn == 0:
                cached_tokens_processed += cached_ids.shape[1] + r.shape[1]
            else:
                cached_tokens_processed += r.shape[1] + 1
            cached_ids = np.concatenate([r, np.array([[90 + turn]])], axis=1)

        savings = 1.0 - (cached_tokens_processed / fresh_tokens_processed)

        print(f"\n{'=' * 60}")
        print("Compute Savings (tokens through transformer)")
        print(f"{'=' * 60}")
        print(f"Fresh total:   {fresh_tokens_processed:6d} tokens")
        print(f"Cached total:  {cached_tokens_processed:6d} tokens")
        print(f"Savings:       {savings * 100:6.1f}%")
        print(f"{'=' * 60}")

        assert savings > 0, (
            f"KV cache should reduce compute: fresh={fresh_tokens_processed} "
            f"vs cached={cached_tokens_processed}"
        )


class TestKVStateBasics:
    """Test NumpyKVState creation, reset, and repr."""

    def test_new_kv_state(self, tiny_model):
        state = tiny_model.new_kv_state()
        assert state.kv_buf_k == []
        assert state.kv_buf_v == []
        assert state.kv_len == []
        assert state.prev_ids is None
        assert state.capacity == 0
        assert state.quantize_kv is False

    def test_kv_state_repr_empty(self, tiny_model):
        state = tiny_model.new_kv_state()
        r = repr(state)
        assert "NumpyKVCache" in r
        assert "capacity=0" in r

    def test_kv_state_reset(self, tiny_model):
        m = tiny_model
        state = m.new_kv_state()
        ids = np.array([[10, 20, 30]])
        m.generate_numpy(ids, max_new_tokens=3, temperature=0.0, kv_state=state)
        assert state.kv_len[0] > 0
        state.reset()
        assert state.kv_buf_k == []
        assert state.kv_buf_v == []
        assert state.kv_len == []
        assert state.prev_ids is None
        assert state.capacity == 0


class TestSingleTurnGeneration:
    """Test basic generation properties."""

    def test_output_shape(self, tiny_model):
        ids = np.array([[10, 20, 30]])
        r = tiny_model.generate_numpy(ids, max_new_tokens=5, temperature=0.0)
        assert r.shape[0] == 1
        assert r.shape[1] == 3 + 5

    def test_output_dtype(self, tiny_model):
        ids = np.array([[10, 20, 30]])
        r = tiny_model.generate_numpy(ids, max_new_tokens=5, temperature=0.0)
        assert r.dtype == np.int64

    def test_prompt_preserved(self, tiny_model):
        ids = np.array([[10, 20, 30]])
        r = tiny_model.generate_numpy(ids, max_new_tokens=5, temperature=0.0)
        np.testing.assert_array_equal(r[0, :3], ids[0])

    def test_greedy_deterministic(self, tiny_model):
        ids = np.array([[10, 20, 30]])
        r1 = tiny_model.generate_numpy(ids, max_new_tokens=5, temperature=0.0)
        r2 = tiny_model.generate_numpy(ids, max_new_tokens=5, temperature=0.0)
        np.testing.assert_array_equal(r1, r2)

    def test_temperature_zero_greedy(self, tiny_model):
        ids = np.array([[10, 20, 30]])
        r = tiny_model.generate_numpy(ids, max_new_tokens=5, temperature=0.0)
        assert r.shape[1] == 8

    def test_max_new_tokens_zero(self, tiny_model):
        ids = np.array([[10, 20, 30]])
        r = tiny_model.generate_numpy(ids, max_new_tokens=0, temperature=0.0)
        assert r.shape[1] == 3

    def test_single_token_input(self, tiny_model):
        ids = np.array([[42]])
        r = tiny_model.generate_numpy(ids, max_new_tokens=3, temperature=0.0)
        assert r.shape[1] == 4

    def test_long_input(self, tiny_model):
        ids = np.array([list(range(20))])
        r = tiny_model.generate_numpy(ids, max_new_tokens=5, temperature=0.0)
        assert r.shape[1] == 25

    def test_generate_returns_generate_result(self, tiny_model):
        from domain.training._internal.slonet import GenerateResult

        ids = np.array([[10, 20, 30]])
        r = tiny_model.generate_numpy(ids, max_new_tokens=5, temperature=0.0)
        assert isinstance(r, GenerateResult)

    def test_generated_ids_property(self, tiny_model):
        ids = np.array([[10, 20, 30]])
        r = tiny_model.generate_numpy(ids, max_new_tokens=5, temperature=0.0)
        assert r.generated_ids.shape[1] == 5

    def test_getitem(self, tiny_model):
        ids = np.array([[10, 20, 30]])
        r = tiny_model.generate_numpy(ids, max_new_tokens=5, temperature=0.0)
        assert r[0, 0] == 10

    def test_array_protocol(self, tiny_model):
        ids = np.array([[10, 20, 30]])
        r = tiny_model.generate_numpy(ids, max_new_tokens=5, temperature=0.0)
        arr = np.asarray(r)
        assert arr.shape == r.shape


class TestCachedVsFreshConsistency:
    """Verify cached and fresh generation produce identical results."""

    def test_same_tokens_greedy(self, tiny_model):
        m = tiny_model
        ids = np.array([[10, 20, 30, 40]])
        max_new = 8

        r_fresh = self._generate_fresh(m, ids, max_new)

        state = m.new_kv_state()
        r_cached = self._generate_cached(m, state, ids, max_new)

        np.testing.assert_array_equal(r_fresh, r_cached)

    def test_cached_prefix_reuse(self, tiny_model):
        m = tiny_model
        ids = np.array([[10, 20, 30]])
        state = m.new_kv_state()
        r1 = m.generate_numpy(ids, max_new_tokens=5, temperature=0.0, kv_state=state)

        ids2 = np.concatenate([r1, np.array([[60]])], axis=1)
        r2 = m.generate_numpy(ids2, max_new_tokens=5, temperature=0.0, kv_state=state)

        r2_fresh = m.generate_numpy(ids2, max_new_tokens=5, temperature=0.0, kv_state=None)

        np.testing.assert_array_equal(r2, r2_fresh)

    def _generate_fresh(self, model, input_ids, max_new_tokens, temperature=0.0):
        return model.generate_numpy(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            kv_state=None,
        )

    def _generate_cached(self, model, state, input_ids, max_new_tokens, temperature=0.0):
        return model.generate_numpy(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            kv_state=state,
        )


class TestMultiTurnPatterns:
    """Test various multi-turn conversation patterns."""

    def test_five_turn_conversation(self, tiny_model):
        m = tiny_model
        ids = np.array([[10, 20, 30]])
        state = m.new_kv_state()
        for turn in range(5):
            r = m.generate_numpy(ids, max_new_tokens=3, temperature=0.0, kv_state=state)
            ids = np.concatenate([r, np.array([[100 + turn]])], axis=1)
        assert state.kv_len[0] > 10

    def test_increasing_input_lengths(self, tiny_model):
        m = tiny_model
        state = m.new_kv_state()
        for length in [3, 5, 8, 12]:
            ids = np.array([list(range(10, 10 + length))])
            r = m.generate_numpy(ids, max_new_tokens=3, temperature=0.0, kv_state=state)
            assert r.shape[1] == length + 3

    def test_short_long_short_pattern(self, tiny_model):
        m = tiny_model
        state = m.new_kv_state()
        ids = np.array([[10, 20]])
        r = m.generate_numpy(ids, max_new_tokens=2, temperature=0.0, kv_state=state)
        ids = np.concatenate([r, np.array([[50]])], axis=1)
        r = m.generate_numpy(ids, max_new_tokens=8, temperature=0.0, kv_state=state)
        ids = np.concatenate([r, np.array([[60]])], axis=1)
        r = m.generate_numpy(ids, max_new_tokens=2, temperature=0.0, kv_state=state)
        assert r.shape[1] > 0

    def test_no_session_id_fresh_each_time(self, tiny_model):
        m = tiny_model
        ids = np.array([[10, 20, 30]])
        r1 = m.generate_numpy(ids, max_new_tokens=5, temperature=0.0)
        r2 = m.generate_numpy(ids, max_new_tokens=5, temperature=0.0)
        np.testing.assert_array_equal(r1, r2)

    def test_state_eviction_on_new_state(self, tiny_model):
        m = tiny_model
        s1 = m.new_kv_state()
        ids = np.array([[10, 20, 30]])
        m.generate_numpy(ids, max_new_tokens=5, temperature=0.0, kv_state=s1)
        s2 = m.new_kv_state()
        m.generate_numpy(ids, max_new_tokens=5, temperature=0.0, kv_state=s2)
        assert s1.kv_len[0] > 0
        assert s2.kv_len[0] > 0


class TestStreamGeneration:
    """Test streaming generation path."""

    def test_stream_yields_tokens(self, tiny_model):
        m = tiny_model
        ids = np.array([[10, 20, 30]])
        tokens = list(m.generate_numpy_stream(ids, max_new_tokens=5, temperature=0.0))
        assert len(tokens) == 5

    def test_stream_tokens_are_ints(self, tiny_model):
        m = tiny_model
        ids = np.array([[10, 20, 30]])
        tokens = list(m.generate_numpy_stream(ids, max_new_tokens=5, temperature=0.0))
        for t in tokens:
            assert isinstance(t, (int, np.integer))

    def test_stream_matches_batch_tokens(self, tiny_model):
        m = tiny_model
        ids = np.array([[10, 20, 30, 40]])
        max_new = 6

        r_batch = m.generate_numpy(ids, max_new_tokens=max_new, temperature=0.0)
        stream_tokens = list(m.generate_numpy_stream(ids, max_new_tokens=max_new, temperature=0.0))

        batch_new = r_batch[0, ids.shape[1] :].tolist()
        assert batch_new == stream_tokens

    def test_stream_with_kv_state(self, tiny_model):
        m = tiny_model
        ids = np.array([[10, 20, 30]])
        state = m.new_kv_state()
        tokens = list(
            m.generate_numpy_stream(ids, max_new_tokens=4, temperature=0.0, kv_state=state)
        )
        assert len(tokens) == 4
        assert state.kv_len[0] > 0

    def test_stream_empty_generation(self, tiny_model):
        m = tiny_model
        ids = np.array([[10, 20, 30]])
        tokens = list(m.generate_numpy_stream(ids, max_new_tokens=0, temperature=0.0))
        assert len(tokens) == 0


class TestGenerationMetrics:
    """Test GenerateResult metrics and properties."""

    def test_result_has_metrics(self, tiny_model):

        ids = np.array([[10, 20, 30]])
        r = tiny_model.generate_numpy(ids, max_new_tokens=5, temperature=0.0)
        assert hasattr(r, "metrics")

    def test_metrics_n_tokens(self, tiny_model):
        ids = np.array([[10, 20, 30]])
        r = tiny_model.generate_numpy(ids, max_new_tokens=5, temperature=0.0)
        assert r.metrics.n_tokens == 5

    def test_metrics_prompt_tokens(self, tiny_model):
        ids = np.array([[10, 20, 30]])
        r = tiny_model.generate_numpy(ids, max_new_tokens=5, temperature=0.0)
        assert r.metrics.prompt_tokens == 3

    def test_metrics_finalize(self, tiny_model):
        from domain.training._internal.slonet import GenerationMetrics

        m = GenerationMetrics()
        m.n_tokens = 10
        m.t_start = 0.1
        m.t_end = 1.0
        m.t_first_token = 0.5
        m.finalize()
        assert abs(m.tokens_per_sec - 10.0 / 0.9) < 0.01
        assert m.decode_ms == 900.0
        assert m.prefill_ms == 400.0

    def test_metrics_total_ms(self, tiny_model):
        from domain.training._internal.slonet import GenerationMetrics

        m = GenerationMetrics()
        m.t_start = 0.0
        m.t_end = 0.5
        assert m.total_ms == 500.0

    def test_metrics_ttft_ms(self, tiny_model):
        from domain.training._internal.slonet import GenerationMetrics

        m = GenerationMetrics()
        m.t_start = 0.0
        m.t_first_token = 0.1
        assert m.ttft_ms == 100.0

    def test_result_eq(self, tiny_model):
        ids = np.array([[10, 20, 30]])
        r1 = tiny_model.generate_numpy(ids, max_new_tokens=3, temperature=0.0)
        r2 = tiny_model.generate_numpy(ids, max_new_tokens=3, temperature=0.0)
        assert r1 == r2

    def test_result_ne_different(self, tiny_model):
        ids = np.array([[10, 20, 30]])
        r1 = tiny_model.generate_numpy(ids, max_new_tokens=3, temperature=0.0)
        r2 = tiny_model.generate_numpy(ids, max_new_tokens=5, temperature=0.0)
        assert r1 != r2


class TestStackCrossTurn:
    """End-to-end cross-turn KV through the production serving stack."""

    class _CharTokenizer:
        """Deterministic char-level tokenizer for the tiny vocab."""

        def __init__(self):
            self.eos_token_id = 0

        def encode(self, text):
            return [ord(c) % 256 for c in text]

        def decode(self, ids):
            return "".join(chr(i % 256) for i in ids)

        @staticmethod
        def chat_stop_ids():
            return ()

    class _StubProvider:
        """Minimal provider exposing the session KV map used by the server."""

        def __init__(self, model):
            from domain.inference._internal.session_kv_manager import SessionKVManager

            self._model = model
            self._kv_manager = SessionKVManager(kv_ttl=3600.0, kv_max_sessions=64)

        def _resolve_session_kv(self, session_id):
            if session_id is None:
                return None
            return self._kv_manager.get_or_create(session_id, self._model.new_kv_state)

        def _cached_tokens(self):
            return self._kv_manager.get_stats().get("active_sessions", 0)

        def _get_model(self):
            return self._model

    @pytest.fixture
    def stack(self, tiny_model):
        from domain.infrastructure._internal.slonet_server import SloNetServer

        provider = self._StubProvider(tiny_model)
        server = SloNetServer(
            model=tiny_model,
            tokenizer=self._CharTokenizer(),
            model_id="test-kv-bench",
            enable_warmup=False,
            provider=provider,
        )
        return server, provider

    @pytest.mark.asyncio
    async def test_three_turn_through_server(self, stack):
        """Three-turn chat through the server reuses KV and grows cached tokens."""
        server, provider = stack
        turns = [
            "Hello there",
            "Hello there, how are you today",
            "Hello there, how are you today? What is your name",
        ]
        times = []

        for i, t in enumerate(turns):
            t0 = time.perf_counter()
            out = await server.generate(t, max_new_tokens=6, temperature=0.0, session_id="sess-1")
            times.append(time.perf_counter() - t0)
            assert isinstance(out, str) and out

        cached = provider._cached_tokens()
        print(f"\n{'=' * 60}")
        print("Stack Cross-Turn Benchmark (SloNetServer.generate)")
        print(f"{'=' * 60}")
        for i, t in enumerate(times):
            print(f"Turn {i + 1}: {t * 1000:8.2f} ms")
        print(f"Total:        {sum(times) * 1000:8.2f} ms")
        print(f"Cached tokens after 3 turns: {cached}")
        print(f"{'=' * 60}")

        assert cached > 0, "cross-turn KV state must grow after 3 turns"
        assert times[1] <= times[0] * 3.0, (
            f"Turn 2 should be near turn 1 (suffix-only), got {times[1] * 1000:.2f} vs {times[0] * 1000:.2f} ms"
        )

    @pytest.mark.asyncio
    async def test_distinct_sessions_do_not_share_state(self, stack):
        server, provider = stack
        await server.generate("First message", max_new_tokens=4, temperature=0.0, session_id="s-a")
        await server.generate("Second message", max_new_tokens=4, temperature=0.0, session_id="s-b")
        assert len(provider._kv_manager.kv_states) == 2
        assert provider._kv_manager.kv_states["s-a"] is not provider._kv_manager.kv_states["s-b"]

    @pytest.mark.asyncio
    async def test_no_session_id_is_fresh_each_call(self, stack):
        server, provider = stack
        await server.generate("No session", max_new_tokens=4, temperature=0.0)
        assert len(provider._kv_manager.kv_states) == 0

    @pytest.mark.asyncio
    async def test_same_session_reuses_state(self, stack):
        server, provider = stack
        await server.generate("Hello", max_new_tokens=4, temperature=0.0, session_id="reuse-sess")
        assert len(provider._kv_manager.kv_states) == 1
        await server.generate(
            "Hello again", max_new_tokens=4, temperature=0.0, session_id="reuse-sess"
        )
        assert len(provider._kv_manager.kv_states) == 1

    @pytest.mark.asyncio
    async def test_server_returns_string(self, stack):
        server, _ = stack
        result = await server.generate(
            "test", max_new_tokens=3, temperature=0.0, session_id="str-test"
        )
        assert isinstance(result, str)


class TestModelProperties:
    def test_num_parameters(self, tiny_model):
        n = tiny_model.num_parameters()
        assert n > 0

    def test_vocab_size(self, tiny_model):
        assert tiny_model.vocab_size == 256

    def test_n_embed(self, tiny_model):
        assert tiny_model.n_embed == 64

    def test_n_layer(self, tiny_model):
        assert tiny_model.n_layer == 2

    def test_block_size(self, tiny_model):
        assert tiny_model.block_size == 128

    def test_max_seq_len(self, tiny_model):
        assert tiny_model.max_seq_len == 128

    def test_metadata(self, tiny_model):
        md = tiny_model.metadata
        assert md["vocab_size"] == 256
        assert md["n_embed"] == 64
        assert md["n_layer"] == 2

    def test_tok_emb_exists(self, tiny_model):
        assert tiny_model.tok_emb is not None

    def test_blocks_exist(self, tiny_model):
        blocks = tiny_model.blocks
        assert len(blocks) == 2

    def test_parameters_have_data(self, tiny_model):
        for p in tiny_model.parameters():
            assert p.data is not None
            assert p.data.size > 0


class TestInputHandling:
    def test_1d_input_reshaped(self, tiny_model):
        ids = np.array([[10, 20, 30]])
        r = tiny_model.generate_numpy(ids, max_new_tokens=3, temperature=0.0)
        assert r.shape[0] == 1

    def test_different_token_values(self, tiny_model):
        for token_id in [0, 1, 127, 255]:
            ids = np.array([[token_id]])
            r = tiny_model.generate_numpy(ids, max_new_tokens=2, temperature=0.0)
            assert r.shape[1] == 3

    def test_max_seq_len_limit(self, tiny_model):
        ids = np.array([list(range(10, 120))])
        r = tiny_model.generate_numpy(ids, max_new_tokens=50, temperature=0.0)
        assert r.shape[1] <= 128

    def test_empty_input_min_tokens(self, tiny_model):
        ids = np.array([[42]])
        r = tiny_model.generate_numpy(ids, max_new_tokens=0, temperature=0.0)
        assert r.shape[1] == 1

    def test_long_prompt_short_gen(self, tiny_model):
        ids = np.array([list(range(20))])
        r = tiny_model.generate_numpy(ids, max_new_tokens=1, temperature=0.0)
        assert r.shape[1] == 21
