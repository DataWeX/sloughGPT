"""Tests for scripts/benchmark_kv_reuse.py cross-turn KV reuse benchmark."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import benchmark_kv_reuse as bk  # noqa: E402


# ── prefix_match ────────────────────────────────────────────────────────────

def test_prefix_match_identical():
    """Equal sequences match fully."""
    assert bk.prefix_match([1, 2, 3], [1, 2, 3]) == 3


def test_prefix_match_partial():
    """First divergence cuts the match short."""
    assert bk.prefix_match([1, 2, 3], [1, 2, 9]) == 2


def test_prefix_match_prefix():
    """A strict prefix of the other sequence matches fully."""
    assert bk.prefix_match([1, 2], [1, 2, 3]) == 2


def test_prefix_match_empty():
    """Empty inputs match zero tokens."""
    assert bk.prefix_match([], [1, 2]) == 0
    assert bk.prefix_match([1, 2], []) == 0


def test_prefix_match_disjoint():
    """No common prefix yields zero."""
    assert bk.prefix_match([7], [8]) == 0


# ── benchmark invariants ────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def model():
    return bk.create_model()


@pytest.fixture(scope="module")
def metrics(model):
    """Run a tiny 2-turn benchmark once and reuse it across tests."""
    return bk.benchmark(model, [7], step=2, turns=2, max_tokens=2, steps=1)


def test_benchmark_structure(metrics):
    """Metrics dict has per-turn rows plus aggregates."""
    assert metrics["turns"] == 2
    assert len(metrics["rows"]) == 2
    assert {"turn", "prompt_len", "reused_tokens", "warm_ms",
            "cold_ms", "speedup", "consistency_pct"} <= set(metrics["rows"][0])


def test_benchmark_prompt_grows(metrics):
    """Later turns carry a longer prompt (history accumulates)."""
    lengths = [r["prompt_len"] for r in metrics["rows"]]
    assert lengths[1] > lengths[0]


def test_benchmark_reuse_grows_monotonically(metrics):
    """Reused tokens increase each turn as the cached history grows."""
    reused = [r["reused_tokens"] for r in metrics["rows"]]
    assert reused[0] == 0
    assert all(b > a for a, b in zip(reused, reused[1:]))


def test_benchmark_warm_cold_consistency(metrics):
    """KV-reused output matches fresh-computed output exactly."""
    assert all(r["consistency_pct"] == 100.0 for r in metrics["rows"])


def test_benchmark_positive_timings(metrics):
    """Warm and cold latencies are positive and finite."""
    for r in metrics["rows"]:
        assert r["warm_ms"] > 0
        assert r["cold_ms"] > 0
        assert r["speedup"] > 0


def test_benchmark_aggregates(metrics):
    """Aggregate fields derive from per-turn rows."""
    assert metrics["total_warm_ms"] == pytest.approx(
        sum(r["warm_ms"] for r in metrics["rows"]))
    assert metrics["total_reused_tokens"] == sum(
        r["reused_tokens"] for r in metrics["rows"])


# ── stack benchmark mode ────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def stack_metrics():
    """Run a tiny 2-turn serving-stack benchmark once and reuse it."""
    model = bk.create_model(vocab=256)
    return bk.benchmark_stack(model, [7], step=2, turns=2, max_tokens=2, steps=1)


def test_stack_benchmark_structure(stack_metrics):
    """Stack metrics expose the same row shape plus cached-token totals."""
    assert stack_metrics["mode"] == "stack"
    assert len(stack_metrics["rows"]) == 2
    assert {"turn", "prompt_len", "reused_tokens", "warm_ms",
            "cold_ms", "speedup", "consistency_pct"} <= set(
        stack_metrics["rows"][0])
    assert stack_metrics["cached_tokens"] > 0


def test_stack_benchmark_reuse_grows(stack_metrics):
    """Server-side session KV grows monotonically across turns."""
    reused = [r["reused_tokens"] for r in stack_metrics["rows"]]
    assert reused[0] == 0
    assert all(b > a for a, b in zip(reused, reused[1:]))


def test_stack_benchmark_consistency(stack_metrics):
    """Warm and cold outputs agree — temp 0 through the server is greedy."""
    assert all(r["consistency_pct"] == 100.0 for r in stack_metrics["rows"])


# ── streaming stack benchmark mode ──────────────────────────────────────────

@pytest.fixture(scope="module")
def stack_stream_metrics():
    """Tiny 2-turn streaming serving-stack benchmark (the /chat/stream path)."""
    model = bk.create_model(vocab=256)
    return bk.benchmark_stack(model, [7], step=2, turns=2, max_tokens=2,
                              steps=1, stream=True)


def test_stack_stream_benchmark_structure(stack_stream_metrics):
    """Streaming mode is flagged and carries cached-token totals."""
    assert stack_stream_metrics["mode"] == "stack-stream"
    assert len(stack_stream_metrics["rows"]) == 2
    assert stack_stream_metrics["cached_tokens"] > 0


def test_stack_stream_benchmark_prompt_grows(stack_stream_metrics):
    """Streaming keeps the full prior turn so prompts accumulate."""
    lengths = [r["prompt_len"] for r in stack_stream_metrics["rows"]]
    assert lengths[1] > lengths[0]


def test_stack_stream_benchmark_reuse_grows(stack_stream_metrics):
    """Session KV reuse grows monotonically under token-by-token streaming."""
    reused = [r["reused_tokens"] for r in stack_stream_metrics["rows"]]
    assert reused[0] == 0
    assert all(b > a for a, b in zip(reused, reused[1:]))


def test_stack_stream_benchmark_consistency(stack_stream_metrics):
    """Streamed warm and cold outputs agree bit-for-bit."""
    assert all(r["consistency_pct"] == 100.0 for r in stack_stream_metrics["rows"])


def test_stack_reuse_matches_prefix_of_prev_output(stack_metrics):
    """Stack reuse equals the cached previous turn (prompt + max_tokens out),
    the honest prefix-match length — not the inflated session state fill."""
    r0, r1 = stack_metrics["rows"]
    assert r0["reused_tokens"] == 0
    assert r1["reused_tokens"] == r0["prompt_len"] + 2  # max_tokens=2


# ── temperature-0 determinism regression ────────────────────────────────────

def test_temp_zero_with_top_p_is_deterministic_greedy():
    """temperature≈0 must be argmax even when the serving stack sets
    top_k/top_p defaults — top-k/nucleus filtering cannot change the argmax.
    Regression for stack-mode non-determinism (unseeded np.random.choice)."""
    from domains.training.slonet import _sample_from_logits

    logits = np.array([[0.1, 0.9, 0.3, 0.2]], dtype=np.float64)
    a = _sample_from_logits(logits, temperature=0.0, top_k=50, top_p=0.9)
    b = _sample_from_logits(logits, temperature=0.0, top_k=50, top_p=0.9)
    assert a == b == 1
    # Same token as the unfiltered argmax — top_p did not change the result.
    c = _sample_from_logits(logits, temperature=0.0, top_k=None, top_p=None)
    assert a == c


# ── int8 quantized KV cache (--quantize-kv) ─────────────────────────────────

@pytest.fixture(scope="module")
def quant_metrics():
    """Tiny 2-turn direct-mode benchmark with the KV cache stored as int8."""
    model = bk.create_model()
    return bk.benchmark(model, [7], step=2, turns=2, max_tokens=2, steps=1,
                        quantize_kv=True)


@pytest.fixture(scope="module")
def quant_stack_metrics():
    """Tiny 2-turn serving-stack benchmark with int8 KV via the server."""
    model = bk.create_model(vocab=256)
    return bk.benchmark_stack(model, [7], step=2, turns=2, max_tokens=2,
                              steps=1, quantize_kv=True)


@pytest.fixture(scope="module")
def quant_stack_stream_metrics():
    """Tiny 2-turn streaming serving-stack benchmark with int8 KV."""
    model = bk.create_model(vocab=256)
    return bk.benchmark_stack(model, [7], step=2, turns=2, max_tokens=2,
                              steps=1, stream=True, quantize_kv=True)


def test_quant_benchmark_reuse_grows(quant_metrics):
    """int8 KV state still accumulates the cross-turn prefix."""
    reused = [r["reused_tokens"] for r in quant_metrics["rows"]]
    assert reused[0] == 0
    assert all(b > a for a, b in zip(reused, reused[1:]))


def test_quant_benchmark_consistency(quant_metrics):
    """Warm int8-reused output matches cold fresh-int8 output bit-for-bit."""
    assert all(r["consistency_pct"] == 100.0 for r in quant_metrics["rows"])


def test_quant_stack_benchmark_reuse_grows(quant_stack_metrics):
    """Server-side int8 session KV grows across turns."""
    reused = [r["reused_tokens"] for r in quant_stack_metrics["rows"]]
    assert reused[0] == 0
    assert all(b > a for a, b in zip(reused, reused[1:]))


def test_quant_stack_benchmark_consistency(quant_stack_metrics):
    """int8 through the serving stack is deterministic greedy."""
    assert all(r["consistency_pct"] == 100.0
               for r in quant_stack_metrics["rows"])


def test_quant_stack_stream_benchmark_reuse_grows(quant_stack_stream_metrics):
    """int8 session KV reuse grows under token-by-token streaming."""
    reused = [r["reused_tokens"] for r in quant_stack_stream_metrics["rows"]]
    assert reused[0] == 0
    assert all(b > a for a, b in zip(reused, reused[1:]))


def test_quant_stack_stream_benchmark_consistency(quant_stack_stream_metrics):
    """Streamed int8 warm and cold outputs agree bit-for-bit."""
    assert all(r["consistency_pct"] == 100.0
               for r in quant_stack_stream_metrics["rows"])


def test_quant_reuse_matches_float_reuse():
    """Reuse is a prefix-match quantity — independent of KV dtype. The int8
    path must report the exact same cached-token growth as float32."""
    model = bk.create_model()
    float_metrics = bk.benchmark(model, [7], step=2, turns=2,
                                 max_tokens=2, steps=1)
    quant_metrics = bk.benchmark(model, [7], step=2, turns=2,
                                 max_tokens=2, steps=1, quantize_kv=True)
    assert [r["reused_tokens"] for r in float_metrics["rows"]] == \
        [r["reused_tokens"] for r in quant_metrics["rows"]]


# ── KV cache memory accounting (kv_memory_kb) ───────────────────────────────

def test_kv_state_memory_empty():
    """An empty state owns no buffers → 0 KiB."""
    from domains.training.slonet import NumpyKVState
    assert bk.kv_state_memory_kb(NumpyKVState()) == 0


def test_kv_state_memory_quantized_is_smaller():
    """int8 state holds the same sequence with a fraction of the memory.

    For head_dim=E and nkv heads: float32 uses 2*E*4 bytes per (token, head);
    int8 uses 2*E*1 + 2*4 (per-token-head float32 scales). At the benchmark
    model's E=16 that is 8E vs 2E+8 bytes → a 3.2x reduction."""
    model = bk.create_model()
    base = np.array([[7]], dtype=np.int64)
    state_f = model.new_kv_state()
    model.generate_numpy(base, max_new_tokens=4, temperature=0.0,
                         kv_state=state_f, quantize_kv=False)
    state_q = model.new_kv_state()
    model.generate_numpy(base, max_new_tokens=4, temperature=0.0,
                         kv_state=state_q, quantize_kv=True)

    mem_f = bk.kv_state_memory_kb(state_f)
    mem_q = bk.kv_state_memory_kb(state_q)
    assert mem_f > 0
    assert mem_q < mem_f
    assert mem_f / mem_q == pytest.approx(3.2, rel=0.05)


def test_benchmark_kv_memory_grows_with_turns(metrics):
    """KV memory grows as the cached sequence lengthens each turn."""
    mem = [r["kv_memory_kb"] for r in metrics["rows"]]
    assert mem[0] > 0
    assert all(b > a for a, b in zip(mem, mem[1:]))


def test_quant_benchmark_kv_memory_below_float():
    """At matched turns the int8 benchmark holds strictly less KV memory."""
    model = bk.create_model()
    float_metrics = bk.benchmark(model, [7], step=2, turns=2,
                                 max_tokens=2, steps=1)
    quant_metrics = bk.benchmark(model, [7], step=2, turns=2,
                                 max_tokens=2, steps=1, quantize_kv=True)
    for rf, rq in zip(float_metrics["rows"], quant_metrics["rows"]):
        assert rq["kv_memory_kb"] < rf["kv_memory_kb"]


def test_stack_benchmark_kv_memory_present(stack_metrics):
    """Stack-mode rows carry the server-side session KV memory."""
    assert all(r["kv_memory_kb"] > 0 for r in stack_metrics["rows"])


# ── float32 vs int8 KV quality (--compare-kv) ───────────────────────────────

@pytest.fixture(scope="module")
def kv_quality():
    """Tiny 2-turn float-vs-int8 KV quality comparison."""
    model = bk.create_model()
    return bk.compare_kv_quality(model, [7], step=2, turns=2, max_tokens=4)


def test_kv_quality_structure(kv_quality):
    """Quality metrics carry per-turn rows and an overall agreement."""
    assert kv_quality["mode"] == "compare-kv"
    assert len(kv_quality["rows"]) == 2
    assert {"turn", "prompt_len", "generated", "identical_pct",
            "prefix_agree", "prefix_pct"} <= set(kv_quality["rows"][0])
    assert 0.0 <= kv_quality["overall_identical_pct"] <= 100.0


def test_kv_quality_lengths_consistent(kv_quality):
    """Generated length stays within [prompt_len, prompt_len + max_tokens]."""
    for r in kv_quality["rows"]:
        assert r["generated"] >= r["prompt_len"]
        assert r["generated"] <= r["prompt_len"] + 4


def test_kv_quality_turn0_perfect(kv_quality):
    """A 4-token generation from identical input is untouched by int8 KV."""
    assert kv_quality["rows"][0]["identical_pct"] == 100.0
    assert kv_quality["rows"][0]["prefix_pct"] == 100.0


def test_kv_quality_high_agreement(kv_quality):
    """int8 KV keeps near-perfect greedy agreement on the tiny model."""
    assert kv_quality["overall_identical_pct"] >= 90.0


def test_kv_quality_metrics_bounds(kv_quality):
    """Prefix agreement cannot exceed the generated length."""
    for r in kv_quality["rows"]:
        assert 0 <= r["prefix_agree"] <= r["generated"]
        assert r["identical_pct"] >= r["prefix_pct"]


def test_kv_quality_main_exit_zero(monkeypatch):
    """`--compare-kv` with healthy agreement exits 0."""
    import benchmark_kv_reuse as _bk  # noqa: F811 (fresh module, cached above)
    monkeypatch.setattr(
        sys, "argv", ["benchmark_kv_reuse.py", "--compare-kv",
                      "--turns", "2", "--max-tokens", "4"])
    assert _bk.main() == 0


def test_kv_quality_main_warns_on_low_agreement(monkeypatch):
    """Agreement below 50% fails loudly with a warning."""
    import benchmark_kv_reuse as _bk
    monkeypatch.setattr(
        _bk, "compare_kv_quality",
        lambda *a, **k: {"mode": "compare-kv", "rows": [], "overall_identical_pct": 20.0})
    monkeypatch.setattr(
        sys, "argv", ["benchmark_kv_reuse.py", "--compare-kv"])
    assert _bk.main() == 1


# ── concurrent sessions (--sessions N) ──────────────────────────────────────

@pytest.fixture(scope="module")
def session_metrics():
    """Tiny 2-session interleaved serving-stack benchmark."""
    model = bk.create_model(vocab=256)
    return bk.benchmark_sessions(model, [7], step=2, turns=2, max_tokens=2,
                                 steps=1, n_sessions=2)


@pytest.fixture(scope="module")
def session_stream_metrics():
    """Tiny 2-session interleaved streaming stack benchmark."""
    model = bk.create_model(vocab=256)
    return bk.benchmark_sessions(model, [7], step=2, turns=2, max_tokens=2,
                                 steps=1, n_sessions=2, stream=True)


def test_sessions_structure(session_metrics):
    """Rows are tagged with session and turn; isolation is asserted."""
    assert session_metrics["mode"] == "stack-sessions"
    assert session_metrics["n_sessions"] == 2
    assert len(session_metrics["rows"]) == 4  # 2 sessions × 2 turns
    assert {"session", "turn", "prompt_len", "reused_tokens",
            "kv_memory_kb", "warm_ms", "cold_ms", "speedup",
            "consistency_pct"} <= set(session_metrics["rows"][0])
    assert session_metrics["isolation_ok"]


def test_sessions_per_session_reuse_grows(session_metrics):
    """Each session's reuse grows monotonically despite interleaving."""
    for reuse in session_metrics["per_session_reuse"]:
        assert reuse[0] == 0
        assert all(b > a for a, b in zip(reuse, reuse[1:]))


def test_sessions_consistency(session_metrics):
    """Warm and cold agree bit-for-bit for every interleaved request."""
    assert all(r["consistency_pct"] == 100.0
               for r in session_metrics["rows"])


def test_sessions_isolation_matches_single_session():
    """Session 0's reuse under interleaving equals a lone-session stack run —
    evidence no session leaks another session's cached prefix."""
    model = bk.create_model(vocab=256)
    solo = bk.benchmark_stack(model, [7], step=2, turns=2, max_tokens=2,
                              steps=1)
    multi = bk.benchmark_sessions(model, [7], step=2, turns=2, max_tokens=2,
                                  steps=1, n_sessions=2)
    solo_reuse = [r["reused_tokens"] for r in solo["rows"]]
    assert multi["per_session_reuse"][0] == solo_reuse
    assert multi["per_session_reuse"][1] == solo_reuse


def test_sessions_streaming_isolation(session_stream_metrics):
    """Streaming interleaved sessions keep isolation and consistency."""
    assert session_stream_metrics["isolation_ok"]
    assert all(r["consistency_pct"] == 100.0
               for r in session_stream_metrics["rows"])
    for reuse in session_stream_metrics["per_session_reuse"]:
        assert reuse[0] == 0
        assert all(b > a for a, b in zip(reuse, reuse[1:]))
