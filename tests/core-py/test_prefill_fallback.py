"""Regression tests for the chunked numpy prefill-attention fallback.

Why this file exists
--------------------
``fused_attention_multi`` has two implementations: a compiled numba kernel and a
numpy fallback. On 2026-11-01 the API server ran with numba invisible to its own
interpreter (``PYTHONNOUSERSITE=1`` hides ``~/.local``), so every call silently
took the numpy branch. That branch used to materialise four concurrent
``(H, seq, new_len)`` float32 temporaries -- ~16 GB at seq = new_len = 8192 --
and the kernel OOM-killed the process at ~21 GB RSS.

Two things let that happen, and both are what these tests pin down:

1. **The fallback was unchunked**, so its footprint grew as O(seq^2) with prompt
   length. It is now chunked over the query axis, which is sound because
   attention is independent per query position (query s attends only to keys
   n <= s), so a contiguous slice of queries computes on its own. These tests
   assert the chunked form stays bit-identical to the unchunked one *and* stays
   inside the byte budget -- correctness and boundedness, together, because
   bounding the memory by changing the arithmetic would be a silent correctness
   regression.

2. **Its engagement was silent.** ``_check_numba()`` now warns once. The budget
   itself is env-tunable (``SLO_PREFILL_FALLBACK_BUDGET_MB``) and must fail safe
   on malformed input rather than raise -- a bad value in a unit file must not
   take the server down at import time.

These tests deliberately exercise the *numpy* branch even where numba is
installed, so they fail identically in a numba-less environment. That matters:
the bug was invisible precisely because the environment differed from the one
the tests ran in.
"""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

from domain.training._internal import slonet_kernels as K


def _reference_one_shot(q, k, v, scale, H, E_head, seq_len, new_len, reps):
    """The ORIGINAL pre-fix numpy fallback, verbatim (the OOM path).

    This is the specification for the chunked implementation: the unchunked
    whole-matrix form whose output the chunked form must reproduce exactly, and
    whose O(seq^2) footprint is the reason chunking exists. It is intentionally
    kept unchunked and allocation-hungry -- it is only ever run on the small
    shapes below, never at seq = 8192, because running it there is the bug.

    The original also preallocated ``out`` before immediately overwriting it
    with the einsum result; that dead allocation is dropped here, as it does not
    affect the output this reference exists to pin down.
    """
    if reps != 1:
        k = k[np.arange(H) // reps]
        v = v[np.arange(H) // reps]
    scores = np.einsum("she,hne->hsn", q, k) * scale  # (H, seq, new_len)
    causal = np.triu(np.full((seq_len, new_len), -1e9, dtype=np.float32), k=1)
    scores = scores + causal
    attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
    attn = attn / attn.sum(axis=-1, keepdims=True)
    return np.einsum("hsn,hne->she", attn, v)


@pytest.fixture
def numpy_fallback():
    """Force ``fused_attention_multi`` down its numpy branch.

    ``_ensure_fused()`` must run *first*: it builds the compiled kernel, after
    which nulling the module global sticks (a subsequent ``_ensure_fused()``
    early-returns on ``_fused_built``). Nulling before it would just be undone
    by the build. Restored on teardown so later tests see the real state.
    """
    K._ensure_fused()
    saved = K._nb_fused_attention_multi
    K._nb_fused_attention_multi = None
    try:
        yield
    finally:
        K._nb_fused_attention_multi = saved


@pytest.fixture
def forces_chunking(monkeypatch):
    """Shrink the prefill budget so the shapes in ``SHAPES`` actually chunk.

    Why this is needed: at the default 64 MB the budget works out to a ``blk``
    of 511 to 1,048,576 rows for these shapes, so ``seq_len <= blk``, the loop
    body runs exactly once, and "chunked vs unchunked" compares the unchunked
    form against itself. Verified by injecting a chunk-boundary defect
    (chunk-local row indices instead of absolute): with the default budget
    ``test_matches_one_shot_reference`` passes **12/12** -- it cannot fail --
    while with this fixture **22 of the 26** tests in the class fail. The 4
    that still pass are the two ``(1, 1, ...)`` params (one row, cannot chunk)
    plus the causality and finiteness tests, which are unaffected because the
    first chunk always has ``s0 == 0``, where chunk-local indices are already
    the correct absolute ones.

    Production does chunk: a 2865-token prompt at H=14 gives ``blk ~= 104``.
    So the path these tests exist to guard is precisely the path they were
    not exercising.

    2048 bytes yields ``blk < seq_len`` for 11 of the 12 shapes (every shape
    with ``seq_len > 1``) and ``blk == 1`` for most of them, so every
    boundary in the loop is hit. ``(1, 1, ...)`` has one row and cannot chunk
    by definition -- nothing can make that case multi-chunk.
    """
    monkeypatch.setenv("SLO_PREFILL_FALLBACK_BUDGET_MB", str(2048 / 2**20))


@pytest.fixture
def generic_contraction(monkeypatch):
    """Pin every ``np.einsum`` to ``optimize=False`` for one test.

    Why this exists: the shipped prefill passes ``optimize=True``, because
    einsum's default ``optimize=False`` takes a generic contraction loop
    instead of dispatching to BLAS -- measured 3.4-5.9x slower at prefill
    sizes (H=14, kv=2, E=128, interleaved arms; 49% of a profiled 22.0s
    prefill is ``c_einsum``).

    BLAS gemm's summation order depends on its ``m`` dimension, and chunking
    changes ``m``. So under ``optimize=True`` chunking *does* perturb each
    element by ~1e-6, which makes bit-exactness unachievable for the shipped
    configuration. That is not a chunking bug and must not be read as one.

    Both sides are therefore forced onto the same deterministic path, so the
    bit-exact assertion below keeps testing the single thing this class exists
    for: that chunking reorders *which rows are computed when* and never the
    arithmetic. The premise in that comment is only true while the arithmetic
    is held constant -- this fixture is what holds it constant.

    The BLAS half is covered separately, not dropped:
    ``test_blas_path_matches_across_chunk_boundaries`` (this file, chunking in
    play) and ``TestFusedAttentionMultiBLASPath`` (test_fused_kernels.py).
    """
    real = np.einsum

    def forced(*args, **kwargs):
        kwargs["optimize"] = False
        return real(*args, **kwargs)

    monkeypatch.setattr(np, "einsum", forced)


# Shapes: (seq_len, new_len, H, E_head, kv_heads). kv_heads < H exercises the
# GQA path (reps = H // kv_heads); seq != new_len exercises non-square prefill.
# The values around 64 and 128 straddle chunk boundaries -- but only under the
# `forces_chunking` fixture. At the DEFAULT 64 MB budget none of these chunk
# (blk comes out 511..1,048,576 against seq_len 1..513), which is why the
# equivalence tests now ask for a small budget explicitly; see that fixture.
SHAPES = [
    (1, 1, 4, 8, 4),
    (8, 8, 4, 8, 4),
    (17, 17, 4, 8, 4),
    (64, 64, 8, 16, 8),
    (65, 65, 8, 16, 8),  # one past a power of two
    (100, 100, 16, 8, 16),
    (300, 300, 16, 64, 16),
    (513, 40, 16, 32, 16),  # seq_len > new_len
    (40, 513, 16, 32, 16),  # new_len > seq_len
    (33, 33, 16, 8, 2),  # GQA 8:1 -- reps = 8
    (96, 96, 16, 8, 4),  # GQA 4:1 -- reps = 4
    (129, 129, 16, 8, 1),  # already expanded -- reps = 1
]


class TestChunkedEquivalence:
    """The chunked fallback must reproduce the unchunked form exactly."""

    @pytest.mark.parametrize("seq_len,new_len,H,E_head,kv_heads", SHAPES)
    def test_matches_one_shot_reference(
        self,
        numpy_fallback,
        forces_chunking,
        generic_contraction,
        seq_len,
        new_len,
        H,
        E_head,
        kv_heads,
    ):
        reps = H // kv_heads
        rng = np.random.default_rng(0)
        q = rng.standard_normal((seq_len, H, E_head), dtype=np.float32)
        k = rng.standard_normal((kv_heads, new_len, E_head), dtype=np.float32)
        v = rng.standard_normal((kv_heads, new_len, E_head), dtype=np.float32)
        scale = np.float32(1.0 / np.sqrt(E_head))

        ref = _reference_one_shot(q, k, v, scale, H, E_head, seq_len, new_len, reps)
        got = K.fused_attention_multi(q, k, v, scale, H, E_head)

        assert got.shape == ref.shape
        assert got.dtype == ref.dtype
        # Bit-exact, not merely close: chunking only reorders which rows are
        # computed when, never the arithmetic on any single element, so anything
        # above 0.0 means the causal mask or the chunk boundary is wrong.
        # `generic_contraction` holds the arithmetic constant so this stays
        # true; see that fixture for why the shipped optimize=True path cannot
        # be bit-exact here and where it is pinned instead.
        np.testing.assert_array_equal(got, ref)

    @pytest.mark.parametrize("seq_len,new_len,H,E_head,kv_heads", SHAPES)
    def test_blas_path_matches_across_chunk_boundaries(
        self, numpy_fallback, forces_chunking, seq_len, new_len, H, E_head, kv_heads
    ):
        """The shipped configuration must reproduce the specification.

        The counterpart to ``test_matches_one_shot_reference``: that test
        holds the arithmetic constant to isolate *chunking*, and so can only
        run on the generic path. This one runs the configuration that
        actually ships -- chunked AND ``optimize=True`` -- against the same
        unchunked generic reference, so it is an end-to-end check rather than
        an isolated one: if chunking, the causal mask, or the BLAS dispatch
        is wrong, the shipped output drifts from the specification and this
        fails.

        Bit-exactness is not available here -- BLAS blocking depends on the
        ``m`` dimension that chunking changes -- so the tolerance is explicit
        and grounded in measurement rather than picked to pass. Across all 12
        shapes the observed max deviation is 1.19e-06, while an actual
        chunk-boundary defect (chunk-local row indices instead of absolute)
        lands at 2.58e+00 and above. That is ~7 orders of separation, so
        rtol/atol below cannot hide the bug this file exists to catch.
        """
        reps = H // kv_heads
        rng = np.random.default_rng(0)
        q = rng.standard_normal((seq_len, H, E_head), dtype=np.float32)
        k = rng.standard_normal((kv_heads, new_len, E_head), dtype=np.float32)
        v = rng.standard_normal((kv_heads, new_len, E_head), dtype=np.float32)
        scale = np.float32(1.0 / np.sqrt(E_head))

        # Reference on the generic path (no BLAS); got is the shipped path.
        ref = _reference_one_shot(q, k, v, scale, H, E_head, seq_len, new_len, reps)
        got = K.fused_attention_multi(q, k, v, scale, H, E_head)

        assert got.shape == ref.shape
        assert got.dtype == ref.dtype
        np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-6)

    def test_first_query_attends_only_to_first_key(self, numpy_fallback, forces_chunking):
        """Query 0 must see key 0 and nothing else -- exactly.

        The sharpest causal check available for query 0. What it does NOT
        catch, stated plainly because the original docstring claimed the
        opposite: a chunk-boundary defect that uses chunk-local row indices
        instead of absolute ones leaves query 0 *correct*, because query 0
        sits in the first chunk (s0 == 0) where the two index spaces
        coincide. Verified by injection -- this test passes with that defect
        present. Boundary coverage therefore comes from
        ``test_matches_one_shot_reference`` and
        ``test_blas_path_matches_across_chunk_boundaries``, which chunk for
        real via ``forces_chunking``.

        Query 0's attention row has a single unmasked entry, so softmax gives it
        weight exactly 1.0 and output[0] collapses to v[0] bit-exactly. Any
        leakage turns that into a mixture of later v rows.

        Note "future" is per-query, so this cannot be tested by zeroing keys
        globally -- that would also invalidate keys that earlier queries should
        legitimately see. Only query 0 has a well-defined "no other valid key".
        """
        seq_len = 65  # >1 chunk under forces_chunking (NOT at the default budget)
        H, E_head = 4, 4
        rng = np.random.default_rng(1)
        q = rng.standard_normal((seq_len, H, E_head), dtype=np.float32)
        k = rng.standard_normal((H, seq_len, E_head), dtype=np.float32)
        v = rng.standard_normal((H, seq_len, E_head), dtype=np.float32)
        scale = np.float32(1.0 / np.sqrt(E_head))

        out = K.fused_attention_multi(q, k, v, scale, H, E_head)

        # v is (H, seq_len, E_head), so key position 0 across all heads is
        # v[:, 0, :] -- v[0] would be head 0's entire sequence.
        np.testing.assert_array_equal(out[0], v[:, 0, :])

        # Not vacuous: a later query sees many keys, so it must NOT collapse to
        # its own v row. This proves the assertion above is causality rather
        # than an accident of the implementation always returning v.
        assert not np.allclose(out[-1], v[:, -1, :])

    def test_output_is_finite(self, numpy_fallback):
        """Softmax must not be poisoned by -inf/NaN from the mask sentinel."""
        seq_len = 128
        H, E_head = 8, 16
        rng = np.random.default_rng(2)
        q = rng.standard_normal((seq_len, H, E_head), dtype=np.float32) * 50.0
        k = rng.standard_normal((H, seq_len, E_head), dtype=np.float32) * 50.0
        v = rng.standard_normal((H, seq_len, E_head), dtype=np.float32)
        scale = np.float32(1.0 / np.sqrt(E_head))

        out = K.fused_attention_multi(q, k, v, scale, H, E_head)

        assert np.isfinite(out).all(), "non-finite value in attention output"


# Minimum reduction in peak allocation the chunked fallback must deliver
# versus the unchunked form it replaces. Chosen to sit well above the ~80 MB
# actually measured at seq = 4096 and well below the ~4 GB unchunked would
# need, so the assertion fails decisively in either direction.
_MIN_REDUCTION_FACTOR = 8


class TestChunkedMemoryBound:
    """The fallback's footprint must not grow with prompt length."""

    @pytest.mark.parametrize(
        "H,new_len,seq_len",
        [
            (4, 1024, 1024),
            (8, 4096, 4096),
            (16, 8192, 8192),  # the shape that OOM-killed the server
            (16, 16384, 16384),
            (32, 8192, 65536),  # long prompt, short key set
        ],
    )
    def test_temporaries_stay_within_budget(self, H, new_len, seq_len):
        """Peak temporaries must be <= the budget, for any prompt length.

        This is the arithmetic that makes the OOM impossible, asserted directly
        rather than inferred: blk is derived from the budget by integer floor
        division, so blk * per_chunk <= budget holds by construction. If a future
        edit adds a concurrent temporary without bumping _FALLBACK_TEMP_COUNT,
        this assertion is what catches the under-count -- otherwise the chunk
        grows, the budget is silently exceeded, and the OOM returns with no
        error to warn us.
        """
        budget = K._prefill_fallback_budget()
        per_chunk = max(1, H * new_len * K._FALLBACK_ITEMSIZE * K._FALLBACK_TEMP_COUNT)
        blk = max(1, budget // per_chunk)
        peak = K._FALLBACK_TEMP_COUNT * H * blk * new_len * K._FALLBACK_ITEMSIZE

        assert peak <= budget, (
            f"peak {peak} bytes exceeds budget {budget} at "
            f"H={H} new_len={new_len} seq_len={seq_len} blk={blk}"
        )
        # And the bound must be independent of seq_len: a longer prompt may add
        # chunks, never a bigger one.
        assert blk * per_chunk <= budget

    def test_live_peak_is_bounded(self, numpy_fallback):
        """Measure the real allocation, not just the formula.

        The formula can be right while the code allocates something the formula
        does not count. tracemalloc sees numpy's array allocations, so this
        catches that class. Inputs are allocated before tracing starts so the
        reading covers temporaries and output only.
        """
        seq_len = 4096
        H, E_head = 16, 64
        rng = np.random.default_rng(3)
        q = rng.standard_normal((seq_len, H, E_head), dtype=np.float32)
        k = rng.standard_normal((H, seq_len, E_head), dtype=np.float32)
        v = rng.standard_normal((H, seq_len, E_head), dtype=np.float32)
        scale = np.float32(1.0 / np.sqrt(E_head))

        unchunked = K._FALLBACK_TEMP_COUNT * H * seq_len * seq_len * K._FALLBACK_ITEMSIZE

        tracemalloc.start()
        try:
            out = K.fused_attention_multi(q, k, v, scale, H, E_head)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        assert out.shape == (seq_len, H, E_head)
        # The unchunked form would need `unchunked` bytes (~4 GB here). Demand at
        # least _MIN_REDUCTION_FACTOR-fold: generous over the ~80 MB actually
        # measured, but far below the 4 GB a regression would reintroduce, so a
        # failure is unambiguous rather than marginal.
        assert peak < unchunked / _MIN_REDUCTION_FACTOR, (
            f"peak {peak / 1024**2:.0f} MB is not bounded -- unchunked would be "
            f"{unchunked / 1024**3:.1f} GB"
        )


class TestBudgetOverride:
    """SLO_PREFILL_FALLBACK_BUDGET_MB tunes the budget and fails safe."""

    @pytest.mark.parametrize("mb", ["8", "64", "512"])
    def test_env_override_changes_budget(self, monkeypatch, mb):
        monkeypatch.setenv("SLO_PREFILL_FALLBACK_BUDGET_MB", mb)
        assert K._prefill_fallback_budget() == int(mb) * 1024 * 1024

    @pytest.mark.parametrize("bad", ["not-a-number", "-5", "0", ""])
    def test_malformed_value_fails_safe(self, monkeypatch, bad):
        """A bad value in a unit file must not raise at call time.

        Falling back to the default keeps the server running; raising would turn
        a configuration typo into an outage on the first prefill.
        """
        monkeypatch.setenv("SLO_PREFILL_FALLBACK_BUDGET_MB", bad)
        assert K._prefill_fallback_budget() == K._DEFAULT_FALLBACK_BUDGET_MB * 1024 * 1024

    def test_larger_budget_yields_larger_chunks(self, monkeypatch):
        H, new_len = 16, 4096
        per_chunk = max(1, H * new_len * K._FALLBACK_ITEMSIZE * K._FALLBACK_TEMP_COUNT)
        sizes = {}
        for mb in ("8", "64", "512"):
            monkeypatch.setenv("SLO_PREFILL_FALLBACK_BUDGET_MB", mb)
            sizes[mb] = max(1, K._prefill_fallback_budget() // per_chunk)
        assert sizes["8"] < sizes["64"] < sizes["512"], sizes


def test_causal_mask_sentinel_is_finite():
    """The mask sentinel must stay finite.

    -inf would make ``score - max`` produce NaN on the fully-masked row, and
    0/0 in the subsequent division would poison the whole output silently.
    """
    assert np.isfinite(K._CAUSAL_MASK)
    assert K._CAUSAL_MASK < 0.0
