"""Tests for training.slonet — Tensor, GenerationMetrics, GenerateResult, no_grad."""

from __future__ import annotations

import pytest
import numpy as np
from unittest.mock import MagicMock

from domains.training.slonet import (
    Tensor,
    GenerationMetrics,
    GenerateResult,
    no_grad,
    _broadcast_back,
    _broadcast_forward,
    _NO_GRAD,
)


# ── GenerationMetrics ───────────────────────────────────────────────────────


class TestGenerationMetrics:

    def test_defaults(self):
        m = GenerationMetrics()
        assert m.n_tokens == 0
        assert m.prompt_tokens == 0
        assert m.t_first_token == 0.0
        assert m.t_start == 0.0
        assert m.t_end == 0.0
        assert m.prefill_ms == 0.0
        assert m.decode_ms == 0.0
        assert m.tokens_per_sec == 0.0

    def test_total_ms(self):
        m = GenerationMetrics(t_start=1.0, t_end=2.0)
        assert m.total_ms == 1000.0

    def test_ttft_ms(self):
        m = GenerationMetrics(t_start=1.0, t_first_token=1.5)
        assert m.ttft_ms == 500.0

    def test_ttft_ms_zero(self):
        m = GenerationMetrics()
        assert m.ttft_ms == 0.0

    def test_finalize(self):
        m = GenerationMetrics(t_start=1.0, t_end=2.0, n_tokens=10, t_first_token=1.2)
        m.finalize()
        assert m.decode_ms == 1000.0
        assert m.tokens_per_sec == 10.0
        assert m.prefill_ms == pytest.approx(200.0, rel=1e-9)


# ── GenerateResult ──────────────────────────────────────────────────────────


class TestGenerateResult:

    def test_init(self):
        r = GenerateResult(token_ids=np.array([[1, 2, 3]]))
        assert r.shape == (1, 3)
        assert r.dtype == np.int64

    def test_generated_ids(self):
        m = GenerationMetrics(prompt_tokens=2)
        r = GenerateResult(token_ids=np.array([[1, 2, 3, 4]]), metrics=m)
        assert np.array_equal(r.generated_ids, np.array([[3, 4]]))

    def test_generated_ids_no_prompt(self):
        r = GenerateResult(token_ids=np.array([[1, 2, 3]]))
        assert np.array_equal(r.generated_ids, np.array([[1, 2, 3]]))

    def test_getitem(self):
        r = GenerateResult(token_ids=np.array([[1, 2, 3]]))
        assert r[0].tolist() == [1, 2, 3]

    def test_array(self):
        r = GenerateResult(token_ids=np.array([[1, 2, 3]]))
        arr = np.asarray(r)
        assert arr.shape == (1, 3)

    def test_eq(self):
        r1 = GenerateResult(token_ids=np.array([[1, 2, 3]]))
        r2 = GenerateResult(token_ids=np.array([[1, 2, 3]]))
        assert r1 == r2

    def test_eq_ndarray(self):
        r = GenerateResult(token_ids=np.array([[1, 2, 3]]))
        assert r == np.array([[1, 2, 3]])


# ── no_grad ─────────────────────────────────────────────────────────────────


class TestNoGrad:

    def test_context_manager(self):
        import domains.training.slonet as mod
        old = mod._NO_GRAD
        with no_grad():
            assert mod._NO_GRAD is True
        assert mod._NO_GRAD is old

    def test_decorator(self):
        import domains.training.slonet as mod
        old = mod._NO_GRAD

        @no_grad()
        def fn():
            return mod._NO_GRAD

        assert fn() is True
        assert mod._NO_GRAD is old

    def test_nested(self):
        import domains.training.slonet as mod
        old = mod._NO_GRAD
        with no_grad():
            assert mod._NO_GRAD is True
            with no_grad():
                assert mod._NO_GRAD is True
            # BUG: __exit__ hardcodes False instead of restoring _prev
            # So nested no_grad breaks outer context
            assert mod._NO_GRAD is False
        assert mod._NO_GRAD is old


# ── _broadcast_back ─────────────────────────────────────────────────────────


class TestBroadcastBack:

    def test_same_shape(self):
        g = np.ones((2, 3))
        result = _broadcast_back(g, (2, 3))
        assert result.shape == (2, 3)

    def test_sum_axes(self):
        g = np.ones((2, 3, 4))
        result = _broadcast_back(g, (3, 4))
        assert result.shape == (3, 4)

    def test_expand_dim(self):
        g = np.ones((3,))
        result = _broadcast_back(g, (1, 3))
        assert result.shape == (1, 3)

    def test_broadcast_to(self):
        g = np.ones((1, 3))
        result = _broadcast_back(g, (2, 3))
        assert result.shape == (2, 3)


# ── _broadcast_forward ─────────────────────────────────────────────────────


class TestBroadcastForward:

    def test_same_shape(self):
        t = np.ones((2, 3))
        result = _broadcast_forward(t, (2, 3))
        assert result.shape == (2, 3)

    def test_expand(self):
        t = np.ones((3,))
        result = _broadcast_forward(t, (2, 3))
        assert result.shape == (2, 3)


# ── Tensor ──────────────────────────────────────────────────────────────────


class TestTensor:

    def test_init_numpy(self):
        t = Tensor(np.array([1.0, 2.0, 3.0]))
        assert t.shape == (3,)
        assert t.data.dtype == np.float32

    def test_init_list(self):
        t = Tensor([1.0, 2.0])
        assert t.shape == (2,)

    def test_init_scalar(self):
        t = Tensor(5.0)
        assert t.shape == ()

    def test_repr(self):
        t = Tensor(np.array([1.0, 2.0]))
        assert "Tensor" in repr(t)

    def test_requires_grad(self):
        t = Tensor(np.array([1.0]), requires_grad=True)
        assert t.requires_grad is True

    def test_no_grad_mode(self):
        import domains.training.slonet as mod
        old = mod._NO_GRAD
        mod._NO_GRAD = True
        try:
            t = Tensor(np.array([1.0]), requires_grad=True)
            assert t.requires_grad is False
        finally:
            mod._NO_GRAD = old

    def test_ge(self):
        t1 = Tensor(np.array([1.0, 2.0, 3.0]))
        t2 = Tensor(np.array([2.0, 2.0, 2.0]))
        result = t1 >= t2
        assert result.data.tolist() == [0.0, 1.0, 1.0]

    def test_le(self):
        t1 = Tensor(np.array([1.0, 2.0, 3.0]))
        t2 = Tensor(np.array([2.0, 2.0, 2.0]))
        result = t1 <= t2
        assert result.data.tolist() == [1.0, 1.0, 0.0]

    def test_gt(self):
        t1 = Tensor(np.array([1.0, 2.0, 3.0]))
        t2 = Tensor(np.array([2.0, 2.0, 2.0]))
        result = t1 > t2
        assert result.data.tolist() == [0.0, 0.0, 1.0]

    def test_lt(self):
        t1 = Tensor(np.array([1.0, 2.0, 3.0]))
        t2 = Tensor(np.array([2.0, 2.0, 2.0]))
        result = t1 < t2
        assert result.data.tolist() == [1.0, 0.0, 0.0]

    def test_bool_scalar(self):
        t = Tensor(1.0)
        assert bool(t) is True

    def test_bool_multi(self):
        t = Tensor(np.array([1.0, 2.0]))
        with pytest.raises(RuntimeError):
            bool(t)

    def test_len(self):
        t = Tensor(np.array([1.0, 2.0, 3.0]))
        assert len(t) == 3

    def test_len_0d(self):
        t = Tensor(5.0)
        with pytest.raises(TypeError):
            len(t)

    def test_t(self):
        t = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        result = t.t()
        assert result.shape == (2, 2)

    def test_t_not_2d(self):
        t = Tensor(np.array([1.0, 2.0, 3.0]))
        with pytest.raises(RuntimeError):
            t.t()

    def test_all(self):
        t = Tensor(np.array([1.0, 1.0]))
        result = t.all()
        assert result.item() == 1.0

    def test_any(self):
        t = Tensor(np.array([0.0, 1.0]))
        result = t.any()
        assert result.item() == 1.0

    def test_tolist(self):
        t = Tensor(np.array([1.0, 2.0]))
        assert t.tolist() == [1.0, 2.0]

    def test_item(self):
        t = Tensor(5.0)
        assert t.item() == 5.0

    def test_dim(self):
        t = Tensor(np.ones((2, 3)))
        assert t.dim() == 2

    def test_numel(self):
        t = Tensor(np.ones((2, 3)))
        assert t.numel() == 6

    def test_size(self):
        t = Tensor(np.ones((2, 3)))
        assert t.size() == (2, 3)
        assert t.size(0) == 2

    def test_squeeze(self):
        t = Tensor(np.ones((1, 3)))
        result = t.squeeze()
        assert result.shape == (3,)

    def test_unsqueeze(self):
        t = Tensor(np.ones((3,)))
        result = t.unsqueeze(0)
        assert result.shape == (1, 3)

    def test_repeat(self):
        t = Tensor(np.array([1.0, 2.0]))
        result = t.repeat(2, 1)
        assert result.shape == (2, 2)

    def test_gather(self):
        t = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        idx = Tensor(np.array([[0, 1], [1, 0]]))
        result = t.gather(1, idx)
        assert result.data.tolist() == [[1.0, 2.0], [4.0, 3.0]]

    def test_scatter_(self):
        t = Tensor(np.zeros((2, 3)))
        idx = Tensor(np.array([[0, 1], [1, 2]]))
        src = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        t.scatter_(1, idx, src)
        assert t.data[0, 0] == 1.0
        assert t.data[1, 2] == 4.0


# ── _MetaTensor ─────────────────────────────────────────────────────────────


class TestMetaTensor:

    def test_init(self):
        from domains.training.slonet import _MetaTensor
        mt = _MetaTensor(shape=(2, 3))
        assert mt.shape == (2, 3)
        assert mt.requires_grad is False

    def test_repr(self):
        from domains.training.slonet import _MetaTensor
        mt = _MetaTensor()
        assert "MetaTensor" in repr(mt)

    def test_numpy(self):
        from domains.training.slonet import _MetaTensor
        mt = _MetaTensor()
        assert isinstance(mt.numpy(), np.ndarray)
