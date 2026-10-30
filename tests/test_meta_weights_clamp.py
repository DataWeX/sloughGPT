"""Meta-weight nudges must stay inside the request schema bounds.

``_apply_meta_weights`` merges feedback nudges as ``value + nudge`` with no
clamp, so a benign default can leave the schema: e.g. top_p 1.0 + 0.05 = 1.05
violates ``GenerateRequest.top_p``'s ``le=1.0`` and is rejected downstream;
``top_k`` can even become a float. Explicit params (``model_fields_set``) pass
through verbatim — pydantic already validated them; only nudged values are
clamped.

Bounds are pinned to ``GenerateRequest`` (apps/api/server/routers/inference.py)
by a drift test, mirroring the defaults-contract pattern in
``test_meta_weights_contract.py``.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest

from apps.api.server.routers.inference import GenerateRequest, _apply_meta_weights

NEUTRAL = SimpleNamespace(temperature=0.7, top_p=0.85, top_k=40, repetition_penalty=1.15)


class _FakeManager:
    """get_adjustment returns whatever extreme weights the test wants."""

    def __init__(self, adj: SimpleNamespace) -> None:
        self._adj = adj
        self.neutral_weights = NEUTRAL

    def get_adjustment(self, **_kwargs: object) -> SimpleNamespace:
        return self._adj


@pytest.fixture
def extreme_manager(monkeypatch: pytest.MonkeyPatch):
    def _install(temp: float, top_p: float, top_k: float, rep: float):
        from domain import feedback

        adj = SimpleNamespace(temperature=temp, top_p=top_p, top_k=top_k, repetition_penalty=rep)
        monkeypatch.setattr(feedback, "get_meta_weight_manager", lambda: _FakeManager(adj))

    return _install


def _fresh_user_id() -> str:
    # Nudges are cached per (user_id, hash(message)) for 5s — a unique user id
    # keeps tests independent without touching the shared cache under its lock.
    return f"clamp-test-{uuid.uuid4()}"


class TestMetaWeightClamps:
    def test_nudged_values_clamped_to_schema_bounds(self, extreme_manager):
        extreme_manager(temp=3.0, top_p=1.2, top_k=5.0, rep=0.2)
        out = _apply_meta_weights(
            temperature=0.7,
            top_p=1.0,  # default left non-explicit: nudge may move it
            top_k=40,
            repetition_penalty=1.15,
            user_message="a unique message for clamp test A",
            user_id=_fresh_user_id(),
        )
        assert out["temperature"] == pytest.approx(2.0)
        assert out["top_p"] == pytest.approx(1.0)
        assert out["top_k"] == 5
        assert isinstance(out["top_k"], int)
        assert out["repetition_penalty"] == pytest.approx(0.5)

    def test_nudged_top_k_is_int_even_when_nudge_is_fractional(self, extreme_manager):
        extreme_manager(temp=0.7, top_p=0.85, top_k=39.5, rep=1.15)
        out = _apply_meta_weights(
            temperature=0.7,
            top_p=0.85,
            top_k=40,
            repetition_penalty=1.15,
            user_message="a unique message for clamp test B",
            user_id=_fresh_user_id(),
        )
        assert isinstance(out["top_k"], int)
        assert 0 <= out["top_k"] <= 500

    def test_explicit_params_pass_through_unclamped_intent(self, extreme_manager):
        # Explicit top_p is user intent: pydantic already bounded it, nudge
        # must not touch it even though the manager is extreme.
        extreme_manager(temp=3.0, top_p=1.2, top_k=5.0, rep=0.2)
        out = _apply_meta_weights(
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.15,
            user_message="a unique message for clamp test C",
            user_id=_fresh_user_id(),
            explicit={"top_p"},
        )
        assert out["top_p"] == pytest.approx(0.9)

    def test_bounds_match_generate_request_schema(self):
        # Drift gate: _META_WEIGHT_BOUNDS must mirror GenerateRequest Fields.
        from apps.api.server.routers.inference import _META_WEIGHT_BOUNDS

        for param, (lo, hi) in _META_WEIGHT_BOUNDS.items():
            field = GenerateRequest.model_fields[param]
            metas = {type(m).__name__: m for m in field.metadata}
            if "Ge" in metas:
                assert metas["Ge"].ge == lo, f"{param} lower bound drifted"
            if "Le" in metas:
                assert metas["Le"].le == hi, f"{param} upper bound drifted"
