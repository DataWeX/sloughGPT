"""Meta-weight nudges: schema bounds at the wire, ONE safe band at the seam.

``_apply_meta_weights`` merges feedback nudges as ``value + nudge``. Two
different facts govern the result (card 20261101_182428):

- ``_SCHEMA_BOUNDS`` — wire validation, pinned to ``GenerateRequest`` Field
  constraints by a drift test (a top_p of 1.05 would be rejected
  downstream; ``top_k`` must stay an int).
- ``domain.feedback.SAFE_BAND`` — the ONE generation-safe authority,
  applied to every feedback-moved non-explicit value inside the merge loop
  itself, so a cache hit, a mock or an unbanded producer cannot restore
  the repetition-loop hazard (rep < 1 REWARDS emitted tokens).

Explicit params (``model_fields_set``) pass through verbatim — pydantic
already validated them; only nudged values are clamped. A silent feedback
layer (nudge == 0) never bands user configuration, and a base already
outside the band makes feedback abstain entirely.
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


@pytest.fixture(autouse=True)
def _nudge_enabled(monkeypatch: pytest.MonkeyPatch):
    """The sampler-side nudge ships dark (SLO_META_WEIGHT_NUDGE=0).

    These tests assert the nudge arithmetic, so they opt in explicitly. The
    default-off behaviour has its own tests in TestNudgeIsDarkByDefault.
    """
    monkeypatch.setenv("SLO_META_WEIGHT_NUDGE", "1")


def _fresh_user_id() -> str:
    # Nudges are cached per (user_id, hash(message)) for 5s — a unique user id
    # keeps tests independent without touching the shared cache under its lock.
    return f"clamp-test-{uuid.uuid4()}"


class TestMetaWeightClamps:
    def test_nudged_values_clamped_to_safe_band(self, extreme_manager):
        # The extreme manager simulates an unbanded producer (mock/refactor);
        # the USE seam must still land inside SAFE_BAND: temp -> 1.1,
        # rep -> 1.0, top_p -> 1.0, top_k -> 10 (all tighter than the
        # schema bounds the old contract asserted: 2.0/0.5/1.0/5).
        extreme_manager(temp=3.0, top_p=1.2, top_k=5.0, rep=0.2)
        out = _apply_meta_weights(
            temperature=0.7,
            top_p=1.0,  # default left non-explicit: nudge may move it
            top_k=40,
            repetition_penalty=1.15,
            user_message="a unique message for clamp test A",
            user_id=_fresh_user_id(),
        )
        assert out["temperature"] == pytest.approx(1.1)
        assert out["top_p"] == pytest.approx(1.0)
        assert out["top_k"] == 10
        assert isinstance(out["top_k"], int)
        assert out["repetition_penalty"] == pytest.approx(1.0)

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
        # Drift gate: _SCHEMA_BOUNDS must mirror GenerateRequest Fields.
        # (Schema validation only — generation safety is SAFE_BAND's job.)
        from apps.api.server.routers.inference import _SCHEMA_BOUNDS

        for param, (lo, hi) in _SCHEMA_BOUNDS.items():
            field = GenerateRequest.model_fields[param]
            metas = {type(m).__name__: m for m in field.metadata}
            if "Ge" in metas:
                assert metas["Ge"].ge == lo, f"{param} lower bound drifted"
            if "Le" in metas:
                assert metas["Le"].le == hi, f"{param} upper bound drifted"


class TestNudgeIsDarkByDefault:
    """Feedback steers CONTEXT, not sampling (card 20261103_071231).

    The sampler nudge duplicated a responsibility ``domain/context`` already
    owns one layer up, and an unbounded integrator turned 473 thumbs-up into
    ``repetition_penalty 0.8`` — a value that rewards tokens the model already
    emitted, so a finished answer replayed until ``max_tokens``.
    """

    REQUEST = {
        "temperature": 0.7,
        "top_p": 0.85,
        "top_k": 40,
        "repetition_penalty": 1.15,
    }

    def _call(self, user_id: str) -> dict:
        return _apply_meta_weights(
            **self.REQUEST,
            user_message="a unique message for the dark-nudge test",
            user_id=user_id,
        )

    @pytest.mark.parametrize("value", [None, "0", "false", "off", ""])
    def test_disabled_returns_the_callers_parameters_untouched(
        self, monkeypatch, extreme_manager, value
    ):
        if value is None:
            monkeypatch.delenv("SLO_META_WEIGHT_NUDGE", raising=False)
        else:
            monkeypatch.setenv("SLO_META_WEIGHT_NUDGE", value)
        # An extreme manager must have no effect at all while the gate is shut.
        extreme_manager(temp=3.0, top_p=1.2, top_k=5.0, rep=0.2)

        assert self._call(_fresh_user_id()) == self.REQUEST

    def test_enabled_applies_the_nudge(self, monkeypatch, extreme_manager):
        monkeypatch.setenv("SLO_META_WEIGHT_NUDGE", "1")
        extreme_manager(temp=3.0, top_p=1.2, top_k=5.0, rep=0.2)

        out = self._call(_fresh_user_id())

        assert out["temperature"] == pytest.approx(1.1)
        assert out["repetition_penalty"] == pytest.approx(1.0)


class TestSafeBandAtTheUseSeam:
    """The band must hold on EVERY path a nudge can arrive on.

    ``get_adjustment()`` bands its output, but the transport stores the
    resulting *nudge* for 5s and merges it later — a cache hit (or a mock,
    or a future unbanded producer) never calls the producer at all, so a
    producer-side-only clamp leaves the repetition-loop hazard one refactor
    away (card 20261101_182428).
    """

    REQUEST = {
        "temperature": 0.7,
        "top_p": 0.85,
        "top_k": 40,
        "repetition_penalty": 1.15,
    }

    def test_cached_unbanded_nudge_is_banded_at_merge(self, monkeypatch):
        """A cache hit carrying a pre-band nudge still lands in the band."""
        import time

        from apps.api.server.routers.inference import _META_WEIGHT_CACHE

        monkeypatch.setenv("SLO_META_WEIGHT_NUDGE", "1")
        user_id = f"clamp-test-{uuid.uuid4()}"
        message = "a unique message for the cache-hit band test"
        key = f"{user_id}:{hash(message)}"
        # Simulate a nudge computed by an UNBANDED producer (pre-band cache
        # entry, mock, refactor): the delta alone would drive temp to 5.7,
        # rep to 0.15 — deep outside SAFE_BAND.
        _META_WEIGHT_CACHE[key] = (
            time.monotonic(),
            {
                "temperature": 5.0,
                "top_p": 0.5,
                "top_k": -100.0,
                "repetition_penalty": -1.0,
            },
        )

        def _producer_must_not_run(**_kwargs):
            raise AssertionError("cache hit must not call get_adjustment()")

        from domain import feedback

        monkeypatch.setattr(
            feedback,
            "get_meta_weight_manager",
            lambda: SimpleNamespace(
                get_adjustment=_producer_must_not_run,
                neutral_weights=NEUTRAL,
            ),
        )

        try:
            out = _apply_meta_weights(
                **self.REQUEST,
                user_message=message,
                user_id=user_id,
            )
        finally:
            _META_WEIGHT_CACHE.pop(key, None)

        assert out["temperature"] == pytest.approx(1.1)
        assert out["top_p"] == pytest.approx(1.0)
        assert out["top_k"] == 10
        assert out["repetition_penalty"] == pytest.approx(1.0)

    def test_base_outside_band_makes_feedback_abstain(self, monkeypatch, extreme_manager):
        """A scope/global default outside the band is user config, not
        feedback's to move — the nudge is dropped, value untouched."""
        monkeypatch.setenv("SLO_META_WEIGHT_NUDGE", "1")
        # 1.5 is schema-valid but above SAFE_BAND's temp ceiling (1.1), as a
        # conversation-scope default can be; adj 1.1 => nudge +0.4.
        extreme_manager(temp=1.1, top_p=0.85, top_k=40, rep=1.15)
        out = _apply_meta_weights(
            temperature=1.5,
            top_p=0.85,
            top_k=40,
            repetition_penalty=1.15,
            user_message="a unique message for the abstain test",
            user_id=_fresh_user_id(),
        )
        assert out["temperature"] == pytest.approx(1.5)

    def test_silent_feedback_never_bands_user_configuration(self, monkeypatch, extreme_manager):
        """nudge == 0 (adj == neutral): no feedback signal, so no banding —
        a user's out-of-band default must survive untouched."""
        monkeypatch.setenv("SLO_META_WEIGHT_NUDGE", "1")
        extreme_manager(temp=0.7, top_p=0.85, top_k=40, rep=1.15)  # adj == neutral
        out = _apply_meta_weights(
            temperature=1.5,  # outside band, but feedback is silent
            top_p=0.85,
            top_k=40,
            repetition_penalty=1.15,
            user_message="a unique message for the silent-feedback test",
            user_id=_fresh_user_id(),
        )
        assert out["temperature"] == pytest.approx(1.5)
