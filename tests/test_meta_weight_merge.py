"""
Tests for feedback meta-weight merging and the OpenAI-compatible /v1/models route.

The meta-weight manager returns ABSOLUTE weights built from its own defaults
(0.7/1.15/0.85/40). The chat pipeline must apply feedback as a bounded delta
on the CALLER's parameters instead of replacing them — otherwise requests lose
their sampling settings and saturated per-user boosts force every chat onto
clamped extremes (temp 1.5, top_k 5, warning spam).
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from domain.feedback import MetaWeights


@pytest.fixture(autouse=True)
def _clean_meta_cache():
    from apps.api.server.routers import inference as inf

    inf._META_WEIGHT_CACHE.clear()
    inf._META_WARNED_EXTREME = None
    yield
    inf._META_WEIGHT_CACHE.clear()
    inf._META_WARNED_EXTREME = None


def _apply(
    adj,
    *,
    temperature=0.7,
    top_p=0.9,
    top_k=40,
    repetition_penalty=1.0,
    msg="message",
):
    from apps.api.server.routers.inference import _apply_meta_weights

    with patch("domain.feedback.get_meta_weight_manager") as mgr:
        mgr.return_value.get_adjustment.return_value = adj
        return _apply_meta_weights(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            user_message=msg,
        )


class TestMetaWeightMerge:
    def test_no_signal_passes_caller_params_through(self):
        """No feedback signal (adj == baseline) → caller's params unchanged."""
        out = _apply(MetaWeights())
        assert out == {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "repetition_penalty": 1.0,
        }

    def test_saturated_positive_feedback_is_bounded_delta(self):
        """Clamped-high manager weights bias the request; they don't replace it."""
        saturated = MetaWeights(temperature=1.5, repetition_penalty=0.8, top_p=1.0, top_k=5)
        out = _apply(saturated, msg="sat-up")
        assert out["temperature"] == pytest.approx(1.0)  # 0.7 + 0.3 (delta +0.8 capped)
        assert out["top_p"] == pytest.approx(1.0)  # 0.9 + 0.1 (delta +0.15 capped)
        assert out["top_k"] == 30  # 40 - 10 (delta -35 capped)
        assert out["repetition_penalty"] == pytest.approx(0.85)  # 1.0 - 0.15 (delta capped)

    def test_saturated_negative_feedback_is_bounded_delta(self):
        cold = MetaWeights(temperature=0.1, repetition_penalty=1.3, top_p=0.1, top_k=200)
        out = _apply(
            cold,
            temperature=1.0,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.0,
            msg="sat-down",
        )
        assert out["temperature"] == pytest.approx(0.7)  # 1.0 - 0.3
        assert out["top_p"] == pytest.approx(0.8)  # 0.9 - 0.1
        assert out["top_k"] == 50  # 40 + 10
        assert out["repetition_penalty"] == pytest.approx(1.15)  # 1.0 + 0.15

    def test_none_params_pass_through(self):
        out = _apply(
            MetaWeights(),
            temperature=None,
            top_p=None,
            top_k=None,
            repetition_penalty=None,
            msg="none",
        )
        assert out == {
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "repetition_penalty": None,
        }

    def test_extreme_warning_emitted_once_per_state(self):
        from apps.api.server.routers import inference as inf

        saturated = MetaWeights(temperature=1.5, repetition_penalty=0.8, top_p=1.0, top_k=5)
        with patch.object(inf.logger, "warning") as warn:
            # user temp 1.0 + 0.3 = 1.3 (> 1.2) and top_k 10 - 10 = 5 (< 10) → extreme
            _apply(saturated, temperature=1.0, top_k=10, msg="warn-1")
            _apply(saturated, temperature=1.0, top_k=10, msg="warn-2")
        extreme_calls = [c for c in warn.call_args_list if "extreme values" in str(c.args[0])]
        assert len(extreme_calls) == 1

    def test_feedback_applies_even_when_cache_hits(self):
        """The merge must run per call so request params stay exact (cache holds adj)."""
        saturated = MetaWeights(temperature=1.5, repetition_penalty=0.8, top_p=1.0, top_k=5)
        first = _apply(saturated, msg="cache-me")
        second = _apply(
            saturated,
            temperature=0.2,
            top_p=0.5,
            top_k=100,
            repetition_penalty=1.2,
            msg="cache-me",
        )
        assert first["temperature"] == pytest.approx(1.0)
        assert second["temperature"] == pytest.approx(0.5)  # 0.2 + 0.3, not first's 1.0
        assert second["top_k"] == 90  # 100 - 10


class TestV1ModelsEndpoint:
    @pytest.fixture
    def client(self):
        from apps.api.server.routers.inference import router
        from tests.conftest import build_test_app

        return TestClient(build_test_app(router))

    def test_lists_models_in_openai_shape(self, client):
        ctrl = MagicMock()
        ctrl.get_current_model.return_value = {"model_id": "Qwen/Qwen2.5-0.5B-Instruct"}
        ctrl.list_hf_models.return_value = [{"model_id": "other-model"}]
        with patch("controllers.models.get_models_controller", return_value=ctrl):
            resp = client.get("/v1/models")
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "list"
        ids = [m["id"] for m in body["data"]]
        assert ids == ["Qwen/Qwen2.5-0.5B-Instruct", "other-model"]
        assert all(m["object"] == "model" for m in body["data"])

    def test_degrades_to_state_model_type(self, client):
        with (
            patch("controllers.models.get_models_controller", side_effect=RuntimeError("boom")),
            patch("state.model_type", "fallback-model"),
        ):
            resp = client.get("/v1/models")
        assert resp.status_code == 200
        assert [m["id"] for m in resp.json()["data"]] == ["fallback-model"]

    def test_empty_listing_still_valid(self, client):
        with (
            patch("controllers.models.get_models_controller", side_effect=RuntimeError("boom")),
            patch("state.model_type", None),
        ):
            resp = client.get("/v1/models")
        assert resp.status_code == 200
        assert resp.json() == {"object": "list", "data": []}
